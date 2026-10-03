"""Reload an ISIC checkpoint and generate held-out prediction evidence.

Run from the repository root after a completed training run::

    /Users/alexanderson/venvs/comp3710/bin/python -m \
      recognition.Siamese_ISIC2020_Melanoma.predict \
      --checkpoint data/runs/baseline_seed42/checkpoint.pt

This command never trains a model, changes a patient split, resamples a
Siamese gallery, or chooses a new decision threshold. It applies the
validation-derived policy saved in the checkpoint.
"""

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from functools import partial
from math import isfinite
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from recognition.Siamese_ISIC2020_Melanoma.dataset import (
    ISICImageDataset,
    SplitIndices,
    load_or_create_split_manifest,
    repository_root,
)
from recognition.Siamese_ISIC2020_Melanoma.modules import (
    BaseCNN,
    ModelName,
    SiameseNetwork,
    build_model,
)
from recognition.Siamese_ISIC2020_Melanoma.train import score_base, score_siamese
from recognition.Siamese_ISIC2020_Melanoma.utils import (
    apply_triage_rule,
    load_checkpoint,
    select_device,
)
from recognition.Siamese_ISIC2020_Melanoma.visualisation import (
    Prediction,
    plot_failure_cases,
    plot_prediction_score_distribution,
)


def parse_prediction_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse paths and runtime settings for checkpoint prediction."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path, metavar="PATH")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        metavar="PATH",
        help="Training metrics JSON; defaults to checkpoint.pt's sibling metrics.json.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Extracted image directory; defaults to repository data/.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="ISIC GroundTruth_v2 CSV; defaults to repository data/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Prediction CSV and SVG directory; defaults beside the checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    return parser.parse_args(argv)


def load_run_for_prediction(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, object]]:
    """Load a saved model through the project model factory in evaluation mode.

    Parameters
    ----------
    checkpoint_path : pathlib.Path
        Saved ``checkpoint.pt`` produced by ``train.py``.
    device : torch.device
        Device selected for inference.

    Returns
    -------
    torch.nn.Module
        Reloaded model in evaluation mode.
    dict of str to object
        Validated checkpoint contents.
    """

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model_name = _checkpoint_model_name(checkpoint)
    model_kwargs = checkpoint.get("model_kwargs", {})
    if not isinstance(model_kwargs, Mapping) or model_kwargs:
        raise ValueError("Only empty model_kwargs are supported by this project")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise ValueError("Checkpoint state_dict must map parameter names to tensors")
    model = build_model(model_name)
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state_dict))
    return model.to(device).eval(), checkpoint


def select_failure_cases(
    predictions: list[Prediction],
    decision_rule: Mapping[str, float],
    maximum_cases: int = 5,
) -> list[Prediction]:
    """Select false-negative and false-positive cases, then near-threshold cases.

    A false negative is a melanoma classified as benign. A false positive is a
    benign lesion classified as melanoma. Remaining places are filled by the
    cases closest to the frozen validation threshold.

    Parameters
    ----------
    predictions : list of Prediction
        Held-out scores paired with the saved decisions.
    decision_rule : mapping of str to float
        Validation-derived threshold and referral margin.
    maximum_cases : int, default=5
        Maximum number of images shown in the SVG montage.

    Returns
    -------
    list of Prediction
        False-negative and false-positive cases when available, followed by
        near-threshold held-out cases.
    """

    if maximum_cases < 1:
        raise ValueError("maximum_cases must be positive")
    threshold = _required_float(decision_rule, "threshold", "decision rule")
    selected: list[Prediction] = []
    false_negative = next(
        (
            prediction
            for prediction in predictions
            if prediction.actual == 1 and prediction.decision == "benign"
        ),
        None,
    )
    false_positive = next(
        (
            prediction
            for prediction in predictions
            if prediction.actual == 0 and prediction.decision == "melanoma"
        ),
        None,
    )
    for prediction in (false_negative, false_positive):
        if prediction is not None:
            selected.append(prediction)
    selected = selected[:maximum_cases]
    selected_names = {prediction.image_name for prediction in selected}
    nearby = sorted(
        (
            prediction
            for prediction in predictions
            if prediction.image_name not in selected_names
        ),
        key=partial(_prediction_distance_to_threshold, threshold=threshold),
    )
    selected.extend(nearby[: maximum_cases - len(selected)])
    return selected


def _prediction_distance_to_threshold(
    prediction: Prediction, threshold: float
) -> tuple[float, str]:
    """Return deterministic closeness of one prediction to a decision boundary."""

    return abs(prediction.score - threshold), prediction.image_name


def create_prediction_figures(
    predictions: list[Prediction],
    decision_rule: Mapping[str, float],
    score_label: str,
    output_dir: Path,
) -> list[Path]:
    """Write score-distribution and selected-case SVG evidence for a run.

    Parameters
    ----------
    predictions : list of Prediction
        All held-out predictions from the reloaded model.
    decision_rule : mapping of str to float
        Validation-only triage rule saved in the checkpoint.
    score_label : str
        Human-readable description of the model score.
    output_dir : pathlib.Path
        Destination directory for generated SVG figures.

    Returns
    -------
    list of pathlib.Path
        Score-distribution and failure-case SVG paths, in that order.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    failures = select_failure_cases(predictions, decision_rule)
    return [
        plot_prediction_score_distribution(
            predictions,
            decision_rule,
            score_label,
            output_dir / "test_score_distribution.svg",
        ),
        plot_failure_cases(failures, output_dir / "failure_cases.svg"),
    ]


def _checkpoint_model_name(checkpoint: Mapping[str, object]) -> ModelName:
    """Read one supported model name from checkpoint metadata."""

    model_name = checkpoint.get("model_name")
    if model_name not in ("base", "siamese"):
        raise ValueError("Checkpoint model_name must be 'base' or 'siamese'")
    return model_name


def _load_metrics(metrics_path: Path, model_name: ModelName) -> dict[str, object]:
    """Read metrics and confirm that they belong to the checkpoint's model."""

    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics JSON does not exist: {metrics_path}")
    with metrics_path.open(encoding="utf-8") as stream:
        loaded: Any = json.load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("Metrics JSON must contain an object")
    metrics = cast(dict[str, object], loaded)
    if metrics.get("model") != model_name:
        raise ValueError("Metrics JSON model does not match checkpoint model_name")
    return metrics


def _build_test_subset(
    checkpoint: Mapping[str, object],
    images_dir: Path | None,
    metadata_csv: Path | None,
) -> tuple[ISICImageDataset, Subset, SplitIndices]:
    """Recreate the exact held-out dataset view recorded by training."""

    preprocessing = _required_mapping(checkpoint, "preprocessing", "checkpoint")
    image_size = _required_int(preprocessing, "image_size", "preprocessing")
    color_mode = _required_text(preprocessing, "color_mode", "preprocessing")
    split_settings = _required_mapping(checkpoint, "split_manifest", "checkpoint")
    manifest_path = _resolve_manifest_path(
        _required_text(split_settings, "path", "split manifest")
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Saved split manifest does not exist: {manifest_path}. "
            "Prediction will not create a replacement split."
        )
    dataset = ISICImageDataset(
        images_dir=images_dir,
        metadata_csv=metadata_csv,
        image_size=image_size,
        color_mode=color_mode,
        training=False,
    )
    split_indices = load_or_create_split_manifest(
        dataset,
        manifest_path,
        _required_float(split_settings, "validation_fraction", "split manifest"),
        _required_float(split_settings, "test_fraction", "split manifest"),
        _required_int(split_settings, "seed", "split manifest"),
    )
    return dataset, Subset(dataset, list(split_indices.test)), split_indices


def _rebuild_siamese_gallery(
    checkpoint: Mapping[str, object],
    dataset: ISICImageDataset,
    split_indices: SplitIndices,
) -> tuple[Subset, int]:
    """Rebuild the exact train-only gallery and its saved neighbour count."""

    gallery_names = checkpoint.get("gallery_image_names")
    if not isinstance(gallery_names, list) or not all(
        isinstance(name, str) for name in gallery_names
    ):
        raise ValueError("Siamese checkpoint needs a list of gallery_image_names")
    if not gallery_names or len(gallery_names) != len(set(gallery_names)):
        raise ValueError("Siamese gallery_image_names must be non-empty and unique")
    indices_by_name = {
        sample.image_name: index for index, sample in enumerate(dataset.samples)
    }
    gallery_indices: list[int] = []
    for name in gallery_names:
        index = indices_by_name.get(name)
        if index is None:
            raise ValueError(f"Saved gallery image is absent from metadata: {name}")
        gallery_indices.append(index)
    training_indices = set(split_indices.train)
    if any(index not in training_indices for index in gallery_indices):
        raise ValueError("Saved Siamese gallery includes a non-training image")
    scoring = _required_mapping(checkpoint, "scoring", "checkpoint")
    neighbors = _required_int(scoring, "neighbors", "scoring")
    return Subset(dataset, gallery_indices), neighbors


def _validated_decision_rule(checkpoint: Mapping[str, object]) -> dict[str, float]:
    """Return the saved validation-only threshold and referral margin."""

    saved_rule = _required_mapping(checkpoint, "decision_rule", "checkpoint")
    return {
        "threshold": _required_float(saved_rule, "threshold", "decision rule"),
        "referral_margin": _required_float(
            saved_rule, "referral_margin", "decision rule"
        ),
    }


def _predict_scores(
    model: nn.Module,
    checkpoint: Mapping[str, object],
    dataset: ISICImageDataset,
    test: Subset,
    split_indices: SplitIndices,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Score the fixed test partition using the checkpoint's frozen context."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    test_loader = DataLoader(test, batch_size=batch_size)
    if isinstance(model, BaseCNN):
        return score_base(model, test_loader, device)
    if isinstance(model, SiameseNetwork):
        gallery, neighbors = _rebuild_siamese_gallery(
            checkpoint, dataset, split_indices
        )
        gallery_loader = DataLoader(gallery, batch_size=batch_size)
        return score_siamese(model, test_loader, gallery_loader, device, neighbors)
    raise TypeError(f"Unsupported checkpoint model: {type(model).__name__}")


def _make_predictions(
    dataset: ISICImageDataset,
    test: Subset,
    labels: Sequence[int],
    scores: Sequence[float],
    decision_rule: Mapping[str, float],
) -> list[Prediction]:
    """Pair deterministic test metadata with model scores and frozen decisions."""

    if len(test.indices) != len(labels) or len(labels) != len(scores):
        raise ValueError("Test metadata, labels, and scores must have equal lengths")
    predictions: list[Prediction] = []
    for index, label, score in zip(test.indices, labels, scores):
        numeric_score = float(score)
        if label not in (0, 1) or not isfinite(numeric_score):
            raise ValueError("Prediction labels must be binary and scores finite")
        sample = dataset.samples[index]
        predictions.append(
            Prediction(
                image_name=sample.image_name,
                patient_id=sample.patient_id,
                image_path=sample.path,
                actual=int(label),
                score=numeric_score,
                decision=apply_triage_rule(numeric_score, decision_rule),
            )
        )
    return predictions


def _write_predictions(path: Path, predictions: Sequence[Prediction]) -> None:
    """Write and print one row for every held-out prediction."""

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("image_name", "patient_id", "actual", "score", "decision")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for prediction in predictions:
            writer.writerow(
                (
                    prediction.image_name,
                    prediction.patient_id,
                    prediction.actual,
                    prediction.score,
                    prediction.decision,
                )
            )
    stdout_writer = csv.writer(sys.stdout)
    stdout_writer.writerow(columns)
    for prediction in predictions:
        stdout_writer.writerow(
            (
                prediction.image_name,
                prediction.patient_id,
                prediction.actual,
                prediction.score,
                prediction.decision,
            )
        )


def _required_mapping(
    mapping: Mapping[str, object], name: str, context: str
) -> Mapping[str, object]:
    """Return a named mapping field or raise a schema-specific error."""

    value = mapping.get(name)
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} needs a mapping field named {name!r}")
    return cast(Mapping[str, object], value)


def _required_text(mapping: Mapping[str, object], name: str, context: str) -> str:
    """Return one non-empty text field from saved metadata."""

    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} needs a non-empty text field named {name!r}")
    return value


def _required_int(mapping: Mapping[str, object], name: str, context: str) -> int:
    """Return one positive integer field from saved metadata."""

    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} needs a positive integer field named {name!r}")
    return value


def _required_float(mapping: Mapping[str, object], name: str, context: str) -> float:
    """Return one finite numeric field from saved metadata."""

    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} needs a numeric field named {name!r}")
    numeric_value = float(value)
    if not isfinite(numeric_value):
        raise ValueError(f"{context} needs a finite field named {name!r}")
    return numeric_value


def _resolve_manifest_path(value: str) -> Path:
    """Resolve a saved split path relative to the repository when necessary."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else repository_root() / path


def _score_label(checkpoint: Mapping[str, object]) -> str:
    """Describe the saved model score without treating it as calibrated by default."""

    return (
        "Melanoma probability"
        if _checkpoint_model_name(checkpoint) == "base"
        else "Melanoma similarity score"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Reload a saved run, infer on its held-out partition, and write SVGs."""

    args = parse_prediction_args(argv)
    device = select_device(args.device)
    model, checkpoint = load_run_for_prediction(args.checkpoint, device)
    model_name = _checkpoint_model_name(checkpoint)
    metrics_path = args.metrics or args.checkpoint.parent / "metrics.json"
    _load_metrics(metrics_path, model_name)
    dataset, test, split_indices = _build_test_subset(
        checkpoint, args.images_dir, args.metadata_csv
    )
    decision_rule = _validated_decision_rule(checkpoint)
    scores, labels = _predict_scores(
        model,
        checkpoint,
        dataset,
        test,
        split_indices,
        args.batch_size,
        device,
    )
    predictions = _make_predictions(
        dataset, test, labels.tolist(), scores.tolist(), decision_rule
    )
    output_dir = args.output_dir or args.checkpoint.parent / "predictions"
    _write_predictions(output_dir / "test_predictions.csv", predictions)
    figures = create_prediction_figures(
        predictions, decision_rule, _score_label(checkpoint), output_dir
    )
    print(f"Reloaded {model_name} checkpoint on {device}", file=sys.stderr)
    for figure in figures:
        print(f"Saved {figure}", file=sys.stderr)


if __name__ == "__main__":
    main()
