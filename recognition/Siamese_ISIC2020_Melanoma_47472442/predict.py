"""Reload an ISIC checkpoint and generate held-out prediction evidence.

Run from the repository root after a completed training run::

    /Users/alexanderson/venvs/comp3710/bin/python -m \
      recognition.Siamese_ISIC2020_Melanoma_47472442.predict \
      --checkpoint data/runs/baseline_v2_seed42/checkpoint.pt

This command never trains a model, changes a patient split, resamples a
Siamese gallery, or chooses a new decision threshold. It applies the
saved policy. Prediction defaults to calibration; final patients require
``--partition final --evaluate-final --evaluation-note``.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Mapping, Sequence
from functools import partial
from math import isfinite
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from recognition.Siamese_ISIC2020_Melanoma_47472442.dataset import (
    ISICImageDataset,
    model_preprocessing,
    repository_root,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.modules import (
    BaseCNN,
    ModelName,
    SiameseNetwork,
    build_model,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.protocol import (
    load_protocol_manifest,
    protocol_identity,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.provenance import (
    file_sha256,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.train import (
    classification_metrics,
    score_base,
    score_siamese,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.utils import (
    apply_triage_rule,
    load_checkpoint,
    select_device,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.visualisation import (
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
    parser.add_argument(
        "--partition",
        choices=("selection", "calibration", "final"),
        default="calibration",
        help="Patient role to score; defaults to calibration.",
    )
    parser.add_argument(
        "--evaluate-final",
        action="store_true",
        help="Explicitly authorise final scoring of a frozen checkpoint.",
    )
    parser.add_argument(
        "--evaluation-note",
        default=None,
        help="Recorded reason for final scoring after model/configuration choices freeze.",
    )
    args = parser.parse_args(argv)
    if args.partition == "final":
        if (
            not args.evaluate_final
            or not args.evaluation_note
            or not args.evaluation_note.strip()
        ):
            parser.error(
                "Final evaluation requires --evaluate-final and --evaluation-note"
            )
    elif args.evaluate_final or args.evaluation_note is not None:
        parser.error("Final evaluation flags require --partition final")
    return args


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
    if checkpoint.get("schema_version") != 2:
        raise ValueError("Use a checkpoint produced by the current training pipeline")
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
    cases closest to the frozen decision threshold.

    Parameters
    ----------
    predictions : list of Prediction
        Held-out scores paired with the saved decisions.
    decision_rule : mapping of str to float
        Saved threshold and referral margin.
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
    partition: str,
) -> list[Path]:
    """Write score-distribution and selected-case SVG evidence for a run.

    Parameters
    ----------
    predictions : list of Prediction
        All held-out predictions from the reloaded model.
    decision_rule : mapping of str to float
        Frozen triage rule saved in the checkpoint.
    score_label : str
        Human-readable description of the model score.
    output_dir : pathlib.Path
        Destination directory for generated SVG figures.
    partition : str
        Saved role used to label the score-distribution output.

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
            output_dir / f"{partition}_score_distribution.svg",
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


def _validate_saved_run(
    checkpoint: Mapping[str, object], metrics: Mapping[str, object], config_path: Path
) -> None:
    """Bind a new checkpoint to its saved settings, metrics and calibration rule."""

    with config_path.open(encoding="utf-8") as stream:
        config: object = json.load(stream)
    if not isinstance(config, Mapping):
        raise ValueError("Saved effective_config.json must contain an object")
    config_hash = file_sha256(config_path)
    if (
        checkpoint.get("effective_config_sha256") != config_hash
        or metrics.get("effective_config_sha256") != config_hash
    ):
        raise ValueError("Saved configuration does not match checkpoint and metrics")
    if (
        metrics.get("schema_version") != 2
        or config.get("model") != _checkpoint_model_name(checkpoint)
        or config.get("seed") != metrics.get("seed")
    ):
        raise ValueError("Metrics and checkpoint do not match the saved configuration")
    if metrics.get("protocol") != checkpoint.get("protocol"):
        raise ValueError("Protocol identity differs between checkpoint and metrics")
    if checkpoint.get("preprocessing") != model_preprocessing():
        raise ValueError("Checkpoint preprocessing is not supported by this loader")
    scoring = _required_mapping(checkpoint, "scoring", "checkpoint")
    expected_neighbors = (
        config.get("neighbors") if config.get("model") == "siamese" else None
    )
    if scoring.get("neighbors") != expected_neighbors:
        raise ValueError("Checkpoint scoring differs from the saved configuration")
    if (
        checkpoint.get("decision_rule") != metrics.get("decision_rule")
        or checkpoint.get("decision_rule_partition") != "calibration"
        or metrics.get("decision_rule_partition") != "calibration"
    ):
        raise ValueError("The decision rule must come from calibration patients")
    if checkpoint.get("best_epoch") != metrics.get("best_epoch"):
        raise ValueError("Checkpoint and metrics disagree on the selected epoch")


@dataclass(frozen=True)
class PredictionData:
    """Deterministic prediction view and the only eligible reference indices."""

    dataset: ISICImageDataset
    subset: Subset
    fit_indices: tuple[int, ...]
    partition: str


def build_prediction_data(
    checkpoint: Mapping[str, object],
    images_dir: Path | None,
    metadata_csv: Path | None,
    partition: str,
) -> PredictionData:
    """Rebuild a saved role, validating fitting membership before scoring.

    Parameters
    ----------
    checkpoint : mapping of str to object
        Checkpoint metadata from the current training pipeline.
    images_dir, metadata_csv : pathlib.Path or None
        Optional relocated inputs; metadata must match the frozen identity.
    partition : {"selection", "calibration", "final"}
        Patient role to score using the saved decision rule.

    Returns
    -------
    PredictionData
        Verified prediction subset and fitting-only gallery eligibility.
    """

    role = partition
    if role not in ("selection", "calibration", "final"):
        raise ValueError("Prediction role must be selection, calibration or final")
    saved_protocol = _required_mapping(checkpoint, "protocol", "checkpoint")
    manifest = _resolve_manifest_path(
        _required_text(saved_protocol, "path", "protocol")
    )
    preprocessing = _required_mapping(checkpoint, "preprocessing", "checkpoint")
    dataset = ISICImageDataset(
        images_dir=images_dir,
        metadata_csv=metadata_csv,
        image_size=_required_int(preprocessing, "image_size", "preprocessing"),
        color_mode=_required_text(preprocessing, "color_mode", "preprocessing"),
        training=False,
    )
    indices = load_protocol_manifest(dataset, manifest)
    if dict(saved_protocol) != protocol_identity(manifest):
        raise ValueError(
            "Frozen protocol identity has changed since this run was fitted"
        )
    fit_patients = sorted({dataset.samples[index].patient_id for index in indices.fit})
    if checkpoint.get("fit_patient_ids") != fit_patients:
        raise ValueError("Checkpoint fitting patients differ from the frozen protocol")
    chosen_indices: tuple[int, ...] = getattr(indices, role)
    return PredictionData(
        dataset, Subset(dataset, list(chosen_indices)), indices.fit, role
    )


def _rebuild_siamese_gallery(
    checkpoint: Mapping[str, object],
    dataset: ISICImageDataset,
    fit_indices: Sequence[int],
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
    training_indices = set(fit_indices)
    if any(index not in training_indices for index in gallery_indices):
        raise ValueError("Saved Siamese gallery includes a non-fitting image")
    scoring = _required_mapping(checkpoint, "scoring", "checkpoint")
    neighbors = _required_int(scoring, "neighbors", "scoring")
    return Subset(dataset, gallery_indices), neighbors


def _validated_decision_rule(checkpoint: Mapping[str, object]) -> dict[str, float]:
    """Return the saved threshold and referral margin without recalibrating."""

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
    subset: Subset,
    fit_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Score the chosen patient role using the checkpoint's frozen context."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    loader = DataLoader(subset, batch_size=batch_size)
    if isinstance(model, BaseCNN):
        return score_base(model, loader, device)
    if isinstance(model, SiameseNetwork):
        gallery, neighbors = _rebuild_siamese_gallery(checkpoint, dataset, fit_indices)
        gallery_loader = DataLoader(gallery, batch_size=batch_size)
        return score_siamese(model, loader, gallery_loader, device, neighbors)
    raise TypeError(f"Unsupported checkpoint model: {type(model).__name__}")


def _make_predictions(
    dataset: ISICImageDataset,
    subset: Subset,
    labels: Sequence[int],
    scores: Sequence[float],
    decision_rule: Mapping[str, float],
) -> list[Prediction]:
    """Pair held-out metadata with model scores and frozen decisions."""

    if len(subset.indices) != len(labels) or len(labels) != len(scores):
        raise ValueError("Metadata, labels, and scores must have equal lengths")
    predictions: list[Prediction] = []
    for index, label, score in zip(subset.indices, labels, scores):
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


def _write_predictions(
    path: Path, predictions: Sequence[Prediction], threshold: float
) -> None:
    """Write and print one row for every held-out prediction."""

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("image_name", "patient_id", "actual", "score", "predicted", "decision")
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
                    int(prediction.score >= threshold),
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
                int(prediction.score >= threshold),
                prediction.decision,
            )
        )


def _required_mapping(
    mapping: Mapping[str, object], name: str, context: str
) -> Mapping[str, object]:
    """Return a named mapping field or raise a schema-specific error."""

    value = mapping.get(name)
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
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
        "Melanoma softmax score"
        if _checkpoint_model_name(checkpoint) == "base"
        else "Melanoma similarity score"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Reload a frozen run and explicitly score its permitted patient role."""

    args = parse_prediction_args(argv)
    device = select_device(args.device)
    model, checkpoint = load_run_for_prediction(args.checkpoint, device)
    model_name = _checkpoint_model_name(checkpoint)
    metrics_path = args.metrics or args.checkpoint.parent / "metrics.json"
    metrics = _load_metrics(metrics_path, model_name)
    _validate_saved_run(
        checkpoint, metrics, args.checkpoint.parent / "effective_config.json"
    )
    data = build_prediction_data(
        checkpoint, args.images_dir, args.metadata_csv, args.partition
    )
    decision_rule = _validated_decision_rule(checkpoint)
    output_dir = (
        args.output_dir or args.checkpoint.parent / "predictions" / data.partition
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    evaluation: dict[str, object] = {
        "schema_version": 1,
        "partition": data.partition,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "metrics_sha256": file_sha256(metrics_path),
        "protocol": checkpoint.get("protocol"),
        "effective_config_sha256": checkpoint.get("effective_config_sha256"),
        "final_evaluation_authorised": bool(args.evaluate_final),
        "evaluation_note": args.evaluation_note,
        "decision_rule": decision_rule,
        "device": str(device),
        "batch_size": args.batch_size,
    }
    evaluation_path = output_dir / "evaluation.json"
    _write_evaluation(evaluation_path, evaluation)
    scores, labels = _predict_scores(
        model,
        checkpoint,
        data.dataset,
        data.subset,
        data.fit_indices,
        args.batch_size,
        device,
    )
    predictions = _make_predictions(
        data.dataset, data.subset, labels.tolist(), scores.tolist(), decision_rule
    )
    prediction_path = output_dir / f"{data.partition}_predictions.csv"
    _write_predictions(prediction_path, predictions, decision_rule["threshold"])
    figures = create_prediction_figures(
        predictions,
        decision_rule,
        _score_label(checkpoint),
        output_dir,
        data.partition,
    )
    evaluation.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": classification_metrics(
                labels, scores, decision_rule["threshold"]
            ),
            "prediction_count": len(predictions),
            "predictions_sha256": file_sha256(prediction_path),
        }
    )
    _write_evaluation(evaluation_path, evaluation)
    print(
        f"Reloaded {model_name} on {device}; partition={data.partition}",
        file=sys.stderr,
    )
    for figure in figures:
        print(f"Saved {figure}", file=sys.stderr)


def _write_evaluation(path: Path, evaluation: Mapping[str, object]) -> None:
    """Persist the explicit evaluation intent and its eventual result."""

    with path.open("w", encoding="utf-8") as stream:
        json.dump(dict(evaluation), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
