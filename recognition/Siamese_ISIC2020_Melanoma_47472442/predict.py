"""Reload an ISIC checkpoint and generate held-out prediction evidence.

Run from the repository root after a completed training run::

    /Users/alexanderson/venvs/comp3710/bin/python -m \
      recognition.Siamese_ISIC2020_Melanoma_47472442.predict \
      --checkpoint data/runs/resnet18_baseline_imagenet_features512_30ep_seed42/checkpoint.pt

This command applies the saved classifier and decision policy without
training, changing patient splits, or selecting a new threshold. Prediction defaults to calibration; final patients require
``--partition final --evaluate-final --evaluation-note``.
Checkpoints must use the current 512-feature classifier architecture.
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from recognition.Siamese_ISIC2020_Melanoma_47472442.dataset import (
    ISICImageDataset,
    repository_root,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.modules import (
    MelanomaModel,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.protocol import (
    ProtocolIndices,
    load_protocol_manifest,
    protocol_identity,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.provenance import (
    file_sha256,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.train import (
    classification_metrics,
    score_model,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.utils import (
    apply_triage_rule,
    count_parameters,
    load_checkpoint,
    memory_usage,
    reset_peak_accelerator_memory,
    select_device,
    synchronised_time,
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
    check_final_evaluation(parser, args)
    return args


def check_final_evaluation(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Require explicit authorisation before scoring final patients."""

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


def load_run_for_prediction(
    checkpoint_path: Path, device: torch.device
) -> tuple[MelanomaModel, dict[str, Any]]:
    """Restore trained weights without downloading ImageNet initialisation.

    Parameters
    ----------
    checkpoint_path : pathlib.Path
        Saved ``checkpoint.pt`` produced by ``train.py``.
    device : torch.device
        Device selected for inference.

    Returns
    -------
    MelanomaModel
        Reloaded ResNet-18 model in evaluation mode.
    dict of str to object
        Saved checkpoint contents.
    """

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = MelanomaModel()
    model.load_state_dict(checkpoint["state_dict"])
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

    threshold = decision_rule["threshold"]
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


def validate_saved_run(
    checkpoint: Mapping[str, Any], metrics: Mapping[str, Any], config_path: Path
) -> None:
    """Check run identity without revalidating the saved file schemas."""

    config_hash = file_sha256(config_path)
    if (
        checkpoint["effective_config_sha256"] != config_hash
        or metrics["effective_config_sha256"] != config_hash
    ):
        raise ValueError("Run settings differ from the saved checkpoint or metrics")
    if (
        checkpoint["protocol"] != metrics["protocol"]
        or checkpoint["decision_rule"] != metrics["decision_rule"]
        or checkpoint["objective"] != metrics["objective"]
    ):
        raise ValueError("Checkpoint and metrics refer to different experiments")


@dataclass(frozen=True)
class PredictionData:
    """Deterministic image view for one saved patient role."""

    dataset: ISICImageDataset
    subset: Subset
    indices: ProtocolIndices
    partition: str


def build_prediction_data(
    checkpoint: Mapping[str, Any],
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
        Prediction subset checked against the saved fitting identity.
    """

    saved_protocol = checkpoint["protocol"]
    manifest = repository_root() / Path(saved_protocol["path"]).expanduser()
    dataset = ISICImageDataset(images_dir, metadata_csv)
    indices = load_protocol_manifest(dataset, manifest)
    if saved_protocol != protocol_identity(manifest):
        raise ValueError("The frozen patient protocol has changed since training")
    fit_patients = sorted({dataset.samples[index].patient_id for index in indices.fit})
    if checkpoint["fit_patient_ids"] != fit_patients:
        raise ValueError("Fitting patients differ from the saved checkpoint")
    return PredictionData(
        dataset,
        Subset(dataset, list(getattr(indices, partition))),
        indices,
        partition,
    )


def make_predictions(
    dataset: ISICImageDataset,
    subset: Subset,
    labels: Sequence[int],
    scores: Sequence[float],
    decision_rule: Mapping[str, float],
) -> list[Prediction]:
    """Pair held-out metadata with model scores and frozen decisions."""

    predictions: list[Prediction] = []
    for index, label, score in zip(subset.indices, labels, scores):
        sample = dataset.samples[index]
        predictions.append(
            Prediction(
                image_name=sample.image_name,
                patient_id=sample.patient_id,
                image_path=sample.path,
                actual=int(label),
                score=float(score),
                decision=apply_triage_rule(float(score), decision_rule),
            )
        )
    return predictions


def _write_predictions(
    path: Path, predictions: Sequence[Prediction], threshold: float
) -> None:
    """Write and print one row for every held-out prediction."""

    columns = ("image_name", "patient_id", "actual", "score", "predicted", "decision")
    rows = [
        (
            p.image_name,
            p.patient_id,
            p.actual,
            p.score,
            int(p.score >= threshold),
            p.decision,
        )
        for p in predictions
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        for output in (stream, sys.stdout):
            writer = csv.writer(output)
            writer.writerow(columns)
            writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    """Reload a frozen run and explicitly score its permitted patient role."""

    args = parse_prediction_args(argv)
    started = time.perf_counter()
    device = select_device(args.device)
    reset_peak_accelerator_memory(device)
    model, checkpoint = load_run_for_prediction(args.checkpoint, device)
    model_name = checkpoint["model_name"]
    metrics_path = args.metrics or args.checkpoint.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validate_saved_run(
        checkpoint, metrics, args.checkpoint.parent / "effective_config.json"
    )
    data = build_prediction_data(
        checkpoint, args.images_dir, args.metadata_csv, args.partition
    )
    decision_rule = checkpoint["decision_rule"]
    output_dir = (
        args.output_dir or args.checkpoint.parent / "predictions" / data.partition
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    evaluation: dict[str, object] = {
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
        "objective": checkpoint["objective"],
        "score_kind": checkpoint["score_kind"],
        "device": str(device),
        "batch_size": args.batch_size,
    }
    evaluation_path = output_dir / "evaluation.json"
    write_evaluation(evaluation_path, evaluation)
    phase_started = synchronised_time(device)
    scores, labels = score_model(
        model, DataLoader(data.subset, batch_size=args.batch_size), device
    )
    scoring_seconds = synchronised_time(device) - phase_started
    predictions = make_predictions(
        data.dataset, data.subset, labels.tolist(), scores.tolist(), decision_rule
    )
    prediction_path = output_dir / f"{data.partition}_predictions.csv"
    _write_predictions(prediction_path, predictions, decision_rule["threshold"])
    figures = create_prediction_figures(
        predictions,
        decision_rule,
        "Melanoma softmax score",
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
            "metrics_at_0_5": classification_metrics(labels, scores, 0.5),
            "prediction_count": len(predictions),
            "predictions_sha256": file_sha256(prediction_path),
            "timings": {
                "scoring_seconds": scoring_seconds,
                "total_seconds": time.perf_counter() - started,
            },
            "profile": {
                "parameter_count": count_parameters(model),
                "scoring_images_per_second": len(predictions) / scoring_seconds,
                "scoring_scope": "image_loading_preprocessing_transfers_forward_softmax_cpu_scores",
                **memory_usage(device),
            },
        }
    )
    write_evaluation(evaluation_path, evaluation)
    print(
        f"Reloaded {model_name} on {device}; partition={data.partition}",
        file=sys.stderr,
    )
    for figure in figures:
        print(f"Saved {figure}", file=sys.stderr)


def write_evaluation(path: Path, evaluation: Mapping[str, Any]) -> None:
    """Persist the explicit evaluation intent and its eventual result."""

    with path.open("w", encoding="utf-8") as stream:
        json.dump(dict(evaluation), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
