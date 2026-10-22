"""Inspect fitting references and calibration-derived referral policies.

Run explicitly after training. Selection is the default evaluation role;
final patients require the same opt-in flags as ``predict.py``.
"""

import argparse
import csv
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from math import ceil
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Subset

from recognition.Siamese_ISIC2020_Melanoma_47472442.predict import (
    build_prediction_data,
    check_final_evaluation,
    load_run_for_prediction,
    make_predictions,
    validate_saved_run,
    write_evaluation,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.provenance import file_sha256
from recognition.Siamese_ISIC2020_Melanoma_47472442.references import (
    encode_images,
    score_neighbors,
    select_gallery,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.train import classification_metrics
from recognition.Siamese_ISIC2020_Melanoma_47472442.utils import (
    count_parameters,
    memory_usage,
    reset_peak_accelerator_memory,
    select_device,
    synchronised_time,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.visualisation import (
    Prediction,
    plot_embedding_map,
    plot_reference_cases,
    plot_referral_workload,
)


def parse_diagnostic_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read checkpoint, evaluation role and reference-analysis settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--partition",
        choices=("selection", "calibration", "final"),
        default="selection",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument("--references-per-class", type=int, default=100)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--evaluate-final", action="store_true")
    parser.add_argument("--evaluation-note")
    args = parser.parse_args(argv)
    check_final_evaluation(parser, args)
    return args


def calibrate_referral_policies(
    labels: np.ndarray,
    scores: np.ndarray,
    disagreement: np.ndarray,
    threshold: float,
    target_recall: float,
) -> dict[str, tuple[float, float] | None]:
    """Choose queue cutoffs from calibration melanomas only.

    Classifier-positive images always enter review. Additional negatives are
    ranked by disagreement then confidence, or by confidence alone. ``None``
    means the existing classifier flags already meet the calibration target.
    Exact cutoff ties are all referred when the policy is applied.
    """

    flagged = scores >= threshold
    confidence = -np.abs(scores - threshold)
    missed = np.flatnonzero((labels == 1) & ~flagged)
    needed = ceil(target_recall * int(labels.sum())) - int(labels[flagged].sum())
    policies: dict[str, tuple[float, float] | None] = {}
    for name, priority in (
        ("confidence", np.zeros(len(scores))),
        ("reference_disagreement", disagreement),
    ):
        cutoff = None
        if needed > 0:
            order = np.lexsort((-confidence[missed], -priority[missed]))
            boundary = missed[order[needed - 1]]
            cutoff = (float(priority[boundary]), float(confidence[boundary]))
        policies[name] = cutoff
    return policies


def referral_summary(
    labels: np.ndarray, reviewed: np.ndarray
) -> dict[str, float | int]:
    """Count review workload and true melanomas routed for human review."""

    captured = int(labels[reviewed].sum())
    return {
        "reviewed_images": int(reviewed.sum()),
        "review_fraction": float(reviewed.mean()),
        "captured_melanomas": captured,
        "melanoma_recall": captured / int(labels.sum()),
        "cleared_melanomas": int(labels.sum()) - captured,
        "reviewed_benign": int(((labels == 0) & reviewed).sum()),
    }


def evaluate_referrals(
    labels: np.ndarray,
    scores: np.ndarray,
    disagreement: np.ndarray,
    threshold: float,
    policies: Mapping[str, tuple[float, float] | None],
) -> tuple[dict[str, object], dict[str, dict[str, list[float]]], dict[str, np.ndarray]]:
    """Apply frozen policies and describe queues without changing their cutoffs.

    Returns
    -------
    dict, dict, dict
        Workload/error-detection metrics, descriptive workload curves and
        per-query referral flags. Curve labels are used for evaluation only.
    """

    flagged = scores >= threshold
    confidence = -np.abs(scores - threshold)
    errors = (labels != flagged).astype(int)
    metrics: dict[str, object] = {"classifier_flags": referral_summary(labels, flagged)}
    curves: dict[str, dict[str, list[float]]] = {}
    referrals: dict[str, np.ndarray] = {}
    for name, cutoff in policies.items():
        priority = (
            disagreement if name == "reference_disagreement" else np.zeros(len(scores))
        )
        reviewed = flagged.copy()
        if cutoff is not None:
            reviewed |= (priority > cutoff[0]) | (
                (priority == cutoff[0]) & (confidence >= cutoff[1])
            )
        uncertainty = disagreement if name == "reference_disagreement" else confidence
        metrics[name] = {
            **referral_summary(labels, reviewed),
            "error_detection_auroc": float(roc_auc_score(errors, uncertainty))
            if len(np.unique(errors)) == 2
            else None,
            "error_detection_average_precision": float(
                average_precision_score(errors, uncertainty)
            )
            if errors.any()
            else None,
        }
        referrals[name] = reviewed
        order = np.lexsort((-confidence, -priority))
        order = order[~flagged[order]]
        changes = (priority[order][1:] != priority[order][:-1]) | (
            confidence[order][1:] != confidence[order][:-1]
        )
        stops = np.r_[0, np.flatnonzero(changes) + 1, len(order)]
        captured = int(labels[flagged].sum()) + np.r_[0, np.cumsum(labels[order])]
        curves[name] = {
            "review_fraction": ((int(flagged.sum()) + stops) / len(scores)).tolist(),
            "melanoma_recall": (captured[stops] / int(labels.sum())).tolist(),
        }
    return metrics, curves, referrals


def select_reference_failures(
    predictions: Sequence[Prediction],
    disagreement: np.ndarray,
    threshold: float,
    maximum: int,
) -> list[int]:
    """Select actual binary errors, covering false negatives and false positives."""

    scores = np.asarray([prediction.score for prediction in predictions])
    labels = np.asarray(
        [prediction.actual for prediction in predictions], dtype=np.int64
    )
    names = np.asarray([prediction.image_name for prediction in predictions])
    order = np.lexsort((names, -disagreement, -np.abs(scores - threshold)))
    errors = order[labels[order] != (scores[order] >= threshold)]
    selected: list[int] = []
    for label in (1, 0):
        matches = errors[labels[errors] == label]
        if len(matches):
            selected.append(int(matches[0]))
    return (selected + [int(index) for index in errors if index not in selected])[
        :maximum
    ]


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write nonempty diagnostic rows, using their keys as CSV headers."""

    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    """Produce reference evidence and calibration-only referral comparisons."""

    args = parse_diagnostic_args(argv)
    started = time.perf_counter()
    device = select_device(args.device)
    reset_peak_accelerator_memory(device)
    model, checkpoint = load_run_for_prediction(args.checkpoint, device)
    run_dir = args.checkpoint.parent
    config_path = run_dir / "effective_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    validate_saved_run(checkpoint, metrics, config_path)
    data = build_prediction_data(
        checkpoint, args.images_dir, args.metadata_csv, args.partition
    )
    gallery = select_gallery(
        Subset(data.dataset, list(data.indices.fit)),
        args.references_per_class,
        config["seed"],
    )
    gallery_samples = [data.dataset.samples[index] for index in gallery.indices]
    reference_labels = np.asarray(
        [sample.label for sample in gallery_samples], dtype=np.int64
    )
    neighbors = min(args.neighbors, len(gallery))
    rule = checkpoint["decision_rule"]
    threshold = rule["threshold"]
    output_dir = args.output_dir or run_dir / "diagnostics" / args.partition
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "status": "started",
        "partition": args.partition,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "effective_config_sha256": checkpoint["effective_config_sha256"],
        "protocol": checkpoint["protocol"],
        "objective": checkpoint["objective"],
        "classifier_threshold": threshold,
        "target_melanoma_recall": args.target_recall,
        "gallery_seed": config["seed"],
        "references_per_class": len(gallery) // 2,
        "neighbors": neighbors,
        "device": str(device),
        "batch_size": args.batch_size,
        "final_evaluation_authorised": bool(args.evaluate_final),
        "evaluation_note": args.evaluation_note,
    }
    report_path = output_dir / "evaluation.json"
    write_evaluation(report_path, report)

    print(f"Encoding {len(gallery)} fitting references", flush=True)
    phase_started = synchronised_time(device)
    _, reference_embeddings = encode_images(model, gallery, args.batch_size, device)
    reference_seconds = synchronised_time(device) - phase_started
    gallery_path = output_dir / "gallery.csv"
    write_csv(
        gallery_path,
        (
            {
                "image_name": sample.image_name,
                "patient_id": sample.patient_id,
                "actual": sample.label,
            }
            for sample in gallery_samples
        ),
    )

    print("Scoring calibration patients and freezing referral cutoffs", flush=True)
    phase_started = synchronised_time(device)
    calibration = Subset(data.dataset, list(data.indices.calibration))
    calibration_scores, calibration_embeddings = encode_images(
        model, calibration, args.batch_size, device
    )
    calibration_encoding_seconds = synchronised_time(device) - phase_started
    neighbor_started = time.perf_counter()
    calibration_neighbors = score_neighbors(
        calibration_embeddings,
        calibration_scores,
        reference_embeddings,
        reference_labels,
        threshold,
        neighbors,
    )
    calibration_neighbor_seconds = time.perf_counter() - neighbor_started
    calibration_labels = np.asarray(
        [data.dataset.samples[index].label for index in calibration.indices],
        dtype=np.int64,
    )
    policies = calibrate_referral_policies(
        calibration_labels,
        calibration_scores,
        calibration_neighbors.disagreement,
        threshold,
        args.target_recall,
    )
    calibration_metrics, _, _ = evaluate_referrals(
        calibration_labels,
        calibration_scores,
        calibration_neighbors.disagreement,
        threshold,
        policies,
    )
    calibration_seconds = time.perf_counter() - phase_started
    report.update(
        {
            "gallery_sha256": file_sha256(gallery_path),
            "referral_cutoffs": policies,
            "policy_calibration_partition": "calibration",
            "calibration": calibration_metrics,
        }
    )
    write_evaluation(report_path, report)

    query_encoding_seconds = query_neighbor_seconds = 0.0
    if args.partition == "calibration":
        scores, embeddings, query_neighbors = (
            calibration_scores,
            calibration_embeddings,
            calibration_neighbors,
        )
    else:
        print(f"Scoring {args.partition} patients with frozen policies", flush=True)
        phase_started = synchronised_time(device)
        scores, embeddings = encode_images(model, data.subset, args.batch_size, device)
        query_encoding_seconds = synchronised_time(device) - phase_started
        neighbor_started = time.perf_counter()
        query_neighbors = score_neighbors(
            embeddings,
            scores,
            reference_embeddings,
            reference_labels,
            threshold,
            neighbors,
        )
        query_neighbor_seconds = time.perf_counter() - neighbor_started
    query_seconds = query_encoding_seconds + query_neighbor_seconds
    labels = np.asarray(
        [data.dataset.samples[index].label for index in data.subset.indices],
        dtype=np.int64,
    )
    predictions = make_predictions(
        data.dataset, data.subset, labels.tolist(), scores.tolist(), rule
    )
    evaluation_metrics, curves, referrals = evaluate_referrals(
        labels, scores, query_neighbors.disagreement, threshold, policies
    )
    np.savez_compressed(
        output_dir / "embeddings.npz",
        reference_embeddings=reference_embeddings.numpy(),
        reference_labels=reference_labels,
        query_embeddings=embeddings.numpy(),
        query_labels=labels,
        reference_image_names=np.asarray(
            [sample.image_name for sample in gallery_samples]
        ),
        query_image_names=np.asarray(
            [prediction.image_name for prediction in predictions]
        ),
    )
    write_csv(
        output_dir / "query_diagnostics.csv",
        (
            {
                "image_name": prediction.image_name,
                "patient_id": prediction.patient_id,
                "actual": prediction.actual,
                "score": prediction.score,
                "predicted": int(prediction.score >= threshold),
                "decision": prediction.decision,
                "confidence_uncertainty": -abs(prediction.score - threshold),
                "neighbor_melanoma_fraction": float(
                    query_neighbors.melanoma_fraction[index]
                ),
                "neighbor_disagreement": float(query_neighbors.disagreement[index]),
                "confidence_review": int(referrals["confidence"][index]),
                "reference_review": int(referrals["reference_disagreement"][index]),
            }
            for index, prediction in enumerate(predictions)
        ),
    )
    write_csv(
        output_dir / "neighbors.csv",
        (
            {
                "image_name": prediction.image_name,
                "rank": rank + 1,
                "reference_image": gallery_samples[reference_index].image_name,
                "reference_patient": gallery_samples[reference_index].patient_id,
                "reference_actual": gallery_samples[reference_index].label,
                "distance": float(distance),
            }
            for index, prediction in enumerate(predictions)
            for rank, (reference_index, distance) in enumerate(
                zip(query_neighbors.indices[index], query_neighbors.distances[index])
            )
        ),
    )
    write_csv(
        output_dir / "referral_curves.csv",
        (
            {"policy": name, "review_fraction": fraction, "melanoma_recall": recall}
            for name, curve in curves.items()
            for fraction, recall in zip(
                curve["review_fraction"], curve["melanoma_recall"]
            )
        ),
    )
    failures = select_reference_failures(
        predictions, query_neighbors.disagreement, threshold, args.max_cases
    )
    if failures:
        plot_reference_cases(
            predictions,
            gallery_samples,
            query_neighbors,
            failures,
            output_dir / "failure_references.svg",
            threshold,
        )
    plot_referral_workload(curves, output_dir / "referral_workload.svg", args.partition)
    plot_embedding_map(
        reference_embeddings.numpy(),
        reference_labels,
        embeddings.numpy(),
        labels,
        output_dir / "embeddings.svg",
        args.partition,
    )
    report.update(
        {
            "status": "complete",
            "evaluation": evaluation_metrics,
            "classifier": classification_metrics(labels, scores, threshold),
            "classifier_at_0_5": classification_metrics(labels, scores, 0.5),
            "query_images": len(predictions),
            "failure_cases": [predictions[index].image_name for index in failures],
            "timings": {
                "reference_encoding_seconds": reference_seconds,
                "calibration_encoding_seconds": calibration_encoding_seconds,
                "calibration_neighbor_seconds": calibration_neighbor_seconds,
                "calibration_seconds": calibration_seconds,
                "query_encoding_seconds": query_encoding_seconds,
                "query_neighbor_seconds": query_neighbor_seconds,
                "query_seconds": query_seconds,
                "total_seconds": time.perf_counter() - started,
            },
            "profile": {
                "parameter_count": count_parameters(model),
                "reference_images_encoded": len(gallery),
                "calibration_images_encoded": len(calibration),
                "query_reuses_calibration": args.partition == "calibration",
                "query_images_encoded": 0
                if args.partition == "calibration"
                else len(predictions),
                "query_images_per_second": len(predictions) / query_seconds
                if args.partition != "calibration"
                else None,
                "query_scope": "image_loading_preprocessing_transfers_forward_softmax_cpu_embeddings_and_neighbors",
                **memory_usage(device),
            },
        }
    )
    write_evaluation(report_path, report)
    print(f"Saved {args.partition} diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
