"""Compare saved diagnostic predictions without loading models or images."""

import argparse
import csv
import json
from collections.abc import Sequence
from operator import itemgetter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from .provenance import file_sha256
from .visualisation import plot_statistical_comparison


Run = tuple[dict[str, Any], dict[str, np.ndarray]]
METRICS = (
    "accuracy",
    "macro_f1",
    "roc_auc",
    "melanoma_average_precision",
    "melanoma_precision",
    "melanoma_recall",
    "benign_precision",
    "benign_recall",
    "confidence_review_fraction",
    "confidence_melanoma_recall",
    "reference_review_fraction",
    "reference_melanoma_recall",
    "reference_minus_confidence_review_fraction",
    "reference_minus_confidence_melanoma_recall",
)
DIAGNOSTIC_SETTINGS = ("target_melanoma_recall", "references_per_class", "neighbors")
TRAINING_SETTINGS = (
    "initial_weights",
    "epochs",
    "head_warmup_epochs",
    "batch_size",
    "learning_rate",
    "encoder_learning_rate",
    "min_learning_rate",
    "weight_decay",
    "rotation_degrees",
    "color_jitter",
    "pairs_per_epoch",
    "contrastive_margin",
    "target_recall",
)


def load_run(directory: Path, partition: str) -> Run:
    """Read completed diagnostic evidence and order it by unique image identity.

    Parameters
    ----------
    directory : pathlib.Path
        Completed current ResNet training run with diagnostic artifacts.
    partition : {"selection", "calibration", "final"}
        Saved diagnostic role to analyse.

    Returns
    -------
    tuple of dict and dict
        Run settings/file identities and aligned NumPy prediction columns.
    """

    folder = directory / "diagnostics" / partition
    report_path = folder / "evaluation.json"
    csv_path = folder / "query_diagnostics.csv"
    config_path = directory / "effective_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config_hash = file_sha256(config_path)
    if report["status"] != "complete" or report["partition"] != partition:
        raise ValueError(
            f"Diagnostics are incomplete or refer to another role: {folder}"
        )
    if report["effective_config_sha256"] != config_hash:
        raise ValueError(
            f"Training settings differ from the diagnostic run: {directory}"
        )
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=itemgetter("image_name"))
    columns = {
        "image_name": np.asarray([row["image_name"] for row in rows], dtype=str),
        "patient_id": np.asarray([row["patient_id"] for row in rows], dtype=str),
        "actual": np.asarray([row["actual"] for row in rows], dtype=np.int64),
        "score": np.asarray([row["score"] for row in rows], dtype=float),
        "predicted": np.asarray([row["predicted"] for row in rows], dtype=np.int64),
        "confidence_review": np.asarray(
            [row["confidence_review"] for row in rows], dtype=np.int64
        ),
        "reference_review": np.asarray(
            [row["reference_review"] for row in rows], dtype=np.int64
        ),
    }
    if len(np.unique(columns["image_name"])) != len(rows):
        raise ValueError(f"Duplicate prediction image IDs: {csv_path}")
    info = {
        "directory": str(directory),
        "seed": config["seed"],
        "config": config,
        "evaluation": report,
        "config_sha256": config_hash,
        "evaluation_sha256": file_sha256(report_path),
        "predictions_sha256": file_sha256(csv_path),
    }
    return info, columns


def check_comparison(groups: Sequence[dict[int, Run]], seeds: Sequence[int]) -> None:
    """Require matching populations and a controlled objective comparison."""

    first_info, first_data = groups[0][seeds[0]]
    for group, model in zip(groups, ("base", "siamese")):
        for info, data in group.values():
            report = info["evaluation"]
            weight = report["objective"]["contrastive_weight"]
            if (model == "base" and weight != 0) or (
                model == "siamese" and weight <= 0
            ):
                raise ValueError(
                    "Compare classification alone with a positive Siamese loss weight"
                )
            if info["config"]["model"] != model or any(
                report[key] != first_info["evaluation"][key]
                for key in ("protocol", *DIAGNOSTIC_SETTINGS)
            ):
                raise ValueError(
                    "Use the two objectives with matching diagnostic settings/protocol"
                )
            if any(
                info["config"].get(key) != first_info["config"].get(key)
                for key in TRAINING_SETTINGS
            ):
                raise ValueError(
                    "Training hyperparameters must match across seeds/objectives"
                )
            if report["objective"] != group[seeds[0]][0]["evaluation"]["objective"]:
                raise ValueError(
                    "Each objective must use the same loss settings across seeds"
                )
            if any(
                not np.array_equal(data[key], first_data[key])
                for key in ("image_name", "patient_id", "actual")
            ):
                raise ValueError(
                    "Every run must evaluate the same images, patients and labels"
                )
    for seed in seeds:
        baseline, siamese = groups[0][seed][0], groups[1][seed][0]
        if (
            baseline["evaluation"]["gallery_sha256"]
            != siamese["evaluation"]["gallery_sha256"]
        ):
            raise ValueError(f"Fitting references differ within seed {seed}")


def metric_values(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    """Evaluate saved binary and review decisions in the METRICS column order."""

    labels = data["actual"][indices]
    predicted = data["predicted"][indices]
    scores = data["score"][indices]
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    correct = matrix.diagonal()
    predicted_counts = matrix.sum(axis=0)
    precision = np.divide(
        correct, predicted_counts, out=np.zeros(2), where=predicted_counts > 0
    )
    recall = correct / matrix.sum(axis=1)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(2),
        where=(precision + recall) > 0,
    )
    confidence = data["confidence_review"][indices]
    reference = data["reference_review"][indices]
    positive = labels == 1
    return np.asarray(
        [
            (labels == predicted).mean(),
            f1.mean(),
            roc_auc_score(labels, scores),
            average_precision_score(labels, scores),
            precision[1],
            recall[1],
            precision[0],
            recall[0],
            confidence.mean(),
            confidence[positive].mean(),
            reference.mean(),
            reference[positive].mean(),
            (reference - confidence).mean(),
            (reference[positive] - confidence[positive]).mean(),
        ],
        dtype=float,
    )


def bootstrap_patients(runs: Sequence[Run], resamples: int, seed: int) -> np.ndarray:
    """Resample complete patient clusters with identical draws for every run.

    Parameters
    ----------
    runs : sequence of tuple
        Aligned run metadata and prediction columns.
    resamples : int
        Number of bootstrap draws.
    seed : int
        NumPy resampling seed, independent of training seeds.

    Returns
    -------
    numpy.ndarray
        Values shaped ``[draws, runs, metrics]``. Patient strata preserve the
        observed counts with/without any melanoma. Repeated draws of a patient
        repeat all its images; image counts may vary between draws.
    """

    data = runs[0][1]
    clusters = [
        np.flatnonzero(data["patient_id"] == patient)
        for patient in np.unique(data["patient_id"])
    ]
    positive = np.asarray([data["actual"][indices].max() for indices in clusters])
    strata = [np.flatnonzero(positive == label) for label in (0, 1)]
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(runs), len(METRICS)))
    for draw in range(resamples):
        indices = np.concatenate(
            [
                clusters[int(index)]
                for stratum in strata
                for index in rng.choice(stratum, size=len(stratum), replace=True)
            ]
        )
        draws[draw] = np.stack([metric_values(columns, indices) for _, columns in runs])
    return draws


def result_rows(
    comparison: str, estimates: np.ndarray, draws: np.ndarray
) -> list[dict[str, Any]]:
    """Summarise seed means, seed SD and conditional patient percentile intervals.

    Parameters
    ----------
    comparison : str
        Objective name or paired difference label.
    estimates : numpy.ndarray
        Original-cohort values shaped ``[seeds, metrics]``.
    draws : numpy.ndarray
        Paired patient draws shaped ``[draws, seeds, metrics]``.

    Returns
    -------
    list of dict
        One row per metric. Seed SD is undefined for one seed. Bootstrap
        intervals condition on the supplied trained runs and frozen cutoffs.
    """

    means = estimates.mean(axis=0)
    lower, upper = np.percentile(draws.mean(axis=1), [2.5, 97.5], axis=0)
    deviations = estimates.std(axis=0, ddof=1) if len(estimates) > 1 else None
    return [
        {
            "comparison": comparison,
            "metric": metric,
            "seeds": len(estimates),
            "estimate": float(means[index]),
            "seed_sd": float(deviations[index]) if deviations is not None else None,
            "patient_ci_lower": float(lower[index]),
            "patient_ci_upper": float(upper[index]),
        }
        for index, metric in enumerate(METRICS)
    ]


def main(argv: Sequence[str] | None = None) -> None:
    """Write an offline paired comparison; never train, infer or tune a cutoff."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--siamese-runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--partition",
        choices=("selection", "calibration", "final"),
        default="selection",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    groups: list[dict[int, Run]] = []
    for directories in (args.baseline_runs, args.siamese_runs):
        loaded = [load_run(directory, args.partition) for directory in directories]
        by_seed = {int(info["seed"]): (info, columns) for info, columns in loaded}
        if len(by_seed) != len(directories):
            raise ValueError("Provide one run per training seed for each objective")
        groups.append(by_seed)
    if groups[0].keys() != groups[1].keys():
        raise ValueError("Baseline and Siamese training seeds must match")
    seeds = sorted(groups[0])
    check_comparison(groups, seeds)
    runs = [group[seed] for group in groups for seed in seeds]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = runs[0][1]
    indices = np.arange(len(data["actual"]))
    estimates = np.stack([metric_values(columns, indices) for _, columns in runs])
    estimates = estimates.reshape(2, len(seeds), len(METRICS))
    print(
        f"Bootstrapping {args.resamples} patient samples across {len(seeds)} paired seeds",
        flush=True,
    )
    draws = bootstrap_patients(runs, args.resamples, args.seed)
    draws = draws.reshape(args.resamples, 2, len(seeds), len(METRICS))
    per_seed: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for name, values, samples in (
        ("baseline", estimates[0], draws[:, 0]),
        ("siamese", estimates[1], draws[:, 1]),
        (
            "siamese_minus_baseline",
            estimates[1] - estimates[0],
            draws[:, 1] - draws[:, 0],
        ),
    ):
        summary.extend(result_rows(name, values, samples))
        for index, seed in enumerate(seeds):
            per_seed.extend(
                {"seed": seed, **row}
                for row in result_rows(
                    name, values[index : index + 1], samples[:, index : index + 1]
                )
            )
    for filename, rows in (("per_seed.csv", per_seed), ("summary.csv", summary)):
        with (args.output_dir / filename).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_statistical_comparison(
        summary, args.output_dir / "comparison.svg", args.partition
    )
    report = {
        "partition": args.partition,
        "training_seeds": seeds,
        "bootstrap_seed": args.seed,
        "resamples": args.resamples,
        "interval": "95_percentile_stratified_paired_patient_cluster_bootstrap",
        "interval_conditions": "fixed_trained_runs_calibration_rules_and_patient_stratum_counts",
        "patients": len(np.unique(data["patient_id"])),
        "melanoma_patients": len(np.unique(data["patient_id"][data["actual"] == 1])),
        "images": len(indices),
        "melanomas": int(data["actual"].sum()),
        "protocol": runs[0][0]["evaluation"]["protocol"],
        "diagnostic_settings": {
            key: runs[0][0]["evaluation"][key] for key in DIAGNOSTIC_SETTINGS
        },
        "inputs": [
            {
                key: value
                for key, value in info.items()
                if key not in ("config", "evaluation")
            }
            for info, _ in runs
        ],
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
