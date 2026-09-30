"""Create report-ready SVG figures from one saved evaluation run.

Run from the repository root::

    python -m src.visualize --run-dir data/runs/base
"""

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve


REQUIRED_PREDICTION_COLUMNS = frozenset(
    {"image_name", "patient_id", "actual", "score", "predicted"}
)


@dataclass(frozen=True, slots=True)
class Prediction:
    """One held-out image prediction saved by ``src.evaluate``.

    Attributes
    ----------
    actual : int
        Ground-truth label: 0 for benign or 1 for melanoma.
    score : float
        Model score for ranking melanoma likelihood.
    """

    actual: int
    score: float


def create_visualisations(run_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Create four SVG figures from one evaluation run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory containing exactly one ``*_metrics.json`` file and its
        matching ``*_test_predictions.csv`` file.
    output_dir : pathlib.Path or None, default=None
        Figure directory. ``None`` writes to ``run_dir / "figures"``.

    Returns
    -------
    list of pathlib.Path
        Paths to training, confusion-matrix, discrimination, and score
        distribution SVGs, in that order.

    Raises
    ------
    FileNotFoundError
        If the run artifacts are missing.
    ValueError
        If the artifacts do not match the evaluator's saved schema.
    """

    metrics_path = _find_metrics_file(run_dir)
    run_name = metrics_path.stem.removesuffix("_metrics")
    predictions_path = run_dir / f"{run_name}_test_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Test prediction CSV does not exist: {predictions_path}"
        )

    metrics = _load_metrics(metrics_path)
    predictions = _load_predictions(predictions_path)
    figure_dir = output_dir if output_dir is not None else run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        figure_dir / f"{run_name}_training.svg",
        figure_dir / f"{run_name}_confusion_matrix.svg",
        figure_dir / f"{run_name}_discrimination.svg",
        figure_dir / f"{run_name}_score_distribution.svg",
    ]
    _plot_training_history(metrics, paths[0])
    _plot_confusion_matrix(metrics, paths[1])
    _plot_discrimination_curves(metrics, predictions, paths[2])
    _plot_score_distribution(metrics, predictions, paths[3])
    return paths


def _find_metrics_file(run_dir: Path) -> Path:
    """Return the only metrics JSON in a single-run directory."""

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    metrics_files = sorted(run_dir.glob("*_metrics.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No *_metrics.json file found in {run_dir}")
    if len(metrics_files) > 1:
        raise ValueError(
            f"Found {len(metrics_files)} metrics files in {run_dir}; use a directory for one run"
        )
    return metrics_files[0]


def _load_metrics(metrics_path: Path) -> dict[str, Any]:
    """Read and minimally validate the evaluator's metrics JSON."""

    with metrics_path.open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    if not isinstance(metrics, dict):
        raise ValueError("Metrics JSON must contain an object")
    required_keys = {"model", "history", "test"}
    missing = sorted(required_keys - set(metrics))
    if missing:
        raise ValueError(
            f"Metrics JSON is missing required key(s): {', '.join(missing)}"
        )
    if not isinstance(metrics["history"], list) or not isinstance(
        metrics["test"], dict
    ):
        raise ValueError("Metrics JSON has invalid history or test values")
    return metrics


def _load_predictions(predictions_path: Path) -> list[Prediction]:
    """Read held-out scores and validate the evaluator's prediction CSV."""

    with predictions_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_PREDICTION_COLUMNS - columns)
        if missing:
            raise ValueError(
                f"Prediction CSV is missing required column(s): {', '.join(missing)}"
            )
        predictions = [_parse_prediction(row) for row in reader]
    if not predictions:
        raise ValueError("Prediction CSV has no rows")
    labels = {prediction.actual for prediction in predictions}
    if labels != {0, 1}:
        raise ValueError("Prediction CSV must contain both benign and melanoma labels")
    return predictions


def _parse_prediction(row: dict[str, str | None]) -> Prediction:
    """Convert one CSV row to numeric held-out prediction data."""

    try:
        actual = int(row["actual"] or "")
        score = float(row["score"] or "")
    except ValueError as error:
        raise ValueError(
            "Prediction CSV contains a non-numeric actual label or score"
        ) from error
    if actual not in (0, 1):
        raise ValueError("Prediction CSV actual labels must be 0 or 1")
    if not np.isfinite(score):
        raise ValueError("Prediction CSV scores must be finite")
    return Prediction(actual=actual, score=score)


def _plot_training_history(metrics: dict[str, Any], output_path: Path) -> None:
    """Plot training loss and validation average precision by epoch."""

    history = metrics["history"]
    try:
        epochs = [int(entry["epoch"]) for entry in history]
        losses = [float(entry["train_loss"]) for entry in history]
        average_precision = [
            float(entry["validation_average_precision"]) for entry in history
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Metrics history must contain epoch, train_loss, and validation_average_precision"
        ) from error
    if not epochs:
        raise ValueError("Metrics history has no epochs")

    figure, loss_axis = plt.subplots(figsize=(7.2, 4.5))
    ap_axis = loss_axis.twinx()
    loss_line = loss_axis.plot(
        epochs, losses, "o-", color="#1f77b4", label="Training loss"
    )
    ap_line = ap_axis.plot(
        epochs,
        average_precision,
        "o-",
        color="#d62728",
        label="Validation average precision",
    )
    loss_axis.set_title(f"Training history — {metrics['model']} model")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Training loss", color="#1f77b4")
    ap_axis.set_ylabel("Validation average precision", color="#d62728")
    loss_axis.set_xticks(epochs)
    loss_axis.grid(axis="y", alpha=0.3)
    loss_axis.legend(
        loss_line + ap_line,
        ("Training loss", "Validation average precision"),
    )
    _save_svg(figure, output_path)


def _plot_confusion_matrix(metrics: dict[str, Any], output_path: Path) -> None:
    """Plot the held-out test confusion matrix."""

    test_metrics = metrics["test"]
    matrix = np.asarray(test_metrics.get("confusion_matrix"), dtype=np.int64)
    if matrix.shape != (2, 2):
        raise ValueError("Metrics test confusion_matrix must have shape [2, 2]")

    figure, axis = plt.subplots(figsize=(5.4, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Image count")
    labels = ("Benign", "Melanoma")
    axis.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="True class",
        title="Held-out test confusion matrix",
    )
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color=color,
            )
    _save_svg(figure, output_path)


def _plot_discrimination_curves(
    metrics: dict[str, Any], predictions: list[Prediction], output_path: Path
) -> None:
    """Plot test ROC and precision-recall curves from saved scores."""

    labels, scores = _labels_and_scores(predictions)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    test_metrics = metrics["test"]
    roc_auc = _metric_value(test_metrics, "roc_auc")
    average_precision = _metric_value(test_metrics, "melanoma_average_precision")

    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    roc_axis.plot(
        false_positive_rate,
        true_positive_rate,
        color="#1f77b4",
        label=f"AUROC = {roc_auc:.3f}",
    )
    roc_axis.plot((0, 1), (0, 1), "--", color="0.5", label="Chance")
    roc_axis.set(
        title="Held-out test ROC curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    roc_axis.grid(alpha=0.3)
    roc_axis.legend(loc="lower right")

    baseline = float(labels.mean())
    pr_axis.plot(
        recall, precision, color="#d62728", label=f"AP = {average_precision:.3f}"
    )
    pr_axis.axhline(
        baseline, linestyle="--", color="0.5", label=f"Prevalence = {baseline:.3f}"
    )
    pr_axis.set(
        title="Held-out test precision-recall curve",
        xlabel="Melanoma recall",
        ylabel="Melanoma precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    pr_axis.grid(alpha=0.3)
    pr_axis.legend(loc="upper right")
    _save_svg(figure, output_path)


def _plot_score_distribution(
    metrics: dict[str, Any], predictions: list[Prediction], output_path: Path
) -> None:
    """Plot class-conditioned held-out score densities and the decision threshold."""

    labels, scores = _labels_and_scores(predictions)
    threshold = _metric_value(metrics["test"], "threshold")
    score_label = (
        "Melanoma probability"
        if metrics.get("score_kind") == "melanoma_probability"
        else "Melanoma similarity score"
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    bins = min(40, max(10, int(np.sqrt(len(scores)) * 2)))
    axis.hist(
        scores[labels == 0],
        bins=bins,
        density=True,
        alpha=0.6,
        color="#1f77b4",
        label="True benign",
    )
    axis.hist(
        scores[labels == 1],
        bins=bins,
        density=True,
        alpha=0.6,
        color="#d62728",
        label="True melanoma",
    )
    axis.axvline(
        threshold, color="black", linestyle="--", label=f"Threshold = {threshold:.3f}"
    )
    axis.set(
        title="Held-out test score distributions",
        xlabel=score_label,
        ylabel="Density",
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    _save_svg(figure, output_path)


def _labels_and_scores(predictions: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
    """Convert saved prediction objects to NumPy arrays for plotting."""

    return (
        np.asarray([prediction.actual for prediction in predictions], dtype=np.int64),
        np.asarray([prediction.score for prediction in predictions], dtype=np.float64),
    )


def _metric_value(metrics: dict[str, Any], name: str) -> float:
    """Return a finite numeric value from a metrics object."""

    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Metrics test section needs a numeric {name}") from error
    if not np.isfinite(value):
        raise ValueError(f"Metrics test section needs a finite {name}")
    return value


def _save_svg(figure: Figure, output_path: Path) -> None:
    """Apply final layout, save one SVG, and release Matplotlib resources."""

    figure.tight_layout()
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """Create SVG figures for a saved run directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    figures = create_visualisations(args.run_dir, args.output_dir)
    for figure in figures:
        print(f"Saved {figure}")


if __name__ == "__main__":
    main()
