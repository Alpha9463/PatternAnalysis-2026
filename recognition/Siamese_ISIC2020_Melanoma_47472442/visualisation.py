"""SVG plotting helpers and offline ROC/PR generation from saved ISIC artifacts.

Run this module with ``--run-dir`` to plot saved prediction scores and the
frozen decision threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

if TYPE_CHECKING:
    from .dataset import ISICSample
    from .references import NeighborScores


@dataclass(frozen=True, slots=True)
class Prediction:
    """One held-out prediction with source metadata for SVG evidence.

    Attributes
    ----------
    image_name : str
        ISIC image identifier.
    patient_id : str
        Patient identifier supplied by the dataset metadata.
    image_path : pathlib.Path
        Source JPEG shown in the failure-case figure.
    actual : int
        Ground-truth class: 0 for benign and 1 for melanoma.
    score : float
        Frozen model score used by the run's decision policy.
    decision : str
        Frozen automated decision: ``"benign"``, ``"refer"``, or
        ``"melanoma"``.
    """

    image_name: str
    patient_id: str
    image_path: Path
    actual: int
    score: float
    decision: str


def plot_training_history(
    history: Sequence[Mapping[str, float]], output_path: Path
) -> Path:
    """Write an SVG of recorded losses, selection metrics and training dynamics.

    Parameters
    ----------
    history : list of mapping
        Per-epoch records with ``epoch``, ``train_loss``, and
        ``selection_average_precision`` fields. Additional panels show recorded
        overall/per-class selection loss, sampling coverage, AUROC, learning
        rates, pair distances and active negatives. Frozen encoder rates are
        omitted on the logarithmic learning-rate axis.
    output_path : pathlib.Path
        SVG destination. Parent directories are created when needed.

    Returns
    -------
    pathlib.Path
        The written SVG path.

    """

    epochs = [int(entry["epoch"]) for entry in history]
    panels = [
        (
            "Training losses",
            "Loss",
            (
                ("train_loss", "Total loss"),
                ("classification_loss", "Cross-entropy"),
                ("contrastive_loss", "Raw contrastive loss"),
            ),
        ),
        (
            "Selection ranking",
            "Score",
            (
                ("selection_average_precision", "Average precision"),
                ("selection_roc_auc", "AUROC"),
            ),
        ),
        (
            "Selection loss",
            "Log loss",
            (
                ("selection_loss", "Overall"),
                ("selection_benign_loss", "Benign"),
                ("selection_melanoma_loss", "Melanoma"),
            ),
        ),
        (
            "Learning rates",
            "Learning rate",
            (
                ("learning_rate", "Classifier"),
                ("encoder_learning_rate", "Encoder (zero = frozen)"),
            ),
        ),
        (
            "Training pair distances",
            "Euclidean distance",
            (
                ("same_pair_distance", "Same class"),
                ("different_pair_distance", "Different class"),
            ),
        ),
        (
            "Different-class pairs inside margin",
            "Fraction",
            (("active_negative_fraction", "Active negatives"),),
        ),
        (
            "Unique fitting examples per epoch",
            "Unique count",
            (
                ("sampled_benign_images", "Benign images"),
                ("sampled_melanoma_images", "Melanoma images"),
                ("sampled_melanoma_patients", "Melanoma patients"),
            ),
        ),
    ]
    panels = [
        panel for panel in panels if any(field in history[0] for field, _ in panel[2])
    ]
    rows, columns = ceil(len(panels) / 3), min(len(panels), 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.5 * columns, 3.7 * rows), squeeze=False
    )
    for axis, (title, ylabel, fields) in zip(axes.flat, panels):
        for field, label in fields:
            if field in history[0]:
                values = [entry[field] for entry in history]
                if field == "encoder_learning_rate":
                    values = [value if value > 0 else np.nan for value in values]
                axis.plot(
                    epochs,
                    values,
                    "o-",
                    markersize=3,
                    label=label,
                )
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize="small")
        if ylabel in ("Learning rate", "Unique count"):
            axis.set_yscale("log")
        if ylabel in ("Score", "Fraction"):
            axis.set_ylim(0, 1)
    for axis in axes.flat:
        axis.set_visible(bool(axis.lines))
    figure.suptitle("Training history")
    figure.tight_layout()
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_roc_precision_recall(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    output_path: Path,
    title: str = "Melanoma classification",
) -> Path:
    """Write ROC and melanoma precision–recall curves to one SVG.

    Parameters
    ----------
    labels : sequence of int
        Ground-truth image labels: 0 for benign and 1 for melanoma.
    scores : sequence of float
        Melanoma ranking scores, with larger values indicating melanoma.
        Signed gallery scores are supported; probabilities are not required.
    threshold : float
        Saved decision threshold. This function
        never selects a threshold from the plotted labels.
    output_path : pathlib.Path
        SVG destination. Parent directories are created when needed.
    title : str, default="Melanoma classification"
        Figure title, followed by image and melanoma counts.

    Returns
    -------
    pathlib.Path
        Written SVG path.

    Raises
    ------
    ValueError
        If inputs are empty, differ in length, lack either binary class,
        contain non-finite scores or threshold, or the output is not SVG.

    Notes
    -----
    PR performance uses average precision, not trapezoidal integration.
    Precision is undefined when the frozen threshold predicts no positives;
    in that case only the ROC operating point is marked.
    """

    actual, ranking_scores = _curve_arrays(labels, scores)
    if not isfinite(threshold):
        raise ValueError("The frozen decision threshold must be finite")
    if output_path.suffix.lower() != ".svg":
        raise ValueError("The ROC/PR output path must have an .svg suffix")

    false_positive_rate, recall, _ = roc_curve(actual, ranking_scores, pos_label=1)
    precision, pr_recall, _ = precision_recall_curve(
        actual, ranking_scores, pos_label=1
    )
    positive_count = int(np.count_nonzero(actual == 1))
    predicted_positive = ranking_scores >= threshold
    true_positives = int(np.count_nonzero(predicted_positive & (actual == 1)))
    false_positives = int(np.count_nonzero(predicted_positive & (actual == 0)))
    point_recall = true_positives / positive_count
    point_fpr = false_positives / int(np.count_nonzero(actual == 0))
    prediction_count = int(np.count_nonzero(predicted_positive))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "isic-roc-pr"}
    ):
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        roc_axis, pr_axis = axes
        roc_axis.plot(
            false_positive_rate,
            recall,
            color="#087e83",
            label=f"AUROC = {roc_auc_score(actual, ranking_scores):.4f}",
        )
        roc_axis.plot(
            [0, 1], [0, 1], color="#8b949e", linestyle="--", label="Chance ranking"
        )
        roc_axis.scatter(
            [point_fpr],
            [point_recall],
            color="#c45c35",
            zorder=3,
            label="Frozen decision threshold",
        )
        roc_axis.set(
            xlabel="Benign false-positive rate",
            ylabel="Melanoma recall",
            title="ROC: sensitivity versus false positives",
        )
        pr_axis.plot(
            pr_recall,
            precision,
            color="#087e83",
            label=f"Average precision = {average_precision_score(actual, ranking_scores):.4f}",
        )
        pr_axis.axhline(
            positive_count / len(actual),
            color="#8b949e",
            linestyle="--",
            label=f"Positive prevalence = {positive_count / len(actual):.2%}",
        )
        if prediction_count:
            pr_axis.scatter(
                [point_recall],
                [true_positives / prediction_count],
                color="#c45c35",
                zorder=3,
                label="Frozen decision threshold",
            )
        else:
            pr_axis.text(
                0.5,
                0.5,
                "No positive predictions at the frozen threshold",
                transform=pr_axis.transAxes,
                horizontalalignment="center",
                fontsize=9,
            )
        pr_axis.set(
            xlabel="Melanoma recall",
            ylabel="Melanoma precision",
            title="Precision–recall on saved image population",
        )
        for axis, location in ((roc_axis, "lower right"), (pr_axis, "upper right")):
            axis.set(xlim=(0, 1), ylim=(0, 1.02))
            axis.legend(loc=location, fontsize=8.5)
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(
            f"{title} · {len(actual):,} images / {positive_count:,} melanomas",
            fontsize=13,
            fontweight="bold",
        )
        _save_svg(
            figure,
            output_path,
            description="Calculated from saved image labels and scores; no model execution.",
        )
    return output_path


def load_roc_artifacts(
    run_directory: Path, predictions_path: Path | None = None
) -> tuple[list[int], list[float], float]:
    """Read saved scores and the frozen binary threshold from a run.

    Parameters
    ----------
    run_directory : pathlib.Path
        Directory containing ``metrics.json`` with ``decision_rule.threshold``.
    predictions_path : pathlib.Path or None, default=None
        CSV with ``actual`` and ``score`` columns. Defaults to the run's
        ``calibration_predictions.csv``. An explicit path allows plotting any
        other saved predictions without executing the model.

    Returns
    -------
    labels : list of int
        Ground-truth image labels in CSV order.
    scores : list of float
        Corresponding saved melanoma scores.
    threshold : float
        Frozen decision threshold recorded in the metrics file.

    Raises
    ------
    ValueError
        If required CSV fields or a finite numeric threshold are missing,
        or prediction rows cannot be parsed.
    """

    with (run_directory / "metrics.json").open(encoding="utf-8") as stream:
        report: object = json.load(stream)
    if not isinstance(report, Mapping):
        raise ValueError("Metrics JSON must contain an object")
    if predictions_path is None:
        predictions_path = run_directory / "calibration_predictions.csv"
    labels: list[int] = []
    scores: list[float] = []
    with predictions_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"actual", "score"}.issubset(reader.fieldnames or []):
            raise ValueError("Prediction CSV needs actual and score columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                labels.append(int(row["actual"]))
                scores.append(float(row["score"]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid prediction values in {predictions_path}, line {line_number}"
                ) from error

    decision_rule = report.get("decision_rule")
    if not isinstance(decision_rule, Mapping):
        raise ValueError("Metrics JSON needs a saved decision_rule")
    threshold = decision_rule.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not isfinite(threshold)
    ):
        raise ValueError("Metrics decision_rule needs a finite numeric threshold")
    return labels, scores, float(threshold)


def _curve_arrays(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Validate paired binary labels and finite ranking scores for ROC/PR."""

    actual = np.asarray(labels)
    ranking_scores = np.asarray(scores, dtype=np.float64)
    if actual.ndim != 1 or ranking_scores.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional")
    if len(actual) == 0 or len(actual) != len(ranking_scores):
        raise ValueError("Labels and scores must be non-empty and equal in length")
    if not np.isin(actual, (0, 1)).all() or np.unique(actual).size != 2:
        raise ValueError(
            "ROC/PR curves require both benign (0) and melanoma (1) labels"
        )
    if not np.isfinite(ranking_scores).all():
        raise ValueError("Prediction scores must be finite")
    return actual.astype(np.int64), ranking_scores


def plot_prediction_score_distribution(
    predictions: Sequence[Prediction],
    decision_rule: Mapping[str, float],
    score_label: str,
    output_path: Path,
) -> Path:
    """Write an SVG of held-out scores and the frozen referral policy.

    Parameters
    ----------
    predictions : sequence of Prediction
        Held-out scores generated from a reloaded model.
    decision_rule : mapping of str to float
        Frozen ``threshold`` and ``referral_margin`` values.
    score_label : str
        Axis label explaining the score's model-specific interpretation.
    output_path : pathlib.Path
        SVG destination. Parent directories are created when needed.

    Returns
    -------
    pathlib.Path
        Written SVG path.
    """

    if not predictions:
        raise ValueError("At least one prediction is required for a score figure")
    threshold, margin = _decision_bounds(decision_rule)
    labels = np.asarray([prediction.actual for prediction in predictions])
    scores = np.asarray([prediction.score for prediction in predictions])
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores must be finite")

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    axis.axvspan(
        threshold - margin,
        threshold + margin,
        color="#ffbf00",
        alpha=0.18,
        label="Referral band",
    )
    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        label=f"Frozen threshold = {threshold:.3f}",
    )
    axis.set(
        title="Held-out image score distributions",
        xlabel=score_label,
        ylabel="Density",
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    _save_svg(figure, output_path)
    return output_path


def plot_failure_cases(predictions: Sequence[Prediction], output_path: Path) -> Path:
    """Write an SVG montage labelled with selected held-out decisions.

    Parameters
    ----------
    predictions : sequence of Prediction
        Failure candidates selected by the prediction driver.
    output_path : pathlib.Path
        SVG destination. Parent directories are created when needed.

    Returns
    -------
    pathlib.Path
        Written SVG path.
    """

    if not predictions:
        raise ValueError("At least one prediction is required for a failure figure")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(3, len(predictions))
    rows = ceil(len(predictions) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.2 * rows))
    flattened_axes = np.atleast_1d(axes).ravel()
    for axis, prediction in zip(flattened_axes, predictions):
        axis.imshow(_load_preview_image(prediction.image_path))
        axis.set_title(_prediction_title(prediction), fontsize=9)
        axis.axis("off")
    for axis in flattened_axes[len(predictions) :]:
        axis.axis("off")
    figure.suptitle("Held-out decision-review cases", fontsize=14)
    figure.subplots_adjust(top=0.86)
    _save_svg(figure, output_path)
    return output_path


def plot_reference_cases(
    predictions: Sequence[Prediction],
    gallery: Sequence[ISICSample],
    neighbors: NeighborScores,
    selected: Sequence[int],
    output_path: Path,
    threshold: float,
) -> Path:
    """Show each query with up to three nearest fitting references in an SVG.

    Parameters
    ----------
    predictions : sequence of Prediction
        Query predictions in the same order as the neighbour arrays.
    gallery : sequence of ISICSample
        Fitting metadata in reference-index order.
    neighbors : NeighborScores
        Nearest-reference indices, distances and agreement statistics.
    selected : sequence of int
        Nonempty list of query positions chosen for the montage.
    output_path : pathlib.Path
        SVG destination; parent directories are created when needed.
    threshold : float
        Frozen binary classifier threshold, separate from the triage band.

    Returns
    -------
    pathlib.Path
        Written SVG path. Neighbour votes are labelled as diagnostic fractions,
        rather than probabilities or causal explanations.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = 1 + min(3, neighbors.indices.shape[1])
    figure, axes = plt.subplots(
        len(selected),
        columns,
        figsize=(3.7 * columns, 4.5 * len(selected)),
        squeeze=False,
    )
    for row, index in zip(axes, selected):
        prediction = predictions[index]
        classifier_label = "melanoma" if prediction.score >= threshold else "benign"
        row[0].imshow(_load_preview_image(prediction.image_path))
        row[0].set_title(
            f"Query · {_prediction_title(prediction)}\n"
            f"classifier={classifier_label}; threshold={threshold:.3f}\n"
            f"patient={prediction.patient_id}\n"
            f"neighbour disagreement={neighbors.disagreement[index]:.1%}\n"
            f"melanoma labels={neighbors.melanoma_fraction[index]:.1%}",
            fontsize=8,
        )
        for axis, reference_index, distance in zip(
            row[1:], neighbors.indices[index, :3], neighbors.distances[index, :3]
        ):
            reference = gallery[int(reference_index)]
            actual = "melanoma" if reference.label == 1 else "benign"
            axis.imshow(_load_preview_image(reference.path))
            axis.set_title(
                f"Fitting reference · {reference.image_name}\n"
                f"actual={actual}; distance={distance:.3f}\n"
                f"patient={reference.patient_id}",
                fontsize=8,
            )
        for axis in row:
            axis.axis("off")
    figure.suptitle("Query decisions and nearest fitting references", fontsize=14)
    _save_svg(
        figure,
        output_path,
        description=(
            "References are fitting images only. Neighbour agreement is a "
            "diagnostic, not a probability or causal explanation."
        ),
    )
    return output_path


def plot_referral_workload(
    curves: Mapping[str, Mapping[str, Sequence[float]]],
    output_path: Path,
    partition: str,
) -> Path:
    """Plot descriptive review workload and melanoma capture for an image role.

    Parameters
    ----------
    curves : mapping of str to mapping
        Named review queues containing ascending ``review_fraction`` values and
        their corresponding ``melanoma_recall`` values.
    output_path : pathlib.Path
        SVG destination; parent directories are created when needed.
    partition : str
        Patient role displayed in the title, such as selection or calibration.

    Returns
    -------
    pathlib.Path
        Written SVG path. These diagnostic curves do not select policy cutoffs.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 5.1))
    for name, curve in curves.items():
        axis.step(
            curve["review_fraction"],
            curve["melanoma_recall"],
            where="post",
            label=name.replace("_", " ").capitalize(),
        )
    axis.set(
        title=f"Review workload · {partition} diagnostic",
        xlabel="Fraction of images routed for review",
        ylabel="Fraction of melanomas routed for review (recall)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    _save_svg(
        figure,
        output_path,
        description=(
            "Descriptive workload and melanoma capture on the named patient "
            "role. Deployment cutoffs are chosen separately on calibration patients."
        ),
    )
    return output_path


def plot_embedding_map(
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    output_path: Path,
    partition: str,
) -> Path:
    """Project fitting references and queries into a fitting-only PCA space.

    Parameters
    ----------
    reference_embeddings, query_embeddings : numpy.ndarray
        Embedding matrices with one image per row and matching feature widths.
    reference_labels, query_labels : numpy.ndarray
        Corresponding binary labels: benign (0) or melanoma (1).
    output_path : pathlib.Path
        SVG destination; parent directories are created when needed.
    partition : str
        Query patient role displayed in the title and legend.

    Returns
    -------
    pathlib.Path
        Written SVG path. PCA is fitted on references only; query labels affect
        plotting colours without changing the fitted projection.
    """

    projection = PCA(n_components=2, svd_solver="full")
    references = projection.fit_transform(reference_embeddings)
    queries = projection.transform(query_embeddings)
    variance = projection.explained_variance_ratio_
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 5.9))
    for label, name, color in (
        (0, "benign", "#1f77b4"),
        (1, "melanoma", "#c45c35"),
    ):
        reference_points = references[reference_labels == label]
        query_points = queries[query_labels == label]
        axis.scatter(
            reference_points[:, 0],
            reference_points[:, 1],
            s=42,
            marker="o",
            facecolors="none",
            edgecolors=color,
            alpha=0.7,
            label=f"Fitting references · {name}",
        )
        axis.scatter(
            query_points[:, 0],
            query_points[:, 1],
            s=20,
            marker="x",
            color=color,
            alpha=0.4,
            label=f"{partition.capitalize()} queries · {name}",
        )
    axis.set(
        title=f"Embedding geometry · {partition} diagnostic",
        xlabel=f"PC1 ({variance[0]:.1%} of fitting-reference variance)",
        ylabel=f"PC2 ({variance[1]:.1%} of fitting-reference variance)",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.15)
    axis.spines[["top", "right"]].set_visible(False)
    _save_svg(
        figure,
        output_path,
        description=(
            "PCA fitted on fitting references only and then applied to query "
            "embeddings. Two-dimensional geometry does not establish classifier quality."
        ),
    )
    return output_path


def plot_statistical_comparison(
    summary: Sequence[Mapping[str, Any]], output_path: Path, partition: str
) -> Path:
    """Plot classifier estimates and paired differences with patient intervals.

    Parameters
    ----------
    summary : sequence of mapping
        Rows from compare.py containing comparison/metric names, estimates and
        patient_ci_lower/patient_ci_upper. Estimates average the supplied seeds.
    output_path : pathlib.Path
        SVG destination.
    partition : str
        Evaluated patient role displayed in the figure title.

    Returns
    -------
    pathlib.Path
        Saved plot. Intervals condition on trained runs and frozen calibration;
        they do not incorporate training variability or selection optimism.
    """

    metrics = (
        ("melanoma_average_precision", "Melanoma AP"),
        ("roc_auc", "AUROC"),
        ("macro_f1", "Macro-F1"),
        ("melanoma_recall", "Melanoma recall"),
    )
    lookup = {(row["comparison"], row["metric"]): row for row in summary}
    positions = np.arange(len(metrics))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for name, offset, color in (
        ("baseline", -0.12, "#1f77b4"),
        ("siamese", 0.12, "#c45c35"),
    ):
        rows = [lookup[name, metric] for metric, _ in metrics]
        y = positions + offset
        axes[0].hlines(
            y,
            [float(row["patient_ci_lower"]) for row in rows],
            [float(row["patient_ci_upper"]) for row in rows],
            color=color,
        )
        axes[0].scatter(
            [float(row["estimate"]) for row in rows], y, color=color, label=name
        )
    rows = [lookup["siamese_minus_baseline", metric] for metric, _ in metrics]
    axes[1].hlines(
        positions,
        [float(row["patient_ci_lower"]) for row in rows],
        [float(row["patient_ci_upper"]) for row in rows],
        color="#505050",
    )
    axes[1].scatter(
        [float(row["estimate"]) for row in rows], positions, color="#505050"
    )
    axes[1].axvline(0, color="#999999", linestyle="--", linewidth=1)
    axes[0].set_yticks(positions, [label for _, label in metrics])
    axes[0].invert_yaxis()
    axes[0].set(title="Classifier performance", xlabel="Estimate", xlim=(0, 1))
    axes[0].legend(loc="best")
    axes[1].set(title="Paired objective difference", xlabel="Siamese minus baseline")
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"Seed means and 95% conditional patient intervals · {partition}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_svg(
        figure,
        output_path,
        description=(
            "Patient-cluster percentile intervals conditional on fixed trained models "
            "and calibration. Paired differences use identical patient draws."
        ),
    )
    return output_path


def _decision_bounds(decision_rule: Mapping[str, float]) -> tuple[float, float]:
    """Return finite threshold and referral margin from a saved rule."""

    try:
        threshold = float(decision_rule["threshold"])
        margin = float(decision_rule["referral_margin"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Decision rule needs finite threshold and referral_margin values"
        ) from error
    if not np.isfinite((threshold, margin)).all() or margin < 0:
        raise ValueError("Decision-rule values must be finite and margin non-negative")
    return threshold, margin


def _load_preview_image(path: Path) -> Image.Image:
    """Decode a lesion preview bounded to 768 pixels per side for an SVG."""

    try:
        with Image.open(path) as source:
            preview = ImageOps.exif_transpose(source).convert("RGB")
            preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
            return preview
    except OSError as error:
        raise OSError(f"Could not decode JPEG for visualisation: {path}") from error


def _prediction_title(prediction: Prediction) -> str:
    """Format one compact visualisation label for a held-out lesion."""

    actual = "melanoma" if prediction.actual == 1 else "benign"
    return (
        f"{prediction.image_name}\n"
        f"actual={actual}; score={prediction.score:.3f}\n"
        f"decision={prediction.decision}"
    )


def _save_svg(
    figure: Figure, output_path: Path, *, description: str | None = None
) -> None:
    """Save a figure as SVG and release its Matplotlib resources."""

    metadata = {"Date": None, "Description": description} if description else None
    try:
        figure.tight_layout()
        figure.savefig(
            output_path, format="svg", bbox_inches="tight", metadata=metadata
        )
    finally:
        plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate an ROC/PR SVG from saved artifacts without model execution.

    Parameters
    ----------
    argv : sequence of str or None, default=None
        Command-line arguments. ``None`` reads the process arguments.
    """

    parser = argparse.ArgumentParser(
        description="Plot ROC and precision–recall curves from saved run artifacts."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing metrics.json and saved prediction CSVs.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Saved CSV override; defaults to <run-dir>/calibration_predictions.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="SVG destination; defaults to <run-dir>/roc_pr.svg.",
    )
    parser.add_argument(
        "--title", default="Melanoma classification", help="Figure title prefix."
    )
    args = parser.parse_args(argv)
    labels, scores, threshold = load_roc_artifacts(args.run_dir, args.predictions)
    output_path = args.output or args.run_dir / "roc_pr.svg"
    saved_path = plot_roc_precision_recall(
        labels, scores, threshold, output_path, args.title
    )
    print(f"Saved {saved_path}")


if __name__ == "__main__":
    main()
