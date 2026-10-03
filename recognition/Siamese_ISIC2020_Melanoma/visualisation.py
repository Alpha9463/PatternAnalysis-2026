"""SVG plotting helpers for ISIC training artifacts."""

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageOps


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
    """Write an SVG showing training loss and validation average precision.

    Parameters
    ----------
    history : list of mapping
        Per-epoch records with ``epoch``, ``train_loss``, and
        ``validation_average_precision`` fields.
    output_path : pathlib.Path
        SVG destination. Parent directories are created when needed.

    Returns
    -------
    pathlib.Path
        The written SVG path.

    Raises
    ------
    ValueError
        If history is empty or lacks the required numeric values.
    """

    try:
        epochs = [int(entry["epoch"]) for entry in history]
        losses = [float(entry["train_loss"]) for entry in history]
        average_precision = [
            float(entry["validation_average_precision"]) for entry in history
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "History needs numeric epoch, train_loss, and validation_average_precision"
        ) from error
    if not epochs:
        raise ValueError("History must contain at least one epoch")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, loss_axis = plt.subplots(figsize=(7.2, 4.5))
    average_precision_axis = loss_axis.twinx()
    loss_line = loss_axis.plot(
        epochs, losses, "o-", color="#1f77b4", label="Training loss"
    )
    average_precision_line = average_precision_axis.plot(
        epochs,
        average_precision,
        "o-",
        color="#d62728",
        label="Validation average precision",
    )
    loss_axis.set(
        title="Training history",
        xlabel="Epoch",
        ylabel="Training loss",
        xticks=epochs,
    )
    average_precision_axis.set_ylabel("Validation average precision")
    loss_axis.grid(axis="y", alpha=0.3)
    loss_axis.legend(
        loss_line + average_precision_line,
        ("Training loss", "Validation average precision"),
    )
    figure.tight_layout()
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return output_path


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
        Validation-derived ``threshold`` and ``referral_margin`` values.
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
        label=f"Validation threshold = {threshold:.3f}",
    )
    axis.set(
        title="Held-out test score distributions",
        xlabel=score_label,
        ylabel="Density",
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    _save_svg(figure, output_path)
    return output_path


def plot_failure_cases(
    predictions: Sequence[Prediction], output_path: Path
) -> Path:
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
    """Decode one source lesion image for inclusion in an SVG montage."""

    try:
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB").copy()
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


def _save_svg(figure: Figure, output_path: Path) -> None:
    """Save a figure as SVG and release its Matplotlib resources."""

    figure.tight_layout()
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)
