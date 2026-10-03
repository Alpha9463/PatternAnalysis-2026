"""SVG plotting helpers for ISIC training artifacts."""

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


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
