"""SVG figures for the report: architecture, training curves, ROC/PR/reliability, embeddings, cases.

    python -m recognition.Siamese_ISIC2020_Melanoma_47472442.visualisation  # redraws the architecture
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .dataset import ISICDataset, MEAN, STD, Sample

FIGURES = Path(__file__).parent / "figures"
COLOURS = {0: "#1f77b4", 1: "#c45c35"}  # benign, melanoma


def _save(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(h_pad=2)
    figure.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(figure)


def _thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.draft("RGB", (512, 512))
        image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((512, 512))
    return image


def plot_architecture(path: Path) -> None:
    """Draw encoder stages and the training and prediction branches.

    Parameters
    ----------
    path : pathlib.Path
        Destination SVG file.
    """
    figure, (encoder, heads) = plt.subplots(2, 1, figsize=(12, 7.4), gridspec_kw={"height_ratios": (1, 2.4)})
    for axis in (encoder, heads):
        axis.set(xlim=(-0.02, 1.02), ylim=(-0.10, 1.10))
        axis.axis("off")
    encoder.set_title("Shared ResNet-18 encoder — the same weights process every image", fontsize=14, pad=14)
    style = {"boxstyle": "round,pad=0.55", "fc": "#edf3f8", "ec": "#2878b5"}
    stages = (
        "RGB image\n3 × 384 × 384",
        "7×7 conv, s2\nBN + ReLU\n3×3 pool, s2\n64 × 96 × 96",
        "Stage 1\n2 basic blocks\n64 × 96 × 96",
        "Stage 2\n2 basic blocks\n128 × 48 × 48",
        "Stage 3\n2 basic blocks\n256 × 24 × 24",
        "Stage 4\n2 basic blocks\n512 × 12 × 12",
        "Global avg pool\nFlatten\n512 features",
    )
    positions = np.linspace(0.06, 0.94, len(stages))
    boxes = [encoder.text(x, 0.5, label, ha="center", va="center", fontsize=12.5, bbox=style) for x, label in zip(positions, stages)]
    for index in range(len(boxes) - 1):
        encoder.annotate("", xy=(positions[index + 1], 0.5), xytext=(positions[index], 0.5), arrowprops={"arrowstyle": "->", "patchA": boxes[index].get_bbox_patch(), "patchB": boxes[index + 1].get_bbox_patch(), "shrinkA": 4, "shrinkB": 4})

    heads.set_title("Classifier and embedding branches", fontsize=14, pad=12)
    heads.set_ylim(-0.24, 1.20)
    nodes = {
        "features": (0.08, 0.55, "Pooled encoder\n512 features", False),
        "logits": (0.34, 0.82, "Linear 512 → 2\nClass logits", False),
        "embedding": (0.34, 0.28, "L2 normalisation\n512-d embedding", False),
        "classification": (0.62, 1.02, "Training only\nCross-entropy loss", True),
        "probability": (0.62, 0.65, "Platt scaling\nMelanoma probability p", False),
        "contrastive": (0.62, 0.28, "Training only\n496 within-batch pairs\nContrastive loss", True),
        "references": (0.62, -0.09, "838 references, same encoder\n15 nearest per class\nReference score r", False),
        "review": (0.92, 0.55, "Fixed review policy\nMelanoma flag /\nreview / benign", False),
    }
    labels = {}
    for name, (x, y, label, training) in nodes.items():
        box = {**style, "fc": "#fff3df", "ec": "#a56b1f"} if training else style
        labels[name] = heads.text(x, y, label, ha="center", va="center", fontsize=13, bbox=box)
    for start, end in (("features", "logits"), ("features", "embedding"), ("logits", "classification"), ("logits", "probability"), ("embedding", "contrastive"), ("embedding", "references"), ("probability", "review"), ("references", "review")):
        connection = "angle,angleA=0,angleB=-90,rad=0" if start == "references" else "arc3"
        heads.annotate("", xy=nodes[end][:2], xytext=nodes[start][:2], arrowprops={"arrowstyle": "->", "patchA": labels[start].get_bbox_patch(), "patchB": labels[end].get_bbox_patch(), "connectionstyle": connection, "shrinkA": 5, "shrinkB": 5})
    heads.text(0.5, -0.34, "Training: 32 images per batch; loss = cross-entropy + λ × contrastive.  λ = 0 (baseline) or 0.5 (Siamese).", ha="center", va="top", fontsize=12)
    _save(figure, path)


def plot_training_history(history: list[dict[str, float]], path: Path) -> None:
    """Training/selection losses and selection AUROC/AP per epoch."""
    epochs = [h["epoch"] for h in history]
    figure, (losses, ranking) = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in (
        ("train_cross_entropy", "Training cross-entropy"),
        ("train_contrastive", "Training contrastive"),
        ("selection_cross_entropy", "Selection cross-entropy"),
    ):
        losses.plot(epochs, [h[key] for h in history], "o-", markersize=3, label=label)
    for key, label in (("selection_auroc", "Selection AUROC"), ("selection_average_precision", "Selection AP")):
        ranking.plot(epochs, [h[key] for h in history], "o-", markersize=3, label=label)
    losses.set(title="Losses", xlabel="Epoch", ylabel="Loss")
    ranking.set(title="Selection ranking", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for axis in (losses, ranking):
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small")
    _save(figure, path)


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarise eight quantile bins, retaining bins with zero melanomas.

    Parameters
    ----------
    labels : numpy.ndarray
        Binary image labels.
    probabilities : numpy.ndarray
        Calibrated melanoma probabilities.

    Returns
    -------
    predicted, observed, positives, counts : tuple of numpy.ndarray
        Mean probability, melanoma fraction, melanoma count and image count
        for each nonempty bin. Equal scores stay in the same bin.
    """
    edges = np.quantile(probabilities, np.linspace(0, 1, 9))
    bins = np.searchsorted(edges[1:-1], probabilities)
    counts = np.bincount(bins, minlength=8)
    positives = np.bincount(bins, weights=labels, minlength=8)
    totals = np.bincount(bins, weights=probabilities, minlength=8)
    occupied = counts > 0
    return totals[occupied] / counts[occupied], positives[occupied] / counts[occupied], positives[occupied].astype(int), counts[occupied]


def plot_curves(labels: np.ndarray, probabilities: np.ndarray, reference_scores: np.ndarray, rule: dict, path: Path, title: str) -> None:
    """ROC and precision-recall for both scores, plus a reliability diagram of the probabilities."""
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), gridspec_kw={"height_ratios": (3.8, 1.4)})
    roc, pr, reliability, bins_axis = axes[0]
    for axis in axes[1]:
        axis.axis("off")
    for scores, name, colour in ((probabilities, "Classifier", "#087e83"), (reference_scores, "References", "#9467bd")):
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        roc.plot(fpr, tpr, color=colour, linewidth=2, label=f"{name} AUC {roc_auc_score(labels, scores):.3f}")
        pr.plot(recall, precision, color=colour, linewidth=2, label=f"{name} AP {average_precision_score(labels, scores):.3f}")
    for scores, key, marker in ((probabilities, "decision_threshold", "o"), (reference_scores, "reference_review_threshold", "s")):
        flagged = scores >= rule[key]
        recall = flagged[labels == 1].mean()
        cutoff = "Binary cutoff" if key == "decision_threshold" else "Review cutoff"
        roc.scatter(flagged[labels == 0].mean(), recall, marker=marker, color="black", zorder=3, label=cutoff)
        pr.scatter(recall, labels[flagged].mean(), marker=marker, color="black", zorder=3, label=cutoff)
    roc.plot([0, 1], [0, 1], "--", color="grey")
    pr.axhline(labels.mean(), ls="--", color="grey", label=f"Prevalence = {labels.mean():.2%}")
    predicted, observed, positives, counts = reliability_bins(labels, probabilities)
    reliability.plot(predicted, observed, "o-", color="#087e83", label="Classifier")
    bins_axis.axis("off")
    bins_axis.set_title("Reliability bins (8)", fontsize=18)
    counts_table = bins_axis.table(
        cellText=[[f"{p:.4g}", f"{positive}/{count}"] for p, positive, count in zip(predicted, positives, counts)],
        colLabels=["Mean p", "Mel./n"], bbox=(0, 0.08, 1, 0.84),
    )
    counts_table.auto_set_font_size(False)
    counts_table.set_fontsize(16)
    for (row, _), cell in counts_table.get_celld().items():
        cell.set_edgecolor("#cbd5df")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#edf3f8")
            cell.set_text_props(weight="bold")
    reliability.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    reliability.set(xscale="log", xlim=(1e-4, 1), ylim=(-0.01, max(0.15, float(observed.max()) * 1.25)))
    roc.set(title="ROC", xlabel="Benign false-positive rate", ylabel="Melanoma recall")
    pr.set(title="Precision-recall", xlabel="Melanoma recall", ylabel="Melanoma precision", ylim=(0, 1))
    reliability.set(title="Reliability", xlabel="Mean predicted probability\n(log scale)", ylabel="Observed melanoma rate")
    for axis, legend_axis in zip((roc, pr, reliability), axes[1]):
        axis.grid(alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=14)
        axis.xaxis.label.set_size(16)
        axis.yaxis.label.set_size(16)
        axis.title.set_size(18)
        legend_axis.legend(*axis.get_legend_handles_labels(), loc="upper left", borderaxespad=0, fontsize=14, frameon=False)
    figure.suptitle(f"{title} · {len(labels):,} images / {labels.sum()} melanomas", fontsize=18)
    _save(figure, path)


def plot_embeddings(gallery: np.ndarray, gallery_labels: np.ndarray, queries: np.ndarray, query_labels: np.ndarray, path: Path, partition: str) -> None:
    """PCA fitted on the training references only, then applied to the queries."""
    pca = PCA(n_components=2).fit(gallery)
    references, projected = pca.transform(gallery), pca.transform(queries)
    figure, axis = plt.subplots(figsize=(8, 6))
    for label, name in ((0, "benign"), (1, "melanoma")):
        axis.scatter(*references[gallery_labels == label].T, s=30, facecolors="none", edgecolors=COLOURS[label], alpha=0.6, label=f"Training references · {name}")
        axis.scatter(*projected[query_labels == label].T, s=14 + 30 * label, marker="x", color=COLOURS[label], alpha=0.35 + 0.5 * label, label=f"{partition} · {name}")
    variance = pca.explained_variance_ratio_
    axis.set(title=f"Embeddings · {partition}", xlabel=f"PC1 ({variance[0]:.0%} of reference variance)", ylabel=f"PC2 ({variance[1]:.0%})")
    axis.legend(fontsize=8)
    _save(figure, path)


def plot_cases(cases: list[dict], path: Path) -> None:
    """Each row: a query image beside its nearest training reference."""
    figure, axes = plt.subplots(len(cases), 2, figsize=(7.5, 3.9 * len(cases)), squeeze=False)
    for (query, reference), case in zip(axes, cases):
        query.imshow(_thumbnail(case["path"]))
        query.set_title(f"{case['image_name']} · actual {case['actual']}\np = {case['probability']:.3g} → {case['decision']}\nreference score = {case['reference_score']:.2f}", fontsize=9)
        reference.imshow(_thumbnail(case["reference_path"]))
        reference.set_title(f"Nearest reference {case['reference_name']}\nactual {case['reference_actual']}", fontsize=9)
        query.axis("off")
        reference.axis("off")
    _save(figure, path)


def plot_case_comparison(samples: list[Sample], predictions: dict[str, dict[str, np.ndarray]], image_size: int, path: Path) -> None:
    """Show original images, actual evaluation crops and both models' results.

    Parameters
    ----------
    samples : list of Sample
        The same selected cases for both models, in display order.
    predictions : dict
        Baseline and Siamese prediction arrays aligned with ``samples``.
    image_size : int
        Evaluation crop size from the saved training configuration.
    path : pathlib.Path
        Destination SVG file.
    """
    dataset = ISICDataset(samples, image_size)
    figure, axes = plt.subplots(len(samples), 2, figsize=(8, 4.6 * len(samples)), squeeze=False)
    for index, (sample, (original, cropped)) in enumerate(zip(samples, axes)):
        normalised, _ = dataset[index]
        image = normalised.permute(1, 2, 0).numpy() * np.asarray(STD) + np.asarray(MEAN)
        original.imshow(_thumbnail(sample.path))
        cropped.imshow(image.clip(0, 1))
        original.set_title(f"{sample.image_name} · actual {('benign', 'melanoma')[sample.label]}\nOriginal image", fontsize=11)
        cropped.set_title(f"Evaluation input (both models)\n{image_size} × {image_size} centre crop", fontsize=11)
        for axis, model in ((original, "baseline"), (cropped, "siamese")):
            values = predictions[model]
            axis.set_xlabel(f"{model.capitalize()}: p={values['probability'][index]:.6g}, r={values['reference_score'][index]:.3f}\nOutput: {values['decision'][index]}", fontsize=10)
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
    _save(figure, path)


if __name__ == "__main__":
    plot_architecture(FIGURES / "architecture.svg")
