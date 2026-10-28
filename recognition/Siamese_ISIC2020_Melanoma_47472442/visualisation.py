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
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

FIGURES = Path(__file__).parent / "figures"
COLOURS = {0: "#1f77b4", 1: "#c45c35"}  # benign, melanoma


def _save(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)


def _thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.draft("RGB", (512, 512))
        image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((512, 512))
    return image


def plot_architecture(path: Path) -> None:
    """Shared encoder feeding the classifier head and the Siamese embedding."""
    figure, axis = plt.subplots(figsize=(12, 4.2))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    box = {"boxstyle": "round,pad=0.7", "fc": "#edf3f8", "ec": "#2878b5"}
    for x, y, text in (
        (0.10, 0.54, "RGB image\n3 × 384 × 384"),
        (0.36, 0.54, "Shared ResNet-18\n512 features"),
        (0.68, 0.79, "Linear 512 → 2\nclass logits"),
        (0.68, 0.28, "Unit-length\n512-d embedding"),
    ):
        axis.text(x, y, text, ha="center", va="center", fontsize=11, bbox=box)
    for start, end in (((0.19, 0.54), (0.26, 0.54)), ((0.46, 0.58), (0.57, 0.76)), ((0.46, 0.50), (0.58, 0.30))):
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->"})
    axis.text(0.88, 0.79, "Cross-entropy (training)\nPlatt-scaled melanoma\nprobability (decision)", ha="center", va="center", fontsize=10)
    axis.text(0.88, 0.28, "Contrastive loss on all\nbatch pairs (Siamese only)\nReference score vs training\nimages (review triage)", ha="center", va="center", fontsize=10)
    axis.text(0.5, 0.03, "Every image in a batch passes through the same weights. Baseline contrastive weight = 0.", ha="center", fontsize=10)
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


def plot_curves(labels: np.ndarray, probabilities: np.ndarray, reference_scores: np.ndarray, rule: dict, path: Path, title: str) -> None:
    """ROC and precision-recall for both scores, plus a reliability diagram of the probabilities."""
    figure, (roc, pr, reliability) = plt.subplots(1, 3, figsize=(16, 4.8))
    for scores, name, colour in ((probabilities, "Classifier", "#087e83"), (reference_scores, "References", "#9467bd")):
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        roc.plot(fpr, tpr, color=colour, label=f"{name} AUROC = {roc_auc_score(labels, scores):.3f}")
        pr.plot(recall, precision, color=colour, label=f"{name} AP = {average_precision_score(labels, scores):.3f}")
    for scores, key, marker in ((probabilities, "decision_threshold", "o"), (reference_scores, "reference_review_threshold", "s")):
        flagged = scores >= rule[key]
        recall = flagged[labels == 1].mean()
        roc.scatter(flagged[labels == 0].mean(), recall, marker=marker, color="black", zorder=3, label=key.replace("_", " "))
        pr.scatter(recall, labels[flagged].mean(), marker=marker, color="black", zorder=3)
    roc.plot([0, 1], [0, 1], "--", color="grey")
    pr.axhline(labels.mean(), ls="--", color="grey", label=f"Prevalence = {labels.mean():.2%}")
    observed, predicted = calibration_curve(labels, probabilities, n_bins=8, strategy="quantile")
    reliability.plot(predicted, observed, "o-", color="#087e83", label="Classifier")
    reliability.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    reliability.set(xscale="log", yscale="log", xlim=(1e-4, 1), ylim=(1e-4, 1))
    roc.set(title="ROC", xlabel="Benign false-positive rate", ylabel="Melanoma recall")
    pr.set(title="Precision-recall", xlabel="Melanoma recall", ylabel="Melanoma precision", ylim=(0, 1))
    reliability.set(title="Reliability (equal-count bins)", xlabel="Mean predicted probability", ylabel="Observed melanoma rate")
    for axis in (roc, pr, reliability):
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(f"{title} · {len(labels):,} images / {labels.sum()} melanomas")
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


if __name__ == "__main__":
    plot_architecture(FIGURES / "architecture.svg")
