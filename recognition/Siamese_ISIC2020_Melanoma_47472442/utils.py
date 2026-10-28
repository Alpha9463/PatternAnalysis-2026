"""Device selection, image scoring, calibration, thresholds and metrics shared by train and predict."""

import random
import resource
import sys
from math import ceil
from typing import Any, cast

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .dataset import ISICDataset, Sample
from .modules import MelanomaModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def image_loader(samples: list[Sample], config: dict, train: bool = False, sampler=None) -> DataLoader:
    """Batch images with parallel JPEG decoding; workers persist across epochs."""
    dataset = ISICDataset(samples, config["image_size"], train)
    return DataLoader(dataset, config["batch_size"], sampler=sampler, num_workers=config["workers"], persistent_workers=True)


@torch.no_grad()
def score_images(model: MelanomaModel, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """Return melanoma logit margins (log-odds), CPU embeddings and labels in loader order."""
    model.eval()
    margins, embeddings, labels = [], [], []
    for images, targets in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits, embedding = model(images.to(device))
        logits = logits.float()
        margins.append((logits[:, 1] - logits[:, 0]).cpu())
        embeddings.append(embedding.float().cpu())
        labels.append(targets)
    return torch.cat(margins).numpy(), torch.cat(embeddings), torch.cat(labels).numpy()


def fit_platt(margins: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Platt scaling: fit p = sigmoid(a * margin + b) on calibration patients.

    Ranking is unchanged; probabilities are corrected for the oversampled training
    prevalence and label smoothing, so thresholds and confidences are interpretable.
    """
    model = LogisticRegression(C=1e6).fit(margins[:, None], labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def melanoma_probability(margins: np.ndarray, platt: tuple[float, float]) -> np.ndarray:
    return 1 / (1 + np.exp(-(platt[0] * margins + platt[1])))


def recall_threshold(labels: np.ndarray, scores: np.ndarray, recall: float) -> float:
    """Highest threshold that still flags at least ``recall`` of the melanomas (score >= threshold)."""
    melanoma_scores = np.sort(scores[labels == 1])[::-1]
    return float(melanoma_scores[ceil(recall * len(melanoma_scores)) - 1])


def specificity_threshold(labels: np.ndarray, scores: np.ndarray, specificity: float) -> float:
    """Threshold leaving ``specificity`` of benign images below it."""
    return float(np.quantile(scores[labels == 0], specificity))


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Per-class precision/recall/F1, macro-F1, accuracy and confusion counts at one threshold."""
    predicted = (scores >= threshold).astype(int)
    # average=None returns class-wise arrays; sklearn's annotations omit this case and numeric zero_division.
    precision, recall, f1, _ = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        precision_recall_fscore_support(labels, predicted, labels=[0, 1], average=None, zero_division=cast(Any, 0)),
    )
    (tn, fp), (fn, tp) = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1.mean()),
        "benign_precision": float(precision[0]),
        "benign_recall": float(recall[0]),
        "melanoma_precision": float(precision[1]),
        "melanoma_recall": float(recall[1]),
        "melanoma_f1": float(f1[1]),
        "flagged_fraction": float(predicted.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def peak_process_memory_gib() -> float:
    """Peak resident memory of this process (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 if sys.platform == "darwin" else 1024) / 2**30


def accelerator_memory_gib(device: torch.device) -> float:
    """GPU memory held now: MPS driver allocation (no peak counter exists) or CUDA peak reserved."""
    if device.type == "cuda":
        return torch.cuda.max_memory_reserved() / 2**30
    return torch.mps.driver_allocated_memory() / 2**30 if device.type == "mps" else 0.0
