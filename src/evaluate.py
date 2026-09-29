"""Train and evaluate a CNN or Siamese model on patient-separated ISIC data.

Run from the repository root, for example::

    python -m src.evaluate --config configs/base.yaml
    python -m src.evaluate --config configs/siamese.yaml

An explicit command-line value overrides the YAML setting::

    python -m src.evaluate --config configs/base.yaml --epochs 10

Both models use RGB 224-pixel images and the same patient split. The Siamese
model scores images against a labelled gallery selected from training data.
Its distance score is a ranking signal, not a calibrated probability.
"""

import argparse
import copy
import csv
import json
import random
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from src.base_cnn import BaseCNN
from src.config import RunConfig, load_run_config
from src.load_images import ISICImageDataset, SiamesePairDataset, split_by_patient
from src.siamese import SiameseNetwork
from src.training_utils import select_device, set_seed, to_cpu, to_device


def subset_labels(subset: Subset) -> np.ndarray:
    """Read labels from metadata without decoding images."""

    dataset = subset.dataset
    if not isinstance(dataset, ISICImageDataset):
        raise TypeError("subset must wrap an ISICImageDataset")
    return np.asarray(
        [dataset.samples[i].label for i in subset.indices], dtype=np.int64
    )


def train_base_epoch(
    model: BaseCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train the direct classifier for one epoch and return mean loss."""

    model.train()
    total_loss = 0.0
    seen = 0
    for images, labels in loader:
        images, labels = to_device(images, device), to_device(labels, device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        seen += len(labels)
    return total_loss / seen


@torch.no_grad()
def score_base(
    model: BaseCNN, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return melanoma probabilities and labels from the direct classifier."""

    model.eval()
    scores, labels = [], []
    for images, targets in loader:
        logits = model(to_device(images, device))
        scores.extend(to_cpu(logits.softmax(dim=1)[:, 1]).tolist())
        labels.extend(targets.tolist())
    return np.asarray(scores), np.asarray(labels, dtype=np.int64)


def train_siamese_epoch(
    model: SiameseNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float = 1.0,
) -> float:
    """Train embeddings with a balanced contrastive loss for one epoch."""

    model.train()
    total_loss = 0.0
    seen = 0
    for first, second, same in loader:
        first = to_device(first, device)
        second = to_device(second, device)
        same = to_device(same, device)
        optimizer.zero_grad(set_to_none=True)
        first_embedding, second_embedding = model(first, second)
        distance = F.pairwise_distance(first_embedding, second_embedding)
        loss = (
            same * distance.square() + (1 - same) * F.relu(margin - distance).square()
        ).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(same)
        seen += len(same)
    return total_loss / seen


def make_gallery(subset: Subset, per_class: int, seed: int) -> Subset:
    """Select a labelled training gallery with broad patient coverage."""

    if per_class < 1:
        raise ValueError("gallery-per-class must be positive")
    dataset = subset.dataset
    if not isinstance(dataset, ISICImageDataset):
        raise TypeError("subset must wrap an ISICImageDataset")
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        by_patient: dict[str, list[int]] = {}
        for index in subset.indices:
            sample = dataset.samples[index]
            if sample.label == label:
                by_patient.setdefault(sample.patient_id, []).append(index)
        patients = sorted(by_patient)
        rng.shuffle(patients)
        for patient in patients[:per_class]:
            selected.append(rng.choice(by_patient[patient]))
        if len(patients) < per_class:
            remaining = [
                i for pid in patients for i in by_patient[pid] if i not in selected
            ]
            selected.extend(
                rng.sample(remaining, min(per_class - len(patients), len(remaining)))
            )
    return Subset(dataset, selected)


@torch.no_grad()
def encode(
    model: SiameseNetwork, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, np.ndarray]:
    """Return CPU embeddings and labels for an image loader."""

    model.eval()
    embeddings, labels = [], []
    for images, targets in loader:
        embeddings.append(to_cpu(model.forward_once(to_device(images, device))))
        labels.extend(targets.tolist())
    return torch.cat(embeddings), np.asarray(labels, dtype=np.int64)


@torch.no_grad()
def score_siamese(
    model: SiameseNetwork,
    loader: DataLoader,
    gallery_loader: DataLoader,
    device: torch.device,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank melanoma by nearest gallery distances from each class.

    The score is mean benign distance minus mean melanoma distance. Higher
    scores mean that the image is closer to melanoma references.
    """

    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    gallery_embeddings, gallery_labels = encode(model, gallery_loader, device)
    references = [gallery_embeddings[gallery_labels == label] for label in (0, 1)]
    if any(len(group) < neighbors for group in references):
        raise ValueError("Gallery needs at least 'neighbors' images from each class")
    model.eval()
    scores, labels = [], []
    for images, targets in loader:
        queries = to_cpu(model.forward_once(to_device(images, device)))
        nearest = [
            torch.cdist(queries, group)
            .topk(neighbors, largest=False)
            .values.mean(dim=1)
            for group in references
        ]
        scores.extend((nearest[0] - nearest[1]).tolist())
        labels.extend(targets.tolist())
    return np.asarray(scores), np.asarray(labels, dtype=np.int64)


def threshold_for_recall(
    labels: np.ndarray, scores: np.ndarray, target_recall: float
) -> float:
    """Pick the highest validation threshold meeting melanoma recall."""

    if not 0 < target_recall <= 1:
        raise ValueError("target-recall must be between 0 and 1")
    positives = np.sort(scores[labels == 1])[::-1]
    if len(positives) == 0:
        raise ValueError("Validation data contains no melanoma images")
    needed = int(np.ceil(target_recall * len(positives)))
    return float(positives[needed - 1])


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Summarize binary classification and ranking on one split."""

    predictions = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    per_class = {}
    f1_scores = []
    for index, name in enumerate(("benign", "melanoma")):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[1 - index, index])
        false_negative = int(matrix[index, 1 - index])
        support = true_positive + false_negative
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_scores.append(f1)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1_scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "melanoma_average_precision": float(average_precision_score(labels, scores)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def parse_run_config(argv: Sequence[str] | None = None) -> RunConfig:
    """Parse a YAML-configured run, with explicit CLI values taking precedence.

    Parameters
    ----------
    argv : sequence of str or None, default=None
        Arguments to parse. ``None`` reads the process command line.

    Returns
    -------
    RunConfig
        Validated training settings.
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, metavar="PATH", default=None)
    parser.add_argument(
        "--model", choices=("base", "siamese"), default=argparse.SUPPRESS
    )
    parser.add_argument("--images-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--metadata-csv", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--validation-fraction", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--test-fraction", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--target-recall", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--pairs-per-epoch", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--gallery-per-class", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--neighbors", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default=argparse.SUPPRESS,
    )
    parsed = vars(parser.parse_args(argv))
    config_path = parsed.pop("config")
    return load_run_config(config_path, parsed)


def main() -> None:
    """Run training, validation threshold selection, and held-out testing."""

    args = parse_run_config()

    set_seed(args.seed)
    device = select_device(args.device)
    dataset = ISICImageDataset(
        args.images_dir, args.metadata_csv, image_size=224, color_mode="RGB"
    )
    train, validation, test = split_by_patient(
        dataset, args.validation_fraction, args.test_fraction, args.seed
    )
    print(
        f"Device: {device}; images: train={len(train)}, validation={len(validation)}, test={len(test)}",
        flush=True,
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    model: nn.Module
    train_one_epoch: Callable[[], float]
    score_images: Callable[[DataLoader], tuple[np.ndarray, np.ndarray]]
    pairs: SiamesePairDataset | None = None
    if args.model == "base":
        base_model = BaseCNN().to(device)
        model = base_model
        optimizer = torch.optim.Adam(base_model.parameters(), lr=args.learning_rate)
        counts = np.bincount(subset_labels(train), minlength=2)
        weights = torch.tensor(
            len(train) / (2 * counts), dtype=torch.float32, device=device
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
        base_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

        train_one_epoch = partial(
            train_base_epoch, base_model, base_loader, optimizer, criterion, device
        )
        score_images = partial(score_base, base_model, device=device)
    else:
        siamese_model = SiameseNetwork().to(device)
        model = siamese_model
        optimizer = torch.optim.Adam(siamese_model.parameters(), lr=args.learning_rate)
        pairs = SiamesePairDataset(train, args.pairs_per_epoch, args.seed)
        pair_loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=True)
        gallery = make_gallery(train, args.gallery_per_class, args.seed)
        gallery_loader = DataLoader(gallery, batch_size=args.batch_size)

        train_one_epoch = partial(
            train_siamese_epoch, siamese_model, pair_loader, optimizer, device
        )
        score_images = partial(
            score_siamese,
            siamese_model,
            gallery_loader=gallery_loader,
            device=device,
            neighbors=args.neighbors,
        )

    best_ap = -1.0
    best_state = None
    history = []
    for epoch in range(args.epochs):
        if pairs is not None:
            pairs.set_epoch(epoch)
        loss = train_one_epoch()
        validation_scores, validation_labels = score_images(validation_loader)
        ap = float(average_precision_score(validation_labels, validation_scores))
        history.append(
            {"epoch": epoch + 1, "train_loss": loss, "validation_average_precision": ap}
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={loss:.4f}, validation AP={ap:.4f}",
            flush=True,
        )
        if ap > best_ap:
            best_ap, best_state = ap, copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No model checkpoint was selected")
    model.load_state_dict(best_state)
    validation_scores, validation_labels = score_images(validation_loader)
    threshold = threshold_for_recall(
        validation_labels, validation_scores, args.target_recall
    )
    test_scores, test_labels = score_images(test_loader)
    report = {
        "model": args.model,
        "device": str(device),
        "seed": args.seed,
        "split_images": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "history": history,
        "validation": metrics(validation_labels, validation_scores, threshold),
        "test": metrics(test_labels, test_scores, threshold),
        "score_kind": "melanoma_probability"
        if args.model == "base"
        else "benign_minus_melanoma_distance",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_seed{args.seed}"
    torch.save(model.state_dict(), args.output_dir / f"{stem}.pt")
    with (args.output_dir / f"{stem}_metrics.json").open("w") as stream:
        json.dump(report, stream, indent=2)
    with (args.output_dir / f"{stem}_test_predictions.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_name", "patient_id", "actual", "score", "predicted"))
        for index, actual, score_value in zip(test.indices, test_labels, test_scores):
            sample = dataset.samples[index]
            writer.writerow(
                (
                    sample.image_name,
                    sample.patient_id,
                    int(actual),
                    float(score_value),
                    int(score_value >= threshold),
                )
            )
    print(json.dumps(report["test"], indent=2))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
