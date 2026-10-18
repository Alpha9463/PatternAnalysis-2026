"""Train, validate, profile, and save ISIC melanoma models.

Run from the repository root::

    /Users/alexanderson/venvs/comp3710/bin/python -m \
      recognition.Siamese_ISIC2020_Melanoma_47472442.train \
      --config recognition/Siamese_ISIC2020_Melanoma_47472442/configs/baseline.yaml

Selection patients choose the best checkpoint; separate calibration patients
choose the triage rule. Final patients are scored only by an explicit
``predict --partition final --evaluate-final`` command after the run is frozen.
"""

import argparse
import copy
import csv
import json
import time
from collections.abc import Sequence
from functools import partial
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Subset

from recognition.Siamese_ISIC2020_Melanoma_47472442.config import (
    RunConfig,
    load_run_config,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.dataset import (
    ISICImageDataset,
    SiamesePairDataset,
    model_preprocessing,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.protocol import (
    ProtocolIndices,
    load_protocol_manifest,
    partition_summary,
    protocol_identity,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.provenance import (
    effective_config,
    file_sha256,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.modules import (
    MelanomaModel,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.utils import (
    apply_triage_rule,
    cpu_state_dict,
    count_parameters,
    measure_inference_latency_ms,
    peak_accelerator_memory_bytes,
    reset_peak_accelerator_memory,
    save_checkpoint,
    select_device,
    select_triage_rule,
    set_seed,
    to_cpu,
    to_device,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.visualisation import (
    plot_training_history,
)


def train_epoch(
    model: MelanomaModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    contrastive_weight: float,
    margin: float,
) -> dict[str, float]:
    """Fit one epoch on identical paired-image batches for either objective.

    Parameters
    ----------
    model : MelanomaModel
        Shared encoder, normalised projection and two-class head.
    loader : torch.utils.data.DataLoader
        Fitting-only pairs with each member's image label.
    optimizer : torch.optim.Optimizer
        Optimiser shared by both experiment configurations.
    device : torch.device
        Training device.
    contrastive_weight, margin : float
        Contrastive multiplier and different-class distance margin. A zero
        multiplier gives the classification-only ablation.

    Returns
    -------
    dict of str to float
        Mean total, classification and unweighted contrastive losses.
    """

    model.train()
    criterion = nn.CrossEntropyLoss()
    totals = {"train_loss": 0.0, "classification_loss": 0.0, "contrastive_loss": 0.0}
    seen = 0
    for first, second, first_targets, second_targets in loader:
        images = to_device(torch.cat((first, second)), device)
        labels = to_device(torch.cat((first_targets, second_targets)), device)
        optimizer.zero_grad(set_to_none=True)
        logits, embeddings = model(images)
        first_embedding, second_embedding = embeddings.chunk(2)
        distance = functional.pairwise_distance(first_embedding, second_embedding)
        same = (first_targets == second_targets).to(device=device, dtype=torch.float32)
        classification_loss = criterion(logits, labels)
        contrastive_loss = (
            same * distance.square()
            + (1 - same) * functional.relu(margin - distance).square()
        ).mean()
        loss = classification_loss + contrastive_weight * contrastive_loss
        loss.backward()
        optimizer.step()
        batch_pairs = len(first_targets)
        totals["train_loss"] += loss.item() * batch_pairs
        totals["classification_loss"] += classification_loss.item() * batch_pairs
        totals["contrastive_loss"] += contrastive_loss.item() * batch_pairs
        seen += batch_pairs
    return {name: value / seen for name, value in totals.items()}


@torch.no_grad()
def score_model(
    model: MelanomaModel, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return image-level melanoma softmax scores and labels for either model."""

    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    for images, targets in loader:
        logits, _ = model(to_device(images, device))
        scores.extend(to_cpu(logits.softmax(dim=1)[:, 1]).tolist())
        labels.extend(targets.tolist())
    return np.asarray(scores), np.asarray(labels, dtype=np.int64)


def classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, object]:
    """Summarise classification and ranking metrics for one fixed threshold."""

    predictions = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
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
    """Parse YAML settings and explicit command-line overrides for one run."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, metavar="PATH", default=None)
    parser.add_argument(
        "--model", choices=("base", "siamese"), default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Initialise ResNet-18 with ImageNet weights (may download once).",
    )
    parser.add_argument("--run-name", default=argparse.SUPPRESS)
    parser.add_argument("--images-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--metadata-csv", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--protocol-manifest", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--target-recall", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--pairs-per-epoch", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--contrastive-weight", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--contrastive-margin", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default=argparse.SUPPRESS,
    )
    parsed = vars(parser.parse_args(argv))
    config_path = parsed.pop("config")
    return load_run_config(config_path, parsed)


@dataclass(frozen=True)
class TrainingData:
    """Dataset views allowed during fitting, selection and calibration.

    Attributes
    ----------
    fit, selection, calibration : torch.utils.data.Subset
        Fitting uses augmented images; all other views are deterministic.
    dataset : ISICImageDataset
        Deterministic metadata/image view used for reporting.
    indices : ProtocolIndices
        Frozen role membership. Final indices are never made into a loader here.
    """

    fit: Subset
    selection: Subset
    calibration: Subset
    dataset: ISICImageDataset
    indices: ProtocolIndices


def build_datasets(args: RunConfig) -> TrainingData:
    """Load the frozen protocol and create development-only dataset views.

    Parameters
    ----------
    args : RunConfig
        Effective training settings including the existing protocol path.

    Returns
    -------
    TrainingData
        Fitting, selection and calibration views, without a final loader.
    """

    dataset = ISICImageDataset(args.images_dir, args.metadata_csv)
    indices = load_protocol_manifest(dataset, args.protocol_manifest)
    augmented = ISICImageDataset(args.images_dir, args.metadata_csv, training=True)
    return TrainingData(
        fit=Subset(augmented, list(indices.fit)),
        selection=Subset(dataset, list(indices.selection)),
        calibration=Subset(dataset, list(indices.calibration)),
        dataset=dataset,
        indices=indices,
    )


@torch.no_grad()
def profile_model(
    model: MelanomaModel,
    selection_loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    """Return parameter count and fixed-batch model-forward latency."""

    images, _ = next(iter(selection_loader))
    device_images = to_device(images, device)
    model.eval()

    return {
        "parameter_count": count_parameters(model),
        "inference_latency_ms": measure_inference_latency_ms(
            partial(model, device_images), device
        ),
        "latency_scope": "classifier_and_embedding_forward_fixed_selection_batch",
        "accelerator_memory_bytes": peak_accelerator_memory_bytes(device),
        "accelerator_memory_kind": (
            "cuda_peak_allocated"
            if device.type == "cuda"
            else ("mps_current_allocated" if device.type == "mps" else None)
        ),
    }


def write_predictions(
    path: Path,
    dataset: ISICImageDataset,
    subset: Subset,
    labels: np.ndarray,
    scores: np.ndarray,
    decision_rule: dict[str, float],
) -> None:
    """Write held-out image scores and frozen triage decisions as CSV."""

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("image_name", "patient_id", "actual", "score", "predicted", "decision")
        )
        for index, actual, score in zip(subset.indices, labels, scores):
            sample = dataset.samples[index]
            writer.writerow(
                (
                    sample.image_name,
                    sample.patient_id,
                    int(actual),
                    float(score),
                    int(score >= decision_rule["threshold"]),
                    apply_triage_rule(float(score), decision_rule),
                )
            )


def main() -> None:
    """Train on fitting patients, select a checkpoint and calibrate its rule.

    Notes
    -----
    This entry point never scores final or development patients.
    Existing run directories are refused to preserve previous artifacts.
    """

    started = time.perf_counter()
    args = parse_run_config()
    set_seed(args.seed)
    device = select_device(args.device)
    run_directory = args.output_dir / args.run_name
    data = build_datasets(args)
    identity = protocol_identity(args.protocol_manifest)
    preprocessing = model_preprocessing()
    config = effective_config(args)
    run_directory.mkdir(parents=True, exist_ok=False)
    config_path = run_directory / "effective_config.json"
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    config_hash = file_sha256(config_path)
    print(
        f"Device: {device}; images: fit={len(data.fit)}, "
        f"selection={len(data.selection)}, calibration={len(data.calibration)}; "
        "final evaluation disabled during training",
        flush=True,
    )
    selection_loader = DataLoader(data.selection, batch_size=args.batch_size)
    calibration_loader = DataLoader(data.calibration, batch_size=args.batch_size)

    reset_peak_accelerator_memory(device)
    model = MelanomaModel(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pairs = SiamesePairDataset(data.fit, args.pairs_per_epoch, args.seed)
    train_loader = DataLoader(
        pairs,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    contrastive_weight = args.contrastive_weight if args.model == "siamese" else 0.0
    objective = {
        "classification_loss": "cross_entropy",
        "contrastive_weight": contrastive_weight,
        "contrastive_margin": args.contrastive_margin,
    }

    best_average_precision = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, float]] = []
    training_seconds = 0.0
    selection_seconds = 0.0
    for epoch in range(args.epochs):
        pairs.set_epoch(epoch)
        phase_started = time.perf_counter()
        losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            contrastive_weight,
            args.contrastive_margin,
        )
        epoch_training_seconds = time.perf_counter() - phase_started
        training_seconds += epoch_training_seconds
        phase_started = time.perf_counter()
        selection_scores, selection_labels = score_model(
            model, selection_loader, device
        )
        epoch_selection_seconds = time.perf_counter() - phase_started
        selection_seconds += epoch_selection_seconds
        average_precision = float(
            average_precision_score(selection_labels, selection_scores)
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                **losses,
                "selection_average_precision": average_precision,
                "training_seconds": epoch_training_seconds,
                "selection_seconds": epoch_selection_seconds,
            }
        )
        plot_training_history(history, run_directory / "training.svg")
        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={losses['train_loss']:.4f}, "
            f"CE={losses['classification_loss']:.4f}, "
            f"contrastive={losses['contrastive_loss']:.4f}, "
            f"selection AP={average_precision:.4f}",
            flush=True,
        )
        if average_precision > best_average_precision:
            best_average_precision = average_precision
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    phase_started = time.perf_counter()
    selection_scores, selection_labels = score_model(model, selection_loader, device)
    selection_seconds += time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    calibration_scores, calibration_labels = score_model(
        model, calibration_loader, device
    )
    decision_rule = select_triage_rule(
        calibration_labels, calibration_scores, args.target_recall
    )
    calibration_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    profile = profile_model(model, selection_loader, device)
    profiling_seconds = time.perf_counter() - phase_started
    fit_patient_ids = sorted(
        {data.dataset.samples[index].patient_id for index in data.indices.fit}
    )
    timings = {
        "training_seconds": training_seconds,
        "selection_seconds": selection_seconds,
        "calibration_seconds": calibration_seconds,
        "profiling_seconds": profiling_seconds,
        "elapsed_before_save_seconds": time.perf_counter() - started,
    }
    report: dict[str, object] = {
        "model": args.model,
        "pretrained": args.pretrained,
        "device": str(device),
        "seed": args.seed,
        "run_name": args.run_name,
        "score_kind": "melanoma_softmax_score",
        "objective": objective,
        "training_budget": {
            "batch_size_pairs": args.batch_size,
            "images_per_full_training_batch": 2 * args.batch_size,
            "pairs_per_epoch": len(pairs),
            "image_presentations_per_epoch": 2 * len(pairs),
            "optimizer_steps_per_epoch": len(train_loader),
            "total_image_presentations": 2 * len(pairs) * args.epochs,
            "total_optimizer_steps": len(train_loader) * args.epochs,
        },
        "protocol": identity,
        "partitions": partition_summary(data.dataset, data.indices),
        "history": history,
        "history_metric_partition": "selection",
        "best_epoch": best_epoch,
        "selection_metric": "melanoma_average_precision",
        "best_selection_average_precision": best_average_precision,
        "decision_rule_partition": "calibration",
        "decision_rule": decision_rule,
        "selection": classification_metrics(
            selection_labels, selection_scores, decision_rule["threshold"]
        ),
        "calibration": classification_metrics(
            calibration_labels, calibration_scores, decision_rule["threshold"]
        ),
        "profile": profile,
        "timings": timings,
        "effective_config_sha256": config_hash,
        "final_evaluated": False,
    }
    checkpoint = {
        "model_name": args.model,
        "state_dict": cpu_state_dict(model),
        "preprocessing": preprocessing,
        "protocol": identity,
        "fit_patient_ids": fit_patient_ids,
        "score_kind": "melanoma_softmax_score",
        "objective": objective,
        "decision_rule": decision_rule,
        "decision_rule_partition": "calibration",
        "best_epoch": best_epoch,
        "effective_config_sha256": config_hash,
    }
    save_checkpoint(run_directory / "checkpoint.pt", checkpoint)
    with (run_directory / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    write_predictions(
        run_directory / "selection_predictions.csv",
        data.dataset,
        data.selection,
        selection_labels,
        selection_scores,
        decision_rule,
    )
    write_predictions(
        run_directory / "calibration_predictions.csv",
        data.dataset,
        data.calibration,
        calibration_labels,
        calibration_scores,
        decision_rule,
    )
    print(json.dumps(report["calibration"], indent=2))
    print(f"Saved development artifacts to {run_directory}")


if __name__ == "__main__":
    main()
