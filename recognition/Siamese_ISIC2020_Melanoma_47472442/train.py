"""Train the baseline or Siamese model, keep the best epoch and freeze its decision rules.

    python -m recognition.Siamese_ISIC2020_Melanoma_47472442.train \
        recognition/Siamese_ISIC2020_Melanoma_47472442/configs/siamese.yaml

Selection patients choose the epoch; calibration patients fit the probability
scaling, thresholds and reference gallery rule. Final patients are never loaded
here; ``predict.py`` scores them with the frozen checkpoint.
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch.nn import functional as F

from .dataset import DATA_DIR, Sample, load_splits, melanoma_sampler, read_samples
from .modules import MelanomaModel, contrastive_loss, nearest_references
from .utils import (
    accelerator_memory_gib,
    fit_platt,
    image_loader,
    melanoma_probability,
    peak_process_memory_gib,
    recall_threshold,
    score_images,
    select_device,
    set_seed,
    specificity_threshold,
    threshold_metrics,
)
from .visualisation import plot_training_history


def train_epoch(model, loader, optimizer, scheduler, device, config) -> dict[str, float]:
    """One pass of cross-entropy plus weighted contrastive loss (weight 0 for the baseline)."""
    model.train()
    totals = np.zeros(2)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits, embeddings = model(images)
        cross_entropy = F.cross_entropy(logits.float(), labels, label_smoothing=config["label_smoothing"])
        contrastive = contrastive_loss(embeddings.float(), labels, config["contrastive_margin"])
        loss = cross_entropy + config["contrastive_weight"] * contrastive
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        totals += (cross_entropy.item(), contrastive.item())
    cross_entropy, contrastive = totals / len(loader)
    return {"train_cross_entropy": cross_entropy, "train_contrastive": contrastive}


def reference_gallery(train: list[Sample], seed: int) -> list[Sample]:
    """Every training melanoma plus an equal number of random training benign images."""
    melanomas = [s for s in train if s.label == 1]
    benign = random.Random(seed).sample([s for s in train if s.label == 0], len(melanomas))
    return melanomas + benign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="YAML hyperparameters, e.g. configs/siamese.yaml")
    parser.add_argument("--run-name", help="output folder under data/runs (default: config name)")
    parser.add_argument("--seed", type=int, help="override the config seed (patient splits stay fixed)")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    run_dir = DATA_DIR / "runs" / (args.run_name or args.config.stem)
    run_dir.mkdir(parents=True)  # fails rather than overwrite an earlier run

    set_seed(config["seed"])
    device = select_device()
    splits = load_splits(read_samples())
    sampler = melanoma_sampler(splits["train"], config["melanoma_fraction"], config["seed"])
    train_loader = image_loader(splits["train"], config, train=True, sampler=sampler)
    selection_loader = image_loader(splits["selection"], config)
    model = MelanomaModel(pretrained=config["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), config["learning_rate"], weight_decay=config["weight_decay"])
    # Linear warm-up for the first 10% of steps, then cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, config["learning_rate"], total_steps=config["epochs"] * len(train_loader), pct_start=0.1
    )

    history: list[dict[str, float]] = []
    best_auroc, peak_gpu = 0.0, 0.0
    best_state, best_epoch = model.state_dict(), 0
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        losses = train_epoch(model, train_loader, optimizer, scheduler, device, config)
        peak_gpu = max(peak_gpu, accelerator_memory_gib(device))
        margins, _, labels = score_images(model, selection_loader, device)
        history.append({
            "epoch": epoch,
            **losses,
            "selection_cross_entropy": float(log_loss(labels, 1 / (1 + np.exp(-margins)))),
            "selection_auroc": float(roc_auc_score(labels, margins)),
            "selection_average_precision": float(average_precision_score(labels, margins)),
        })
        print(", ".join(f"{key}={value:.4f}" for key, value in history[-1].items()), flush=True)
        plot_training_history(history, run_dir / "training.svg")
        # AUROC uses every selection image; AP over 41 melanomas is too noisy to pick epochs
        if history[-1]["selection_auroc"] > best_auroc:
            best_auroc, best_epoch = history[-1]["selection_auroc"], epoch
            best_state = copy.deepcopy(model.state_dict())
    training_hours = (time.perf_counter() - started) / 3600
    model.load_state_dict(best_state)

    # Freeze every decision rule on calibration patients
    gallery = reference_gallery(splits["train"], config["seed"])
    _, gallery_embeddings, gallery_labels = score_images(model, image_loader(gallery, config), device)
    margins, embeddings, labels = score_images(model, image_loader(splits["calibration"], config), device)
    platt = fit_platt(margins, labels)
    probabilities = melanoma_probability(margins, platt)
    _, reference_scores = nearest_references(
        embeddings, gallery_embeddings, torch.from_numpy(gallery_labels), config["neighbours"]
    )
    rule = {
        "platt": platt,
        "decision_threshold": specificity_threshold(labels, probabilities, config["target_specificity"]),
        "review_threshold": recall_threshold(labels, probabilities, config["review_recall"]),
        "reference_review_threshold": recall_threshold(labels, reference_scores.numpy(), config["review_recall"]),
    }
    torch.save({
        "config": config,
        "state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
        "rule": rule,
        "gallery_names": [s.image_name for s in gallery],
        "gallery_embeddings": gallery_embeddings,
        "gallery_labels": torch.from_numpy(gallery_labels),
    }, run_dir / "checkpoint.pt")

    report = {
        "best_epoch": best_epoch,
        "rule": rule,
        "calibration": {
            "auroc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "decision": threshold_metrics(labels, probabilities, rule["decision_threshold"]),
            "reference_review": threshold_metrics(labels, reference_scores.numpy(), rule["reference_review_threshold"]),
            "classifier_review": threshold_metrics(labels, probabilities, rule["review_threshold"]),
        },
        "profile": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_hours": training_hours,
            "peak_gpu_memory_gib": peak_gpu,
            "peak_process_memory_gib": peak_process_memory_gib(),
        },
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("best_epoch", "rule", "calibration")}, indent=2))


if __name__ == "__main__":
    main()
