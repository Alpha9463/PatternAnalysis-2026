"""Runtime, artifact, profiling, and triage helpers for ISIC experiments."""

import random
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable local runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, Apple MPS, or CPU, with an optional explicit choice.

    Parameters
    ----------
    requested : {"auto", "cpu", "mps", "cuda"}, default="auto"
        Requested execution device. ``"auto"`` prefers CUDA, then MPS, then
        CPU.

    Returns
    -------
    torch.device
        Device selected for training and inference.
    """

    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    return torch.device(requested)


def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to the selected device."""

    return data.to(device)


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Detach a tensor from autograd and move it to CPU memory."""

    return tensor.detach().cpu()


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy a model state dictionary into CPU tensors for portable loading."""

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable and non-trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(path: Path, checkpoint: Mapping[str, object]) -> None:
    """Persist one complete experiment checkpoint under an ignored run directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(checkpoint), path)


def load_checkpoint(
    path: Path, map_location: torch.device | str | None = None
) -> dict[str, Any]:
    """Load the dictionary saved by the training pipeline."""

    return torch.load(path, map_location=map_location)


def select_triage_rule(
    labels: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
    referral_fraction: float = 0.1,
) -> dict[str, float]:
    """Derive a validation-only threshold and referral band.

    The threshold is the highest value that retains the requested melanoma
    recall on validation data. The referral margin covers the closest fraction
    of validation scores around that threshold.
    """

    positive_scores = np.sort(scores[labels == 1])[::-1]
    required_positives = int(np.ceil(target_recall * len(positive_scores)))
    threshold = float(positive_scores[required_positives - 1])
    margin = float(np.quantile(np.abs(scores - threshold), referral_fraction))
    return {
        "threshold": threshold,
        "referral_margin": margin,
        "target_recall": float(target_recall),
        "referral_fraction": float(referral_fraction),
    }


def apply_triage_rule(score: float, rule: Mapping[str, float]) -> str:
    """Return the frozen automated decision for one model score."""

    threshold = rule["threshold"]
    margin = rule["referral_margin"]
    if score >= threshold + margin:
        return "melanoma"
    if score <= threshold - margin:
        return "benign"
    return "refer"


def reset_peak_accelerator_memory(device: torch.device) -> None:
    """Reset available accelerator-memory counters before one profile measurement."""

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.empty_cache()


def peak_accelerator_memory_bytes(device: torch.device) -> int | None:
    """Return peak CUDA allocation or current MPS allocation, if available."""

    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return None


def measure_inference_latency_ms(
    inference: Callable[[], object],
    device: torch.device,
    warmup_iterations: int = 3,
    repetitions: int = 10,
) -> float:
    """Measure mean synchronous inference latency for one fixed input batch."""

    for _ in range(warmup_iterations):
        inference()
    _synchronise(device)
    start = time.perf_counter()
    for _ in range(repetitions):
        inference()
    _synchronise(device)
    return (time.perf_counter() - start) * 1000 / repetitions


def _synchronise(device: torch.device) -> None:
    """Wait for queued work on an accelerator before timing or profiling."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
