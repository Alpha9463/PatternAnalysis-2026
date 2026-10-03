"""Runtime, artifact, profiling, and triage helpers for ISIC experiments."""

import random
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeAlias, overload

import numpy as np
import torch


DeviceData: TypeAlias = torch.Tensor | tuple["DeviceData", ...] | list["DeviceData"]
InferenceCall: TypeAlias = Callable[[], object]


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

    Raises
    ------
    ValueError
        If the requested device is unknown or unavailable.
    """

    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unknown device: {requested}")
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


@overload
def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor: ...


@overload
def to_device(
    data: tuple[DeviceData, ...], device: torch.device
) -> tuple[DeviceData, ...]: ...


@overload
def to_device(data: list[DeviceData], device: torch.device) -> list[DeviceData]: ...


def to_device(data: DeviceData, device: torch.device) -> DeviceData:
    """Move a tensor or nested tuple/list of tensors to a device."""

    if isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    if isinstance(data, list):
        return [to_device(item, device) for item in data]
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
) -> dict[str, object]:
    """Load and minimally validate one experiment checkpoint."""

    loaded: Any = torch.load(path, map_location=map_location)
    if not isinstance(loaded, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    return loaded


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

    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be one-dimensional arrays of equal size")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be between 0 and 1")
    if not 0 <= referral_fraction < 1:
        raise ValueError("referral_fraction must be between 0 and 1")
    positive_scores = np.sort(scores[labels == 1])[::-1]
    if len(positive_scores) == 0:
        raise ValueError("Validation data contains no melanoma images")
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

    try:
        threshold = rule["threshold"]
        margin = rule["referral_margin"]
    except KeyError as error:
        raise ValueError("Triage rule needs threshold and referral_margin") from error
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
        mps = getattr(torch, "mps", None)
        empty_cache = getattr(mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def peak_accelerator_memory_bytes(device: torch.device) -> int | None:
    """Return peak CUDA allocation or current MPS allocation, if available."""

    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps":
        mps = getattr(torch, "mps", None)
        current_allocated_memory = getattr(mps, "current_allocated_memory", None)
        if callable(current_allocated_memory):
            value = current_allocated_memory()
            if isinstance(value, int):
                return value
    return None


def measure_inference_latency_ms(
    inference: InferenceCall,
    device: torch.device,
    warmup_iterations: int = 3,
    repetitions: int = 10,
) -> float:
    """Measure mean synchronous inference latency for one fixed input batch."""

    if warmup_iterations < 0 or repetitions < 1:
        raise ValueError("warmup_iterations must be non-negative and repetitions positive")
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
        mps = getattr(torch, "mps", None)
        synchronise = getattr(mps, "synchronize", None)
        if callable(synchronise):
            synchronise()
