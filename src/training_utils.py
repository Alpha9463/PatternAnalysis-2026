"""Small utilities shared by local model-training scripts."""

import random
from typing import TypeAlias, overload

import numpy as np
import torch

DeviceData: TypeAlias = torch.Tensor | tuple["DeviceData", ...] | list["DeviceData"]


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for repeatable runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, Apple MPS, or CPU, with optional explicit choice.

    Parameters
    ----------
    requested : {"auto", "cpu", "mps", "cuda"}, default="auto"
        Requested device. ``"auto"`` prefers CUDA, then MPS, then CPU.

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
    """Move a tensor or a nested tuple/list of tensors to a device."""

    if isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    if isinstance(data, list):
        return [to_device(item, device) for item in data]
    return data.to(device)


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Detach a tensor from autograd and move it to CPU memory."""

    return tensor.detach().cpu()
