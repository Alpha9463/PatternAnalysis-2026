"""Validated configuration for reproducible local training runs."""

from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from recognition.Siamese_ISIC2020_Melanoma_47472442.dataset import (
    default_data_dir,
    repository_root,
)


ModelName = Literal["base", "siamese"]
DeviceName = Literal["auto", "cpu", "mps", "cuda"]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Settings required to train and evaluate one model.

    Omitted fields use defaults selected for a first local experiment. In
    particular, ``device="auto"`` lets ``utils.select_device`` pick
    CUDA, then MPS, then CPU.

    Parameters
    ----------
    model : {"base", "siamese"}, default="base"
        Model architecture to train.
    images_dir : pathlib.Path or None, default=None
        Extracted ISIC image directory. ``None`` uses the loader default.
    metadata_csv : pathlib.Path or None, default=None
        Ground-truth CSV path. ``None`` uses the loader default.
    output_dir : pathlib.Path
        Directory for checkpoints, metrics, and prediction CSVs.
    split_manifest : pathlib.Path
        Patient-level partition shared by baseline and Siamese runs.
    run_name : str, default="base_seed42"
        Directory name for one reproducible training run.
    epochs : int, default=5
        Number of training epochs.
    batch_size : int, default=16
        Images or pairs processed in each optimisation step.
    learning_rate : float, default=0.001
        Adam learning rate.
    seed : int, default=42
        Seed for splitting, pairs, and training.
    validation_fraction : float, default=0.15
        Patient fraction reserved for validation.
    test_fraction : float, default=0.15
        Patient fraction reserved for final testing.
    target_recall : float, default=0.9
        Validation target used to choose the melanoma threshold.
    pairs_per_epoch : int, default=4096
        Balanced Siamese training pairs per epoch.
    gallery_per_class : int, default=100
        Training reference images per class for Siamese scoring.
    neighbors : int, default=5
        Same-class gallery neighbours used for each Siamese score.
    device : {"auto", "cpu", "mps", "cuda"}, default="auto"
        Training device. ``"auto"`` prefers CUDA, then MPS, then CPU.
    """

    model: ModelName = "base"
    images_dir: Path | None = None
    metadata_csv: Path | None = None
    output_dir: Path = default_data_dir() / "runs"
    split_manifest: Path = default_data_dir() / "splits" / "isic_seed42.json"
    run_name: str = "base_seed42"
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-3
    seed: int = 42
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    target_recall: float = 0.9
    pairs_per_epoch: int = 4096
    gallery_per_class: int = 100
    neighbors: int = 5
    device: DeviceName = "auto"

    def __post_init__(self) -> None:
        """Reject invalid values before images are loaded or training starts."""

        if self.model not in ("base", "siamese"):
            raise ValueError("model must be 'base' or 'siamese'")
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', 'mps', or 'cuda'")
        for name in (
            "epochs",
            "batch_size",
            "pairs_per_epoch",
            "gallery_per_class",
            "neighbors",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not _is_positive_number(self.learning_rate):
            raise ValueError("learning_rate must be positive")
        for name in ("validation_fraction", "test_fraction"):
            value = getattr(self, name)
            if not _is_fraction(value):
                raise ValueError(f"{name} must be between 0 and 1")
        if not _is_positive_number(self.target_recall) or self.target_recall > 1:
            raise ValueError("target_recall must be between 0 and 1")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError(
                "validation_fraction and test_fraction must sum to less than 1"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.images_dir is not None and not isinstance(self.images_dir, Path):
            raise ValueError("images_dir must be a path or null")
        if self.metadata_csv is not None and not isinstance(self.metadata_csv, Path):
            raise ValueError("metadata_csv must be a path or null")
        for name in ("output_dir", "split_manifest"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a path")
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise ValueError("run_name must be a non-empty string")


_CONFIG_KEYS = frozenset(field.name for field in fields(RunConfig))
_PATH_KEYS = frozenset(
    {"images_dir", "metadata_csv", "output_dir", "split_manifest"}
)


def _is_positive_number(value: object) -> bool:
    """Return whether a finite integer or float is greater than zero."""

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
        and value > 0
    )


def _is_fraction(value: object) -> bool:
    """Return whether a finite integer or float is strictly between zero and one."""

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
        and 0 < value < 1
    )


def load_run_config(
    config_path: Path | None = None, overrides: Mapping[str, Any] | None = None
) -> RunConfig:
    """Load a YAML configuration and apply explicit command-line overrides.

    Parameters
    ----------
    config_path : pathlib.Path or None, default=None
        YAML file to load. ``None`` starts from the built-in defaults.
    overrides : mapping of str to object or None, default=None
        Explicit command-line values. These take precedence over YAML values.

    Returns
    -------
    RunConfig
        Validated settings for one experiment.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If the YAML document is not a mapping, contains an unknown key, or
        has an invalid setting.
    """

    values: dict[str, Any] = {}
    if config_path is not None:
        values.update(_load_yaml_mapping(config_path))
    if overrides is not None:
        values.update(overrides)
    unknown_keys = sorted(set(values) - _CONFIG_KEYS)
    if unknown_keys:
        names = ", ".join(unknown_keys)
        raise ValueError(f"Unknown configuration key(s): {names}")
    for key in _PATH_KEYS:
        if values.get(key) is not None:
            value = values[key]
            if not isinstance(value, (str, Path)):
                raise ValueError(f"{key} must be a path or null")
            values[key] = _resolve_repository_path(value)
    return RunConfig(**values)


def _resolve_repository_path(value: str | Path) -> Path:
    """Return an absolute path, anchoring relative configuration to the repo."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else repository_root() / path


def _load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    """Read one non-empty YAML mapping from disk."""

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ValueError(
            "Configuration file must contain a mapping of string keys to values"
        )
    return loaded
