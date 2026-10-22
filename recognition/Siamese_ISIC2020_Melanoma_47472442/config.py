"""YAML configuration and defaults for local ResNet-18 runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from recognition.Siamese_ISIC2020_Melanoma_47472442.dataset import (
    DEFAULT_COLOR_JITTER,
    DEFAULT_ROTATION_DEGREES,
    default_data_dir,
    repository_root,
)
from recognition.Siamese_ISIC2020_Melanoma_47472442.modules import ModelName


DeviceName = Literal["auto", "cpu", "mps", "cuda"]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Settings for one ResNet-18 experiment.

    Parameters
    ----------
    model : {"base", "siamese"}, default="base"
        Direct classification or contrastive embedding learning.
    images_dir, metadata_csv : pathlib.Path or None
        Input paths; ``None`` finds the dataset under repository ``data/``.
    output_dir, protocol_manifest : pathlib.Path
        Run output root and frozen patient assignments.
    run_name : str
        Unique output directory name.
    epochs, batch_size : int
        Optimisation epochs and pairs per training batch. Scoring uses
        ``batch_size`` individual images per batch.
    head_warmup_epochs : int
        Initial classifier-only epochs, included in ``epochs``.
    learning_rate, encoder_learning_rate, min_learning_rate : float
        Initial classifier/encoder AdamW rates and shared cosine-decay minimum.
    weight_decay : float
        Decoupled AdamW weight decay.
    rotation_degrees, color_jitter : float
        Maximum fitting rotation angle and relative brightness/contrast variation.
    seed : int
        Model initialisation and sampling seed.
    target_recall : float
        Calibration target for the melanoma threshold.
    pairs_per_epoch : int
        Paired-image presentations per epoch for both configurations.
    contrastive_weight : float
        Metric-loss multiplier for the Siamese objective; baseline uses zero.
    contrastive_margin : float
        Minimum target distance for different-class pairs.
    device : {"auto", "cpu", "mps", "cuda"}, default="auto"
        ``"auto"`` prefers CUDA, then MPS, then CPU.
    """

    model: ModelName = "base"
    images_dir: Path | None = None
    metadata_csv: Path | None = None
    output_dir: Path = default_data_dir() / "runs"
    protocol_manifest: Path = default_data_dir() / "splits" / "isic_protocol.json"
    run_name: str = "resnet18_baseline_imagenet_staged_30ep_seed42"
    epochs: int = 30
    head_warmup_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-4
    encoder_learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    rotation_degrees: float = DEFAULT_ROTATION_DEGREES
    color_jitter: float = DEFAULT_COLOR_JITTER
    seed: int = 42
    target_recall: float = 0.9
    pairs_per_epoch: int = 4096
    contrastive_weight: float = 0.1
    contrastive_margin: float = 1.0
    device: DeviceName = "auto"


def load_run_config(
    config_path: Path | None = None, overrides: Mapping[str, Any] | None = None
) -> RunConfig:
    """Read YAML settings and apply explicit CLI overrides.

    Parameters
    ----------
    config_path : pathlib.Path or None, default=None
        YAML file; omitted settings use ``RunConfig`` defaults.
    overrides : mapping or None, default=None
        CLI values taking precedence over YAML settings.

    Returns
    -------
    RunConfig
        Effective settings with paths resolved relative to the repository.
    """

    values: dict[str, Any] = {}
    if config_path is not None:
        values.update(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    values.update(overrides or {})
    for key in ("images_dir", "metadata_csv", "output_dir", "protocol_manifest"):
        if values.get(key) is not None:
            path = Path(values[key]).expanduser()
            values[key] = path if path.is_absolute() else repository_root() / path
    return RunConfig(**values)
