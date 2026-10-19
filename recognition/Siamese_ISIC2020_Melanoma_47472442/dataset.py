"""Lazy RGB image loading and patient-separated pairs for ResNet-18.

Images and GroundTruth_v2 metadata default to repository ``data/``. Every
image uses explicit resize, centre crop and RGB normalisation transforms.
"""

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Subset
from torchvision import transforms


DEFAULT_METADATA_NAME = "ISIC_2020_Training_GroundTruth_v2.csv"
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def repository_root() -> Path:
    """Return the PatternAnalysis repository root."""

    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    """Return the repository-local directory for untracked data."""

    return repository_root() / "data"


def model_preprocessing() -> dict[str, object]:
    """Describe the fixed ResNet-18 input pipeline saved with each run."""

    return {
        "image_size": 224,
        "color_mode": "RGB",
        "resize_shorter_side": 256,
        "resize": "bilinear",
        "antialias": True,
        "crop": "center",
        "scaling": "float32_0_1",
        "normalization": {"mean": list(_MEAN), "std": list(_STD)},
        "exif_transpose": True,
        "layout": "CHW",
        "fitting_augmentation": ["horizontal_flip_0.5", "vertical_flip_0.5"],
    }


@dataclass(frozen=True)
class ISICSample:
    """Image identifier, patient, binary label and JPEG path."""

    image_name: str
    patient_id: str
    label: int
    path: Path


class ISICImageDataset(Dataset):
    """Load ISIC RGB images with fixed resize, crop and normalisation.

    Parameters
    ----------
    images_dir, metadata_csv : str or pathlib.Path or None, default=None
        Extracted JPEG root and GroundTruth_v2 CSV. ``None`` searches
        repository ``data/``.
    training : bool, default=False
        Apply random horizontal and vertical flips before preprocessing.

    Attributes
    ----------
    samples : list of ISICSample
        CSV records with patient IDs and JPEG paths; pixels are loaded lazily.
    """

    def __init__(
        self,
        images_dir: str | Path | None = None,
        metadata_csv: str | Path | None = None,
        training: bool = False,
    ) -> None:
        self.images_dir = (
            Path(images_dir).expanduser() if images_dir else default_data_dir()
        )
        self.metadata_csv = (
            Path(metadata_csv).expanduser()
            if metadata_csv
            else next(default_data_dir().rglob(DEFAULT_METADATA_NAME))
        )
        self.training = training
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
            ]
        )
        image_paths = {
            path.stem: path
            for path in self.images_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg"}
        }
        with self.metadata_csv.open(newline="", encoding="utf-8-sig") as stream:
            self.samples = [
                ISICSample(
                    row["image_name"],
                    row["patient_id"],
                    int(row["target"]),
                    image_paths[row["image_name"]],
                )
                for row in csv.DictReader(stream)
            ]

    def __len__(self) -> int:
        """Return the number of labelled images."""

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one image and label.

        Parameters
        ----------
        index : int
            Position in the metadata records.

        Returns
        -------
        image : torch.Tensor
            Normalised float32 RGB tensor with shape [3, 224, 224].
        label : torch.Tensor
            Int64 scalar: benign (0) or melanoma (1).
        """

        sample = self.samples[index]
        with Image.open(sample.path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if self.training:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if random.random() < 0.5:
                image = ImageOps.flip(image)
        return cast(torch.Tensor, self.transform(image)), torch.tensor(
            sample.label, dtype=torch.long
        )


class SiamesePairDataset(Dataset):
    """Draw balanced same-label and different-label pairs from training data.

    Parameters
    ----------
    subset : torch.utils.data.Subset
        Patient-separated training subset of an ISICImageDataset.
    pairs_per_epoch : int or None, default=None
        Number of sampled pairs per epoch. Defaults to the number of images.
    seed : int, default=42
        Base seed for reproducible pair selection.

    Notes
    -----
    Pair members always come from different patients. Call ``set_epoch``
    before each training epoch to draw a new deterministic set of pairs.
    Each image's class label is returned for classification; matching labels
    identify positive pairs for the contrastive objective.
    """

    def __init__(
        self, subset: Subset, pairs_per_epoch: int | None = None, seed: int = 42
    ) -> None:
        self.dataset = cast(ISICImageDataset, subset.dataset)
        self.indices = list(subset.indices)
        self.pairs_per_epoch = (
            len(self.indices) if pairs_per_epoch is None else pairs_per_epoch
        )
        self.seed = seed
        self.epoch = 0
        self.by_label: dict[int, dict[str, list[int]]] = {
            0: defaultdict(list),
            1: defaultdict(list),
        }
        for index in self.indices:
            sample = self.dataset.samples[index]
            self.by_label[sample.label][sample.patient_id].append(index)

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible pair stream for an epoch.

        Parameters
        ----------
        epoch : int
            Zero-based training epoch.
        """

        self.epoch = epoch

    def __len__(self) -> int:
        """Return the number of pairs drawn per epoch."""

        return self.pairs_per_epoch

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load a cross-patient image pair and both class labels.

        Parameters
        ----------
        index : int
            Pair position in this epoch.

        Returns
        -------
        first : torch.Tensor
            First RGB image.
        second : torch.Tensor
            Second RGB image from a different patient.
        first_label : torch.Tensor
            First image's int64 class label: benign (0) or melanoma (1).
        second_label : torch.Tensor
            Second image's int64 class label: benign (0) or melanoma (1).
        """

        rng = random.Random(self.seed + self.epoch * self.pairs_per_epoch + index)
        first_label = rng.randrange(2)
        same = index % 2 == 0
        second_label = first_label if same else 1 - first_label
        first_patient = rng.choice(sorted(self.by_label[first_label]))
        possible_second = sorted(
            pid for pid in self.by_label[second_label] if pid != first_patient
        )
        second_patient = rng.choice(possible_second)
        first_index = rng.choice(self.by_label[first_label][first_patient])
        second_index = rng.choice(self.by_label[second_label][second_patient])
        first, first_label_tensor = self.dataset[first_index]
        second, second_label_tensor = self.dataset[second_index]
        return first, second, first_label_tensor, second_label_tensor


def main() -> None:
    """Print the record count and first preprocessed image's shape and label."""

    parser = argparse.ArgumentParser(description="Inspect the local ISIC dataset")
    parser.add_argument("images_dir", type=Path, nargs="?")
    parser.add_argument("metadata_csv", type=Path, nargs="?")
    args = parser.parse_args()
    dataset = ISICImageDataset(args.images_dir, args.metadata_csv)
    image, label = dataset[0]
    print(f"Loaded {len(dataset)} records")
    print(f"First image: {dataset.samples[0].image_name}")
    print(f"Shape: {tuple(image.shape)}, label: {int(label)}")


if __name__ == "__main__":
    main()
