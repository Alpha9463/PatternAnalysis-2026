"""ISIC 2020 loading, patient-level splits, augmentation and class-balanced sampling.

Images and ``ISIC_2020_Training_GroundTruth_v2.csv`` live under repository ``data/``.
Create the fixed patient split once with
``python -m recognition.Siamese_ISIC2020_Melanoma_47472442.dataset``.
"""

import csv
import json
import random
from pathlib import Path
from typing import NamedTuple

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
METADATA_CSV = DATA_DIR / "ISIC_2020_Training_GroundTruth_v2.csv"
SPLIT_FILE = DATA_DIR / "splits" / "isic_protocol.json"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet RGB statistics


class Sample(NamedTuple):
    image_name: str
    patient_id: str
    label: int  # 0 benign, 1 melanoma
    path: Path


def read_samples(metadata_csv: Path = METADATA_CSV) -> list[Sample]:
    """Read labelled images in CSV order, locating each JPEG anywhere under ``data/``."""
    paths = {path.stem: path for path in DATA_DIR.rglob("*.jpg")}
    with metadata_csv.open(newline="") as stream:
        return [
            Sample(row["image_name"], row["patient_id"], int(row["target"]), paths[row["image_name"]])
            for row in csv.DictReader(stream)
        ]


def _split_patients(labels: dict[str, int], fractions: tuple[float, ...], seed: int) -> list[list[str]]:
    """Shuffle melanoma and melanoma-free patients separately, cut off each fraction, keep the rest last."""
    rng = random.Random(seed)
    groups: list[list[str]] = [[] for _ in range(len(fractions) + 1)]
    for label in (0, 1):
        patients = sorted(patient for patient, target in labels.items() if target == label)
        rng.shuffle(patients)
        start = 0
        for group, fraction in zip(groups, fractions):
            count = round(len(patients) * fraction)
            group.extend(patients[start : start + count])
            start += count
        groups[-1].extend(patients[start:])
    return [sorted(group) for group in groups]


def create_splits(samples: list[Sample], output: Path = SPLIT_FILE) -> None:
    """Assign whole patients to roles, stratified by whether the patient has any melanoma.

    ``development`` patients were examined in early experiments; they now only add
    training data (see ``load_splits``) and are never used for evaluation.
    """
    labels: dict[str, int] = {}
    for sample in samples:
        labels[sample.patient_id] = max(labels.get(sample.patient_id, 0), sample.label)
    tuning, development, pool = _split_patients(labels, (0.15, 0.15), 42)
    final, fit = _split_patients({p: labels[p] for p in pool}, (0.2,), 20261007)
    calibration, selection = _split_patients({p: labels[p] for p in tuning}, (0.5,), 20261008)
    partitions = dict(fit=fit, selection=selection, calibration=calibration, final=final, development=development)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"partitions": partitions}, indent=2) + "\n")


def load_splits(samples: list[Sample], split_file: Path = SPLIT_FILE) -> dict[str, list[Sample]]:
    """Return image samples for ``train``, ``selection``, ``calibration`` and ``final`` patients."""
    partitions = json.loads(split_file.read_text())["partitions"]
    roles = {
        "train": set(partitions["fit"] + partitions["development"]),
        "selection": set(partitions["selection"]),
        "calibration": set(partitions["calibration"]),
        "final": set(partitions["final"]),
    }
    return {role: [s for s in samples if s.patient_id in patients] for role, patients in roles.items()}


class ISICDataset(Dataset):
    """Decode, augment (training only) and normalise lesion images to ``image_size`` squares."""

    def __init__(self, samples: list[Sample], image_size: int, train: bool = False) -> None:
        self.samples = samples
        self.image_size = image_size
        geometry = (
            [
                # Random crops of 35-100% of the image vary scale and position of the lesion
                transforms.RandomResizedCrop(image_size, scale=(0.35, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45, fill=[round(255 * m) for m in MEAN]),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            ]
            if train
            else [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
        )
        self.transform = transforms.Compose([*geometry, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            # JPEG draft mode decodes straight to >= image_size: ~4x faster on 6000x4000 photos
            image.draft("RGB", (self.image_size, self.image_size))
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), sample.label


def melanoma_sampler(samples: list[Sample], melanoma_fraction: float, seed: int) -> WeightedRandomSampler:
    """Draw one epoch of ``len(samples)`` images, a ``melanoma_fraction`` share of them melanoma.

    Moderate oversampling keeps melanomas in most batches (needed for contrastive
    pairs) without repeating the few training melanomas so often they are memorised.
    """
    melanomas = sum(s.label for s in samples)
    weights = [melanoma_fraction / melanomas if s.label else (1 - melanoma_fraction) / (len(samples) - melanomas) for s in samples]
    return WeightedRandomSampler(weights, len(samples), generator=torch.Generator().manual_seed(seed))


if __name__ == "__main__":
    all_samples = read_samples()
    create_splits(all_samples)
    for role, group in load_splits(all_samples).items():
        print(f"{role}: {len(group)} images, {len({s.patient_id for s in group})} patients, {sum(s.label for s in group)} melanomas")
