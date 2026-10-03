"""Load labelled ISIC 2020 JPEGs without holding the image archive in memory.

Notes
-----
Use the training JPEG archive and its GroundTruth_v2 CSV from the ISIC 2020
challenge. Images are decoded only when ``dataset[index]`` is called. Patient
IDs remain available in ``dataset.samples`` for ``split_by_patient``.
By default, images and the CSV are found under the repository's ``data``
directory, regardless of the current working directory.

Examples
--------
Create a dataset and read one labelled image::

    dataset = ISICImageDataset()
    image, label = dataset[0]  # float32 [3, 224, 224], int64 scalar
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Subset


DEFAULT_METADATA_NAME = "ISIC_2020_Training_GroundTruth_v2.csv"
_PARTITION_NAMES = ("train", "validation", "test")


def repository_root() -> Path:
    """Return the PatternAnalysis repository root from this project directory."""

    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    """Return the repository-local directory reserved for untracked data."""

    return repository_root() / "data"


@dataclass(frozen=True)
class ISICSample:
    """Metadata and file location for one labelled image.

    Attributes
    ----------
    image_name : str
        ISIC image identifier without the JPEG extension.
    patient_id : str
        Patient identifier supplied by the metadata CSV.
    lesion_id : str or None
        Lesion identifier, if present in the metadata CSV.
    label : int
        Binary target: 0 for benign and 1 for melanoma.
    path : pathlib.Path
        Path to the matching JPEG file.
    """

    image_name: str
    patient_id: str
    lesion_id: str | None
    label: int
    path: Path


@dataclass(frozen=True)
class SplitIndices:
    """Dataset indices for one fixed patient-level train/validation/test split.

    Attributes
    ----------
    train : tuple of int
        Dataset indices assigned to training patients.
    validation : tuple of int
        Dataset indices assigned to validation patients.
    test : tuple of int
        Dataset indices assigned to held-out test patients.
    """

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


class ISICImageDataset(Dataset):
    """Load single images from the labelled ISIC 2020 training archive.

    Parameters
    ----------
    images_dir : str or pathlib.Path or None, default=None
        Directory containing the extracted training JPEGs, possibly in nested
        folders. ``None`` searches the repository's ``data`` directory.
    metadata_csv : str or pathlib.Path or None, default=None
        Path to the ISIC 2020 training GroundTruth_v2 CSV. ``None`` searches
        the repository's ``data`` directory and its subfolders.
    image_size : int, tuple of int, or None, default=224
        Square side length, ``(width, height)``, or ``None`` to retain each
        image's original size.
    color_mode : {"RGB", "L"}, default="RGB"
        Load colour images or single-channel grayscale images.
    transform : callable or None, default=None
        Optional function receiving a resized PIL image and returning a
        tensor. Without one, images become channel-first float32 tensors
        scaled to [0, 1].
    training : bool, default=False
        Apply the declared random flips after deterministic resize. Keep this
        ``False`` for validation and test data.

    Attributes
    ----------
    samples : list of ISICSample
        Labelled file records, including patient IDs for later splitting.

    Notes
    -----
    Construction checks that every metadata image has a JPEG file. Image
    pixels are decoded lazily by ``__getitem__``.

    Raises
    ------
    FileNotFoundError
        If the metadata CSV, JPEG directory, or a listed image is missing.
    ValueError
        If an option, image ID, or metadata value is invalid.
    """

    def __init__(
        self,
        images_dir: str | Path | None = None,
        metadata_csv: str | Path | None = None,
        image_size: int | tuple[int, int] | None = 224,
        color_mode: str = "RGB",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        training: bool = False,
    ) -> None:
        data_dir = default_data_dir()
        self.images_dir = (
            Path(images_dir).expanduser()
            if images_dir is not None
            else data_dir
        )
        if metadata_csv is None:
            matches = sorted(data_dir.rglob(DEFAULT_METADATA_NAME))
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple {DEFAULT_METADATA_NAME} files found under {data_dir}; "
                    "pass metadata_csv explicitly"
                )
            self.metadata_csv = (
                matches[0] if matches else data_dir / DEFAULT_METADATA_NAME
            )
        else:
            self.metadata_csv = Path(metadata_csv).expanduser()
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.images_dir}"
            )
        if not self.metadata_csv.is_file():
            raise FileNotFoundError(f"Metadata CSV does not exist: {self.metadata_csv}")
        if color_mode not in {"RGB", "L"}:
            raise ValueError("color_mode must be 'RGB' or 'L'")
        self.color_mode = color_mode
        self.image_size = self._validate_image_size(image_size)
        self.transform = transform
        self.training = training

        image_paths = self._index_images(self.images_dir)
        self.samples = self._read_metadata(self.metadata_csv, image_paths)

    @staticmethod
    def _validate_image_size(
        size: int | tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        """Convert a square side length to a PIL-compatible size.

        Parameters
        ----------
        size : int, tuple of int, or None
            Square side length, ``(width, height)``, or ``None``.

        Returns
        -------
        tuple of int or None
            Validated ``(width, height)`` pair or ``None``.

        Raises
        ------
        ValueError
            If a supplied width or height is not a positive integer.
        TypeError
            If ``size`` is an unsupported scalar, such as a float.
        """

        if size is None:
            return None
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2 or any(
            not isinstance(side, int) or side <= 0 for side in size
        ):
            raise ValueError(
                "image_size must be a positive int, (width, height), or None"
            )
        return size

    @staticmethod
    def _index_images(images_dir: Path) -> dict[str, Path]:
        """Index JPEG paths by ISIC image identifier.

        Parameters
        ----------
        images_dir : pathlib.Path
            Root directory to search recursively.

        Returns
        -------
        dict of str to pathlib.Path
            JPEG paths keyed by filename stem.

        Raises
        ------
        FileNotFoundError
            If no JPEG files are found.
        ValueError
            If multiple JPEG files have the same filename stem.
        """

        paths: dict[str, Path] = {}
        for path in images_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            if path.stem in paths:
                raise ValueError(f"Multiple JPEG files have image ID {path.stem}")
            paths[path.stem] = path
        if not paths:
            raise FileNotFoundError(f"No JPEG images found under {images_dir}")
        return paths

    @staticmethod
    def _read_metadata(
        metadata_csv: Path, image_paths: dict[str, Path]
    ) -> list[ISICSample]:
        """Match CSV labels and patient IDs to indexed JPEG paths.

        Parameters
        ----------
        metadata_csv : pathlib.Path
            CSV containing ``image_name``, ``patient_id``, and ``target``.
        image_paths : dict of str to pathlib.Path
            JPEG paths keyed by image identifier.

        Returns
        -------
        list of ISICSample
            Labelled image records in CSV order.

        Raises
        ------
        FileNotFoundError
            If a metadata image has no matching JPEG.
        ValueError
            If required columns or rows are missing, image IDs repeat, or a
            target is not 0 or 1.
        """

        samples: list[ISICSample] = []
        missing_images: list[str] = []
        seen_names: set[str] = set()
        with metadata_csv.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"image_name", "patient_id", "target"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"Metadata CSV needs columns: {', '.join(sorted(required))}"
                )
            for row_number, row in enumerate(reader, start=2):
                image_name = (row["image_name"] or "").strip()
                patient_id = (row["patient_id"] or "").strip()
                target = (row["target"] or "").strip()
                if not image_name or not patient_id:
                    raise ValueError(
                        f"Missing image_name or patient_id on CSV row {row_number}"
                    )
                if image_name in seen_names:
                    raise ValueError(f"Duplicate image_name in metadata: {image_name}")
                seen_names.add(image_name)
                if target not in {"0", "1"}:
                    raise ValueError(
                        f"Invalid target {target!r} for {image_name}; expected 0 or 1"
                    )
                path = image_paths.get(image_name)
                if path is None:
                    missing_images.append(image_name)
                    continue
                samples.append(
                    ISICSample(
                        image_name=image_name,
                        patient_id=patient_id,
                        lesion_id=(row.get("lesion_id") or "").strip() or None,
                        label=int(target),
                        path=path,
                    )
                )
        if missing_images:
            examples = ", ".join(missing_images[:5])
            raise FileNotFoundError(
                f"No JPEG found for {len(missing_images)} metadata images "
                f"(first IDs: {examples})"
            )
        if not samples:
            raise ValueError(f"No labelled image rows found in {metadata_csv}")
        return samples

    def __len__(self) -> int:
        """Return the number of labelled images.

        Returns
        -------
        int
            Number of metadata rows with matching JPEGs.
        """

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode and return one image with its binary label.

        Parameters
        ----------
        index : int
            Position in ``samples``.

        Returns
        -------
        image : torch.Tensor
            Channel-first image tensor. With the default transform, values
            are float32 in [0, 1].
        label : torch.Tensor
            Scalar int64 target: 0 for benign or 1 for melanoma.

        Raises
        ------
        OSError
            If the JPEG cannot be decoded.
        """

        sample = self.samples[index]
        try:
            with Image.open(sample.path) as source:
                image = ImageOps.exif_transpose(source).convert(self.color_mode)
                if self.image_size is not None:
                    image = image.resize(self.image_size, Image.Resampling.BILINEAR)
                image = self._augment(image)
        except OSError as exc:
            raise OSError(f"Could not decode JPEG image: {sample.path}") from exc

        if self.transform is None:
            pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
            if self.color_mode == "L":
                pixels = pixels[None, :, :]
            else:
                pixels = pixels.transpose(2, 0, 1)
            image_tensor = torch.from_numpy(pixels)
        else:
            image_tensor = self.transform(image)
        label = torch.tensor(sample.label, dtype=torch.long)
        return image_tensor, label

    def _augment(self, image: Image.Image) -> Image.Image:
        """Apply declared training-only augmentations to one resized image."""

        if not self.training:
            return image
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if random.random() < 0.5:
            image = ImageOps.flip(image)
        return image


def split_by_patient(
    dataset: ISICImageDataset,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Create reproducible train, validation, and test image subsets.

    Patients are stratified by whether any of their images is melanoma. All
    images from one patient stay in the same subset.

    Parameters
    ----------
    dataset : ISICImageDataset
        Dataset containing labelled images and patient identifiers.
    validation_fraction : float, default=0.15
        Fraction of patients reserved for validation.
    test_fraction : float, default=0.15
        Fraction of patients reserved for the final test.
    seed : int, default=42
        Seed used to shuffle patient identifiers.

    Returns
    -------
    train : torch.utils.data.Subset
        Images from training patients.
    validation : torch.utils.data.Subset
        Images from validation patients.
    test : torch.utils.data.Subset
        Images from test patients.

    Raises
    ------
    ValueError
        If fractions are invalid or too few patients exist in either stratum.
    """

    indices = split_indices_by_patient(
        dataset, validation_fraction, test_fraction, seed
    )
    return (
        Subset(dataset, list(indices.train)),
        Subset(dataset, list(indices.validation)),
        Subset(dataset, list(indices.test)),
    )


def split_indices_by_patient(
    dataset: ISICImageDataset,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Return patient-disjoint indices for a reproducible stratified split.

    Each patient is represented by their highest image label so a patient with
    any melanoma image belongs to the melanoma stratum. The returned index
    sets preserve metadata order within each partition.

    Parameters
    ----------
    dataset : ISICImageDataset
        Dataset containing labelled images and patient identifiers.
    validation_fraction : float, default=0.15
        Fraction of patients reserved for validation.
    test_fraction : float, default=0.15
        Fraction of patients reserved for held-out testing.
    seed : int, default=42
        Seed used to shuffle patient identifiers inside each stratum.

    Returns
    -------
    SplitIndices
        Patient-disjoint indices for training, validation, and test data.

    Raises
    ------
    ValueError
        If fractions are invalid or too few patients exist in either stratum.
    """

    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")

    patient_labels: dict[str, int] = {}
    for sample in dataset.samples:
        patient_labels[sample.patient_id] = max(
            patient_labels.get(sample.patient_id, 0), sample.label
        )

    rng = random.Random(seed)
    partition: dict[str, str] = {}
    for label in (0, 1):
        patients = sorted(
            patient_id
            for patient_id, target in patient_labels.items()
            if target == label
        )
        if len(patients) < 3:
            raise ValueError(f"At least three patients are needed in stratum {label}")
        rng.shuffle(patients)
        n_validation = max(1, round(len(patients) * validation_fraction))
        n_test = max(1, round(len(patients) * test_fraction))
        if n_validation + n_test >= len(patients):
            raise ValueError(f"Fractions leave no training patients in stratum {label}")
        for patient_id in patients[:n_validation]:
            partition[patient_id] = "validation"
        for patient_id in patients[n_validation : n_validation + n_test]:
            partition[patient_id] = "test"
        for patient_id in patients[n_validation + n_test :]:
            partition[patient_id] = "train"

    grouped_indices: dict[str, list[int]] = {
        name: [] for name in _PARTITION_NAMES
    }
    for index, sample in enumerate(dataset.samples):
        grouped_indices[partition[sample.patient_id]].append(index)
    return SplitIndices(
        train=tuple(grouped_indices["train"]),
        validation=tuple(grouped_indices["validation"]),
        test=tuple(grouped_indices["test"]),
    )


def load_or_create_split_manifest(
    dataset: ISICImageDataset,
    manifest_path: str | Path,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Load or create the fixed patient-level split for an experiment family.

    The JSON manifest records image names and patient IDs, not only integer
    positions. Reloading verifies the metadata still matches before returning
    current dataset indices, so baseline and Siamese runs use the same split.

    Parameters
    ----------
    dataset : ISICImageDataset
        Dataset whose metadata determines the manifest content.
    manifest_path : str or pathlib.Path
        JSON manifest path, normally under the ignored repository ``data``
        directory.
    validation_fraction : float, default=0.15
        Fraction of patients reserved for validation when creating a manifest.
    test_fraction : float, default=0.15
        Fraction of patients reserved for test data when creating a manifest.
    seed : int, default=42
        Patient-shuffle seed recorded in the manifest.

    Returns
    -------
    SplitIndices
        Validated current dataset indices for the fixed split.

    Raises
    ------
    ValueError
        If the manifest is malformed or no longer matches the metadata.
    """

    path = Path(manifest_path).expanduser()
    if path.is_file():
        return _load_split_manifest(
            dataset, path, validation_fraction, test_fraction, seed
        )

    indices = split_indices_by_patient(
        dataset, validation_fraction, test_fraction, seed
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "partitions": {
            name: [
                {
                    "image_name": dataset.samples[index].image_name,
                    "patient_id": dataset.samples[index].patient_id,
                }
                for index in getattr(indices, name)
            ]
            for name in _PARTITION_NAMES
        },
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return indices


def _load_split_manifest(
    dataset: ISICImageDataset,
    path: Path,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> SplitIndices:
    """Validate one JSON split manifest and return current dataset indices."""

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Split manifest must be a JSON object: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported split manifest schema: {path}")
    if payload.get("seed") != seed:
        raise ValueError(f"Split manifest seed differs from requested seed: {path}")
    if payload.get("validation_fraction") != validation_fraction:
        raise ValueError(
            f"Split manifest validation fraction differs from configuration: {path}"
        )
    if payload.get("test_fraction") != test_fraction:
        raise ValueError(
            f"Split manifest test fraction differs from configuration: {path}"
        )
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != set(_PARTITION_NAMES):
        raise ValueError(f"Split manifest needs all partitions: {path}")

    sample_indices = {
        sample.image_name: index for index, sample in enumerate(dataset.samples)
    }
    if len(sample_indices) != len(dataset.samples):
        raise ValueError("Dataset metadata has duplicate image names")
    seen_images: set[str] = set()
    patient_partitions: dict[str, str] = {}
    grouped_indices: dict[str, list[int]] = {name: [] for name in _PARTITION_NAMES}
    for name in _PARTITION_NAMES:
        records = partitions[name]
        if not isinstance(records, list):
            raise ValueError(f"Split manifest partition {name!r} must be a list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Split manifest record in {name!r} must be an object")
            image_name = record.get("image_name")
            patient_id = record.get("patient_id")
            if not isinstance(image_name, str) or not isinstance(patient_id, str):
                raise ValueError(f"Split manifest record in {name!r} is incomplete")
            if image_name in seen_images:
                raise ValueError(f"Duplicate image in split manifest: {image_name}")
            index = sample_indices.get(image_name)
            if index is None:
                raise ValueError(f"Manifest image is missing from metadata: {image_name}")
            current_patient_id = dataset.samples[index].patient_id
            if current_patient_id != patient_id:
                raise ValueError(
                    f"Manifest patient differs for {image_name}: "
                    f"{patient_id!r} != {current_patient_id!r}"
                )
            previous_partition = patient_partitions.setdefault(patient_id, name)
            if previous_partition != name:
                raise ValueError(
                    f"Patient {patient_id} appears in multiple manifest partitions"
                )
            seen_images.add(image_name)
            grouped_indices[name].append(index)
    if seen_images != set(sample_indices):
        raise ValueError("Split manifest does not cover exactly the current metadata")
    return SplitIndices(
        train=tuple(grouped_indices["train"]),
        validation=tuple(grouped_indices["validation"]),
        test=tuple(grouped_indices["test"]),
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
    Targets are 1 for matching labels and 0 for different labels.
    """

    def __init__(
        self, subset: Subset, pairs_per_epoch: int | None = None, seed: int = 42
    ) -> None:
        if not isinstance(subset.dataset, ISICImageDataset):
            raise TypeError("subset must wrap an ISICImageDataset")
        self.dataset = subset.dataset
        self.indices = list(subset.indices)
        self.pairs_per_epoch = (
            len(self.indices) if pairs_per_epoch is None else pairs_per_epoch
        )
        if self.pairs_per_epoch <= 0:
            raise ValueError("pairs_per_epoch must be positive")
        self.seed = seed
        self.epoch = 0
        self.by_label: dict[int, dict[str, list[int]]] = {
            0: defaultdict(list),
            1: defaultdict(list),
        }
        for index in self.indices:
            sample = self.dataset.samples[index]
            self.by_label[sample.label][sample.patient_id].append(index)
        if any(len(self.by_label[label]) < 2 for label in (0, 1)):
            raise ValueError(
                "Each label needs images from at least two training patients"
            )

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load a pair and its same-label target.

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
        same_label : torch.Tensor
            Float32 scalar, 1 for matching labels and 0 otherwise.
        """

        if not 0 <= index < len(self):
            raise IndexError(index)
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
        first, _ = self.dataset[first_index]
        second, _ = self.dataset[second_index]
        return first, second, torch.tensor(float(same), dtype=torch.float32)


def main() -> None:
    """Check an extracted archive by decoding its first labelled image.

    Notes
    -----
    Optional command-line arguments override the repository's ``data``
    directory. The command prints the record count, first image ID, tensor
    shape, and label.
    """

    parser = argparse.ArgumentParser(description="Check a local ISIC 2020 JPEG archive")
    parser.add_argument(
        "images_dir", type=Path, nargs="?", help="Extracted training JPEG directory"
    )
    parser.add_argument(
        "metadata_csv", type=Path, nargs="?", help="ISIC 2020 GroundTruth_v2 CSV"
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()
    dataset = ISICImageDataset(
        args.images_dir,
        args.metadata_csv,
        image_size=args.image_size,
        color_mode="L" if args.grayscale else "RGB",
    )
    image, label = dataset[0]
    print(f"Loaded {len(dataset)} records")
    print(f"First image: {dataset.samples[0].image_name}")
    print(f"Shape: {tuple(image.shape)}, label: {int(label)}")


if __name__ == "__main__":
    main()
