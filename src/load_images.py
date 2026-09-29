"""Load labelled ISIC 2020 JPEGs without holding the image archive in memory.

Notes
-----
Use the training JPEG archive and its GroundTruth_v2 CSV from the ISIC 2020
challenge. Images are decoded only when ``dataset[index]`` is called. Patient
IDs remain available in ``dataset.samples`` for a later patient-level split.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_METADATA_NAME = "ISIC_2020_Training_GroundTruth_v2.csv"


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
    ) -> None:
        self.images_dir = (
            Path(images_dir).expanduser() if images_dir is not None else DEFAULT_DATA_DIR
        )
        if metadata_csv is None:
            matches = sorted(DEFAULT_DATA_DIR.rglob(DEFAULT_METADATA_NAME))
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple {DEFAULT_METADATA_NAME} files found under {DEFAULT_DATA_DIR}; "
                    "pass metadata_csv explicitly"
                )
            self.metadata_csv = matches[0] if matches else DEFAULT_DATA_DIR / DEFAULT_METADATA_NAME
        else:
            self.metadata_csv = Path(metadata_csv).expanduser()
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.images_dir}")
        if not self.metadata_csv.is_file():
            raise FileNotFoundError(f"Metadata CSV does not exist: {self.metadata_csv}")
        if color_mode not in {"RGB", "L"}:
            raise ValueError("color_mode must be 'RGB' or 'L'")
        self.color_mode = color_mode
        self.image_size = self._validate_image_size(image_size)
        self.transform = transform

        image_paths = self._index_images(self.images_dir)
        self.samples = self._read_metadata(self.metadata_csv, image_paths)

    @staticmethod
    def _validate_image_size(size: int | tuple[int, int] | None) -> tuple[int, int] | None:
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
        if len(size) != 2 or any(not isinstance(side, int) or side <= 0 for side in size):
            raise ValueError("image_size must be a positive int, (width, height), or None")
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
    def _read_metadata(metadata_csv: Path, image_paths: dict[str, Path]) -> list[ISICSample]:
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
                raise ValueError(f"Metadata CSV needs columns: {', '.join(sorted(required))}")
            for row_number, row in enumerate(reader, start=2):
                image_name = (row["image_name"] or "").strip()
                patient_id = (row["patient_id"] or "").strip()
                target = (row["target"] or "").strip()
                if not image_name or not patient_id:
                    raise ValueError(f"Missing image_name or patient_id on CSV row {row_number}")
                if image_name in seen_names:
                    raise ValueError(f"Duplicate image_name in metadata: {image_name}")
                seen_names.add(image_name)
                if target not in {"0", "1"}:
                    raise ValueError(f"Invalid target {target!r} for {image_name}; expected 0 or 1")
                path = image_paths.get(image_name)
                if path is None:
                    missing_images.append(image_name)
                    continue
                samples.append(ISICSample(
                    image_name=image_name,
                    patient_id=patient_id,
                    lesion_id=(row.get("lesion_id") or "").strip() or None,
                    label=int(target),
                    path=path,
                ))
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
