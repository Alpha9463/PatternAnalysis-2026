"""Create, load and summarise the fixed patient split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .provenance import file_sha256

if TYPE_CHECKING:
    from .dataset import ISICImageDataset


_ROLES = ("fit", "selection", "calibration", "final", "development")
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class ProtocolIndices:
    """Image indices for each patient group.

    Attributes
    ----------
    fit, selection, calibration, final : tuple of int
        Fitting, checkpoint selection, calibration and final evaluation.
    development : tuple of int
        Previously examined patients, excluded from the new comparison.
    """

    fit: tuple[int, ...]
    selection: tuple[int, ...]
    calibration: tuple[int, ...]
    final: tuple[int, ...]
    development: tuple[int, ...]


def _split_patients(
    labels: dict[str, int], fractions: tuple[float, ...], seed: int
) -> list[list[str]]:
    """Stratify patients by melanoma status, returning the reserved groups and remainder."""

    rng = random.Random(seed)
    groups: list[list[str]] = [[] for _ in range(len(fractions) + 1)]
    for label in (0, 1):
        patients = sorted(
            patient for patient, target in labels.items() if target == label
        )
        rng.shuffle(patients)
        offset = 0
        for group, fraction in zip(groups, fractions):
            count = round(len(patients) * fraction)
            group.extend(patients[offset : offset + count])
            offset += count
        groups[-1].extend(patients[offset:])
    return [sorted(group) for group in groups]


def create_protocol_manifest(metadata_csv: Path, output_path: Path) -> None:
    """Save the deterministic patient assignments.

    Parameters
    ----------
    metadata_csv, output_path : pathlib.Path
        Labelled ISIC metadata and destination JSON file.
    """

    labels: dict[str, int] = {}
    with metadata_csv.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            patient = row["patient_id"]
            labels[patient] = max(labels.get(patient, 0), int(row["target"]))
    tuning, development, fitting_pool = _split_patients(labels, (0.15, 0.15), 42)
    final, fit = _split_patients(
        {patient: labels[patient] for patient in fitting_pool}, (0.2,), 20261007
    )
    calibration, selection = _split_patients(
        {patient: labels[patient] for patient in tuning}, (0.5,), 20261008
    )
    partitions = dict(zip(_ROLES, (fit, selection, calibration, final, development)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"partitions": partitions}, indent=2) + "\n")


def load_protocol_manifest(
    dataset: ISICImageDataset, manifest_path: Path
) -> ProtocolIndices:
    """Map saved patient assignments to dataset indices.

    Parameters
    ----------
    dataset : ISICImageDataset
        Dataset supplying image metadata.
    manifest_path : pathlib.Path
        Saved patient assignments.

    Returns
    -------
    ProtocolIndices
        Image indices for each patient role.
    """

    with manifest_path.open(encoding="utf-8") as stream:
        partitions: dict[str, list[str]] = json.load(stream)["partitions"]
    patients = {role: set(partitions[role]) for role in _ROLES}
    return ProtocolIndices(
        **{
            role: tuple(
                index
                for index, sample in enumerate(dataset.samples)
                if sample.patient_id in patients[role]
            )
            for role in _ROLES
        }
    )


def protocol_identity(manifest_path: Path) -> dict[str, object]:
    """Record which saved split a run used.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Protocol file inside the repository.

    Returns
    -------
    dict of str to object
        Repository-relative path and content hash.
    """

    return {
        "path": str(manifest_path.resolve().relative_to(_DATA_DIR.parent)),
        "manifest_sha256": file_sha256(manifest_path),
    }


def partition_summary(
    dataset: ISICImageDataset, indices: ProtocolIndices
) -> dict[str, object]:
    """Count images, patients and classes in each role.

    Parameters
    ----------
    dataset : ISICImageDataset
        Dataset supplying image metadata.
    indices : ProtocolIndices
        Image indices grouped by patient role.

    Returns
    -------
    dict of str to object
        Per-role image, patient, benign and melanoma counts.
    """

    summary: dict[str, object] = {}
    for role in _ROLES:
        group = [dataset.samples[index] for index in getattr(indices, role)]
        melanoma = sum(sample.label for sample in group)
        summary[role] = {
            "images": len(group),
            "patients": len({sample.patient_id for sample in group}),
            "benign": len(group) - melanoma,
            "melanoma": melanoma,
        }
    return summary


def main() -> None:
    """Create the split from labelled metadata without opening images."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=_DATA_DIR / "ISIC_2020_Training_GroundTruth_v2.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=_DATA_DIR / "splits" / "isic_protocol.json"
    )
    args = parser.parse_args()
    create_protocol_manifest(args.metadata_csv, args.output)
    print(f"Saved patient split to {args.output}")


if __name__ == "__main__":
    main()
