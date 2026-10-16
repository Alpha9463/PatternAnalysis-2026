"""Serialize run settings and identify input files by their contents."""

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RunConfig


def file_sha256(path: Path) -> str:
    """Hash a file without reading all its contents into memory.

    Parameters
    ----------
    path : pathlib.Path
        Input or saved artifact to identify.

    Returns
    -------
    str
        SHA-256 digest of the file's bytes.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def effective_config(config: RunConfig) -> dict[str, object]:
    """Convert validated settings to JSON-compatible values.

    Parameters
    ----------
    config : RunConfig
        Settings after defaults, YAML and command-line overrides are applied.

    Returns
    -------
    dict of str to object
        Configuration fields with paths converted to strings.
    """

    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(config).items()
    }
