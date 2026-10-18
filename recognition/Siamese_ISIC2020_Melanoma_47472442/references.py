"""Fitting-reference selection, image encoding and neighbour diagnostics."""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import ISICImageDataset
from .modules import MelanomaModel
from .utils import to_cpu, to_device


@dataclass(frozen=True)
class NeighborScores:
    """Nearest fitting references and their agreement with classifier decisions.

    Attributes
    ----------
    indices : numpy.ndarray
        Reference positions with shape ``[queries, neighbors]``.
    distances : numpy.ndarray
        Euclidean embedding distances with shape ``[queries, neighbors]``.
    melanoma_fraction : numpy.ndarray
        Fraction of neighbouring reference labels that are melanoma per query.
    disagreement : numpy.ndarray
        Fraction of neighbours opposing each query's binary classifier decision.
    """

    indices: np.ndarray
    distances: np.ndarray
    melanoma_fraction: np.ndarray
    disagreement: np.ndarray


def select_gallery(fit: Subset, per_class: int, seed: int) -> Subset:
    """Select balanced fitting references with patient diversity within each class.

    Parameters
    ----------
    fit : torch.utils.data.Subset
        Fitting indices from an ISIC image dataset.
    per_class : int
        Maximum number of references per class.
    seed : int
        Seed for patient selection and one image per patient per class.

    Returns
    -------
    torch.utils.data.Subset
        Equal class counts, limited by the smaller unique-patient count.
    """

    dataset = cast(ISICImageDataset, fit.dataset)
    by_label: dict[int, dict[str, list[int]]] = {
        0: defaultdict(list),
        1: defaultdict(list),
    }
    for index in sorted(fit.indices):
        sample = dataset.samples[index]
        by_label[sample.label][sample.patient_id].append(index)
    count = min(per_class, len(by_label[0]), len(by_label[1]))
    rng = random.Random(seed)
    selected: list[int] = []
    for label in (0, 1):
        patients = sorted(by_label[label])
        rng.shuffle(patients)
        for patient in patients[:count]:
            selected.append(rng.choice(by_label[label][patient]))
    return Subset(dataset, selected)


def encode_images(
    model: MelanomaModel,
    subset: Subset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    """Encode images in subset order without gradients.

    Parameters
    ----------
    model : MelanomaModel
        Loaded classifier and embedding model on the requested device.
    subset : torch.utils.data.Subset
        Images from a dataset with deterministic evaluation preprocessing.
    batch_size : int
        Number of images per model call.
    device : torch.device
        Device hosting the model.

    Returns
    -------
    scores : numpy.ndarray
        Melanoma softmax scores in subset order.
    embeddings : torch.Tensor
        CPU embeddings in subset order with shape ``[images, 128]``.
    """

    model.eval()
    score_batches: list[torch.Tensor] = []
    embedding_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for images, _ in DataLoader(subset, batch_size=batch_size, shuffle=False):
            logits, embeddings = model(to_device(images, device))
            score_batches.append(to_cpu(logits.softmax(dim=1)[:, 1]))
            embedding_batches.append(to_cpu(embeddings))
    return torch.cat(score_batches).numpy().astype(float), torch.cat(embedding_batches)


def score_neighbors(
    query_embeddings: torch.Tensor,
    query_scores: np.ndarray,
    reference_embeddings: torch.Tensor,
    reference_labels: np.ndarray,
    threshold: float,
    neighbors: int,
) -> NeighborScores:
    """Compare classifier decisions with nearby fitting-reference labels.

    Parameters
    ----------
    query_embeddings, reference_embeddings : torch.Tensor
        Image embeddings with matching feature dimensions.
    query_scores : numpy.ndarray
        Melanoma scores in query order.
    reference_labels : numpy.ndarray
        Binary labels in fitting-reference order.
    threshold : float
        Saved classifier threshold; scores at or above it predict melanoma.
    neighbors : int
        Requested neighbours per query, capped by the reference count.

    Returns
    -------
    NeighborScores
        Nearest-reference positions, distances, melanoma fraction and disagreement.

    Notes
    -----
    Neighbour fractions describe fitting-reference labels, not probabilities.
    """

    distances, indices = torch.cdist(
        to_cpu(query_embeddings), to_cpu(reference_embeddings)
    ).topk(k=min(neighbors, len(reference_labels)), largest=False)
    neighbor_indices = indices.numpy()
    neighbor_labels = reference_labels[neighbor_indices]
    predictions = (query_scores >= threshold)[:, None]
    return NeighborScores(
        indices=neighbor_indices,
        distances=distances.numpy(),
        melanoma_fraction=(neighbor_labels == 1).mean(axis=1),
        disagreement=(neighbor_labels != predictions).mean(axis=1),
    )
