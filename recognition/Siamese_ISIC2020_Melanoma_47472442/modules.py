"""Shared ResNet-18 encoder, melanoma classifier head and Siamese contrastive loss.

Encoder: He et al., Deep Residual Learning, https://arxiv.org/abs/1512.03385
Contrastive loss: Hadsell et al., CVPR 2006, https://doi.org/10.1109/CVPR.2006.100
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class MelanomaModel(nn.Module):
    """ResNet-18 shared by every image: one pass gives class logits and an embedding.

    The 512 backbone features feed a linear benign/melanoma classifier; the same
    features scaled to unit length form the embedding compared between images.
    """

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(512, 2)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[batch, 2]`` logits and ``[batch, 512]`` unit-length embeddings."""
        features = self.encoder(images)
        return self.classifier(features), F.normalize(features, dim=1)


def contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Contrastive loss over every image pair in the batch (Siamese pairs share all weights).

    Same-class pairs are pulled together (squared distance); different-class pairs
    are pushed until they are at least ``margin`` apart. Positive and negative pair
    means are added so the many benign-benign pairs do not swamp melanoma pairs.
    """
    squared = (2 - 2 * embeddings @ embeddings.T).clamp_min(1e-12)  # unit vectors: |a-b|^2 = 2 - 2a.b
    same = labels[:, None] == labels[None, :]
    pairs = torch.ones_like(same).triu(diagonal=1)  # each unordered pair once, no self-pairs
    positive, negative = (same & pairs).float(), (~same & pairs).float()
    pull = squared
    push = F.relu(margin - squared.sqrt()).square()
    # clamp_min(1) keeps the rare melanoma-free batch from dividing by zero
    return (pull * positive).sum() / positive.sum().clamp_min(1) + (push * negative).sum() / negative.sum().clamp_min(1)


def nearest_references(
    queries: torch.Tensor, references: torch.Tensor, reference_labels: torch.Tensor, neighbours: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Siamese comparison with labelled training references.

    Returns each query's nearest reference index ``[queries]`` and its reference
    score ``d_benign / (d_benign + d_melanoma)``, where each ``d`` is the mean distance
    to the ``neighbours`` closest references of that class: 0.5 is equidistant and
    higher is more melanoma-like. Unlike a neighbour vote it is continuous, so a
    recall threshold never collapses onto a tie.
    """
    distances = torch.cdist(queries, references)
    benign, melanoma = (distances[:, reference_labels == label].topk(neighbours, largest=False).values.mean(dim=1) for label in (0, 1))
    return distances.argmin(dim=1), benign / (benign + melanoma)
