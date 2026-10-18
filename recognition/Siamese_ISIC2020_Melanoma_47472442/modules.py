"""Shared classifier and metric-learning model for ISIC melanoma triage."""

from typing import Literal, cast

import torch
from torch import nn
from torch.nn import functional as functional
from torchvision.models import ResNet18_Weights, resnet18


ModelName = Literal["base", "siamese"]


def _resnet18_encoder(pretrained: bool) -> nn.Module:
    """Create a ResNet-18 encoder returning 512 features per image."""

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    encoder = cast(nn.Module, resnet18(weights=weights))
    encoder.fc = nn.Identity()
    return encoder


class MelanomaModel(nn.Module):
    """Classify RGB lesions using their normalised image embeddings.

    Parameters
    ----------
    pretrained : bool, default=True
        Initialise ResNet-18 with ImageNet-1K V1 weights.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = _resnet18_encoder(pretrained)
        self.projection = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, 2)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute classification logits and embeddings in one encoder pass.

        Parameters
        ----------
        images : torch.Tensor
            Normalised RGB images with shape ``[batch, 3, 224, 224]``.

        Returns
        -------
        logits : torch.Tensor
            Raw benign and melanoma logits with shape ``[batch, 2]``.
        embeddings : torch.Tensor
            L2-normalised image embeddings with shape ``[batch, 128]``.
        """

        features = self.encoder(images)
        embeddings = functional.normalize(self.projection(features), p=2, dim=1)
        return self.classifier(embeddings), embeddings
