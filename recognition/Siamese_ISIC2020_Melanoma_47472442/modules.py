"""Shared classifier and metric-learning model for ISIC melanoma triage."""

from typing import Literal, cast

import torch
from torch import nn
from torch.nn import functional as functional
from torchvision.models import resnet18


ModelName = Literal["base", "siamese"]


class MelanomaModel(nn.Module):
    """Classify RGB lesions using a ResNet-18 initialised with random weights."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = cast(nn.Module, resnet18(weights=None))
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(512, 2)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute classification logits and embeddings in one encoder pass.

        Parameters
        ----------
        images : torch.Tensor
            Normalised RGB images with shape ``[batch, 3, 224, 224]``.

        Returns
        -------
        logits : torch.Tensor
            Benign and melanoma logits from raw backbone features, with shape
            ``[batch, 2]``.
        embeddings : torch.Tensor
            L2-normalised backbone features with shape ``[batch, 512]`` for
            contrastive learning and reference comparison.
        """

        features = self.encoder(images)
        embeddings = functional.normalize(features, p=2, dim=1)
        return self.classifier(features), embeddings
