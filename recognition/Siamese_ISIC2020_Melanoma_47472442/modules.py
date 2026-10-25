"""Shared classifier and metric-learning model for ISIC melanoma triage.

Uses Torchvision's ResNet-18 implementation of He et al. (2016):
https://arxiv.org/abs/1512.03385
https://docs.pytorch.org/vision/0.28/models/generated/torchvision.models.resnet18.html
"""

from typing import Literal, cast

import torch
from torch import nn
from torch.nn import functional as functional
from torchvision.models import ResNet18_Weights, resnet18


ModelName = Literal["base", "siamese"]


class MelanomaModel(nn.Module):
    """Classify RGB lesions with Torchvision's ResNet-18 backbone.

    Parameters
    ----------
    weights : torchvision.models.ResNet18_Weights or None, default=None
        Encoder initialisation; training supplies ImageNet weights. Checkpoint
        restoration uses ``None`` and loads the saved trained tensors directly.
    """

    def __init__(self, weights: ResNet18_Weights | None = None) -> None:
        super().__init__()
        self.encoder = cast(nn.Module, resnet18(weights=weights))
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
