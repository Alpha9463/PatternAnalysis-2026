"""ResNet-18 classifier and Siamese encoder for ISIC melanoma triage."""

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


class BaseCNN(nn.Module):
    """Classify RGB lesions with ResNet-18 and a two-class linear head.

    Parameters
    ----------
    pretrained : bool, default=True
        Initialise the encoder with ImageNet weights.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = _resnet18_encoder(pretrained)
        self.classifier = nn.Linear(512, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map normalised RGB images [N, 3, 224, 224] to logits [N, 2]."""

        return self.classifier(self.encoder(images))


class SiameseNetwork(nn.Module):
    """Embed two RGB lesions using the same ResNet-18 weights.

    Parameters
    ----------
    pretrained : bool, default=True
        Initialise the encoder with ImageNet weights.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = _resnet18_encoder(pretrained)
        self.projection = nn.Linear(512, 128)

    def forward_once(self, images: torch.Tensor) -> torch.Tensor:
        """Map normalised RGB images [N, 3, 224, 224] to unit vectors [N, 128]."""

        return functional.normalize(self.projection(self.encoder(images)), p=2, dim=1)

    def forward(
        self, first_images: torch.Tensor, second_images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return embeddings for both image batches through the shared encoder."""

        return self.forward_once(first_images), self.forward_once(second_images)


def build_model(
    model_name: ModelName, pretrained: bool = True
) -> BaseCNN | SiameseNetwork:
    """Construct a classifier or Siamese model.

    Parameters
    ----------
    model_name : {"base", "siamese"}
        Model type to construct.
    pretrained : bool, default=True
        Use ImageNet initialisation. Checkpoint prediction passes ``False``.

    Returns
    -------
    BaseCNN or SiameseNetwork
        ResNet-18 model with the corresponding output head.
    """

    return BaseCNN(pretrained) if model_name == "base" else SiameseNetwork(pretrained)
