"""PyTorch model definitions for ISIC 2020 melanoma triage.

The module intentionally contains only model components. Image loading,
preprocessing, losses, metrics, and visualisation remain outside this file.
"""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as functional


ModelName = Literal["base", "siamese"]


class BaseCNN(nn.Module):
    """Classify RGB lesion images as benign or melanoma.

    Notes
    -----
    The ISIC loader returns ``[batch, 3, 224, 224]`` float32 images. This
    model returns raw logits ordered as benign (0) and melanoma (1), suitable
    for direct use with ``nn.CrossEntropyLoss``.
    """

    def __init__(self) -> None:
        """Initialise the convolutional feature extractor and classifier."""

        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Produce benign and melanoma logits for one batch of RGB images.

        Parameters
        ----------
        images : torch.Tensor
            RGB images with shape ``[batch, 3, height, width]``.

        Returns
        -------
        torch.Tensor
            Raw benign and melanoma logits with shape ``[batch, 2]``.
        """

        features = self.cnn(images)
        flattened = torch.flatten(features, start_dim=1)
        return self.fc(flattened)


class SiameseNetwork(nn.Module):
    """Embed two RGB lesion images with the same convolutional encoder.

    Parameters
    ----------
    embedding_dim : int, default=128
        Number of features in each L2-normalised image embedding.

    Notes
    -----
    Inputs have shape ``[batch, 3, height, width]``. Adaptive pooling permits
    positive spatial dimensions beyond the default 224-pixel square images.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        """Initialise the shared RGB encoder and embedding projection."""

        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward_once(self, images: torch.Tensor) -> torch.Tensor:
        """Encode one batch of RGB images into unit-length embeddings.

        Parameters
        ----------
        images : torch.Tensor
            RGB images with shape ``[batch, 3, height, width]``.

        Returns
        -------
        torch.Tensor
            Unit-length embeddings with shape ``[batch, embedding_dim]``.
        """

        features = torch.flatten(self.cnn(images), start_dim=1)
        return functional.normalize(self.fc(features), p=2, dim=1)

    def forward(
        self, first_images: torch.Tensor, second_images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode two image batches through the same network weights.

        Parameters
        ----------
        first_images : torch.Tensor
            First batch of RGB images.
        second_images : torch.Tensor
            Second batch of RGB images.

        Returns
        -------
        first : torch.Tensor
            Unit-length embeddings for ``first_images``.
        second : torch.Tensor
            Unit-length embeddings for ``second_images``.
        """

        return self.forward_once(first_images), self.forward_once(second_images)


def build_model(model_name: ModelName) -> BaseCNN | SiameseNetwork:
    """Construct one supported model with its standard configuration.

    Parameters
    ----------
    model_name : {"base", "siamese"}
        Architecture selected for the run.

    Returns
    -------
    BaseCNN or SiameseNetwork
        Newly constructed model with RGB inputs.

    Raises
    ------
    ValueError
        If ``model_name`` is unsupported.
    """

    if model_name == "base":
        return BaseCNN()
    if model_name == "siamese":
        return SiameseNetwork()
    raise ValueError(f"Unsupported model: {model_name}")
