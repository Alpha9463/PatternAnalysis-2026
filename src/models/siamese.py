"""Shared RGB image encoder for Siamese melanoma similarity learning."""

import torch
from torch import nn
from torch.nn import functional as F


class SiameseNetwork(nn.Module):
    """Embed two RGB lesion images with the same CNN weights.

    Parameters
    ----------
    embedding_dim : int, default=128
        Number of features in each normalized image embedding.

    Notes
    -----
    Inputs from the default ISIC loader have shape ``[batch, 3, 224, 224]``.
    Adaptive pooling also permits other positive spatial dimensions.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
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

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one batch of images.

        Parameters
        ----------
        x : torch.Tensor
            RGB images with shape ``[batch, 3, height, width]``.

        Returns
        -------
        torch.Tensor
            Unit-length embeddings with shape ``[batch, embedding_dim]``.
        """

        features = torch.flatten(self.cnn(x), start_dim=1)
        return F.normalize(self.fc(features), p=2, dim=1)

    def forward(
        self, input1: torch.Tensor, input2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a pair of image batches with shared weights.

        Parameters
        ----------
        input1 : torch.Tensor
            First batch of RGB images.
        input2 : torch.Tensor
            Second batch of RGB images.

        Returns
        -------
        first : torch.Tensor
            Embeddings of the first batch.
        second : torch.Tensor
            Embeddings of the second batch.
        """

        return self.forward_once(input1), self.forward_once(input2)
