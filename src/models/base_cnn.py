import torch
from torch import nn


class BaseCNN(nn.Module):
    """Classify RGB lesion images as benign or melanoma.

    Notes
    -----
    The default ISIC loader returns ``[batch, 3, 224, 224]`` float32 images.
    This model returns two raw logits per image, ordered as benign (0) and
    melanoma (1). Pass the logits directly to ``nn.CrossEntropyLoss``.
    """

    def __init__(self):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce one pair of class logits for each input image.

        Parameters
        ----------
        x : torch.Tensor
            Batch of RGB images with shape ``[batch, 3, height, width]``.

        Returns
        -------
        torch.Tensor
            Raw benign and melanoma logits with shape ``[batch, 2]``.
        """

        output = self.cnn(x)
        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)
        return output
