from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Convolutional block used in the baseline CNN."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Apply convolutional block."""
        return self.block(images)


class BaselineCNN(nn.Module):
    """Simple convolutional baseline for playing card classification."""

    def __init__(
        self,
        input_channels: int,
        conv_channels: Sequence[int],
        kernel_size: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()

        if not conv_channels:
            raise ValueError("conv_channels must contain at least one value.")

        feature_blocks: list[nn.Module] = []
        current_channels = input_channels

        for output_channels in conv_channels:
            feature_blocks.append(
                ConvBlock(
                    in_channels=current_channels,
                    out_channels=int(output_channels),
                    kernel_size=kernel_size,
                )
            )
            current_channels = int(output_channels)

        self.features = nn.Sequential(
            *feature_blocks,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(current_channels, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return class logits for input images."""
        features = self.features(images)
        return self.classifier(features)
