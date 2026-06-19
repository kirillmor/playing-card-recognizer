from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 classifier with a replaceable classification head."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool,
        dropout: float,
    ) -> None:
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)

        classifier = self.model.classifier
        if not isinstance(classifier, nn.Sequential):
            raise TypeError("Expected EfficientNet classifier to be nn.Sequential.")

        final_linear = classifier[-1]
        if not isinstance(final_linear, nn.Linear):
            raise TypeError("Expected last EfficientNet classifier layer to be nn.Linear.")

        in_features = final_linear.in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return class logits."""
        return self.model(images)

    def freeze_backbone(self) -> None:
        """Freeze all parameters except the classification head."""
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all model parameters."""
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def get_parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
    ) -> list[dict]:
        """Return optimizer parameter groups with separate learning rates."""
        backbone_parameters = []
        head_parameters = []

        for name, parameter in self.model.named_parameters():
            if name.startswith("classifier."):
                head_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)

        return [
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
                "name": "backbone",
            },
            {
                "params": head_parameters,
                "lr": head_lr,
                "name": "head",
            },
        ]
