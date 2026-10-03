"""Transfer learning by feature extraction with a pre-trained ResNet-50."""

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_feature_extractor(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Builds a ResNet-50 whose backbone is frozen and whose head is new.

    Every backbone weight stops receiving gradients. The classification head
    is replaced by a fresh linear layer with one output per class, which is
    the only trainable part.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_feature_extraction_mode(model: nn.Module) -> None:
    """Keeps the frozen backbone in evaluation mode while the head trains.

    Batch normalization layers update their running statistics in training
    mode even when their weights are frozen, so the backbone must stay in
    evaluation mode to remain a fixed feature extractor.
    """
    model.eval()
    model.fc.train()


def trainable_parameters(model: nn.Module) -> list:
    """Returns the parameters that receive gradients."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
