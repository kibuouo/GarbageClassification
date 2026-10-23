"""ResNet-18 model factory for garbage image classification."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build a ResNet-18 classifier with the requested number of classes."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_trainable(model, stage):
    if stage not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"Unknown stage: {stage}")
    for name, param in model.named_parameters():
        param.requires_grad = stage == "scratch" or name.startswith("fc.") or (stage == "finetune" and name.startswith("layer4."))


def training_mode(model, stage):
    """Keep frozen BatchNorm running statistics fixed as well as weights."""
    model.train()
    if stage != "scratch":
        model.eval()
        model.fc.train()
        if stage == "finetune":
            model.layer4.train()
