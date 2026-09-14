from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

def build_model(num_classes: int, dropout: float = 0.2, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, num_classes))

    return model

def build_criterion(label_smoothing: float = 0.0) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

def build_optimizer(model: nn.Module, backbone_lr: float, head_lr: float, weight_decay: float) -> Optimizer:
    return Adam(
        [
            {"params": model.features.parameters(), "lr": backbone_lr},
            {"params": model.classifier.parameters(), "lr": head_lr}
        ],
        weight_decay=weight_decay
    )

def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False

def unfreeze_backbone(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = True

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

def count_total_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)

def load_trained_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = build_model(num_classes=num_classes, pretrained=False)
    return load_checkpoint(model, checkpoint_path, device)