from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, LRScheduler, ReduceLROnPlateau, SequentialLR
from torch.utils.data import DataLoader
from src.config import HyperParams
from src.model import build_optimizer, freeze_backbone, save_checkpoint, unfreeze_backbone
from src.utils import EarlyStopping, TrainingHistory

@dataclass(slots=True)
class EpochMetrics:
    loss: float
    accuracy: float
    f1_macro: float

def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer | None
) -> EpochMetrics:
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.set_grad_enabled(train):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    correct_predictions = sum(
        prediction == label for prediction, label in zip(all_predictions, all_labels)
    )

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=correct_predictions / total_samples,
        f1_macro=f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    )

def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: str,
    warmup_epochs: int,
    total_epochs: int
) -> LRScheduler | ReduceLROnPlateau:
    if scheduler_name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    if scheduler_name == "none":
        return ConstantLR(optimizer, factor=1.0, total_iters=max(total_epochs, 1))

    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1))
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    return CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))

@dataclass(slots=True)
class FitResult:
    history: TrainingHistory
    best_epoch: int
    best_valid_f1_macro: float

class TwoStageTrainer:
    def __init__(self, device: torch.device, criterion: nn.Module) -> None:
        self.device = device
        self.criterion = criterion

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        params: HyperParams,
        max_epochs: int,
        checkpoint_path: Path | None
    ) -> FitResult:
        model = model.to(self.device)
        history = TrainingHistory()
        best_f1_macro = 0.0
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None

        freeze_epochs = min(params.freeze_epochs, max_epochs)
        if freeze_epochs > 0:
            freeze_backbone(model)
            head_optimizer = Adam(model.classifier.parameters(), lr=params.head_lr, weight_decay=params.weight_decay)

            for epoch in range(1, freeze_epochs + 1):
                train_metrics = run_epoch(model, train_loader, self.device, self.criterion, head_optimizer)
                valid_metrics = run_epoch(model, valid_loader, self.device, self.criterion, None) if valid_loader is not None else None
                valid_args = (
                    (valid_metrics.loss, valid_metrics.accuracy, valid_metrics.f1_macro) if valid_metrics else (None, None, None)
                )
                history.append(train_metrics.loss, train_metrics.accuracy, train_metrics.f1_macro, *valid_args)

                if valid_metrics is not None:
                    print(
                        f"[freeze] epoch {epoch}/{max_epochs} train_loss={train_metrics.loss:.4f} "
                        f"valid_loss={valid_metrics.loss:.4f} valid_f1={valid_metrics.f1_macro:.4f}"
                    )
                    if valid_metrics.f1_macro > best_f1_macro:
                        best_f1_macro = valid_metrics.f1_macro
                        best_epoch = epoch
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    print(f"[freeze] epoch {epoch}/{max_epochs} train_loss={train_metrics.loss:.4f}")

            unfreeze_backbone(model)

        remaining_epochs = max_epochs - freeze_epochs
        if remaining_epochs > 0:
            optimizer = build_optimizer(model, params.backbone_lr, params.head_lr, params.weight_decay)
            scheduler = build_scheduler(optimizer, params.scheduler_name, params.warmup_epochs, remaining_epochs)
            early_stopping = EarlyStopping(patience=params.early_stopping_patience) if valid_loader is not None else None

            for offset in range(1, remaining_epochs + 1):
                epoch = freeze_epochs + offset
                train_metrics = run_epoch(model, train_loader, self.device, self.criterion, optimizer)
                valid_metrics = run_epoch(model, valid_loader, self.device, self.criterion, None) if valid_loader is not None else None
                valid_args = (
                    (valid_metrics.loss, valid_metrics.accuracy, valid_metrics.f1_macro) if valid_metrics else (None, None, None)
                )
                history.append(train_metrics.loss, train_metrics.accuracy, train_metrics.f1_macro, *valid_args)

                if valid_metrics is not None:
                    print(
                        f"[finetune] epoch {epoch}/{max_epochs} train_loss={train_metrics.loss:.4f} "
                        f"valid_loss={valid_metrics.loss:.4f} valid_f1={valid_metrics.f1_macro:.4f}"
                    )
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(valid_metrics.f1_macro)
                    else:
                        scheduler.step()

                    if valid_metrics.f1_macro > best_f1_macro:
                        best_f1_macro = valid_metrics.f1_macro
                        best_epoch = epoch
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

                    if early_stopping.step(valid_metrics.f1_macro):
                        print(f"Early stopping at epoch {epoch}")
                        break
                else:
                    print(f"[finetune] epoch {epoch}/{max_epochs} train_loss={train_metrics.loss:.4f}")
                    if not isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step()

        if best_state is not None:
            model.load_state_dict(best_state)
        if checkpoint_path is not None:
            save_checkpoint(model, checkpoint_path)

        return FitResult(history=history, best_epoch=best_epoch, best_valid_f1_macro=best_f1_macro)
