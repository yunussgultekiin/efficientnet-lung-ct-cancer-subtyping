from __future__ import annotations
import random
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass(slots=True)
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_score: float | None = field(default=None, init=False)
    counter: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

@dataclass(slots=True)
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    train_f1_macro: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    valid_accuracy: list[float] = field(default_factory=list)
    valid_f1_macro: list[float] = field(default_factory=list)

    def append(
        self,
        train_loss: float,
        train_accuracy: float,
        train_f1_macro: float,
        valid_loss: float | None = None,
        valid_accuracy: float | None = None,
        valid_f1_macro: float | None = None
    ) -> None:
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)
        self.train_f1_macro.append(train_f1_macro)
        if valid_loss is not None:
            self.valid_loss.append(valid_loss)
            self.valid_accuracy.append(valid_accuracy)
            self.valid_f1_macro.append(valid_f1_macro)

    def plot(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(self.train_loss) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        panels = (
            (axes[0], "Loss", self.train_loss, self.valid_loss),
            (axes[1], "Accuracy", self.train_accuracy, self.valid_accuracy),
            (axes[2], "F1 (macro)", self.train_f1_macro, self.valid_f1_macro)
        )

        for ax, title, train_values, valid_values in panels:
            ax.plot(epochs, train_values, label="train")
            if valid_values:
                ax.plot(epochs, valid_values, label="valid")
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / "training_history.png")
        plt.close(fig)