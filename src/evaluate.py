from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from src.config import CLASS_NAMES, CONFIG, NUM_CLASSES, PLOT_DIR, REPORT_DIR, TEST_DEV_SPLIT_PATH
from src.dataset import build_dataloader
from src.model import load_trained_model
from src.splits import load_test_dev_split

@dataclass(slots=True)
class EvaluationResult:
    y_true: list[int]
    y_pred: list[int]

class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    def predict(self, dataloader: DataLoader) -> EvaluationResult:
        self.model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1).cpu().tolist()

                y_true.extend(labels.tolist())
                y_pred.extend(predictions)

        return EvaluationResult(y_true=y_true, y_pred=y_pred)

    def save_classification_report(self, train_result: EvaluationResult, test_result: EvaluationResult, report_dir: Path) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)

        train_accuracy = accuracy_score(train_result.y_true, train_result.y_pred)
        test_accuracy = accuracy_score(test_result.y_true, test_result.y_pred)
        train_f1_macro = f1_score(train_result.y_true, train_result.y_pred, average="macro")
        test_f1_macro = f1_score(test_result.y_true, test_result.y_pred, average="macro")

        lines = [
            "Train (dev pool) vs held-out test",
            "=" * 40,
            f"train accuracy={train_accuracy:.4f} macro_f1={train_f1_macro:.4f}",
            f"test  accuracy={test_accuracy:.4f} macro_f1={test_f1_macro:.4f}",
            f"gap   accuracy={train_accuracy - test_accuracy:+.4f} macro_f1={train_f1_macro - test_f1_macro:+.4f}",
            "",
            "Train (dev pool) classification report",
            "-" * 40,
            classification_report(train_result.y_true, train_result.y_pred, target_names=list(CLASS_NAMES)),
            "Held-out test classification report",
            "-" * 40,
            classification_report(test_result.y_true, test_result.y_pred, target_names=list(CLASS_NAMES))
        ]

        (report_dir / "classification_report.txt").write_text("\n".join(lines))

    def save_confusion_matrix(self, result: EvaluationResult, plot_dir: Path) -> None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        matrix = confusion_matrix(result.y_true, result.y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(CLASS_NAMES))

        fig, ax = plt.subplots(figsize=(8, 8))
        display.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
        fig.tight_layout()
        fig.savefig(plot_dir / "confusion_matrix.png")
        plt.close(fig)

class EvaluationPipeline:
    def __init__(self) -> None:
        self.device = CONFIG.device
        self.train_loader = self._build_loader("dev")
        self.test_loader = self._build_loader("test")
        self.model = load_trained_model(CONFIG.best_model_path, NUM_CLASSES, self.device)
        self.evaluator = Evaluator(model=self.model, device=self.device)

    def _build_loader(self, split: str) -> DataLoader:
        rows = [row for row in load_test_dev_split(TEST_DEV_SPLIT_PATH) if row.split == split]
        items = [(row.relative_path, row.class_name) for row in rows]
        return build_dataloader(items, CONFIG.image_size, CONFIG.batch_size, CONFIG.num_workers, train=False)

    def run(self) -> None:
        train_result = self.evaluator.predict(self.train_loader)
        test_result = self.evaluator.predict(self.test_loader)
        self.evaluator.save_classification_report(train_result, test_result, REPORT_DIR)
        self.evaluator.save_confusion_matrix(test_result, PLOT_DIR)