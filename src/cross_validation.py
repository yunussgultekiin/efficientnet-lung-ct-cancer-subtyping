from __future__ import annotations
import dataclasses
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from src.config import CLASS_NAMES, CONFIG, CV_FOLDS_PATH, HyperParams, MODEL_DIR, NUM_CLASSES, PLOT_DIR, REPORT_DIR
from src.dataset import build_dataloader
from src.model import build_criterion, build_model
from src.splits import FoldRow, load_cv_folds
from src.training import TwoStageTrainer
from src.utils import set_seed

def load_dev_rows(cv_folds_path: Path = CV_FOLDS_PATH) -> list[FoldRow]:
    return load_cv_folds(cv_folds_path)

def to_dataset_items(rows: list[FoldRow]) -> list[tuple[str, str]]:
    return [(row.relative_path, row.class_name) for row in rows]

@dataclass(slots=True)
class FoldResult:
    fold: int
    best_epoch: int
    accuracy: float
    f1_macro: float
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    f1_per_class: dict[str, float]
    confusion: list[list[int]]

def _evaluate(model: nn.Module, loader, device: torch.device):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())

    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), average=None, zero_division=0
    )
    f1_macro = statistics.fmean(f1)
    matrix = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES)).tolist()
    as_dict = lambda values: {name: float(value) for name, value in zip(CLASS_NAMES, values)}
    return accuracy, f1_macro, as_dict(precision), as_dict(recall), as_dict(f1), matrix

class CrossValidationRunner:
    def __init__(
        self,
        params: HyperParams,
        max_epochs: int = CONFIG.max_epochs,
        cv_folds_path: Path = CV_FOLDS_PATH,
        checkpoint_dir: Path | None = MODEL_DIR / "cv",
        plot_dir: Path | None = PLOT_DIR / "cv",
        seed: int = CONFIG.seed
    ) -> None:
        self.params = params
        self.max_epochs = max_epochs
        self.cv_folds_path = cv_folds_path
        self.checkpoint_dir = checkpoint_dir
        self.plot_dir = plot_dir
        self.seed = seed

    def run(self) -> list[FoldResult]:
        dev_rows = load_dev_rows(self.cv_folds_path)
        n_folds = max(row.fold for row in dev_rows) + 1
        device = CONFIG.device

        fold_results: list[FoldResult] = []

        for fold in range(n_folds):
            fold_result, history = self._run_fold(fold, dev_rows, device)
            fold_results.append(fold_result)
            print(f"fold {fold}: best_epoch={fold_result.best_epoch} accuracy={fold_result.accuracy:.4f} f1_macro={fold_result.f1_macro:.4f}")

            if self.plot_dir is not None:
                history.plot(self.plot_dir / f"fold_{fold}")

        if self.checkpoint_dir is not None:
            self._write_summary(fold_results)
            self._plot_summary(fold_results)
            self._write_cv_result(fold_results)

        return fold_results

    def _run_fold(self, fold: int, dev_rows: list[FoldRow], device: torch.device):
        set_seed(self.seed + fold)
        train_rows = [row for row in dev_rows if row.fold != fold]
        valid_rows = [row for row in dev_rows if row.fold == fold]

        train_loader = build_dataloader(to_dataset_items(train_rows), CONFIG.image_size, self.params.batch_size, CONFIG.num_workers, train=True)
        valid_loader = build_dataloader(to_dataset_items(valid_rows), CONFIG.image_size, self.params.batch_size, CONFIG.num_workers, train=False)

        model = build_model(NUM_CLASSES, dropout=self.params.dropout)
        criterion = build_criterion(label_smoothing=self.params.label_smoothing)
        trainer = TwoStageTrainer(device, criterion)
        checkpoint_path = (self.checkpoint_dir / f"fold_{fold}.pt") if self.checkpoint_dir is not None else None
        fit_result = trainer.fit(model, train_loader, valid_loader, self.params, self.max_epochs, checkpoint_path)

        accuracy, f1_macro, precision, recall, f1, matrix = _evaluate(model, valid_loader, device)
        fold_result = FoldResult(fold, fit_result.best_epoch, accuracy, f1_macro, precision, recall, f1, matrix)
        return fold_result, fit_result.history

    def _write_summary(self, fold_results: list[FoldResult], report_path: Path = REPORT_DIR / "cv_summary.txt") -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        accuracies = [result.accuracy for result in fold_results]
        f1_macros = [result.f1_macro for result in fold_results]

        lines = ["5-fold cross-validation summary", "=" * 40, ""]
        for result in fold_results:
            lines.append(f"fold {result.fold}: best_epoch={result.best_epoch} accuracy={result.accuracy:.4f} f1_macro={result.f1_macro:.4f}")

        lines.append("")
        lines.append(f"accuracy: {statistics.fmean(accuracies):.4f} +/- {statistics.pstdev(accuracies):.4f}")
        lines.append(f"f1_macro: {statistics.fmean(f1_macros):.4f} +/- {statistics.pstdev(f1_macros):.4f}")
        lines.append("")
        lines.append("per-class f1 (mean +/- std across folds):")
        for class_name in CLASS_NAMES:
            values = [result.f1_per_class[class_name] for result in fold_results]
            lines.append(f"  {class_name}: {statistics.fmean(values):.4f} +/- {statistics.pstdev(values):.4f}")

        report_path.write_text("\n".join(lines))

    def _plot_summary(self, fold_results: list[FoldResult], plot_path: Path = PLOT_DIR / "cv_summary.png") -> None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        means = [statistics.fmean(result.f1_per_class[name] for result in fold_results) for name in CLASS_NAMES]
        stds = [statistics.pstdev([result.f1_per_class[name] for result in fold_results]) for name in CLASS_NAMES]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(CLASS_NAMES, means, yerr=stds, capsize=4)
        ax.set_ylabel("F1 (macro, mean +/- std across folds)")
        ax.set_title("Per-class F1 across 5-fold CV")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    def _write_cv_result(self, fold_results: list[FoldResult], result_path: Path = REPORT_DIR / "cv_result.json") -> None:
        payload = {
            "params": self.params.to_dict(),
            "fold_best_epochs": [result.best_epoch for result in fold_results]
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, indent=2))

class FinalRetrainer:
    def __init__(
        self,
        params: HyperParams,
        final_epochs: int,
        cv_folds_path: Path = CV_FOLDS_PATH,
        checkpoint_path: Path = CONFIG.best_model_path
    ) -> None:
        self.params = params
        self.final_epochs = final_epochs
        self.cv_folds_path = cv_folds_path
        self.checkpoint_path = checkpoint_path

    def run(self) -> None:
        dev_rows = load_dev_rows(self.cv_folds_path)
        train_loader = build_dataloader(to_dataset_items(dev_rows), CONFIG.image_size, self.params.batch_size, CONFIG.num_workers, train=True)

        model = build_model(NUM_CLASSES, dropout=self.params.dropout)
        criterion = build_criterion(label_smoothing=self.params.label_smoothing)
        trainer = TwoStageTrainer(CONFIG.device, criterion)

        final_scheduler_name = "cosine" if self.params.scheduler_name == "plateau" else self.params.scheduler_name
        final_params = dataclasses.replace(self.params, scheduler_name=final_scheduler_name)
        trainer.fit(model, train_loader, None, final_params, self.final_epochs, self.checkpoint_path)

        print(f"Final model retrained on {len(dev_rows)} images for {self.final_epochs} epochs -> {self.checkpoint_path}")

    @staticmethod
    def derive_final_epochs(fold_best_epochs: list[int]) -> int:
        return max(round(sum(fold_best_epochs) / len(fold_best_epochs)), 1)

    @staticmethod
    def load_from_cv_result(path: Path) -> tuple[HyperParams, int]:
        payload = json.loads(path.read_text())
        params = HyperParams(**payload["params"])
        final_epochs = FinalRetrainer.derive_final_epochs(payload["fold_best_epochs"])
        return params, final_epochs
