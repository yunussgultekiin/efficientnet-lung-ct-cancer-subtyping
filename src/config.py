from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw" / "Data"
SPLITS_DIR: Path = PROJECT_ROOT / "data" / "splits"
MANIFEST_PATH: Path = SPLITS_DIR / "dataset_manifest.csv"
COLLISIONS_LOG_PATH: Path = SPLITS_DIR / "cross_label_hash_collisions.csv"
TEST_DEV_SPLIT_PATH: Path = SPLITS_DIR / "test_dev_split.csv"
CV_FOLDS_PATH: Path = SPLITS_DIR / "cv_folds.csv"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
MODEL_DIR: Path = OUTPUT_DIR / "models"
PLOT_DIR: Path = OUTPUT_DIR / "plots"
REPORT_DIR: Path = OUTPUT_DIR / "reports"

CLASS_NAMES: tuple[str, ...] = (
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
)
CLASS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}

CLASS_KEYWORDS: dict[str, str] = {
    "adenocarcinoma": "adenocarcinoma",
    "large.cell.carcinoma": "large.cell.carcinoma",
    "squamous.cell.carcinoma": "squamous.cell.carcinoma",
    "normal": "normal"
}

IMAGE_EXTENSION: str = ".png"
IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")
NUM_CLASSES: int = len(CLASS_NAMES)

@dataclass(frozen=True, slots=True)
class HyperParams:
    freeze_epochs: int = 4
    backbone_lr: float = 5e-5
    head_lr: float = 5e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    batch_size: int = 16
    scheduler_name: str = "cosine"
    warmup_epochs: int = 3
    early_stopping_patience: int = 8
    label_smoothing: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "HyperParams":
        return HyperParams(**data)

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_name: str = "efficientnet_b0"
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    max_epochs: int = 100
    seed: int = 42
    n_folds: int = 5
    best_model_path: Path = MODEL_DIR / "best_efficientnet_b0.pt"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG: TrainingConfig = TrainingConfig()