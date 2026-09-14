from __future__ import annotations
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from src.config import CLASS_TO_INDEX, RAW_DATA_DIR

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
    )

class ManifestImageDataset(Dataset):
    def __init__(self, items: list[tuple[str, str]], raw_dir: Path, transform: transforms.Compose) -> None:
        self.items = items
        self.raw_dir = raw_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        relative_path, class_name = self.items[index]
        image = Image.open(self.raw_dir / relative_path).convert("RGB")
        return self.transform(image), CLASS_TO_INDEX[class_name]

def build_dataloader(
    items: list[tuple[str, str]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    raw_dir: Path = RAW_DATA_DIR
) -> DataLoader:
    dataset = ManifestImageDataset(items, raw_dir, build_transforms(image_size, train))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )