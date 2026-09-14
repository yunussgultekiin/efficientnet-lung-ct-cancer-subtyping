from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from src.config import CLASS_NAMES, CONFIG, IMAGE_EXTENSION, NUM_CLASSES
from src.dataset import build_transforms
from src.model import load_trained_model

@dataclass(slots=True)
class Prediction:
    image_path: Path
    class_name: str
    confidence: float
    probabilities: dict[str, float]

class Predictor:
    def __init__(self, checkpoint_path: Path = CONFIG.best_model_path) -> None:
        self.device = CONFIG.device
        self.model = load_trained_model(checkpoint_path, NUM_CLASSES, self.device)
        self.model.eval()
        self.transform = build_transforms(CONFIG.image_size, train=False)

    def predict(self, image_path: Path) -> Prediction:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)

        predicted_index = int(probabilities.argmax().item())

        return Prediction(
            image_path=image_path,
            class_name=CLASS_NAMES[predicted_index],
            confidence=float(probabilities[predicted_index].item()),
            probabilities={
                class_name: float(probability)
                for class_name, probability in zip(CLASS_NAMES, probabilities.tolist())
            }
        )

class BatchPredictor:
    def __init__(self, predictor: Predictor | None = None) -> None:
        self.predictor = predictor or Predictor()

    def predict_folder(self, folder: Path, extension: str = IMAGE_EXTENSION) -> list[Prediction]:
        image_paths = sorted(
            path for path in folder.glob(f"*{extension}") if path.is_file()
        )
        return [self.predictor.predict(image_path) for image_path in image_paths]

    def save_results(self, predictions: list[Prediction], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image_path", "class_name", "confidence"])

            for prediction in predictions:
                writer.writerow(
                    [str(prediction.image_path), prediction.class_name, f"{prediction.confidence:.4f}"]
                )