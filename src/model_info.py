from __future__ import annotations
import json
from pathlib import Path
import torch
import torchinfo
import torchvision
from torchinfo import summary
from torchvision.models import EfficientNet_B0_Weights
from src.config import CONFIG, NUM_CLASSES, OUTPUT_DIR
from src.model import build_model

class ModelInfoExporter:
    def __init__(self, output_path: Path = OUTPUT_DIR / "model_info.json") -> None:
        self.output_path = output_path

    def run(self) -> None:
        model = build_model(NUM_CLASSES)
        input_size = (1, 3, CONFIG.image_size, CONFIG.image_size)
        model_summary = summary(model, input_size=input_size, verbose=0)

        info = {
            "file_name": self.output_path.name,
            "artifact_type": "model_info",
            "model_name": "efficientnet_b0",
            "class_name": type(model).__name__,
            "model_family": "efficientnet_2d",
            "origin": "torchvision, ImageNet-pretrained backbone + custom classifier head",
            "source_module": "src/model.py",
            "task_type": "classification",
            "representation": "2d",
            "weight_strategy": "pretrained_finetune",
            "weights": EfficientNet_B0_Weights.DEFAULT.name,
            "input_size": list(input_size),
            "total_parameters": model_summary.total_params,
            "trainable_parameters": model_summary.trainable_params,
            "non_trainable_parameters": model_summary.total_params - model_summary.trainable_params,
            "total_mult_adds": model_summary.total_mult_adds,
            "estimated_input_size_mb": model_summary.to_megabytes(model_summary.total_input),
            "estimated_forward_backward_size_mb": model_summary.to_megabytes(model_summary.total_output_bytes),
            "estimated_parameter_size_mb": model_summary.to_megabytes(model_summary.total_param_bytes),
            "estimated_total_size_mb": model_summary.to_megabytes(
                model_summary.total_input + model_summary.total_output_bytes + model_summary.total_param_bytes
            ),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "torchinfo_version": torchinfo.__version__,
            "architecture": str(model_summary).splitlines(),
            "layers": [self._describe_layer(layer) for layer in model_summary.summary_list]
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
        print(f"Model architecture info written to {self.output_path}")

    def _describe_layer(self, layer) -> dict:
        return {
            "name": layer.var_name,
            "class_name": layer.class_name,
            "input_size": layer.input_size,
            "output_size": layer.output_size,
            "num_parameters": layer.num_params,
            "trainable": layer.trainable == "True"
        }

if __name__ == "__main__":
    ModelInfoExporter().run()
