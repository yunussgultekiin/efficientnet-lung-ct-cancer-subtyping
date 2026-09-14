from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.config import HyperParams, REPORT_DIR
from src.cross_validation import CrossValidationRunner, FinalRetrainer
from src.data_audit import DatasetAuditor
from src.evaluate import EvaluationPipeline
from src.model_info import ModelInfoExporter
from src.predict import BatchPredictor, Predictor
from src.splits import SplitBuilder

class Pipeline:
    def audit_data(self) -> None:
        DatasetAuditor().run()

    def make_splits(self, force: bool) -> None:
        SplitBuilder().run(force=force)

    def cv(self, params_path: Path | None) -> None:
        if params_path is None:
            params = HyperParams()
        else:
            payload = json.loads(params_path.read_text())
            params = HyperParams.from_dict(payload["params"] if "params" in payload else payload)
        CrossValidationRunner(params).run()

    def final_train(self) -> None:
        params, final_epochs = FinalRetrainer.load_from_cv_result(REPORT_DIR / "cv_result.json")
        FinalRetrainer(params, final_epochs).run()

    def evaluate(self) -> None:
        EvaluationPipeline().run()

    def model_info(self) -> None:
        ModelInfoExporter().run()

    def predict(self, path: Path) -> None:
        if path.is_dir():
            self._predict_folder(path)
        else:
            self._predict_image(path)

    def _predict_image(self, image_path: Path) -> None:
        prediction = Predictor().predict(image_path)
        print(f"class: {prediction.class_name}")
        print(f"confidence: {prediction.confidence:.4f}")

    def _predict_folder(self, folder: Path) -> None:
        batch_predictor = BatchPredictor()
        predictions = batch_predictor.predict_folder(folder)
        batch_predictor.save_results(predictions, REPORT_DIR / "predictions.csv")

        for prediction in predictions:
            print(f"{prediction.image_path.name}: {prediction.class_name} ({prediction.confidence:.4f})")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EfficientNet-B0 lung cancer classification pipeline"
    )

    parser.add_argument(
        "command",
        choices=["audit-data", "make-splits", "cv", "final-train", "evaluate", "model-info", "predict"]
    )
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--params", type=Path, default=None, help="JSON file of HyperParams for the cv command")
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files for make-splits")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    pipeline = Pipeline()

    if args.command == "audit-data":
        pipeline.audit_data()
    elif args.command == "make-splits":
        pipeline.make_splits(args.force)
    elif args.command == "cv":
        pipeline.cv(args.params)
    elif args.command == "final-train":
        pipeline.final_train()
    elif args.command == "evaluate":
        pipeline.evaluate()
    elif args.command == "model-info":
        pipeline.model_info()
    elif args.command == "predict":
        if args.path is None:
            raise ValueError("--path is required for the predict command")
        pipeline.predict(args.path)

if __name__ == "__main__":
    main()
