from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from src.config import CONFIG, CV_FOLDS_PATH, MANIFEST_PATH, TEST_DEV_SPLIT_PATH

@dataclass(slots=True)
class ManifestRow:
    file_id: int
    relative_path: str
    class_name: str
    cluster_id: int
    is_canonical: bool

@dataclass(slots=True)
class TestDevRow:
    file_id: int
    relative_path: str
    class_name: str
    cluster_id: int
    split: str

@dataclass(slots=True)
class FoldRow:
    file_id: int
    relative_path: str
    class_name: str
    cluster_id: int
    fold: int

def load_manifest(path: Path = MANIFEST_PATH) -> list[ManifestRow]:
    with path.open(newline="") as file:
        return [
            ManifestRow(
                file_id=int(row["file_id"]),
                relative_path=row["relative_path"],
                class_name=row["class_name"],
                cluster_id=int(row["cluster_id"]),
                is_canonical=row["is_canonical"] == "True"
            )
            for row in csv.DictReader(file)
        ]

def canonical_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    return [row for row in rows if row.is_canonical]

def load_test_dev_split(path: Path = TEST_DEV_SPLIT_PATH) -> list[TestDevRow]:
    with path.open(newline="") as file:
        return [
            TestDevRow(
                file_id=int(row["file_id"]),
                relative_path=row["relative_path"],
                class_name=row["class_name"],
                cluster_id=int(row["cluster_id"]),
                split=row["split"]
            )
            for row in csv.DictReader(file)
        ]

def load_cv_folds(path: Path = CV_FOLDS_PATH) -> list[FoldRow]:
    with path.open(newline="") as file:
        return [
            FoldRow(
                file_id=int(row["file_id"]),
                relative_path=row["relative_path"],
                class_name=row["class_name"],
                cluster_id=int(row["cluster_id"]),
                fold=int(row["fold"])
            )
            for row in csv.DictReader(file)
        ]

class SplitBuilder:
    def __init__(
        self,
        manifest_path: Path = MANIFEST_PATH,
        test_dev_path: Path = TEST_DEV_SPLIT_PATH,
        cv_folds_path: Path = CV_FOLDS_PATH,
        n_splits: int = CONFIG.n_folds,
        seed: int = CONFIG.seed
    ) -> None:
        self.manifest_path = manifest_path
        self.test_dev_path = test_dev_path
        self.cv_folds_path = cv_folds_path
        self.n_splits = n_splits
        self.seed = seed

    def run(self, force: bool = False) -> None:
        if not force and (self.test_dev_path.exists() or self.cv_folds_path.exists()):
            raise FileExistsError(
                f"{self.test_dev_path} or {self.cv_folds_path} already exists. "
                "Pass force=True to regenerate (this invalidates every prior CV result)."
            )

        rows = load_manifest(self.manifest_path)
        canon = canonical_rows(rows)
        test_dev_assignment = self._make_test_dev_split(canon)
        self._write_test_dev_split(canon, test_dev_assignment)

        dev_rows = [row for row in canon if test_dev_assignment[row.file_id] == "dev"]
        fold_assignment = self._make_cv_folds(dev_rows)
        self._write_cv_folds(dev_rows, fold_assignment)

        test_count = sum(1 for value in test_dev_assignment.values() if value == "test")
        print(f"test: {test_count} images, dev pool: {len(dev_rows)} images, {self.n_splits}-fold CV folds written")

    def _make_test_dev_split(self, canon: list[ManifestRow]) -> dict[int, str]:
        splitter = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        labels = [row.class_name for row in canon]
        groups = [row.cluster_id for row in canon]
        dev_idx, test_idx = next(splitter.split(canon, labels, groups))
        assignment = {canon[i].file_id: "dev" for i in dev_idx}
        assignment.update({canon[i].file_id: "test" for i in test_idx})
        return assignment

    def _make_cv_folds(self, dev_rows: list[ManifestRow]) -> dict[int, int]:
        splitter = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        labels = [row.class_name for row in dev_rows]
        groups = [row.cluster_id for row in dev_rows]
        assignment: dict[int, int] = {}
        for fold, (_, valid_idx) in enumerate(splitter.split(dev_rows, labels, groups)):
            for i in valid_idx:
                assignment[dev_rows[i].file_id] = fold
        return assignment

    def _write_test_dev_split(self, canon: list[ManifestRow], assignment: dict[int, str]) -> None:
        self.test_dev_path.parent.mkdir(parents=True, exist_ok=True)
        with self.test_dev_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["file_id", "relative_path", "class_name", "cluster_id", "split"])
            for row in canon:
                writer.writerow([row.file_id, row.relative_path, row.class_name, row.cluster_id, assignment[row.file_id]])

    def _write_cv_folds(self, dev_rows: list[ManifestRow], assignment: dict[int, int]) -> None:
        with self.cv_folds_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["file_id", "relative_path", "class_name", "cluster_id", "fold"])
            for row in dev_rows:
                writer.writerow([row.file_id, row.relative_path, row.class_name, row.cluster_id, assignment[row.file_id]])

if __name__ == "__main__":
    SplitBuilder().run()
