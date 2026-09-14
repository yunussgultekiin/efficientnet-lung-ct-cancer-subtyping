from __future__ import annotations
import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import imagehash
from PIL import Image
from src.config import (
    CLASS_KEYWORDS,
    CLASS_NAMES,
    COLLISIONS_LOG_PATH,
    IMAGE_EXTENSIONS,
    MANIFEST_PATH,
    RAW_DATA_DIR
)

ORIGINAL_SPLITS: tuple[str, ...] = ("train", "valid", "test")

@dataclass(slots=True)
class RawFileRecord:
    file_id: int
    relative_path: str
    original_split: str
    class_name: str
    md5: str
    phash: str

class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_a] = root_b

class DatasetAuditor:
    def __init__(
        self,
        raw_dir: Path = RAW_DATA_DIR,
        manifest_path: Path = MANIFEST_PATH,
        collisions_path: Path = COLLISIONS_LOG_PATH
    ) -> None:
        self.raw_dir = raw_dir
        self.manifest_path = manifest_path
        self.collisions_path = collisions_path

    def run(self) -> None:
        records = self._scan_raw_files()
        cluster_ids, collisions = self._build_clusters(records)
        excluded_indices = {index for pair in collisions for index in pair}
        canonical_ids = self._choose_canonical(records, cluster_ids, excluded_indices)

        self._write_manifest(records, cluster_ids, canonical_ids, excluded_indices)
        self._write_collisions(records, collisions)
        self._print_summary(records, canonical_ids, collisions)

    def _clean_class_name(self, folder_name: str) -> str:
        for keyword, class_name in CLASS_KEYWORDS.items():
            if keyword in folder_name:
                return class_name

        raise ValueError(f"Unknown class folder: {folder_name}")

    def _scan_raw_files(self) -> list[RawFileRecord]:
        records: list[RawFileRecord] = []
        file_id = 0

        for split in ORIGINAL_SPLITS:
            split_dir = self.raw_dir / split
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue

                class_name = self._clean_class_name(class_dir.name)

                for path in sorted(class_dir.iterdir()):
                    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue

                    data = path.read_bytes()
                    md5 = hashlib.md5(data).hexdigest()
                    with Image.open(path) as image:
                        phash = str(imagehash.phash(image.convert("RGB")))

                    records.append(
                        RawFileRecord(
                            file_id=file_id,
                            relative_path=str(path.relative_to(self.raw_dir)),
                            original_split=split,
                            class_name=class_name,
                            md5=md5,
                            phash=phash
                        )
                    )
                    file_id += 1

        return records

    def _build_clusters(self, records: list[RawFileRecord]) -> tuple[list[int], list[tuple[int, int]]]:
        union_find = _UnionFind(len(records))
        by_md5: dict[str, list[int]] = {}
        by_phash: dict[str, list[int]] = {}

        for index, record in enumerate(records):
            by_md5.setdefault(record.md5, []).append(index)
            by_phash.setdefault(record.phash, []).append(index)

        collisions: list[tuple[int, int]] = []

        for group in list(by_md5.values()) + list(by_phash.values()):
            anchor = group[0]
            for index in group[1:]:
                if records[index].class_name == records[anchor].class_name:
                    union_find.union(anchor, index)
                else:
                    collisions.append((anchor, index))

        roots = [union_find.find(index) for index in range(len(records))]
        remap: dict[int, int] = {}
        cluster_ids: list[int] = []
        for root in roots:
            if root not in remap:
                remap[root] = len(remap)
            cluster_ids.append(remap[root])

        return cluster_ids, collisions

    def _choose_canonical(
        self,
        records: list[RawFileRecord],
        cluster_ids: list[int],
        excluded_indices: set[int]
    ) -> set[int]:
        best_index_by_cluster: dict[int, int] = {}
        for index, cluster_id in enumerate(cluster_ids):
            if index in excluded_indices:
                continue
            current = best_index_by_cluster.get(cluster_id)
            if current is None or records[index].relative_path < records[current].relative_path:
                best_index_by_cluster[cluster_id] = index
        return set(best_index_by_cluster.values())

    def _write_manifest(
        self,
        records: list[RawFileRecord],
        cluster_ids: list[int],
        canonical_ids: set[int],
        excluded_indices: set[int]
    ) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "file_id",
                    "relative_path",
                    "original_split",
                    "class_name",
                    "md5",
                    "phash",
                    "cluster_id",
                    "is_canonical",
                    "excluded_cross_label"
                ]
            )
            for record, cluster_id in zip(records, cluster_ids):
                writer.writerow(
                    [
                        record.file_id,
                        record.relative_path,
                        record.original_split,
                        record.class_name,
                        record.md5,
                        record.phash,
                        cluster_id,
                        record.file_id in canonical_ids,
                        record.file_id in excluded_indices
                    ]
                )

    def _write_collisions(self, records: list[RawFileRecord], collisions: list[tuple[int, int]]) -> None:
        self.collisions_path.parent.mkdir(parents=True, exist_ok=True)
        with self.collisions_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["relative_path_a", "class_a", "relative_path_b", "class_b"])
            for i, j in collisions:
                writer.writerow(
                    [records[i].relative_path, records[i].class_name, records[j].relative_path, records[j].class_name]
                )

    def _print_summary(
        self,
        records: list[RawFileRecord],
        canonical_ids: set[int],
        collisions: list[tuple[int, int]]
    ) -> None:
        raw_counts = Counter(record.class_name for record in records)
        kept_counts = Counter(records[index].class_name for index in canonical_ids)

        print("Dataset audit summary")
        print("-" * 40)
        for class_name in CLASS_NAMES:
            raw = raw_counts[class_name]
            kept = kept_counts[class_name]
            print(f"{class_name}: raw={raw} unique_kept={kept} duplication_factor={raw / kept:.2f}x")

        print(f"\ntotal raw files: {len(records)}")
        print(f"total unique images kept: {len(canonical_ids)}")
        print(f"cross-label hash collisions found (both sides excluded from the dataset): {len(collisions)}")

if __name__ == "__main__":
    DatasetAuditor().run()