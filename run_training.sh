#!/usr/bin/env bash
set -euo pipefail

DATASET_URL="https://huggingface.co/datasets/yunussgultekiin/lung-cancer-dataset/resolve/main/Data-20240107T052410Z-001.zip"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "== 1/6: Python environment =="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "== 2/6: GPU check =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "nvidia-smi not found — training will run on CPU and will be very slow."
fi
python -c "import torch; print('torch sees CUDA:', torch.cuda.is_available())"

echo
echo "== 3/6: Dataset =="
if [ ! -d "data/raw/Data" ]; then
    echo "data/raw/Data not found, downloading from Hugging Face..."
    mkdir -p data/raw
    curl -sL "$DATASET_URL" -o /tmp/lung_cancer_dataset.zip
    unzip -q /tmp/lung_cancer_dataset.zip -d data/raw
    rm -f /tmp/lung_cancer_dataset.zip
else
    echo "data/raw/Data already present, skipping download."
fi

echo
echo "== 4/6: Model architecture summary =="
python -m src.main model-info

echo
echo "== 5/6: Dataset audit + splits =="
if [ ! -f "data/splits/dataset_manifest.csv" ]; then
    python -m src.main audit-data
else
    echo "data/splits/dataset_manifest.csv already present, skipping audit-data."
fi
if [ ! -f "data/splits/cv_folds.csv" ]; then
    python -m src.main make-splits
else
    echo "data/splits/cv_folds.csv already present, skipping make-splits."
fi

echo
echo "== 6/6: 5-fold CV -> final-train -> evaluate =="
python -m src.main cv
python -m src.main final-train
python -m src.main evaluate

echo
echo "Done. See outputs/reports and outputs/plots for results."