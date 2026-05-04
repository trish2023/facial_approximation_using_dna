#!/usr/bin/env bash
set -e

echo "[RUNNER] Starting full project execution (Setup -> PCA -> Training -> Experiments)..."

# 1. Environment Check
if [ ! -d ".venv" ]; then
    echo "[RUNNER] Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

PYTHON=".venv/bin/python"

# 2. PCA Pipeline
echo "[RUNNER] Running PCA preparation..."
bash scripts/pca_prep.sh

# 3. Stage 2 Training Data
echo "[RUNNER] Preparing Stage 2 training data..."
$PYTHON scripts/stage2_training_data.py

# 4. Stage 2 Model Training
echo "[RUNNER] Training Stage 2 Random Forest classifier..."
$PYTHON scripts/stage2_random_forest.py

# 5. Clustering Experiments
echo "[RUNNER] Running baseline clustering experiments..."
$PYTHON scripts/clustering_experiments.py

# 6. Advanced Clustering Experiments
echo "[RUNNER] Running advanced clustering experiments (UMAP/t-SNE)..."
$PYTHON scripts/clustering_experiments_advanced.py

# 7. Visualizations
echo "[RUNNER] Generating PCA plots..."
$PYTHON scripts/plot_pca.py
$PYTHON scripts/plot_pca_subgroups.py

echo "[RUNNER] Full project execution complete!"
ls -R outputs/
