#!/bin/bash
# =============================================================================
# PROMETHEE - Complete Setup and Training Script for RunPod
# =============================================================================
#
# Usage:
#   1. Upload eodhd_sp500.db to /workspace/
#   2. Run: bash setup_and_train.sh
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "PROMETHEE - SOTA Financial Analysis LLM"
echo "=============================================="

# =============================================================================
# CONFIGURATION
# =============================================================================

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/Promethee_LLM"
DB_PATH="$WORKSPACE/eodhd_sp500.db"

# Training parameters
MAX_NEWS=200000
MAX_SEC=30000
EPOCHS=20
BATCH_SIZE=4
GRAD_ACCUM=8
LR="3e-4"
EMBED_DIM=256

# =============================================================================
# STEP 1: Check database
# =============================================================================

echo ""
echo "[1/7] Checking database..."

if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Database not found at $DB_PATH"
    echo "Please upload eodhd_sp500.db to $WORKSPACE/"
    echo ""
    echo "Options:"
    echo "  - Use RunPod UI to upload"
    echo "  - Use scp: scp eodhd_sp500.db root@<IP>:$WORKSPACE/"
    exit 1
fi

echo "Database found: $DB_PATH"
ls -lh "$DB_PATH"

# =============================================================================
# STEP 2: Clone/Update repo
# =============================================================================

echo ""
echo "[2/7] Setting up repository..."

cd "$WORKSPACE"

if [ -d "$REPO_DIR" ]; then
    echo "Repository exists, pulling latest..."
    cd "$REPO_DIR"
    git pull origin main
else
    echo "Cloning repository..."
    git clone https://github.com/Olympe19100/Promethee_LLM.git
    cd "$REPO_DIR"
fi

# =============================================================================
# STEP 3: Install dependencies
# =============================================================================

echo ""
echo "[3/7] Installing dependencies..."

pip install --upgrade pip

# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers & Mamba
pip install transformers accelerate
pip install mamba-ssm causal-conv1d

# ML & Utils
pip install scikit-learn joblib
pip install tqdm loguru numpy pandas

# Optional: TensorBoard
pip install tensorboard

echo "Dependencies installed."

# =============================================================================
# STEP 4: Create directories
# =============================================================================

echo ""
echo "[4/7] Creating directories..."

mkdir -p data models runs

# Link database
if [ ! -f "data/eodhd_sp500.db" ]; then
    ln -s "$DB_PATH" data/eodhd_sp500.db
    echo "Linked database to data/eodhd_sp500.db"
fi

# =============================================================================
# STEP 5: Compute Geometric Features (TDA, Ricci, Takens, Fisher-Rao)
# =============================================================================

echo ""
echo "[5/8] Computing geometric features..."
echo "  - TDA (Persistent Homology)"
echo "  - Ricci Curvature (Correlation Network)"
echo "  - Takens Embedding (Phase Space)"
echo "  - Fisher-Rao Distance (Information Geometry)"
echo ""

# Install geometric dependencies if needed
pip install networkx ripser 2>/dev/null || echo "Some geometric libs may not be available, using fallbacks"

python scripts/compute_geometric_features.py \
    --db_path data/eodhd_sp500.db \
    --start_date 2015-01-01 \
    --end_date 2025-12-31 \
    --window_days 60 \
    --step_days 5

echo ""
echo "Geometric features computed and stored in database."

# =============================================================================
# STEP 6: Prepare training data (with geometric features)
# =============================================================================

echo ""
echo "[6/8] Preparing training data (DATA-DRIVEN + GEOMETRIC)..."

python scripts/prepare_promethee_training.py \
    --db_path data/eodhd_sp500.db \
    --output_path data/promethee_semantic.jsonl \
    --max_news $MAX_NEWS \
    --max_sec $MAX_SEC \
    --news_sample_rate 0.3 \
    --shuffle \
    --save_thresholds data/thresholds.json

echo ""
echo "Training data prepared:"
wc -l data/promethee_semantic.jsonl

# Split train/val
echo ""
echo "Splitting train/val (90/10)..."

python -c "
import json
import random

with open('data/promethee_semantic.jsonl', 'r') as f:
    lines = f.readlines()

random.seed(42)
random.shuffle(lines)
split = int(len(lines) * 0.9)

with open('data/train.jsonl', 'w') as f:
    f.writelines(lines[:split])

with open('data/val.jsonl', 'w') as f:
    f.writelines(lines[split:])

print(f'Train: {split}, Val: {len(lines) - split}')
"

# =============================================================================
# STEP 7: Train Promethee
# =============================================================================

echo ""
echo "[7/8] Training Promethee LLM..."
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE x $GRAD_ACCUM = $(($BATCH_SIZE * $GRAD_ACCUM))"
echo "  Learning rate: $LR"
echo "  Embed dim: $EMBED_DIM"
echo ""

# Start TensorBoard in background
tensorboard --logdir runs/ --port 6006 --bind_all &
TENSORBOARD_PID=$!
echo "TensorBoard started on port 6006 (PID: $TENSORBOARD_PID)"

# Train
python scripts/train_promethee_semantic.py \
    --base_model state-spaces/mamba-1.4b-hf \
    --train_file data/train.jsonl \
    --val_file data/val.jsonl \
    --output_dir models/promethee_semantic \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LR \
    --warmup_ratio 0.1 \
    --grad_clip 1.0 \
    --use_muon \
    --use_contrastive \
    --contrastive_weight 0.1 \
    --embed_dim $EMBED_DIM \
    --max_input_length 2048 \
    --max_output_length 1024 \
    --fp16 \
    --save_every 5

echo ""
echo "Promethee training complete!"

# =============================================================================
# STEP 8: Train SVM Classifier
# =============================================================================

echo ""
echo "[8/8] Training SVM classifier on embeddings..."

python scripts/train_svm_classifier.py \
    --model_path models/promethee_semantic \
    --data_path data/promethee_semantic.jsonl \
    --output_path models/promethee_svm.joblib \
    --max_samples 50000 \
    --return_threshold 0.01 \
    --kernel rbf \
    --C 1.0

echo ""
echo "=============================================="
echo "TRAINING COMPLETE!"
echo "=============================================="
echo ""
echo "Models saved to:"
echo "  - Promethee LLM: models/promethee_semantic/"
echo "  - SVM Classifier: models/promethee_svm.joblib"
echo ""
echo "Data-driven thresholds: data/thresholds.json"
echo ""
echo "To use for inference, see scripts/inference.py"
echo "=============================================="

# Kill TensorBoard
kill $TENSORBOARD_PID 2>/dev/null || true
