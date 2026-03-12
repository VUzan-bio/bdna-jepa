#!/bin/bash
# B-JEPA v5.0 Deployment — Vast.ai A100
# ═══════════════════════════════════════
# MLM + JEPA-CLS Hybrid (JEPA-DNA architecture)
set -e

echo "═══════════════════════════════════════════════════"
echo "  B-JEPA v5.0 — MLM + JEPA-CLS Deployment"
echo "═══════════════════════════════════════════════════"

cd /workspace/bdna-jepa

# ── Kill previous runs ──
echo "[1/4] Killing previous training..."
pkill -f pretrain_v4 2>/dev/null || true
pkill -f pretrain_v5 2>/dev/null || true
kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null || true
sleep 3
echo "  Done."

# ── Pull latest code ──
echo "[2/4] Pulling latest code..."
git pull origin main 2>/dev/null || echo "  (git pull skipped)"

# ── Verify files ──
echo "[3/4] Checking files..."
ls -lh bdna_jepa/models/pretrain_v5.py 2>/dev/null && echo "  ✓ pretrain_v5.py" || { echo "  ✗ MISSING pretrain_v5.py"; exit 1; }
ls -lh data/processed/pretrain_2M.csv 2>/dev/null && echo "  ✓ pretrain_2M.csv" || { echo "  ✗ MISSING data"; exit 1; }
ls -lh data/tokenizer/bpe_4096.json 2>/dev/null && echo "  ✓ bpe_4096.json" || { echo "  ✗ MISSING tokenizer"; exit 1; }

pip install -q umap-learn scikit-learn matplotlib pandas 2>/dev/null || true

# ── Launch ──
echo ""
echo "[4/4] Launching v5.0..."
mkdir -p outputs/checkpoints/v5.0/viz

nohup python bdna_jepa/models/pretrain_v5.py \
    --data-path data/processed/pretrain_2M.csv \
    --tokenizer-path data/tokenizer/bpe_4096.json \
    --output-dir outputs \
    --epochs 20 \
    --batch-size 64 \
    --lr 3e-4 \
    --min-lr 1e-6 \
    --warmup-epochs 3 \
    --embed-dim 576 \
    --num-layers 12 \
    --num-heads 9 \
    --ff-dim 2304 \
    --max-seq-len 512 \
    --predictor-dim 384 \
    --predictor-depth 3 \
    --predictor-heads 6 \
    --mask-ratio-start 0.20 \
    --mask-ratio-end 0.30 \
    --mean-span-len 3.0 \
    --mlm-weight 1.0 \
    --jepa-weight 1.0 \
    --vicreg-weight 1.0 \
    --vicreg-var-weight 25.0 \
    --vicreg-cov-weight 1.0 \
    --gc-adv-weight 1.0 \
    --grad-accum 1 \
    --save-every 5 \
    --log-every 50 \
    --wandb-project bdna-jepa \
    --wandb-run-name bjepa-v5.0-mlm-jepa-cls \
    2>&1 | tee outputs/v5.0_train.log &

sleep 10

echo ""
echo "═══════════════════════════════════════════════════"
echo "  v5.0 launched!"
echo "  Log: outputs/v5.0_train.log"
echo "  Checkpoints: outputs/checkpoints/v5.0/"
echo "  Viz: outputs/checkpoints/v5.0/viz/"
echo "═══════════════════════════════════════════════════"
echo ""
echo "KEY DIFFERENCES from v4.x:"
echo "  MLM cross-entropy as PRIMARY loss (inherent anti-collapse)"
echo "  JEPA on [CLS] only (global semantic layer)"
echo "  VICReg on CLS (prevents CLS-specific collapse)"
echo "  Span masking 20-30% (not I-JEPA block masking)"
echo ""
echo "Expected epoch 1 metrics:"
echo "  RankMe > 300  (MLM forces full-rank encoder)"
echo "  MLM acc ~ 0.10-0.30 (learning to predict tokens)"
echo "  JEPA cos ~ 0.1-0.5 (CLS prediction starting)"
echo ""
echo "RED FLAGS:"
echo "  RankMe < 100  → encoder collapsing despite MLM"
echo "  MLM acc > 0.95 before epoch 5  → overfitting"
echo "  MLM loss not decreasing  → learning rate issue"
echo ""

tail -f outputs/v5.0_train.log
