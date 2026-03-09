#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# B-JEPA v4.2 — Clean Restart Setup Script
# Run in Vast.ai Jupyter terminal on A100
# ═══════════════════════════════════════════════════════════════════════════
set -e

echo "═══════════════════════════════════════════════════════════════════"
echo "  B-JEPA v4.2 Clean Setup — $(date)"
echo "═══════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────
# STEP 0: Navigate and clone
# ─────────────────────────────────────────────────────
cd /workspace

# Clean any previous clone
if [ -d "bdna-jepa" ]; then
    echo "[0] Removing old clone..."
    rm -rf bdna-jepa
fi

echo "[0] Cloning repo..."
git clone https://github.com/VUzan-bio/bdna-jepa.git
cd bdna-jepa

# Check out the v4.1 base (your last working commit)
echo "[0] Checking out base commit..."
git checkout 967556a81e22fd899f5ab8c5e517c8990f310a27
git checkout -b "v4.2-clean-restart"

echo "[0] Repo ready at $(pwd)"
echo ""

# ─────────────────────────────────────────────────────
# STEP 1: Verify codebase structure
# ─────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "  STEP 1: Verify codebase structure"
echo "═══════════════════════════════════════════════════════════════════"

echo "[1a] Package structure:"
find bdna_jepa -name "*.py" | sort

echo ""
echo "[1b] Configs:"
find configs -name "*.yaml" | sort

echo ""
echo "[1c] Scripts:"
ls -la scripts/*.py 2>/dev/null || echo "  No scripts/*.py found"

echo ""
echo "[1d] Key classes/functions in criterion.py:"
grep -n "class \|def forward\|def __init__" bdna_jepa/losses/criterion.py | head -30

echo ""
echo "[1e] Key classes in trainer.py:"
grep -n "class \|def train\|def _train_step\|vicreg\|uniformity\|log" bdna_jepa/training/trainer.py | head -30

echo ""
echo "[1f] LossConfig in config.py:"
grep -A 30 "class LossConfig" bdna_jepa/config.py | head -35

echo ""
echo "[1g] Check if v4.1 patch was already applied:"
grep -n "UniformityLoss\|uniformity\|L2-norm\|normalize" bdna_jepa/losses/criterion.py 2>/dev/null | head -10
echo "---"
grep -n "uniformity" bdna_jepa/training/trainer.py 2>/dev/null | head -5
echo "---"
grep -n "weight_uniformity" bdna_jepa/config.py 2>/dev/null | head -5

echo ""
echo "[1h] Check EMA parameters:"
grep -rn "ema\|EMA\|momentum\|tau" bdna_jepa/models/ bdna_jepa/training/ bdna_jepa/config.py 2>/dev/null | grep -i "ema\|momentum\|tau" | head -20

echo ""
echo "[1i] Check VICReg implementation:"
grep -n "class VICReg\|def.*vicreg\|gamma\|variance_loss\|_variance" bdna_jepa/losses/criterion.py | head -15

echo ""
echo "[1j] Current v4.1 config (if exists):"
cat configs/training/v4.1.yaml 2>/dev/null || echo "  v4.1.yaml not found"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  STEP 1 COMPLETE — Review output above before proceeding"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "NEXT: Run 'bash setup_v42_step2.sh' after reviewing Step 1 output"
