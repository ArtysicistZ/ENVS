#!/bin/bash
# MCTS SFT v2.1 — Base Model, 30% Data (per-task leaf hardcap) + n8 Evaluation
#
# Same recipe as v2.1 but with max_leaves_per_task=10 applied at dataset
# build time. Easier tasks (many successful leaves) get trimmed to 10;
# hard tasks (<10 successful leaves) are untouched. Target: ~28% of the
# v2.1 training data.
#
# Does NOT modify the full-data v2.1 path. To run full-data v2.1, keep
# using scripts/mcts/run_mcts_sft_v2.1.sh.
#
# Steps:
#   1. Train MCTS SFT v2.1 30pct (1 epoch, 8 GPUs)
#   2. Kill GPU processes to free VRAM
#   3. Run n8 evaluation on 300 tasks (112 VMs across 4 servers)
#
# Usage:
#   bash scripts/mcts/run_mcts_sft_v2.1_base_30pct.sh

set -euo pipefail

cd /home/kevinzyz/yincheng/arpo

VENV="/home/kevinzyz/yincheng/arpo/.venv/bin"
PYTHON="${VENV}/python"
TORCHRUN="${VENV}/torchrun"

CONFIG="configs/mcts_sft_v2.1_base_30pct.yaml"
OUTPUT_DIR="checkpoints/mcts_sft_v2.1_base_30pct/beta05_2e-6"
EPOCH_MODEL="${OUTPUT_DIR}/epoch_1"
EVAL_N8_CONFIG="configs/sft_eval_300tasks_clean_n8.yaml"
EVAL_N8_DIR="${OUTPUT_DIR}/eval_n8"
RUN_NAME="mcts_sft_v2.1_base_30pct_cap10_beta05_2e-6"

# ---- wandb ----
export WANDB_PROJECT="ARPO"
export WANDB_ENTITY="artysicistz-university-of-pennsylvania"

echo "=============================================="
echo "  MCTS SFT v2.1 base 30pct — Training + n8 Eval"
echo "  Config: ${CONFIG}"
echo "  Output: ${OUTPUT_DIR}"
echo "  max_leaves_per_task: 10 (cap at dataset build time)"
echo "  wandb project: ${WANDB_PROJECT}"
echo "=============================================="

# ==============================
# Step 1: Train
# ==============================
echo ""
echo "[Step 1/3] Training MCTS SFT v2.1 base 30pct..."
echo "  lr=2e-6, beta=0.5, max_step_ratio=2.0, max_grad_norm=1.0"
echo "  effective_batch=32 (1 x 4 accum x 8 GPUs)"
echo "  ~28% of v2.1 data (per-task leaf cap = 10)"
echo ""

sudo -E ${TORCHRUN} --nproc_per_node=8 scripts/train_mcts_sft_v2.py \
    --config "${CONFIG}" \
    --run_name "${RUN_NAME}"

echo ""
echo "[Step 1/3] Training complete. Checkpoint at ${EPOCH_MODEL}"

# Verify checkpoint exists
if [ ! -f "${EPOCH_MODEL}/config.json" ]; then
    echo "ERROR: Checkpoint not found at ${EPOCH_MODEL}/config.json"
    exit 1
fi

# ==============================
# Step 2: Free GPU memory
# ==============================
echo ""
echo "[Step 2/3] Freeing GPU memory..."
sleep 5

# Kill any leftover GPU processes from training
sudo pkill -f "torchrun.*train_mcts_sft_v2" 2>/dev/null || true
sudo pkill -f "torch.distributed" 2>/dev/null || true
sleep 5

echo "  GPU memory freed."

# ==============================
# Step 3: n8 Evaluation (300 tasks)
# ==============================
echo ""
echo "[Step 3/3] Running n8 evaluation on 300 tasks..."
echo "  Model: ${EPOCH_MODEL}"
echo "  Envs: 112 (4 servers)"
echo ""

sudo -E ${PYTHON} -m verl.trainer.main \
    config="${EVAL_N8_CONFIG}" \
    worker.actor.model.model_path="${EPOCH_MODEL}" \
    trainer.experiment_name="mcts_sft_v2.1_base_30pct_eval_n8" \
    trainer.save_checkpoint_path="${EVAL_N8_DIR}"

echo ""
echo "[Step 3/3] n8 evaluation complete."
echo "  Results at: ${EVAL_N8_DIR}/eval_results_at_0.json"

# ==============================
# Summary
# ==============================
echo ""
echo "=============================================="
echo "  Pipeline complete!"
echo "  Checkpoint: ${EPOCH_MODEL}"
echo "  n8 results: ${EVAL_N8_DIR}/eval_results_at_0.json"
echo "=============================================="

# Print quick summary if results exist
if [ -f "${EVAL_N8_DIR}/eval_results_at_0.json" ]; then
    ${PYTHON} -c "
import json
with open('${EVAL_N8_DIR}/eval_results_at_0.json') as f:
    data = json.load(f)
total = len(data)
doable = sum(1 for v in data.values() if v.get('n_success', 0) > 0)
sr = sum(v.get('success_rate', 0) for v in data.values()) / total
print(f'  n8 Results: {doable}/{total} doable ({doable/total:.1%}), avg SR={sr:.3f}')
"
fi
