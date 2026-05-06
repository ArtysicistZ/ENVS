#!/bin/bash
# MCTS SFT v2.1 — Base Model, 30% Data (per-task leaf cap), TRAINING ONLY.
#
# Identical to run_mcts_sft_v2.1_base_30pct.sh except the n=8 eval step is
# removed. Use this when you just want the trained checkpoint.
#
# Usage (already-in-tmux):
#   bash scripts/mcts/run_mcts_sft_v2.1_base_30pct_train_only.sh
#
# Launch from outside tmux:
#   tmux new -d -s arpo_mcts_sft_30pct \
#       'bash scripts/mcts/run_mcts_sft_v2.1_base_30pct_train_only.sh'

set -euo pipefail

cd /home/kevinzyz/yincheng/arpo

VENV="/home/kevinzyz/yincheng/arpo/.venv/bin"
TORCHRUN="${VENV}/torchrun"

CONFIG="configs/mcts_sft_v2.1_base_30pct.yaml"
OUTPUT_DIR="checkpoints/mcts_sft_v2.1_base_30pct/beta05_5.6e-6"
EPOCH_MODEL="${OUTPUT_DIR}/epoch_1"
RUN_NAME="mcts_sft_v2.1_base_30pct_cap10_beta05_5.6e-6"

# ---- wandb ----
export WANDB_PROJECT="ARPO"
export WANDB_ENTITY="artysicistz-university-of-pennsylvania"

# Deterministic per-process hashing. Python 3 randomizes hash() by default,
# which would cause per-rank dataset divergence in any leaf-cap / subsampling
# code path that keys on hash(task_id). The training code uses zlib.crc32 now
# to avoid this, but PYTHONHASHSEED=0 is a second line of defense — if ANY
# Python hash() call ever slips into dataset construction, all ranks agree.
export PYTHONHASHSEED=0

echo "=============================================="
echo "  MCTS SFT v2.1 base 30pct — TRAINING ONLY"
echo "  Config: ${CONFIG}"
echo "  Output: ${OUTPUT_DIR}"
echo "  max_leaves_per_task: 10, 1 epoch (no data repetition)"
echo "  effective_batch=32 (1 x 4 accum x 8 GPUs) — identical to v2.1"
echo "  lr=5.6e-6 (2e-6 x 2.8, scaled for 2.8x fewer optimizer steps)"
echo "  wandb: project=${WANDB_PROJECT} entity=${WANDB_ENTITY}"
echo "=============================================="

sudo -E ${TORCHRUN} --nproc_per_node=8 scripts/train_mcts_sft_v2.py \
    --config "${CONFIG}" \
    --run_name "${RUN_NAME}"

echo ""
echo "[Training complete] Checkpoint at ${EPOCH_MODEL}"

if [ ! -f "${EPOCH_MODEL}/config.json" ]; then
    echo "ERROR: Checkpoint not found at ${EPOCH_MODEL}/config.json"
    exit 1
fi

echo "=============================================="
echo "  Done. No eval step was run."
echo "  Checkpoint: ${EPOCH_MODEL}"
echo "=============================================="
