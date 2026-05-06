#!/usr/bin/env bash
# 4.4 chain (2026-04-28 overnight). Priority order per user:
#
# Phase A: 4 priority n=1 t=0 evals
#   1. v2.1_no_mask CLEAN n1 t0
#   2. v2.1_no_mask NOISY n1 t0
#   3. v3_noisy_combined CLEAN n1 t0
#   4. v3_noisy_combined NOISY n1 t0
# Phase B: 60% sweep (train + eval n8 t1 + eval n1 t0)
# Phase C: rest of backfills (v2_clean_aligned, v3_aligned, v2.1_base_30pct,
#          ARPO Run-2 step35, Run-3 step140, ARPO from v21nm step35)
#
# Pools: 4.4 + 4.7 (96 envs) throughout.
# Disjoint from 4.6 chain ({4.6} only).

set -uo pipefail
cd /mnt/kevinzyz/arpo_local

LOG_DIR=/mnt/kevinzyz/arpo_local/logs
mkdir -p "$LOG_DIR"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

ENV_VERL=(
  RAY_TMPDIR=/mnt/ray_tmp
  PYTHONUNBUFFERED=1
  HYDRA_FULL_ERROR=1
  MKL_SERVICE_FORCE_INTEL=1
  MKL_THREADING_LAYER=GNU
  RAY_memory_usage_threshold=0.95
  RAY_memory_monitor_refresh_ms=0
)

run_eval() {
  local label="$1" cfg="$2" results="$3"
  local full_results="/mnt/kevinzyz/arpo_local/$results"
  if [ -f "$full_results" ] && [ "$(jq 'length' "$full_results" 2>/dev/null)" = "300" ]; then
    echo "[44 $(stamp)] SKIP $label (already 300)"
    return 0
  fi
  echo "[44 $(stamp)] LAUNCH $label"
  sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main \
    config=configs/$cfg \
    2>&1 | tee "$LOG_DIR/eval_${label}.log"
}

# ===== Phase A: priority 4 evals =====
echo "[44 $(stamp)] Phase A: priority 4 n=1 t=0 evals"
run_eval v21nm_clean_n1t0       eval_clean_300tasks_v21nm_n1_t0.yaml             checkpoints/clean_eval/v21nm_n1_t0/eval_results_at_0.json
run_eval v21nm_noisy_n1t0       eval_noisy_300tasks_v21nm_n1_t0.yaml             checkpoints/noisy_eval/v21nm_n1_t0/eval_results_at_0.json
run_eval v3combined_clean_n1t0  eval_clean_300tasks_v3_combined_44_n1_t0.yaml    checkpoints/clean_eval/v3_combined_n1_t0/eval_results_at_0.json
run_eval v3combined_noisy_n1t0  eval_noisy_300tasks_v3_combined_n1_t0.yaml       checkpoints/noisy_eval/v3_combined_n1_t0/eval_results_at_0.json
echo "[44 $(stamp)] Phase A done"

# ===== Phase B: 60% sweep =====
echo "[44 $(stamp)] Phase B: 60% sweep"
PREP60_FLAG=/mnt/kevinzyz/arpo_local/checkpoints/mcts_trajectories_v2_subsample_60pct/task_index.json
while [ ! -f "$PREP60_FLAG" ]; do sleep 30; done
echo "[44 $(stamp)] 60% prep ready"

TRAIN_ENV=(
  HF_HOME=/home/kevinzyz/.cache/huggingface
  WANDB_PROJECT=ARPO
  WANDB_ENTITY=artysicistz-university-of-pennsylvania
)

TRAIN60_OUT=/mnt/kevinzyz/arpo_local/checkpoints/mcts_sft_v2_subsample_60pct/beta05_3.143e-6/epoch_1
if [ -f "$TRAIN60_OUT/model.safetensors.index.json" ]; then
  echo "[44 $(stamp)] 60% SFT already trained, skipping"
else
  echo "[44 $(stamp)] LAUNCH 60% SFT training"
  set -a; [ -f /mnt/kevinzyz/arpo_local/.env ] && source /mnt/kevinzyz/arpo_local/.env; set +a
  sudo -E env "${TRAIN_ENV[@]}" \
    WANDB_RUN_NAME=mcts_sft_v2_subsample_60pct_3.143e-6 \
    WANDB_API_KEY="$WANDB_API_KEY" \
    /mnt/kevinzyz/arpo_local/.venv/bin/torchrun --nproc_per_node=8 \
    scripts/train_mcts_sft_v2.py \
    --config configs/mcts_sft_v2_subsample_60pct.yaml \
    --run_name mcts_sft_v2_subsample_60pct_3.143e-6 \
    --model_path /home/kevinzyz/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/683d002dd99d8f95104d31e70391a39348857f4e \
    2>&1 | tee "$LOG_DIR/sft_60pct_train.log"
fi

run_eval 60pct_n8t1  eval_clean_300tasks_v2_subsample_60pct_n8_t1.yaml  checkpoints/clean_eval/v2_subsample_60pct_n8_t1/eval_results_at_0.json
run_eval 60pct_n1t0  eval_clean_300tasks_v2_subsample_60pct_n1_t0.yaml  checkpoints/clean_eval/v2_subsample_60pct_n1_t0/eval_results_at_0.json
echo "[44 $(stamp)] Phase B done"

# ===== Phase C: rest of backfills =====
echo "[44 $(stamp)] Phase C: remaining backfills"
run_eval v2_clean_aligned_n1t0     eval_clean_300tasks_v2_clean_aligned_n1_t0.yaml      checkpoints/clean_eval/v2_clean_aligned_n1_t0/eval_results_at_0.json
run_eval v3_aligned_n1t0           eval_clean_300tasks_v3_aligned_n1_t0.yaml            checkpoints/clean_eval/v3_aligned_n1_t0/eval_results_at_0.json
run_eval v21_base_30pct_n1t0       eval_clean_300tasks_v21_base_30pct_n1_t0.yaml        checkpoints/clean_eval/v21_base_30pct_n1_t0/eval_results_at_0.json
run_eval arpo_run2_step35_n1t0     eval_clean_300tasks_arpo_run2_step35_n1_t0.yaml      checkpoints/clean_eval/arpo_run2_step35_n1_t0/eval_results_at_0.json
run_eval arpo_run3_step140_n1t0    eval_clean_300tasks_arpo_run3_step140_n1_t0.yaml     checkpoints/clean_eval/arpo_run3_step140_n1_t0/eval_results_at_0.json
run_eval arpo_from_v21nm_step35_n1t0 eval_clean_300tasks_arpo_from_v21nm_step35_n1_t0.yaml checkpoints/clean_eval/arpo_from_v21nm_step35_n1_t0/eval_results_at_0.json
echo "[44 $(stamp)] chain done"
