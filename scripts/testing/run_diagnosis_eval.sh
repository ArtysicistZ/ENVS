#!/usr/bin/env bash
# Diagnosis eval: run val_only on the SFT init AND ARPO step_35 on the same
# 86 trainable tasks (n=1, T=0, seed=42) with ALL trajectories saved
# (success + failed). Joins by task_id give per-task before/after diff.
#
# Single-node only: VM at 10.100.4.4 (48 envs). Does NOT touch 10.100.4.7.
#
# Outputs:
#   checkpoints/clean_eval/arpo_86tasks_clean_from_v2.1_no_mask_diagnosis/
#     arpo_step35/{eval_results_at_0.json, trajectories_at_0.jsonl}
#     sft_init/{eval_results_at_0.json, trajectories_at_0.jsonl}
#
# Usage:
#   tmux new-session -d -s diag_eval \
#     "bash /mnt/kevinzyz/arpo_local/scripts/testing/run_diagnosis_eval.sh \
#      2>&1 | tee /mnt/kevinzyz/arpo_local/logs/diagnosis_eval.log"

set -uo pipefail
cd /mnt/kevinzyz/arpo_local

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

LOG_DIR=/mnt/kevinzyz/arpo_local/logs
mkdir -p "$LOG_DIR"

ENV_VERL=(
  RAY_TMPDIR=/mnt/ray_tmp
  PYTHONUNBUFFERED=1
  HYDRA_FULL_ERROR=1
  MKL_SERVICE_FORCE_INTEL=1
  MKL_THREADING_LAYER=GNU
  RAY_memory_usage_threshold=0.95
  RAY_memory_monitor_refresh_ms=0
)

# Pass 1: ARPO step_35
echo "[diag-eval $(stamp)] PASS 1/2: ARPO step_35"
sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main \
  config=configs/eval_diagnosis_86trainable_arpo_v21nm_step35.yaml \
  2>&1 | tee "$LOG_DIR/diagnosis_eval_arpo_step35.log"
PASS1_RC=${PIPESTATUS[0]}
echo "[diag-eval $(stamp)] PASS 1 exit code: $PASS1_RC"

sleep 15  # let GPUs/Ray fully release

# Pass 2: SFT init
echo "[diag-eval $(stamp)] PASS 2/2: SFT init (mcts_sft_v2.1_no_mask)"
sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main \
  config=configs/eval_diagnosis_86trainable_sft_v21nm.yaml \
  2>&1 | tee "$LOG_DIR/diagnosis_eval_sft_init.log"
PASS2_RC=${PIPESTATUS[0]}
echo "[diag-eval $(stamp)] PASS 2 exit code: $PASS2_RC"

echo "[diag-eval $(stamp)] both passes done"
