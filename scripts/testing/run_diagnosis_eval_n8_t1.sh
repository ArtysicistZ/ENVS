#!/usr/bin/env bash
# Diagnosis eval, n=8 T=1 variant: same 86 trainable tasks, 8 stochastic
# rollouts per task per model. Tests robustness of the n=1 modal-policy
# bucket counts (SFT_win 14, ARPO_win 17, etc.).
#
# Single-node only: VM at 10.100.4.4 (48 envs). Does NOT touch 10.100.4.7.
# Estimated wall time: ~5h30min (15 batches/pass × ~11 min × 2 passes).
#
# Outputs (separate from n=1 results):
#   checkpoints/clean_eval/arpo_86tasks_clean_from_v2.1_no_mask_diagnosis/
#     arpo_step35_n8_t1/{eval_results_at_0.json, trajectories_at_0.jsonl}
#     sft_init_n8_t1/{eval_results_at_0.json, trajectories_at_0.jsonl}
#
# Usage:
#   tmux new-session -d -s diag_eval_n8 \
#     "bash /mnt/kevinzyz/arpo_local/scripts/testing/run_diagnosis_eval_n8_t1.sh \
#      2>&1 | tee /mnt/kevinzyz/arpo_local/logs/diagnosis_eval_n8_t1.log"

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
echo "[diag-eval-n8 $(stamp)] PASS 1/2: ARPO step_35 (n=8, T=1)"
sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main \
  config=configs/eval_diagnosis_86trainable_arpo_v21nm_step35_n8_t1.yaml \
  2>&1 | tee "$LOG_DIR/diagnosis_eval_arpo_step35_n8_t1.log"
PASS1_RC=${PIPESTATUS[0]}
echo "[diag-eval-n8 $(stamp)] PASS 1 exit code: $PASS1_RC"

sleep 15  # let GPUs/Ray fully release

# Pass 2: SFT init
echo "[diag-eval-n8 $(stamp)] PASS 2/2: SFT init (n=8, T=1)"
sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main \
  config=configs/eval_diagnosis_86trainable_sft_v21nm_n8_t1.yaml \
  2>&1 | tee "$LOG_DIR/diagnosis_eval_sft_init_n8_t1.log"
PASS2_RC=${PIPESTATUS[0]}
echo "[diag-eval-n8 $(stamp)] PASS 2 exit code: $PASS2_RC"

echo "[diag-eval-n8 $(stamp)] both passes done"
