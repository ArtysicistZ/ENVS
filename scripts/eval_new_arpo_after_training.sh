#!/usr/bin/env bash
# Sidecar watcher: after the new ARPO clean-from-v2.1_no_mask training finishes,
# eval its step 35 and step 70 checkpoints on clean 300.
#
# Triggers off the existence of the global_step_70 checkpoint dir AND the
# absence of the verl training process. Pools 4.4 + 4.7 (96 envs) — same
# as the training, since training is done by then.
#
# Usage:
#   tmux new-session -d -s eval_new_arpo \
#     "bash /mnt/kevinzyz/arpo_local/scripts/eval_new_arpo_after_training.sh \
#      2>&1 | tee /mnt/kevinzyz/arpo_local/logs/eval_new_arpo.log"

set -uo pipefail
cd /mnt/kevinzyz/arpo_local

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

CKPT_BASE=/mnt/kevinzyz/arpo_local/checkpoints/OSWorld-ARPO/arpo_86tasks_clean_from_v2.1_no_mask
S70_FLAG="$CKPT_BASE/global_step_70/actor"
LOG_DIR=/mnt/kevinzyz/arpo_local/logs

ENV_VERL=(
  RAY_TMPDIR=/mnt/ray_tmp
  PYTHONUNBUFFERED=1
  HYDRA_FULL_ERROR=1
  MKL_SERVICE_FORCE_INTEL=1
  MKL_THREADING_LAYER=GNU
  RAY_memory_usage_threshold=0.95
  RAY_memory_monitor_refresh_ms=0
)

echo "[eval-new-arpo $(stamp)] starting; waiting for step 70 ckpt at $S70_FLAG"

# Wait for step_70/actor to exist (means ARPO training finished and saved final ckpt)
while [ ! -d "$S70_FLAG" ]; do sleep 30; done
echo "[eval-new-arpo $(stamp)] step 70 ckpt detected"

# Also wait for the ARPO training verl process to fully exit (so GPUs free)
while pgrep -f "config=configs/arpo_86tasks_clean_from_v2_clean_aligned.yaml" > /dev/null; do
  sleep 15
done
echo "[eval-new-arpo $(stamp)] ARPO training process exited; GPUs free"
sleep 10  # safety margin for GPU memory release

# Run step 35 eval, then step 70 eval, serially.
EVAL_STEP35_RESULTS=/mnt/kevinzyz/arpo_local/checkpoints/clean_eval/arpo_from_v21nm_step35_n8_t1/eval_results_at_0.json
if [ -f "$EVAL_STEP35_RESULTS" ] && [ "$(jq 'length' "$EVAL_STEP35_RESULTS" 2>/dev/null)" = "300" ]; then
  echo "[eval-new-arpo $(stamp)] SKIP step 35 (already 300)"
else
  echo "[eval-new-arpo $(stamp)] LAUNCH eval step 35"
  sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main \
    config=configs/eval_clean_300tasks_arpo_from_v21nm_step35_n8_t1.yaml \
    2>&1 | tee "$LOG_DIR/eval_clean_arpo_step35.log"
fi

EVAL_STEP70_RESULTS=/mnt/kevinzyz/arpo_local/checkpoints/clean_eval/arpo_from_v21nm_step70_n8_t1/eval_results_at_0.json
if [ -f "$EVAL_STEP70_RESULTS" ] && [ "$(jq 'length' "$EVAL_STEP70_RESULTS" 2>/dev/null)" = "300" ]; then
  echo "[eval-new-arpo $(stamp)] SKIP step 70 (already 300)"
else
  echo "[eval-new-arpo $(stamp)] LAUNCH eval step 70"
  sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main \
    config=configs/eval_clean_300tasks_arpo_from_v21nm_step70_n8_t1.yaml \
    2>&1 | tee "$LOG_DIR/eval_clean_arpo_step70.log"
fi

echo "[eval-new-arpo $(stamp)] both evals done; sidecar exiting"
