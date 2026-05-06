#!/usr/bin/env bash
# Standalone launcher: waits for the 2nd 4.6 CLEAN eval (v3_noisy_combined)
# to reach length=300, then launches the ARPO clean-from-v2.1_no_mask training
# on 4.6. Runs independently of the main orchestrator — just polls for the
# results file and triggers ssh launch when ready.
#
# Pool: 4.6 only (48 envs). Disjoint from the 4.4 chain's pool 4.4+4.7,
# so this can run in parallel with the 4.4 eval chain.
#
# Usage:
#   tmux new-session -d -s arpo_46_launcher \
#     "bash /mnt/kevinzyz/arpo_local/scripts/launch_46_arpo_after_evals.sh \
#      2>&1 | tee /mnt/kevinzyz/arpo_local/logs/launch_46_arpo.log"

set -uo pipefail

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Sentinel: 2nd 4.6 CLEAN eval results path on 4.6.
RESULTS_REMOTE="/mnt/kevinzyz/arpo_main/checkpoints/clean_eval/mcts_sft_v3_noisy_combined_n8_t1/eval_results_at_0.json"

echo "[arpo46-launcher $(stamp)] starting; waiting for 2nd 4.6 eval to finish ($RESULTS_REMOTE)"

while true; do
  status=$(ssh 10.100.4.6 "[ -f $RESULTS_REMOTE ] && [ \"\$(jq 'length' $RESULTS_REMOTE 2>/dev/null)\" = '300' ] && echo READY || echo WAITING")
  if [ "$status" = "READY" ]; then break; fi
  sleep 30
done

echo "[arpo46-launcher $(stamp)] 2nd eval finished. Launching ARPO clean-from-v2.1_no_mask on 4.6."

ssh 10.100.4.6 "cd /mnt/kevinzyz/arpo_main && \
  mkdir -p logs && \
  sudo -E RAY_TMPDIR=/mnt/ray_tmp PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 \
    MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU \
    RAY_memory_usage_threshold=0.95 RAY_memory_monitor_refresh_ms=0 \
    .venv/bin/python -m verl.trainer.main \
    config=configs/arpo_86tasks_clean_from_v2.1_no_mask.yaml \
    2>&1 | tee logs/arpo_clean_from_v2.1_no_mask.log"

echo "[arpo46-launcher $(stamp)] ARPO finished. Launcher exiting."
