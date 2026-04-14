#!/usr/bin/env bash
# Full ARPO + v4 Noise training launcher.
#
# Topology:
#   - 86 trainable OSWorld tasks (test_86tasks_trainable.json)
#   - 8x A100-80GB on this GPU node (10.100.4.8)
#   - 2 env_servers × 48 docker slots = 96 envs:
#       10.100.4.6:15001  ← run start_noise_env_server.sh on that host first
#       10.100.4.8:15001  ← run start_noise_env_server.sh here first
#
# v4 noise pipeline summary:
#   - synchronized within-group noise via seed=(task_id, training_step) (CRN)
#   - SR→count mapping: SR<0.10→0, SR<0.25→1, SR<0.50→1-3, SR≥0.50→3-5 fires
#   - bucket-spaced random fire-step indices over the 15-step rollout
#   - adaptive: live curriculum EMA SR feeds the SR bucket every step
#   - hard tasks (SR<0.10) get zero noise to preserve training signal
#
# Usage:
#   bash scripts/training/run_arpo_8gpu_noise_full.sh
#
# Pass-through args after CLI overrides for verl.trainer.main are accepted:
#   bash scripts/training/run_arpo_8gpu_noise_full.sh trainer.total_episodes=5

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# ─── Pre-flight: both env_servers must be healthy ──────────────────────────
echo "=== Pre-flight: env_server health on 10.100.4.6 + 10.100.4.8 ==="
ALL_OK=1
for url in http://10.100.4.6:15001 http://10.100.4.8:15001; do
  if ! resp=$(curl -sf --max-time 5 "$url/health/containers" 2>/dev/null); then
    echo "  ✗ $url unreachable"
    ALL_OK=0
    continue
  fi
  state=$(echo "$resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"{d.get(\"healthy\")}/{d.get(\"target_pool_size\")}")' 2>/dev/null || echo 'parse-fail')
  if [[ "$state" != "48/48" ]]; then
    echo "  ✗ $url: $state (expected 48/48)"
    ALL_OK=0
  else
    echo "  ✓ $url: $state"
  fi
done
if [[ $ALL_OK -ne 1 ]]; then
  echo ""
  echo "ERROR: one or both env_servers not ready."
  echo "Start each with:"
  echo "  bash $ROOT_DIR/scripts/servers/start_noise_env_server.sh"
  exit 1
fi

# ─── Pre-flight: GPU memory check (need ≥ 35 GB free per GPU) ───────────────
echo ""
echo "=== Pre-flight: GPU memory ==="
gpu_check=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', ' '{ if ($2 < 35000) printf "GPU %s: only %s MiB free (need 35000)\n", $1, $2 }')
if [[ -n "$gpu_check" ]]; then
  echo "$gpu_check"
  echo "WARNING: some GPUs have <35 GB free. vllm may fail during memory profiling."
  echo "If shared with other users, consider lower gpu_memory_utilization."
fi
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader

# ─── Pre-flight: MCTS SR file present (so curriculum can read priors) ──────
SR_FILE="${ROOT_DIR}/checkpoints/mcts_trajectories_v2/combined_all/collection_results.json"
if [[ ! -f "$SR_FILE" ]]; then
  echo ""
  echo "WARNING: MCTS SR file missing at $SR_FILE"
  echo "Curriculum will start at noise_initial_tier=2 for all tasks."
  echo "(For full training, this file should contain all 86 task SRs.)"
fi

# ─── Activate venv + Ray + env tuning ──────────────────────────────────────
if [[ -f /home/kevinzyz/hansenzuishuai/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /home/kevinzyz/hansenzuishuai/.venv/bin/activate
fi

export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.85}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

# Clear stale Ray logs (best-effort)
[[ -f scripts/utils/clear_ray_logs.sh ]] && bash scripts/utils/clear_ray_logs.sh 2>/dev/null || true

# ─── Launch ────────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
LOG="${ROOT_DIR}/checkpoints/arpo_86tasks_v4_noise_${TS}.log"
mkdir -p "$(dirname "$LOG")"

echo ""
echo "=== Launching ARPO + v4 noise (full 86 tasks) ==="
echo "  config: configs/arpo_8gpu_noise_full.yaml"
echo "  log:    $LOG"
echo ""

python -m verl.trainer.main \
    config=configs/arpo_8gpu_noise_full.yaml \
    "$@" \
    2>&1 | tee "$LOG"
