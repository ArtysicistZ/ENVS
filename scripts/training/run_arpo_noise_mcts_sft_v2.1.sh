#!/usr/bin/env bash
# ARPO + v4 Noise from MCTS SFT v2.1 checkpoint.
#
# Topology:
#   - 86 trainable OSWorld tasks
#   - 8x A100-80GB on this GPU node
#   - 2 env_servers × 48 docker slots = 96 envs:
#       10.100.4.4:15001
#       10.100.4.7:15001
#
# Usage:
#   bash scripts/training/run_arpo_noise_mcts_sft_v2.1.sh
#   bash scripts/training/run_arpo_noise_mcts_sft_v2.1.sh trainer.total_episodes=5

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/arpo_8gpu_noise_mcts_sft_v2.1.yaml"

# ─── Pre-flight: both env_servers must be healthy ──────────────────────────
echo "=== Pre-flight: env_server health on 10.100.4.4 + 10.100.4.7 ==="
ALL_OK=1
for url in http://10.100.4.4:15001 http://10.100.4.7:15001; do
  if ! resp=$(curl -sf --max-time 5 "$url/health/containers" 2>/dev/null); then
    echo "  ✗ $url unreachable"
    ALL_OK=0
    continue
  fi
  state=$(echo "$resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); h=d.get("healthy"); t=d.get("target_pool_size"); print(str(h)+"/"+str(t))' 2>/dev/null || echo 'parse-fail')
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
  echo "On each host run: OSWORLD_POOL_SIZE=48 python scripts/servers/remote_env_server.py"
  exit 1
fi

# ─── Pre-flight: GPU memory check ─────────────────────────────────────────
echo ""
echo "=== Pre-flight: GPU memory ==="
gpu_check=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', ' '{ if ($2 < 35000) printf "GPU %s: only %s MiB free (need 35000)\n", $1, $2 }')
if [[ -n "$gpu_check" ]]; then
  echo "$gpu_check"
  echo "WARNING: some GPUs have <35 GB free. vllm may fail during memory profiling."
fi
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader

# ─── Activate venv + Ray + env tuning ──────────────────────────────────────
if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.95}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

# Clear stale Ray logs (best-effort)
[[ -f scripts/utils/clear_ray_logs.sh ]] && bash scripts/utils/clear_ray_logs.sh 2>/dev/null || true

# ─── Launch ────────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
LOG="${ROOT_DIR}/checkpoints/arpo_mcts_sft_v2.1_noise_${TS}.log"
mkdir -p "$(dirname "$LOG")"

echo ""
echo "=== Launching ARPO + v4 noise (MCTS SFT v2.1 base) ==="
echo "  config: $CONFIG"
echo "  model:  checkpoints/mcts_sft_v2.1/beta05_2e-6/epoch_1"
echo "  envs:   10.100.4.4 + 10.100.4.7 (48+48)"
echo "  log:    $LOG"
echo ""

python -m verl.trainer.main \
    config="$CONFIG" \
    "$@" \
    2>&1 | tee "$LOG"
