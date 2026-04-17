#!/usr/bin/env bash
# ARPO Production Training — 8x A100-80GB
#
# Usage:
#   bash scripts/training/run_arpo_8gpu.sh                          # production (86 tasks)
#   bash scripts/training/run_arpo_8gpu.sh config=configs/arpo_smoke_8gpu.yaml  # smoke test (4 tasks)
#
# All extra arguments are forwarded to verl.trainer.main as CLI overrides.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Activate venv
if [[ -f /home/kevinzyz/hansenzuishuai/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /home/kevinzyz/hansenzuishuai/.venv/bin/activate
fi

# Ray memory tuning — reduce false-positive worker kills
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.8}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

# General env
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

# Clear stale Ray logs
if [[ -f scripts/utils/clear_ray_logs.sh ]]; then
  bash scripts/utils/clear_ray_logs.sh || true
fi

# Default config is production; override via CLI args
python -m verl.trainer.main config=configs/arpo_8gpu.yaml "$@"
