#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

if [[ -f /home/kevinzyz/hansenzuishuai/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /home/kevinzyz/hansenzuishuai/.venv/bin/activate
fi

python -m verl.trainer.main config=configs/smoke_remote_env_8gpu_a100.yaml "$@"
