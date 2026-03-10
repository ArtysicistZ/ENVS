#!/usr/bin/env bash
# Run ARPO smoke training (remote env) on one of several servers.
# Use when the current machine has no free GPUs (e.g. all 8 in use by another job).
#
# Usage:
#   ./scripts/run_remote_env_on_server.sh           # Run on THIS machine (you must be on the target server)
#   ./scripts/run_remote_env_on_server.sh 1         # SSH to SERVER_1 and run there
#   ./scripts/run_remote_env_on_server.sh 2         # SSH to SERVER_2 and run there
#   ... 3, 4 for SERVER_3, SERVER_4
#   DeepX: server 2 is occupied — use 3 or 4.
#
# Prereqs on the target server:
#   - Repo cloned, deps installed, env activated.
#   - env.remote_server_url in configs/smoke_remote_env.yaml points to your EC2 remote env (e.g. http://34.229.141.88:15001).
#   - At least 4 free GPUs (config uses n_gpus_per_node: 4).

set -e

# DeepX cluster: server indices 1–4. Server 2 is occupied — use 3 or 4.
# Use hostnames so SSH works from inside the cluster (IPs may timeout from other nodes).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SERVER_1="${ARPO_SERVER_1:-deepx-a100-40g-1}"
SERVER_2="${ARPO_SERVER_2:-deepx-a100-40g-2}"   # occupied
SERVER_3="${ARPO_SERVER_3:-deepx-a100-40g-3}"
SERVER_4="${ARPO_SERVER_4:-deepx-a100-40g-4}"

# Key folder names (by index) — keys live under ssh_keys_for_users/ssh_keys_for_users_<KEY_HOST>/
KEY_HOST_1="${ARPO_KEY_HOST_1:-20.51.139.132}"
KEY_HOST_2="${ARPO_KEY_HOST_2:-172.172.143.249}"
KEY_HOST_3="${ARPO_KEY_HOST_3:-172.174.34.71}"
KEY_HOST_4="${ARPO_KEY_HOST_4:-172.190.164.51}"

get_server() {
  case "$1" in
    1) echo "$SERVER_1" ;;
    2) echo "$SERVER_2" ;;
    3) echo "$SERVER_3" ;;
    4) echo "$SERVER_4" ;;
    *) echo "" ;;
  esac
}

get_ssh_key() {
  local idx="$1"
  local key_host
  case "$idx" in
    1) key_host="$KEY_HOST_1" ;;
    2) key_host="$KEY_HOST_2" ;;
    3) key_host="$KEY_HOST_3" ;;
    4) key_host="$KEY_HOST_4" ;;
    *) echo "" ; return ;;
  esac
  echo "$REPO_ROOT/ssh_keys_for_users/ssh_keys_for_users_${key_host}/ssh_keys_for_users/kevinzyz_id_ed25519"
}

run_training() {
  local repo_dir="${1:-.}"
  cd "$repo_dir"
  export PYTHONUNBUFFERED=1
  export HYDRA_FULL_ERROR=1
  export VERL_LOGGING_LEVEL=DEBUG
  export MKL_SERVICE_FORCE_INTEL=1
  export MKL_THREADING_LAYER=GNU
  # ARPO-style Ray tuning: reduce false-positive worker kills from Ray's memory monitor.
  export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.8}"
  export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
  if [[ -f scripts/utils/clear_ray_logs.sh ]]; then
    bash scripts/utils/clear_ray_logs.sh || true
  fi
  echo "Starting training on $(hostname) with config=configs/smoke_remote_env.yaml"
  python -m verl.trainer.main config=configs/smoke_remote_env.yaml
}

if [[ -n "${1:-}" ]]; then
  idx="$1"
  host=$(get_server "$idx")
  if [[ -z "$host" ]]; then
    echo "Unknown server index: $idx (use 1–4). Set ARPO_SERVER_1 … ARPO_SERVER_4 or edit this script."
    exit 1
  fi
  if [[ "$idx" == "2" ]]; then
    echo "Note: server 2 is occupied on DeepX — prefer 3 or 4. Continuing anyway."
  fi
  key=$(get_ssh_key "$idx")
  if [[ ! -f "$key" ]]; then
    echo "Key not found: $key"
    echo "Keys must be under repo: ssh_keys_for_users/ssh_keys_for_users_<IP>/ssh_keys_for_users/kevinzyz_id_ed25519"
    exit 1
  fi
  chmod 600 "$key" 2>/dev/null || true
  # Repo path on remote (set ARPO_REMOTE_DIR if different, e.g. /home/kevinzyz/hansenzuishuai)
  remote_dir="${ARPO_REMOTE_DIR:-$REPO_ROOT}"
  echo "SSH to $host (server $idx) using key $key, run training in $remote_dir"
  ssh -i "$key" -o StrictHostKeyChecking=accept-new "$host" "\
    export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 VERL_LOGGING_LEVEL=DEBUG MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU \
      RAY_memory_usage_threshold=\${RAY_memory_usage_threshold:-0.8} RAY_memory_monitor_refresh_ms=\${RAY_memory_monitor_refresh_ms:-0}; \
    cd $remote_dir && ( [ -f scripts/utils/clear_ray_logs.sh ] && bash scripts/utils/clear_ray_logs.sh || true ) && \
    python -m verl.trainer.main config=configs/smoke_remote_env.yaml"
else
  run_training "$REPO_ROOT"
fi
