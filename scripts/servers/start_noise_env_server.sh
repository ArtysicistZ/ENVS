#!/usr/bin/env bash
# Start a v4-noise-aware env_server on the current host with POOL_SIZE=48.
#
# Usage on each of 10.100.4.6 and 10.100.4.8:
#   bash scripts/servers/start_noise_env_server.sh
#
# Idempotent: kills any existing remote_env_server.py before starting fresh.
# Logs to checkpoints/env_server_<host>_<timestamp>.log.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Activate venv
if [[ -f /home/kevinzyz/hansenzuishuai/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /home/kevinzyz/hansenzuishuai/.venv/bin/activate
fi

# Kill any existing env_server (both stale code and accidental dups).
echo "[$(hostname)] Stopping existing remote_env_server (if any) ..."
pkill -INT -f 'scripts/servers/remote_env_server' 2>/dev/null || true
sleep 2
pkill -KILL -f 'scripts/servers/remote_env_server' 2>/dev/null || true
sleep 1

# Verify nothing left over
if pgrep -af 'scripts/servers/remote_env_server' | grep -v grep > /dev/null; then
  echo "[$(hostname)] ERROR: failed to stop existing env_server"
  pgrep -af 'scripts/servers/remote_env_server' | grep -v grep
  exit 1
fi

# Pre-flight: docker daemon reachable?
if ! docker info >/dev/null 2>&1; then
  echo "[$(hostname)] ERROR: docker daemon not reachable"
  exit 1
fi

# Launch with POOL_SIZE=48
HOST=$(hostname -s)
TS=$(date +%Y%m%d_%H%M%S)
LOG="${ROOT_DIR}/checkpoints/env_server_${HOST}_${TS}.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(hostname)] Starting env_server with POOL_SIZE=48 → log=$LOG"
nohup env OSWORLD_POOL_SIZE=48 PROVIDER=docker \
  python scripts/servers/remote_env_server.py \
  > "$LOG" 2>&1 &

PID=$!
echo "[$(hostname)] env_server pid=$PID"

# Wait for it to come up (max 180s — pre-warm of 48 containers can take that long)
echo "[$(hostname)] Waiting for /health/containers to report 48/48 ..."
for i in $(seq 1 90); do
  state=$(curl -sf --max-time 2 http://localhost:15001/health/containers 2>/dev/null \
          | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"{d.get(\"healthy\")}/{d.get(\"target_pool_size\")}")' 2>/dev/null \
          || echo 'down')
  if [[ "$state" == "48/48" ]]; then
    echo "[$(hostname)] READY (48/48 healthy after $((i*2))s)"
    echo "[$(hostname)] Tail log: tail -f $LOG"
    exit 0
  fi
  sleep 2
done

echo "[$(hostname)] ERROR: env_server did not reach 48/48 within 180s"
echo "[$(hostname)] Last few log lines:"
tail -20 "$LOG"
exit 1
