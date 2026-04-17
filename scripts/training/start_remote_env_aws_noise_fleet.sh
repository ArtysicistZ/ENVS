#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
set -a
[ -f .env ] && source .env >/dev/null 2>&1 || true
set +a
: "${AWS_REGION:?AWS_REGION must be set}"
: "${AWS_SUBNET_ID:?AWS_SUBNET_ID must be set}"
: "${AWS_SECURITY_GROUP_ID:?AWS_SECURITY_GROUP_ID must be set}"
mkdir -p docs/research
start_one() {
  local port="$1"
  local log="docs/research/remote_env_server_aws_${port}.log"
  if ss -ltn | rg -q ":${port}\b"; then
    echo "port ${port} already listening; skipping"
    return 0
  fi
  nohup env PROVIDER=aws AWS_REGION="$AWS_REGION" AWS_SUBNET_ID="$AWS_SUBNET_ID" AWS_SECURITY_GROUP_ID="$AWS_SECURITY_GROUP_ID" REMOTE_MAX_SLOTS=48 OSWORLD_POOL_SIZE=48 REMOTE_ACTION_PAUSE_SEC=0.8 REMOTE_REPEAT_ACTION_THRESHOLD=3 REMOTE_REPEAT_ACTION_PENALTY=0.5 REMOTE_ENV_HOST=127.0.0.1 REMOTE_ENV_PORT="$port" PYTHONPATH="$REPO_ROOT:$REPO_ROOT/OSWorld" "$REPO_ROOT/.venv/bin/python" scripts/servers/remote_env_server.py > "$log" 2>&1 &
  echo "$!"
}
start_one 16001
start_one 16002
for port in 16001 16002; do
  for _ in $(seq 1 20); do
    if curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" || true
  echo
 done
