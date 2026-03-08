#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
CURRENT_PWD="$(pwd -P)"
DESKTOP_HOME="${OSWORLD_RESET_HOME:-${HOME}}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(python -c 'import sys; print(sys.executable)')"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(python3 -c 'import sys; print(sys.executable)')"
  else
    PYTHON_BIN=""
  fi
fi

if [[ "${CURRENT_PWD}" == "${DESKTOP_HOME}" || "${CURRENT_PWD}" == "${DESKTOP_HOME}/"* ]]; then
  cat >&2 <<EOF
Do not start the reset stack from inside ${DESKTOP_HOME}.
Current working directory: ${CURRENT_PWD}

The overlay service mounts on top of ${DESKTOP_HOME}, so running this command
from inside that tree can make osworld-home-overlay.service fail with EBUSY.

Use:
  cd /tmp
  bash ${REPO_ROOT}/OSWorld/desktop_env/providers/aws/scripts/start_local_reset_stack.sh
EOF
  exit 1
fi

echo "Starting OSWorld reset stack from ${REPO_ROOT}"
sudo PYTHON_BIN="${PYTHON_BIN}" bash "${REPO_ROOT}/OSWorld/desktop_env/providers/aws/scripts/install_resetd.sh"

wait_for_health() {
  local url="$1"
  local name="$2"
  local timeout="${3:-60}"
  local deadline=$((SECONDS + timeout))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for ${name} health at ${url}" >&2
  return 1
}

wait_for_health "http://127.0.0.1:5001/health" "reset daemon" 30
wait_for_health "http://127.0.0.1:5000/health" "OSWorld server" 90

echo
echo "Reset daemon health:"
curl http://127.0.0.1:5001/health
echo
echo
echo "OSWorld server health:"
curl http://127.0.0.1:5000/health
echo
echo
echo "Services:"
sudo systemctl --no-pager --full status osworld-home-overlay.service osworld-resetd.service osworld-server.service
