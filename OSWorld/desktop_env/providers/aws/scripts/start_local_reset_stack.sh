#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
CURRENT_PWD="$(pwd -P)"
START_TS="${SECONDS}"
DESKTOP_USER="${OSWORLD_RESET_USER:-user}"
DESKTOP_HOME="${OSWORLD_RESET_HOME:-/home/${DESKTOP_USER}}"
if [[ "${DESKTOP_HOME}" == "/home/ubuntu" || "${DESKTOP_HOME}" == /home/ubuntu/* ]]; then
  cat >&2 <<EOF
/home/ubuntu is protected from OSWorld reset/install mutations.

It can still be used for the Python runtime and for reading files, but it must
never be used as the disposable reset workspace target.

Refusing to continue with:
  OSWORLD_RESET_HOME=${DESKTOP_HOME}
EOF
  exit 1
fi
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
ENV_ARGS=("PYTHON_BIN=${PYTHON_BIN}")
if [ -n "${OSWORLD_RESET_USER:-}" ]; then
  ENV_ARGS+=("OSWORLD_RESET_USER=${OSWORLD_RESET_USER}")
fi
if [ -n "${OSWORLD_RESET_HOME:-}" ]; then
  ENV_ARGS+=("OSWORLD_RESET_HOME=${OSWORLD_RESET_HOME}")
fi

sudo env "${ENV_ARGS[@]}" bash "${REPO_ROOT}/OSWorld/desktop_env/providers/aws/scripts/install_resetd.sh"

wait_for_health() {
  local url="$1"
  local name="$2"
  local timeout="${3:-60}"
  local deadline=$((SECONDS + timeout))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if curl --max-time 5 -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for ${name} health at ${url}" >&2
  return 1
}

wait_for_health "http://127.0.0.1:5001/health" "reset daemon" 30
wait_for_health "http://127.0.0.1:5000/health" "OSWorld server" 90

STRICT_SCREENSHOT_READINESS="${OSWORLD_STRICT_SCREENSHOT_READINESS:-0}"

screenshot_looks_valid() {
  local image_path="$1"
  "${PYTHON_BIN}" - "${image_path}" <<'PY'
import sys
from PIL import Image, ImageStat

image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
stats = ImageStat.Stat(image)
means = stats.mean
stddev = stats.stddev

# Reject a trivially dark frame.
if max(means) < 8:
    raise SystemExit(1)

# Reject a nearly uniform frame, which is typically the blank X root window.
if max(stddev) < 2.0:
    raise SystemExit(1)
PY
}

wait_for_screenshot() {
  local url="$1"
  local timeout="${2:-30}"
  local deadline=$((SECONDS + timeout))
  local tmpfile
  tmpfile="$(mktemp)"
  trap 'rm -f "${tmpfile}"' RETURN
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if curl --max-time 8 -fsS "${url}" -o "${tmpfile}" >/dev/null 2>&1 && [ -s "${tmpfile}" ]; then
      if [ "${STRICT_SCREENSHOT_READINESS}" = "1" ] || [ "${STRICT_SCREENSHOT_READINESS}" = "true" ]; then
        if screenshot_looks_valid "${tmpfile}"; then
          rm -f "${tmpfile}"
          trap - RETURN
          return 0
        fi
      else
        rm -f "${tmpfile}"
        trap - RETURN
        return 0
      fi
    fi
    : > "${tmpfile}"
    sleep 1
  done
  rm -f "${tmpfile}"
  trap - RETURN
  echo "Timed out waiting for screenshot readiness at ${url}" >&2
  return 1
}

wait_for_screenshot "http://127.0.0.1:5000/screenshot?cursor=0" 30

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
echo
echo "Cold start completed in $((SECONDS - START_TS))s"
