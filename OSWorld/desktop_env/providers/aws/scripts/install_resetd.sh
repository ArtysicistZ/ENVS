#!/bin/bash
set -euo pipefail

# Provision the clean-room reset architecture into an AMI or a long-lived VM.
# Run this only when the machine is in a known-clean state.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${AWS_DIR}/../../../.." && pwd)"
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

CONTROL_ROOT="${OSWORLD_CONTROL_PLANE_ROOT:-/opt/osworld}"
RESET_ROOT="${CONTROL_ROOT}/reset"
SERVER_ROOT="${CONTROL_ROOT}/server"
BASELINE_HOME="${OSWORLD_RESET_BASELINE_HOME:-${CONTROL_ROOT}/baseline/home-user}"
BASELINE_DCONF="${OSWORLD_RESET_DCONF_SNAPSHOT:-${CONTROL_ROOT}/baseline/dconf/user.dconf}"
STATE_ROOT="${OSWORLD_RESET_STATE_ROOT:-/var/lib/osworld-reset}"
SESSION_ROOT="${OSWORLD_RESET_SESSION_ROOT:-/var/lib/osworld/session}"
if [ -n "${OSWORLD_RESET_USER:-}" ]; then
  DESKTOP_USER="${OSWORLD_RESET_USER}"
elif [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
  DESKTOP_USER="${SUDO_USER}"
else
  DESKTOP_USER="$(id -un)"
fi

DESKTOP_HOME="${OSWORLD_RESET_HOME:-$(getent passwd "${DESKTOP_USER}" | cut -d: -f6)}"
DESKTOP_UID="$(id -u "${DESKTOP_USER}")"
DESKTOP_RUNTIME_DIR="/run/user/${DESKTOP_UID}"

if [ -z "${DESKTOP_HOME}" ] || [ ! -d "${DESKTOP_HOME}" ]; then
  echo "Could not determine a valid desktop home for user '${DESKTOP_USER}'" >&2
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN explicitly before running install_resetd.sh" >&2
  exit 1
fi

CURRENT_PWD="$(pwd -P)"
if [[ "${CURRENT_PWD}" == "${DESKTOP_HOME}" || "${CURRENT_PWD}" == "${DESKTOP_HOME}/"* ]]; then
  cat >&2 <<EOF
Install must be launched from outside the target desktop home.
Current working directory: ${CURRENT_PWD}
Target desktop home:       ${DESKTOP_HOME}

Reason:
  osworld-home-overlay.service mounts an overlay on top of ${DESKTOP_HOME}.
  If your shell is currently inside that directory tree, the mount can fail.

Run this instead:
  cd /tmp
  bash ${REPO_ROOT}/OSWorld/desktop_env/providers/aws/scripts/start_local_reset_stack.sh
EOF
  exit 1
fi

echo "[1/6] Installing reset stack for desktop user '${DESKTOP_USER}' (${DESKTOP_HOME})"

render_unit() {
  local src="$1"
  local dst="$2"
  sed \
    -e "s|__DESKTOP_USER__|${DESKTOP_USER}|g" \
    -e "s|__DESKTOP_HOME__|${DESKTOP_HOME}|g" \
    -e "s|__DESKTOP_RUNTIME_DIR__|${DESKTOP_RUNTIME_DIR}|g" \
    -e "s|__REPO_ROOT__|${REPO_ROOT}|g" \
    -e "s|__PYTHON_BIN__|${PYTHON_BIN}|g" \
    "${src}" > "${dst}"
}

restart_or_dump() {
  local unit="$1"
  if systemctl restart "${unit}"; then
    return 0
  fi

  echo >&2
  echo "Failed to restart ${unit}" >&2
  echo "--- systemctl status ${unit} ---" >&2
  systemctl --no-pager --full status "${unit}" >&2 || true
  echo "--- journalctl -u ${unit} -n 200 ---" >&2
  journalctl -u "${unit}" -n 200 --no-pager >&2 || true
  exit 1
}

install -d -m 0755 "${RESET_ROOT}" "${SERVER_ROOT}" "$(dirname "${BASELINE_HOME}")" "$(dirname "${BASELINE_DCONF}")" "${STATE_ROOT}" "${SESSION_ROOT}"

echo "[2/6] Ensuring control directories exist under ${CONTROL_ROOT}"
install -d -m 0755 "${RESET_ROOT}" "${SERVER_ROOT}"

echo "[3/6] Rendering systemd units"
install -d -m 0755 /etc/systemd/system
render_unit "${SCRIPT_DIR}/systemd/osworld-home-overlay.service" /etc/systemd/system/osworld-home-overlay.service
render_unit "${SCRIPT_DIR}/systemd/osworld-resetd.service" /etc/systemd/system/osworld-resetd.service
render_unit "${SCRIPT_DIR}/systemd/osworld-server.service" /etc/systemd/system/osworld-server.service

if [ ! -d "${BASELINE_HOME}" ]; then
  echo "[4/6] Seeding baseline home from ${DESKTOP_HOME} to ${BASELINE_HOME} (first install can take time)"
  cp -a "${DESKTOP_HOME}" "${BASELINE_HOME}"
else
  echo "[4/6] Baseline home already exists at ${BASELINE_HOME}, skipping seed copy"
fi

if [ ! -f "${BASELINE_DCONF}" ]; then
  echo "[5/6] Capturing baseline dconf"
  runuser -u "${DESKTOP_USER}" -- dconf dump / > "${BASELINE_DCONF}" 2>/dev/null || true
else
  echo "[5/6] Baseline dconf already exists at ${BASELINE_DCONF}, skipping capture"
fi

echo "[6/6] Reloading services and preparing baseline metadata"
systemctl daemon-reload
systemctl enable osworld-home-overlay.service
systemctl enable osworld-resetd.service
systemctl enable osworld-server.service
systemctl stop osworld-server.service || true
systemctl stop osworld-resetd.service || true
modprobe overlay || true
restart_or_dump osworld-home-overlay.service
restart_or_dump osworld-resetd.service
restart_or_dump osworld-server.service

OSWORLD_RESET_USER="${DESKTOP_USER}" \
OSWORLD_RESET_HOME="${DESKTOP_HOME}" \
OSWORLD_SERVER_URL="http://127.0.0.1:5000" \
"${PYTHON_BIN}" "${REPO_ROOT}/OSWorld/desktop_env/providers/aws/reset_runtime.py" prepare-baseline

echo "Reset stack install completed."
