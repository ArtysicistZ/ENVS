#!/bin/bash
set -euo pipefail

# Provision the clean-room reset architecture into an AMI or a long-lived VM.
# Run this only when the machine is in a known-clean state.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${AWS_DIR}/../../../.." && pwd)"

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

echo "[1/6] Installing reset stack for desktop user '${DESKTOP_USER}' (${DESKTOP_HOME})"

render_unit() {
  local src="$1"
  local dst="$2"
  sed \
    -e "s|__DESKTOP_USER__|${DESKTOP_USER}|g" \
    -e "s|__DESKTOP_HOME__|${DESKTOP_HOME}|g" \
    -e "s|__DESKTOP_RUNTIME_DIR__|${DESKTOP_RUNTIME_DIR}|g" \
    "${src}" > "${dst}"
}

install -d -m 0755 "${RESET_ROOT}" "${SERVER_ROOT}" "$(dirname "${BASELINE_HOME}")" "$(dirname "${BASELINE_DCONF}")" "${STATE_ROOT}" "${SESSION_ROOT}"

echo "[2/6] Copying runtime and server files into ${CONTROL_ROOT}"
install -m 0644 "${AWS_DIR}/reset_runtime.py" "${RESET_ROOT}/reset_runtime.py"
install -m 0644 "${AWS_DIR}/reset_daemon.py" "${RESET_ROOT}/reset_daemon.py"
install -m 0644 "${REPO_ROOT}/OSWorld/desktop_env/server/main.py" "${SERVER_ROOT}/main.py"
install -m 0644 "${REPO_ROOT}/OSWorld/desktop_env/server/runtime_paths.py" "${SERVER_ROOT}/runtime_paths.py"

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
systemctl restart osworld-home-overlay.service
systemctl restart osworld-resetd.service
systemctl restart osworld-server.service

OSWORLD_RESET_USER="${DESKTOP_USER}" \
OSWORLD_RESET_HOME="${DESKTOP_HOME}" \
OSWORLD_SERVER_URL="http://127.0.0.1:5000" \
python3 "${RESET_ROOT}/reset_runtime.py" prepare-baseline

echo "Reset stack install completed."
