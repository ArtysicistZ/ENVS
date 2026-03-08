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

has_x_socket() {
  find /tmp/.X11-unix -maxdepth 1 -type s -name 'X*' 2>/dev/null | grep -q .
}

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

wait_for_x_socket() {
  local timeout="${1:-30}"
  local deadline=$((SECONDS + timeout))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if find /tmp/.X11-unix -maxdepth 1 -type s -name 'X*' 2>/dev/null | grep -q .; then
      return 0
    fi
    sleep 1
  done
  return 1
}

fail_no_desktop() {
  cat >&2 <<EOF
No live X desktop session was detected on this VM.

OSWorld server requires a real graphical desktop. This VM currently has:
  - no connectable X display socket under /tmp/.X11-unix
  - no usable display manager service started by this installer

This usually means one of:
  1. this is not the desktop AMI / OSWorld VM image
  2. the desktop stack was not installed on the VM
  3. the graphical session is not running yet

Quick checks:
  ls -l /tmp/.X11-unix
  systemctl list-unit-files | grep -E 'gdm|lightdm|sddm'
  loginctl list-sessions

Do not use this VM for OSWorld if it is headless. The reset stack can run, but
osworld-server cannot start without X.
EOF
  exit 1
}

detect_display_manager_unit() {
  service_exists() {
    local unit="$1"
    [ -e "/etc/systemd/system/${unit}" ] || [ -e "/lib/systemd/system/${unit}" ] || [ -e "/usr/lib/systemd/system/${unit}" ]
  }

  if [ -n "${OSWORLD_DISPLAY_MANAGER_SERVICE:-}" ]; then
    if service_exists "${OSWORLD_DISPLAY_MANAGER_SERVICE}"; then
      echo "${OSWORLD_DISPLAY_MANAGER_SERVICE}"
      return 0
    fi
  fi

  if dpkg -s gdm3 >/dev/null 2>&1; then
    if service_exists "gdm3.service"; then
      echo "gdm3.service"
      return 0
    fi
    if service_exists "gdm.service"; then
      echo "gdm.service"
      return 0
    fi
  fi
  if dpkg -s lightdm >/dev/null 2>&1 && service_exists "lightdm.service"; then
    echo "lightdm.service"
    return 0
  fi
  if dpkg -s sddm >/dev/null 2>&1 && service_exists "sddm.service"; then
    echo "sddm.service"
    return 0
  fi

  local candidates=(
    "display-manager.service"
    "gdm3.service"
    "gdm.service"
    "lightdm.service"
    "sddm.service"
  )
  local unit
  for unit in "${candidates[@]}"; do
    if service_exists "${unit}"; then
      echo "${unit}"
      return 0
    fi
  done
  return 1
}

install -d -m 0755 "${RESET_ROOT}" "${SERVER_ROOT}" "$(dirname "${BASELINE_HOME}")" "$(dirname "${BASELINE_DCONF}")" "${STATE_ROOT}" "${SESSION_ROOT}"

PROVISION_DESKTOP_MODE="${OSWORLD_PROVISION_DESKTOP:-auto}"
if ! has_x_socket; then
  if [ "${PROVISION_DESKTOP_MODE}" = "0" ] || [ "${PROVISION_DESKTOP_MODE}" = "false" ]; then
    fail_no_desktop
  fi
  echo "No live X socket detected; provisioning or refreshing the OSWorld desktop stack"
  bash "${SCRIPT_DIR}/provision_osworld_desktop.sh" "${DESKTOP_USER}" "${DESKTOP_HOME}"
fi

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
DISPLAY_MANAGER_UNIT="$(detect_display_manager_unit || true)"
if [ -n "${DISPLAY_MANAGER_UNIT}" ]; then
  echo "Stopping display manager before overlay remount: ${DISPLAY_MANAGER_UNIT}"
  systemctl stop "${DISPLAY_MANAGER_UNIT}" || true
fi
restart_or_dump osworld-home-overlay.service
if [ -n "${DISPLAY_MANAGER_UNIT}" ]; then
  echo "Starting display manager on top of overlay-mounted home: ${DISPLAY_MANAGER_UNIT}"
  restart_or_dump "${DISPLAY_MANAGER_UNIT}"
  if ! wait_for_x_socket 30; then
    echo "Display manager did not produce an X socket within 30s: ${DISPLAY_MANAGER_UNIT}" >&2
    fail_no_desktop
  fi
else
  echo "No known display manager unit detected; relying on existing graphical session"
  if ! wait_for_x_socket 5; then
    fail_no_desktop
  fi
fi
restart_or_dump osworld-resetd.service
restart_or_dump osworld-server.service

OSWORLD_RESET_USER="${DESKTOP_USER}" \
OSWORLD_RESET_HOME="${DESKTOP_HOME}" \
OSWORLD_SERVER_URL="http://127.0.0.1:5000" \
"${PYTHON_BIN}" "${REPO_ROOT}/OSWorld/desktop_env/providers/aws/reset_runtime.py" prepare-baseline

echo "Reset stack install completed."
