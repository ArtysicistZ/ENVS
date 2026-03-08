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
APP_ROOT="${CONTROL_ROOT}/app"
CONTROL_PLANE_HASH_ROOT="${OSWORLD_CONTROL_PLANE_HASH_ROOT:-${APP_ROOT}/OSWorld}"
BASELINE_HOME="${OSWORLD_RESET_BASELINE_HOME:-${CONTROL_ROOT}/baseline/home-user}"
BASELINE_DCONF="${OSWORLD_RESET_DCONF_SNAPSHOT:-${CONTROL_ROOT}/baseline/dconf/user.dconf}"
STATE_ROOT="${OSWORLD_RESET_STATE_ROOT:-/var/lib/osworld-reset}"
CONTROL_PLANE_STAMP_PATH="${OSWORLD_CONTROL_PLANE_STAMP_PATH:-${STATE_ROOT}/control_plane_build_id}"
SESSION_ROOT="${OSWORLD_RESET_SESSION_ROOT:-/var/lib/osworld/session}"
SYSTEM_DIST_PACKAGES="${OSWORLD_SYSTEM_DIST_PACKAGES:-/usr/lib/python3/dist-packages}"
BASELINE_REBUILD_THRESHOLD_KB="${OSWORLD_RESET_BASELINE_REBUILD_THRESHOLD_KB:-2097152}"
BASELINE_MODE="${OSWORLD_RESET_BASELINE_MODE:-minimal}"
if [ -n "${OSWORLD_RESET_USER:-}" ]; then
  DESKTOP_USER="${OSWORLD_RESET_USER}"
elif id -u osworld >/dev/null 2>&1; then
  DESKTOP_USER="osworld"
else
  DESKTOP_USER="osworld"
fi

if ! id -u "${DESKTOP_USER}" >/dev/null 2>&1; then
  useradd -m -s /bin/bash "${DESKTOP_USER}"
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
  local display_num="${1:-:0}"
  local display_index="${display_num##*:}"
  [ -S "/tmp/.X11-unix/X${display_index}" ]
}

render_unit() {
  local src="$1"
  local dst="$2"
  sed \
    -e "s|__APP_ROOT__|${APP_ROOT}|g" \
    -e "s|__CONTROL_PLANE_HASH_ROOT__|${CONTROL_PLANE_HASH_ROOT}|g" \
    -e "s|__CONTROL_PLANE_STAMP_PATH__|${CONTROL_PLANE_STAMP_PATH}|g" \
    -e "s|__BASELINE_MODE__|${BASELINE_MODE}|g" \
    -e "s|__DESKTOP_USER__|${DESKTOP_USER}|g" \
    -e "s|__DESKTOP_HOME__|${DESKTOP_HOME}|g" \
    -e "s|__DESKTOP_RUNTIME_DIR__|${DESKTOP_RUNTIME_DIR}|g" \
    -e "s|__REPO_ROOT__|${REPO_ROOT}|g" \
    -e "s|__PYTHON_BIN__|${PYTHON_BIN}|g" \
    -e "s|__SYSTEM_DIST_PACKAGES__|${SYSTEM_DIST_PACKAGES}|g" \
    "${src}" > "${dst}"
}

normalize_home_tree_ownership() {
  local target="$1"
  if [ ! -d "${target}" ]; then
    return 0
  fi
  chown -R "${DESKTOP_USER}:${DESKTOP_USER}" "${target}"
}

baseline_needs_rebuild() {
  if [ "${OSWORLD_RESET_REBUILD_BASELINE:-0}" = "1" ] || [ "${OSWORLD_RESET_REBUILD_BASELINE:-false}" = "true" ]; then
    return 0
  fi

  if [ ! -d "${BASELINE_HOME}" ]; then
    return 1
  fi

  if [ -f "${STATE_ROOT}/metadata.json" ]; then
    local expected_user existing_mode
    expected_user="$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
path = Path(${STATE_ROOT@Q}) / "metadata.json"
data = json.loads(path.read_text(encoding="utf-8"))
print(data.get("expected_session_user", ""))
PY
)"
    existing_mode="$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
path = Path(${STATE_ROOT@Q}) / "metadata.json"
data = json.loads(path.read_text(encoding="utf-8"))
print(data.get("baseline_mode", ""))
PY
)"
    if [ -n "${expected_user}" ] && [ "${expected_user}" != "${DESKTOP_USER}" ]; then
      return 0
    fi
    if [ -n "${existing_mode}" ] && [ "${existing_mode}" != "${BASELINE_MODE}" ]; then
      return 0
    fi
  fi

  local baseline_kb
  baseline_kb="$(du -sk "${BASELINE_HOME}" | awk '{print $1}')"
  if [ "${baseline_kb}" -gt "${BASELINE_REBUILD_THRESHOLD_KB}" ]; then
    return 0
  fi

  if [ ! -s "${CONTROL_PLANE_STAMP_PATH}" ]; then
    return 0
  fi

  return 1
}

sync_control_plane_app() {
  local src_root="${REPO_ROOT}/OSWorld"
  local dst_root="${APP_ROOT}/OSWorld"

  install -d -m 0755 "${APP_ROOT}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete \
      --exclude "__pycache__" \
      --exclude "*.pyc" \
      "${src_root}/" "${dst_root}/"
  else
    rm -rf "${dst_root}"
    cp -a "${src_root}" "${dst_root}"
    find "${dst_root}" -type d -name "__pycache__" -prune -exec rm -rf {} +
    find "${dst_root}" -type f -name "*.pyc" -delete
  fi
  chown -R root:root "${APP_ROOT}"
}

lock_down_control_plane_app() {
  local dst_root="${APP_ROOT}/OSWorld"
  chown -R root:root "${dst_root}"
  find "${dst_root}" -type d -exec chmod 0555 {} +
  find "${dst_root}" -type f -exec chmod 0444 {} +
}

write_control_plane_build_stamp() {
  install -d -m 0755 "$(dirname "${CONTROL_PLANE_STAMP_PATH}")"
  "${PYTHON_BIN}" - <<PY > "${CONTROL_PLANE_STAMP_PATH}.tmp"
import hashlib
import os
from pathlib import Path

root = Path(${CONTROL_PLANE_HASH_ROOT@Q}).resolve()
digest = hashlib.sha256()

for current_root, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
    for name in sorted(filenames):
        if name.endswith(".pyc"):
            continue
        path = Path(current_root) / name
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        stat = path.stat()
        digest.update(str(stat.st_mode & 0o7777).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

print(digest.hexdigest())
PY
  mv "${CONTROL_PLANE_STAMP_PATH}.tmp" "${CONTROL_PLANE_STAMP_PATH}"
  chmod 0644 "${CONTROL_PLANE_STAMP_PATH}"
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

check_server_python_deps() {
  local checker="${APP_ROOT}/OSWorld/desktop_env/providers/aws/scripts/check_osworld_server_deps.py"
  echo "[2b/6] Checking Python dependencies for osworld-server with ${PYTHON_BIN}"
  if PYTHONPATH="${APP_ROOT}/OSWorld:${SYSTEM_DIST_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" "${checker}"; then
    return 0
  fi

  echo >&2
  echo "osworld-server Python dependency preflight failed for ${PYTHON_BIN}" >&2
  echo "The missing modules above must be installed into the same interpreter used by osworld-server." >&2
  exit 1
}

ensure_accessibility_python_deps() {
  if PYTHONPATH="${SYSTEM_DIST_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c "import pyatspi" >/dev/null 2>&1; then
    :
  else
    echo "[2b/6] Installing Ubuntu accessibility runtime dependency python3-pyatspi"
    apt-get install -y python3-pyatspi
  fi
}

ensure_python_module() {
  local module_name="$1"
  local package_name="$2"
  if PYTHONPATH="${APP_ROOT}/OSWorld:${SYSTEM_DIST_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('${module_name}') else 1)" \
    >/dev/null 2>&1; then
    return 0
  fi

  echo "[2b/6] Installing missing Python runtime dependency ${package_name} into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install "${package_name}"
}

ensure_server_python_runtime_deps() {
  ensure_python_module "waitress" "waitress"
  ensure_python_module "flask" "flask"
  ensure_python_module "lxml" "lxml"
  ensure_python_module "PIL" "Pillow"
  ensure_python_module "pyautogui" "pyautogui"
  ensure_python_module "pygetwindow" "PyGetWindow"
  ensure_python_module "requests" "requests"
  ensure_python_module "Xlib" "python-xlib"
}

seed_minimal_baseline_home() {
  install -d -m 0755 "${BASELINE_HOME}"
  if [ -d /etc/skel ]; then
    cp -a /etc/skel/. "${BASELINE_HOME}/"
  fi

  install -d -o "${DESKTOP_USER}" -g "${DESKTOP_USER}" -m 0755 \
    "${BASELINE_HOME}/Desktop" \
    "${BASELINE_HOME}/Documents" \
    "${BASELINE_HOME}/Downloads" \
    "${BASELINE_HOME}/Music" \
    "${BASELINE_HOME}/Pictures" \
    "${BASELINE_HOME}/Public" \
    "${BASELINE_HOME}/Templates" \
    "${BASELINE_HOME}/Videos"

  install -d -o "${DESKTOP_USER}" -g "${DESKTOP_USER}" -m 0700 \
    "${BASELINE_HOME}/.config" \
    "${BASELINE_HOME}/.cache" \
    "${BASELINE_HOME}/.local" \
    "${BASELINE_HOME}/.local/share" \
    "${BASELINE_HOME}/.local/state"

  touch "${BASELINE_HOME}/.Xauthority"
  chown "${DESKTOP_USER}:${DESKTOP_USER}" "${BASELINE_HOME}/.Xauthority"
}

wait_for_x_socket() {
  local timeout="${1:-30}"
  local display_num="${2:-:0}"
  local display_index="${display_num##*:}"
  local deadline=$((SECONDS + timeout))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if [ -S "/tmp/.X11-unix/X${display_index}" ]; then
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
  - no usable graphical session produced by the installer

This usually means one of:
  1. this is not the desktop AMI / OSWorld VM image
  2. the desktop stack was not installed on the VM
  3. Xorg or the desktop session failed to start

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

install -d -m 0755 "${RESET_ROOT}" "${SERVER_ROOT}" "${APP_ROOT}" "$(dirname "${BASELINE_HOME}")" "$(dirname "${BASELINE_DCONF}")" "${STATE_ROOT}" "${SESSION_ROOT}"

DISPLAY_NUM="${OSWORLD_DISPLAY_NUMBER:-:0}"
PROVISION_DESKTOP_MODE="${OSWORLD_PROVISION_DESKTOP:-auto}"
if ! has_x_socket "${DISPLAY_NUM}"; then
  if [ "${PROVISION_DESKTOP_MODE}" = "0" ] || [ "${PROVISION_DESKTOP_MODE}" = "false" ]; then
    fail_no_desktop
  fi
  echo "No live X socket detected; provisioning or refreshing the OSWorld desktop stack"
  bash "${SCRIPT_DIR}/provision_osworld_desktop.sh" "${DESKTOP_USER}" "${DESKTOP_HOME}" "${PYTHON_BIN}"
fi

echo "[2/6] Syncing OSWorld control-plane app into ${APP_ROOT}"
sync_control_plane_app
echo "[2a/6] Locking down staged control-plane app permissions"
lock_down_control_plane_app
echo "[2a/6] Writing control-plane build stamp to ${CONTROL_PLANE_STAMP_PATH}"
write_control_plane_build_stamp
ensure_accessibility_python_deps
ensure_server_python_runtime_deps
check_server_python_deps

echo "[3/6] Rendering systemd units"
install -d -m 0755 /etc/systemd/system
render_unit "${SCRIPT_DIR}/systemd/osworld-home-overlay.service" /etc/systemd/system/osworld-home-overlay.service
render_unit "${SCRIPT_DIR}/systemd/osworld-resetd.service" /etc/systemd/system/osworld-resetd.service
render_unit "${SCRIPT_DIR}/systemd/osworld-graphical-session.service" /etc/systemd/system/osworld-graphical-session.service
render_unit "${SCRIPT_DIR}/systemd/osworld-server.service" /etc/systemd/system/osworld-server.service

if baseline_needs_rebuild; then
  echo "[4/6] Rebuilding baseline home at ${BASELINE_HOME}"
  rm -rf "${BASELINE_HOME}"
  rm -f "${BASELINE_DCONF}"
fi

if [ ! -d "${BASELINE_HOME}" ]; then
  if [ "${BASELINE_MODE}" = "copy-home" ]; then
    echo "[4/6] Seeding baseline home from ${DESKTOP_HOME} to ${BASELINE_HOME} (first install can take time)"
    cp -a "${DESKTOP_HOME}" "${BASELINE_HOME}"
  else
    echo "[4/6] Building minimal baseline home at ${BASELINE_HOME}"
    seed_minimal_baseline_home
  fi
else
  echo "[4/6] Baseline home already exists at ${BASELINE_HOME}, skipping seed copy"
fi
echo "[4b/6] Normalizing baseline home ownership for ${DESKTOP_USER}"
normalize_home_tree_ownership "${BASELINE_HOME}"

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
systemctl enable osworld-graphical-session.service
systemctl enable osworld-server.service
systemctl stop osworld-server.service || true
systemctl stop osworld-graphical-session.service || true
systemctl stop osworld-resetd.service || true
modprobe overlay || true
DISPLAY_MANAGER_UNIT="$(detect_display_manager_unit || true)"
if [ -n "${DISPLAY_MANAGER_UNIT}" ]; then
  echo "Stopping display manager so OSWorld can own :0 directly: ${DISPLAY_MANAGER_UNIT}"
  systemctl stop "${DISPLAY_MANAGER_UNIT}" || true
fi
restart_or_dump osworld-home-overlay.service
restart_or_dump osworld-graphical-session.service
if ! wait_for_x_socket 30 "${DISPLAY_NUM}"; then
  echo "OSWorld graphical session did not produce an X socket within 30s" >&2
  echo "--- systemctl status osworld-graphical-session.service ---" >&2
  systemctl --no-pager --full status osworld-graphical-session.service >&2 || true
  echo "--- journalctl -u osworld-graphical-session.service -n 200 ---" >&2
  journalctl -u osworld-graphical-session.service -n 200 --no-pager >&2 || true
  fail_no_desktop
fi
restart_or_dump osworld-resetd.service
restart_or_dump osworld-server.service

OSWORLD_RESET_USER="${DESKTOP_USER}" \
OSWORLD_RESET_HOME="${DESKTOP_HOME}" \
OSWORLD_CONTROL_PLANE_ROOT="${CONTROL_PLANE_HASH_ROOT}" \
OSWORLD_CONTROL_PLANE_STAMP_PATH="${CONTROL_PLANE_STAMP_PATH}" \
OSWORLD_RESET_BASELINE_MODE="${BASELINE_MODE}" \
OSWORLD_SERVER_URL="http://127.0.0.1:5000" \
"${PYTHON_BIN}" "${APP_ROOT}/OSWorld/desktop_env/providers/aws/reset_runtime.py" prepare-baseline

echo "Reset stack install completed."
