#!/bin/bash
set -euo pipefail

DESKTOP_USER="${1:?desktop user required}"
DESKTOP_HOME="${2:?desktop home required}"
DESKTOP_RUNTIME_DIR="${3:?desktop runtime dir required}"

DISPLAY_NUM="${OSWORLD_DISPLAY_NUMBER:-:0}"
DISPLAY_INDEX="${DISPLAY_NUM##*:}"
SCREEN_GEOMETRY="${OSWORLD_SCREEN_GEOMETRY:-1920x1080x24}"
XDG_CONFIG_HOME="${DESKTOP_HOME}/.config"
XDG_CACHE_HOME="${DESKTOP_HOME}/.cache"
XDG_DATA_HOME="${DESKTOP_HOME}/.local/share"
XDG_STATE_HOME="${DESKTOP_HOME}/.local/state"

install_autostart_override() {
  local desktop_file="$1"
  local override_dir="${XDG_CONFIG_HOME}/autostart"
  local override_path="${override_dir}/${desktop_file}"
  cat > "${override_path}" <<EOF
[Desktop Entry]
Type=Application
Name=${desktop_file}
Hidden=true
X-GNOME-Autostart-enabled=false
EOF
  chown "${DESKTOP_USER}:${DESKTOP_USER}" "${override_path}"
}

ensure_writable_session_home() {
  local paths=(
    "${DESKTOP_HOME}"
    "${XDG_CONFIG_HOME}"
    "${XDG_CONFIG_HOME}/autostart"
    "${XDG_CONFIG_HOME}/mutter"
    "${XDG_CONFIG_HOME}/mutter/sessions"
    "${XDG_CACHE_HOME}"
    "${DESKTOP_HOME}/.local"
    "${XDG_DATA_HOME}"
    "${XDG_DATA_HOME}/dbus-1"
    "${XDG_DATA_HOME}/dbus-1/services"
    "${XDG_DATA_HOME}/evolution"
    "${XDG_DATA_HOME}/flatpak"
    "${XDG_DATA_HOME}/flatpak/db"
    "${XDG_DATA_HOME}/gnome-shell"
    "${XDG_DATA_HOME}/keyrings"
    "${XDG_STATE_HOME}"
  )
  local path
  for path in "${paths[@]}"; do
    install -d -m 0700 "${path}"
    chown "${DESKTOP_USER}:${DESKTOP_USER}" "${path}"
  done
  touch "${DESKTOP_HOME}/.Xauthority" "${DESKTOP_HOME}/.ICEauthority"
  chown "${DESKTOP_USER}:${DESKTOP_USER}" "${DESKTOP_HOME}/.Xauthority" "${DESKTOP_HOME}/.ICEauthority"

  if ! runuser -u "${DESKTOP_USER}" -- test -w "${DESKTOP_HOME}"; then
    echo "Desktop home is not writable for ${DESKTOP_USER}: ${DESKTOP_HOME}" >&2
    ls -ld "${DESKTOP_HOME}" >&2 || true
    exit 1
  fi
  if ! runuser -u "${DESKTOP_USER}" -- test -w "${XDG_DATA_HOME}"; then
    echo "XDG data home is not writable for ${DESKTOP_USER}: ${XDG_DATA_HOME}" >&2
    ls -ld "${DESKTOP_HOME}" "${DESKTOP_HOME}/.local" "${XDG_DATA_HOME}" >&2 || true
    exit 1
  fi

  install_autostart_override "gnome-initial-setup-first-login.desktop"
  install_autostart_override "gnome-initial-setup-copy-worker.desktop"
  install_autostart_override "update-notifier.desktop"
  install_autostart_override "tracker-miner-fs-3.desktop"
}

cleanup() {
  if [ -n "${GNOME_PID:-}" ] && kill -0 "${GNOME_PID}" 2>/dev/null; then
    kill "${GNOME_PID}" 2>/dev/null || true
    wait "${GNOME_PID}" 2>/dev/null || true
  fi
  if [ -n "${XVFB_PID:-}" ] && kill -0 "${XVFB_PID}" 2>/dev/null; then
    kill "${XVFB_PID}" 2>/dev/null || true
    wait "${XVFB_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

install -d -m 0700 -o "${DESKTOP_USER}" -g "${DESKTOP_USER}" "${DESKTOP_RUNTIME_DIR}"
install -d -m 1777 /tmp/.X11-unix
ensure_writable_session_home

rm -f "/tmp/.X${DISPLAY_INDEX}-lock"
rm -f "/tmp/.X11-unix/X${DISPLAY_INDEX}"

Xvfb "${DISPLAY_NUM}" -screen 0 "${SCREEN_GEOMETRY}" -ac -nolisten tcp &
XVFB_PID=$!

deadline=$((SECONDS + 20))
while [ "${SECONDS}" -lt "${deadline}" ]; do
  if [ -S "/tmp/.X11-unix/X${DISPLAY_INDEX}" ]; then
    break
  fi
  sleep 1
done

if [ ! -S "/tmp/.X11-unix/X${DISPLAY_INDEX}" ]; then
  echo "Xvfb did not create ${DISPLAY_NUM} socket" >&2
  exit 1
fi

runuser -u "${DESKTOP_USER}" -- env \
  DISPLAY="${DISPLAY_NUM}" \
  HOME="${DESKTOP_HOME}" \
  XAUTHORITY="${DESKTOP_HOME}/.Xauthority" \
  XDG_RUNTIME_DIR="${DESKTOP_RUNTIME_DIR}" \
  XDG_SESSION_TYPE=x11 \
  DESKTOP_SESSION=ubuntu \
  XDG_CONFIG_HOME="${XDG_CONFIG_HOME}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  XDG_DATA_HOME="${XDG_DATA_HOME}" \
  XDG_STATE_HOME="${XDG_STATE_HOME}" \
  dbus-run-session /usr/bin/gnome-session --session=ubuntu &
GNOME_PID=$!

wait "${GNOME_PID}"
