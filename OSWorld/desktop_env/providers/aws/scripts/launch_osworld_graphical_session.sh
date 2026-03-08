#!/bin/bash
set -euo pipefail

DESKTOP_USER="${1:?desktop user required}"
DESKTOP_HOME="${2:?desktop home required}"
DESKTOP_RUNTIME_DIR="${3:?desktop runtime dir required}"

DISPLAY_NUM="${OSWORLD_DISPLAY_NUMBER:-:0}"
DISPLAY_INDEX="${DISPLAY_NUM##*:}"
SCREEN_GEOMETRY="${OSWORLD_SCREEN_GEOMETRY:-1920x1080x24}"

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

touch "${DESKTOP_HOME}/.Xauthority"
chown "${DESKTOP_USER}:${DESKTOP_USER}" "${DESKTOP_HOME}/.Xauthority"

runuser -u "${DESKTOP_USER}" -- env \
  DISPLAY="${DISPLAY_NUM}" \
  HOME="${DESKTOP_HOME}" \
  XAUTHORITY="${DESKTOP_HOME}/.Xauthority" \
  XDG_RUNTIME_DIR="${DESKTOP_RUNTIME_DIR}" \
  XDG_SESSION_TYPE=x11 \
  DESKTOP_SESSION=ubuntu \
  dbus-run-session /usr/bin/gnome-session --session=ubuntu &
GNOME_PID=$!

wait "${GNOME_PID}"
