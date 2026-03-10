#!/bin/bash
set -euo pipefail

DESKTOP_USER="${1:?desktop user required}"
DESKTOP_HOME="${2:?desktop home required}"
DESKTOP_RUNTIME_DIR="${3:?desktop runtime dir required}"

DISPLAY_NUM="${OSWORLD_DISPLAY_NUMBER:-:0}"
DISPLAY_INDEX="${DISPLAY_NUM##*:}"
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
    "${XDG_CONFIG_HOME}/dconf"
    "${XDG_CONFIG_HOME}/Code"
    "${XDG_CONFIG_HOME}/Code/User"
    "${XDG_CONFIG_HOME}/google-chrome"
    "${XDG_CONFIG_HOME}/google-chrome/Default"
    "${XDG_CONFIG_HOME}/libreoffice"
    "${XDG_CONFIG_HOME}/libreoffice/4"
    "${XDG_CONFIG_HOME}/libreoffice/4/user"
    "${XDG_CONFIG_HOME}/vlc"
    "${XDG_CONFIG_HOME}/GIMP"
    "${XDG_CONFIG_HOME}/GIMP/2.10"
    "${XDG_CACHE_HOME}"
    "${DESKTOP_HOME}/.local"
    "${XDG_DATA_HOME}"
    "${XDG_DATA_HOME}/dbus-1"
    "${XDG_DATA_HOME}/dbus-1/services"
    "${XDG_DATA_HOME}/evolution"
    "${XDG_DATA_HOME}/flatpak"
    "${XDG_DATA_HOME}/flatpak/db"
    "${XDG_DATA_HOME}/keyrings"
    "${XDG_STATE_HOME}"
    "${DESKTOP_HOME}/.thunderbird"
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
  install_autostart_override "gnome-keyring-secrets.desktop"
  install_autostart_override "gnome-keyring-pkcs11.desktop"
  install_autostart_override "gnome-keyring-ssh.desktop"
}

cleanup() {
  if [ -n "${SESSION_PID:-}" ] && kill -0 "${SESSION_PID}" 2>/dev/null; then
    kill "${SESSION_PID}" 2>/dev/null || true
    wait "${SESSION_PID}" 2>/dev/null || true
  fi
  if [ -n "${DBUS_PID:-}" ] && kill -0 "${DBUS_PID}" 2>/dev/null; then
    kill "${DBUS_PID}" 2>/dev/null || true
    wait "${DBUS_PID}" 2>/dev/null || true
  fi
  if [ -n "${XORG_PID:-}" ] && kill -0 "${XORG_PID}" 2>/dev/null; then
    kill "${XORG_PID}" 2>/dev/null || true
    wait "${XORG_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

install -d -m 0700 -o "${DESKTOP_USER}" -g "${DESKTOP_USER}" "${DESKTOP_RUNTIME_DIR}"
install -d -m 1777 /tmp/.X11-unix
ensure_writable_session_home

rm -f "/tmp/.X${DISPLAY_INDEX}-lock"
rm -f "/tmp/.X11-unix/X${DISPLAY_INDEX}"

# Use the provisioned dummy-driver Xorg display instead of Xvfb. GNOME Shell on
# Xvfb can stay visually black even though the display and screenshot endpoint
# are technically alive. The dummy Xorg path paints a real desktop surface.
Xorg "${DISPLAY_NUM}" -config /etc/X11/xorg.conf -noreset -nolisten tcp &
XORG_PID=$!

deadline=$((SECONDS + 20))
while [ "${SECONDS}" -lt "${deadline}" ]; do
  if [ -S "/tmp/.X11-unix/X${DISPLAY_INDEX}" ]; then
    break
  fi
  sleep 1
done

if [ ! -S "/tmp/.X11-unix/X${DISPLAY_INDEX}" ]; then
  echo "Xorg did not create ${DISPLAY_NUM} socket" >&2
  exit 1
fi

# Make the managed display immediately usable and visibly non-black even before
# the desktop session finishes painting its own background.
DISPLAY="${DISPLAY_NUM}" xset s off -dpms s noblank >/dev/null 2>&1 || true
DISPLAY="${DISPLAY_NUM}" xsetroot -solid "#2E3440" -cursor_name left_ptr >/dev/null 2>&1 || true

DBUS_SOCKET="${DESKTOP_RUNTIME_DIR}/bus"
rm -f "${DBUS_SOCKET}"
runuser -u "${DESKTOP_USER}" -- env \
  HOME="${DESKTOP_HOME}" \
  XDG_RUNTIME_DIR="${DESKTOP_RUNTIME_DIR}" \
  DBUS_SESSION_BUS_ADDRESS="unix:path=${DBUS_SOCKET}" \
  dbus-daemon --session --address="unix:path=${DBUS_SOCKET}" --nofork --nopidfile &
DBUS_PID=$!

deadline=$((SECONDS + 10))
while [ "${SECONDS}" -lt "${deadline}" ]; do
  if [ -S "${DBUS_SOCKET}" ]; then
    break
  fi
  sleep 1
done
if [ ! -S "${DBUS_SOCKET}" ]; then
  echo "D-Bus session socket was not created at ${DBUS_SOCKET}" >&2
  exit 1
fi

SESSION_COMMON_ENV=(
  DISPLAY="${DISPLAY_NUM}"
  HOME="${DESKTOP_HOME}"
  XAUTHORITY="${DESKTOP_HOME}/.Xauthority"
  XDG_RUNTIME_DIR="${DESKTOP_RUNTIME_DIR}"
  DBUS_SESSION_BUS_ADDRESS="unix:path=${DBUS_SOCKET}"
  XDG_SESSION_TYPE=x11
  GDK_BACKEND=x11
  LIBGL_ALWAYS_SOFTWARE=1
  MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
  __GLX_VENDOR_LIBRARY_NAME=mesa
  XDG_CONFIG_HOME="${XDG_CONFIG_HOME}"
  XDG_CACHE_HOME="${XDG_CACHE_HOME}"
  XDG_DATA_HOME="${XDG_DATA_HOME}"
  XDG_STATE_HOME="${XDG_STATE_HOME}"
)

# Select desktop session: full GNOME Shell (AWS) or GNOME Flashback (Docker).
# The ubuntu.session file only exists when ubuntu-session is installed (AWS AMI).
GNOME_SESSION_DIR="/usr/share/gnome-session/sessions"

# GNOME Shell (ubuntu.session) requires logind sessions which only GDM can
# provide; it cannot run in headless/container environments without a display
# manager.  AWS VMs have GDM so they get full GNOME Shell.  Docker containers
# use GNOME Flashback (Metacity) which provides the same GNOME ecosystem
# (nautilus, gnome-settings-daemon, gnome-panel) without needing logind.
if [ -f "${GNOME_SESSION_DIR}/ubuntu.session" ] && command -v gdm3 >/dev/null 2>&1; then
  # Full GNOME Shell desktop (AWS VMs with GDM)
  runuser -u "${DESKTOP_USER}" -- env \
    "${SESSION_COMMON_ENV[@]}" \
    XDG_CURRENT_DESKTOP=ubuntu:GNOME \
    XDG_SESSION_DESKTOP=ubuntu \
    DESKTOP_SESSION=ubuntu \
    CLUTTER_BACKEND=x11 \
    GSK_RENDERER=cairo \
    /usr/bin/gnome-session --session=ubuntu &
  SESSION_PID=$!
elif [ -f "${GNOME_SESSION_DIR}/gnome-flashback-metacity.session" ]; then
  # GNOME Flashback with Metacity (Docker containers / headless)
  runuser -u "${DESKTOP_USER}" -- env \
    "${SESSION_COMMON_ENV[@]}" \
    XDG_CURRENT_DESKTOP=GNOME-Flashback:GNOME \
    XDG_SESSION_DESKTOP=gnome-flashback-metacity \
    DESKTOP_SESSION=gnome-flashback-metacity \
    gnome-session --session=gnome-flashback-metacity &
  SESSION_PID=$!
else
  echo "FATAL: No supported desktop session found in ${GNOME_SESSION_DIR}/" >&2
  ls "${GNOME_SESSION_DIR}/" >&2 2>/dev/null || true
  exit 1
fi

wait "${SESSION_PID}"
