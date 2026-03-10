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
}

cleanup() {
  if [ -n "${GNOME_PID:-}" ] && kill -0 "${GNOME_PID}" 2>/dev/null; then
    kill "${GNOME_PID}" 2>/dev/null || true
    wait "${GNOME_PID}" 2>/dev/null || true
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

if command -v gnome-session >/dev/null 2>&1; then
  runuser -u "${DESKTOP_USER}" -- env \
    "${SESSION_COMMON_ENV[@]}" \
    XDG_CURRENT_DESKTOP=ubuntu:GNOME \
    XDG_SESSION_DESKTOP=ubuntu \
    DESKTOP_SESSION=ubuntu \
    CLUTTER_BACKEND=x11 \
    GSK_RENDERER=cairo \
    /usr/bin/gnome-session --session=ubuntu &
  SESSION_PID=$!
else
  # Lightweight fallback: openbox window manager when gnome-session is unavailable
  # (e.g., minimal Docker containers). Xorg is already running above.
  WM_BIN="$(command -v openbox || command -v fluxbox || command -v twm || true)"
  if [ -z "${WM_BIN}" ]; then
    echo "No window manager found (gnome-session, openbox, fluxbox, twm); Xorg-only mode" >&2
    # Stay alive as long as Xorg runs
    wait "${XORG_PID}"
    exit 0
  fi
  runuser -u "${DESKTOP_USER}" -- env \
    "${SESSION_COMMON_ENV[@]}" \
    "${WM_BIN}" &
  SESSION_PID=$!

  # Start desktop accessories (panel + wallpaper) in the OpenBox fallback.
  # These run in the same X session context so they can connect to the display.
  (
    sleep 3  # wait for openbox to initialize

    # Wallpaper: generate an Ubuntu-like gradient if none exists, then apply it
    WALLPAPER="${DESKTOP_HOME}/wallpaper.png"
    if [ ! -f "${WALLPAPER}" ]; then
      python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (1920, 1080), (44, 0, 30))
draw = ImageDraw.Draw(img)
for y in range(1080):
    r = int(44 + (77-44) * y / 1080)
    g = int(0 + (20-0) * y / 1080)
    b = int(30 + (60-30) * y / 1080)
    draw.line([(0, y), (1919, y)], fill=(r, g, b))
img.save('${WALLPAPER}')
" 2>/dev/null || true
      chown "${DESKTOP_USER}:${DESKTOP_USER}" "${WALLPAPER}" 2>/dev/null || true
    fi

    if command -v feh >/dev/null 2>&1 && [ -f "${WALLPAPER}" ]; then
      runuser -u "${DESKTOP_USER}" -- env \
        "${SESSION_COMMON_ENV[@]}" \
        feh --bg-fill "${WALLPAPER}" >/dev/null 2>&1 || true
    else
      # Fallback: at least set a non-black root background
      DISPLAY="${DISPLAY_NUM}" xsetroot -solid "#2C001E" -cursor_name left_ptr >/dev/null 2>&1 || true
    fi

    # Panel/taskbar: tint2 provides a Windows/GNOME-like taskbar at the bottom
    if command -v tint2 >/dev/null 2>&1; then
      runuser -u "${DESKTOP_USER}" -- env \
        "${SESSION_COMMON_ENV[@]}" \
        tint2 >/dev/null 2>&1 &
    fi
  ) &
fi

wait "${SESSION_PID}"
