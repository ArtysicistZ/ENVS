#!/bin/bash
set -euo pipefail

DESKTOP_USER="${1:?desktop user required}"
DESKTOP_HOME="${2:?desktop home required}"

export DEBIAN_FRONTEND=noninteractive

echo "Provisioning Ubuntu desktop stack for OSWorld on this VM"

if ! id "${DESKTOP_USER}" >/dev/null 2>&1; then
  echo "Desktop user does not exist: ${DESKTOP_USER}" >&2
  exit 1
fi

echo "[desktop 1/5] Installing desktop and OSWorld runtime packages"
apt-get update
echo "gdm3 shared/default-x-display-manager select gdm3" | debconf-set-selections || true
apt-get install -y \
  ubuntu-desktop \
  gdm3 \
  dbus-x11 \
  gnome-screenshot \
  wmctrl \
  ffmpeg \
  socat \
  xclip \
  xauth \
  x11-xserver-utils \
  xserver-xorg-video-dummy \
  python3-tk \
  python3-dev

echo "[desktop 2/5] Enabling graphical boot target"
systemctl set-default graphical.target

echo "[desktop 3/5] Configuring GDM autologin and Xorg"
install -d -m 0755 /etc/gdm3
cat > /etc/gdm3/custom.conf <<EOF
[daemon]
AutomaticLoginEnable=true
AutomaticLogin=${DESKTOP_USER}
WaylandEnable=false
EOF

install -d -m 0755 /etc/X11/xorg.conf.d
cat > /etc/X11/xorg.conf.d/10-dummy.conf <<'EOF'
Section "Device"
    Identifier "DummyDevice"
    Driver "dummy"
    VideoRam 32768
EndSection

Section "Monitor"
    Identifier "DummyMonitor"
    HorizSync 28.0-80.0
    VertRefresh 48.0-75.0
    Modeline "1920x1080" 172.80 1920 2048 2248 2576 1080 1083 1088 1120
EndSection

Section "Screen"
    Identifier "DummyScreen"
    Device "DummyDevice"
    Monitor "DummyMonitor"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1920x1080"
    EndSubSection
EndSection
EOF

echo "[desktop 4/5] Enabling display manager"
systemctl enable gdm3.service

echo "[desktop 5/5] Preparing desktop user home"
install -d -o "${DESKTOP_USER}" -g "${DESKTOP_USER}" -m 0755 "${DESKTOP_HOME}"
touch "${DESKTOP_HOME}/.Xauthority"
chown "${DESKTOP_USER}:${DESKTOP_USER}" "${DESKTOP_HOME}/.Xauthority"

echo "Desktop provisioning completed."
