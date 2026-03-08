#!/bin/bash
set -euo pipefail

DESKTOP_USER="${1:?desktop user required}"
DESKTOP_HOME="${2:?desktop home required}"
PYTHON_BIN="${3:-python3}"

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
  xinit \
  xvfb \
  dbus-x11 \
  dbus-user-session \
  gnome-screenshot \
  wmctrl \
  ffmpeg \
  socat \
  xclip \
  xauth \
  x11-xserver-utils \
  xserver-xorg-video-dummy \
  python3-pip \
  python3-tk \
  python3-dev

if ! "${PYTHON_BIN}" -c "import pyatspi" >/dev/null 2>&1; then
  echo "[desktop 1b/5] Installing pyatspi2 into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install pyatspi2
fi

echo "[desktop 2/5] Setting multi-user boot target for managed OSWorld session"
systemctl set-default multi-user.target

echo "[desktop 3/5] Configuring GDM autologin and Xorg"
install -d -m 0755 /etc/gdm3
cat > /etc/gdm3/custom.conf <<EOF
[daemon]
AutomaticLoginEnable=true
AutomaticLogin=${DESKTOP_USER}
WaylandEnable=false
EOF

cat > /etc/X11/xorg.conf <<'EOF'
Section "ServerLayout"
    Identifier "DummyLayout"
    Screen 0 "DummyScreen"
EndSection

Section "Monitor"
    Identifier "DummyMonitor"
    HorizSync 28.0-80.0
    VertRefresh 48.0-75.0
    Modeline "1920x1080" 172.80 1920 2048 2248 2576 1080 1083 1088 1120
EndSection

Section "Device"
    Identifier "DummyDevice"
    Driver "dummy"
    VideoRam 256000
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

echo "[desktop 4/5] Disabling GDM autostart in favor of managed OSWorld session"
systemctl disable gdm3.service || true
systemctl disable gdm.service || true
systemctl stop gdm3.service || true
systemctl stop gdm.service || true
systemctl stop display-manager.service || true
systemctl mask gdm3.service || true
systemctl mask gdm.service || true
systemctl mask display-manager.service || true

echo "[desktop 5/5] Preparing desktop user home"
install -d -o "${DESKTOP_USER}" -g "${DESKTOP_USER}" -m 0755 "${DESKTOP_HOME}"
touch "${DESKTOP_HOME}/.Xauthority"
chown "${DESKTOP_USER}:${DESKTOP_USER}" "${DESKTOP_HOME}/.Xauthority"

echo "Desktop provisioning completed."
