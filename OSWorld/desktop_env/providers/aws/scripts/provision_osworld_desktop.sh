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

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

ensure_google_chrome() {
  if command_exists google-chrome; then
    return 0
  fi
  if command_exists google-chrome-stable; then
    ln -sf "$(command -v google-chrome-stable)" /usr/local/bin/google-chrome
    return 0
  fi

  local arch
  arch="$(dpkg --print-architecture)"
  case "${arch}" in
    amd64)
      local chrome_deb="/tmp/google-chrome-stable_current_amd64.deb"
      echo "[desktop 1a/5] Installing Google Chrome"
      curl -LfsS "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb" -o "${chrome_deb}"
      apt-get install -y "${chrome_deb}"
      rm -f "${chrome_deb}"
      ;;
    arm64)
      echo "[desktop 1a/5] Installing Chromium compatibility runtime for google-chrome"
      apt-get install -y chromium-browser || apt-get install -y chromium
      local chromium_cmd=""
      if command_exists chromium-browser; then
        chromium_cmd="$(command -v chromium-browser)"
      elif command_exists chromium; then
        chromium_cmd="$(command -v chromium)"
      fi
      if [ -n "${chromium_cmd}" ]; then
        cat > /usr/local/bin/google-chrome <<EOF
#!/bin/sh
exec "${chromium_cmd}" --user-data-dir="\${HOME}/.config/google-chrome" "\$@"
EOF
        chmod 0755 /usr/local/bin/google-chrome
      fi
      ;;
    *)
      echo "Unsupported architecture for Chrome provisioning: ${arch}" >&2
      exit 1
      ;;
  esac

  if command_exists google-chrome-stable && ! command_exists google-chrome; then
    ln -sf "$(command -v google-chrome-stable)" /usr/local/bin/google-chrome
  fi

  if ! command_exists google-chrome && ! command_exists google-chrome-stable; then
    echo "Failed to provision a google-chrome-compatible browser" >&2
    exit 1
  fi
}

ensure_vscode() {
  if command_exists code; then
    return 0
  fi

  local arch vscode_url vscode_deb
  arch="$(dpkg --print-architecture)"
  case "${arch}" in
    amd64)
      vscode_url="https://update.code.visualstudio.com/latest/linux-deb-x64/stable"
      ;;
    arm64)
      vscode_url="https://update.code.visualstudio.com/latest/linux-deb-arm64/stable"
      ;;
    *)
      echo "Unsupported architecture for VS Code provisioning: ${arch}" >&2
      exit 1
      ;;
  esac

  vscode_deb="/tmp/vscode-latest.deb"
  echo "[desktop 1a/5] Installing VS Code"
  curl -LfsS "${vscode_url}" -o "${vscode_deb}"
  apt-get install -y "${vscode_deb}"
  rm -f "${vscode_deb}"

  if ! command_exists code; then
    echo "Failed to provision VS Code" >&2
    exit 1
  fi
}

echo "[desktop 1/5] Installing desktop and OSWorld runtime packages"
apt-get update
echo "gdm3 shared/default-x-display-manager select gdm3" | debconf-set-selections || true
apt-get install -y \
  ubuntu-desktop \
  gdm3 \
  openbox \
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
  python3-pyatspi \
  python3-pip \
  python3-tk \
  python3-dev \
  python-is-python3 \
  libglib2.0-bin \
  scrot \
  curl \
  wget \
  ca-certificates \
  gnupg \
  jq \
  unzip \
  zip \
  git \
  expect \
  psmisc \
  sqlite3 \
  gnome-terminal \
  thunderbird \
  gimp \
  vlc \
  libreoffice

ensure_google_chrome
ensure_vscode

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
