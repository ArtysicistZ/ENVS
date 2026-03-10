# OSWorld Docker container — Ubuntu 22.04 with systemd as PID 1
# Build: docker build -t osworld:latest -f docker/Dockerfile .
# Run:   docker run --privileged --cgroupns host \
#          --tmpfs /run --tmpfs /run/lock --tmpfs /tmp \
#          -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
#          osworld:latest

FROM ubuntu:22.04

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# ── Base system + systemd ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    systemd \
    systemd-sysv \
    dbus \
    && rm -rf /var/lib/apt/lists/*

# Remove systemd units that don't make sense in a container
RUN find /etc/systemd/system \
         /lib/systemd/system \
         /usr/lib/systemd/system \
    \( -name "systemd-udevd*" \
    -o -name "systemd-udevd-kernel.socket" \
    -o -name "proc-sys-fs-binfmt_misc.automount" \
    -o -name "sys-fs-fuse-connections.mount" \
    -o -name "dev-hugepages.mount" \
    -o -name "dev-mqueue.mount" \) \
    -exec rm -f {} + 2>/dev/null || true

# ── Desktop stack + OSWorld runtime dependencies ────────────────────────────
# GNOME Flashback provides the real GNOME desktop (panels, nautilus, settings)
# without requiring logind sessions (which GNOME Shell needs but containers
# cannot provide).  This is the standard approach for headless GNOME in VNC/
# container environments.  AWS VMs use full GNOME Shell via GDM login.
RUN apt-get update && apt-get install -y \
    gnome-session-flashback \
    gnome-settings-daemon \
    nautilus \
    ubuntu-wallpapers \
    yaru-theme-gtk \
    yaru-theme-icon \
    xauth \
    x11-xserver-utils \
    xserver-xorg-video-dummy \
    dbus-x11 \
    dbus-user-session \
    at-spi2-core \
    gnome-terminal \
    gnome-screenshot \
    wmctrl \
    ffmpeg \
    socat \
    xclip \
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
    kmod \
    rsync \
    sqlite3 \
    cron \
    pulseaudio \
    pulseaudio-utils \
    sudo \
    thunderbird \
    gimp \
    vlc \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# ── Bypass GNOME GL acceleration check ──────────────────────────────────────
# gnome-session-check-accelerated returns code 512 with software rendering,
# which causes gnome-session to abort.  Replace it with a no-op so GNOME
# Flashback (which uses Metacity, not Mutter) can start.
RUN ACCEL_BIN="$(command -v gnome-session-check-accelerated 2>/dev/null \
        || echo /usr/libexec/gnome-session-check-accelerated)" \
    && if [ -f "$ACCEL_BIN" ]; then \
        printf '#!/bin/sh\nexit 0\n' > "$ACCEL_BIN" \
        && chmod +x "$ACCEL_BIN"; \
    fi

# ── Desktop defaults (wallpaper, theme, panel layout) ───────────────────────
RUN printf '\
[org.gnome.desktop.background]\n\
picture-uri="file:///usr/share/backgrounds/warty-final-ubuntu.png"\n\
picture-options="zoom"\n\
\n\
[org.gnome.desktop.interface:GNOME-Flashback]\n\
gtk-theme="Yaru"\n\
icon-theme="Yaru"\n\
cursor-theme="Yaru"\n\
\n\
[org.gnome.gnome-panel.general:GNOME-Flashback]\n\
default-layout="gnome-flashback"\n\
\n\
[org.gnome.desktop.screensaver]\n\
lock-enabled=false\n\
idle-activation-enabled=false\n\
\n\
[org.gnome.desktop.session]\n\
idle-delay=uint32 0\n\
\n\
[org.gnome.settings-daemon.plugins.power]\n\
sleep-inactive-ac-type="nothing"\n\
sleep-inactive-battery-type="nothing"\n\
idle-dim=false\n\
' > /usr/share/glib-2.0/schemas/90_osworld.gschema.override \
    && glib-compile-schemas /usr/share/glib-2.0/schemas/

# ── Disable gnome-keyring D-Bus activation ──────────────────────────────────
# Without this, Chrome triggers gnome-keyring-daemon via D-Bus on first launch,
# which pops up a "Choose password for new keyring" dialog.  Removing the
# D-Bus service files makes Chrome fall back to basic password storage.
RUN rm -f /usr/share/dbus-1/services/org.gnome.keyring.service \
          /usr/share/dbus-1/services/org.freedesktop.secrets.service

# ── Google Chrome + VS Code ─────────────────────────────────────────────────
RUN curl -fsSL https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
        -o /tmp/chrome.deb \
    && curl -fsSL "https://update.code.visualstudio.com/latest/linux-deb-x64/stable" \
        -o /tmp/vscode.deb \
    && apt-get install -y /tmp/chrome.deb /tmp/vscode.deb \
    && rm -f /tmp/chrome.deb /tmp/vscode.deb \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages for OSWorld server and reset runtime ────────────────────
RUN pip3 install --no-cache-dir \
    flask \
    waitress \
    fastapi \
    uvicorn \
    pillow \
    requests \
    psutil \
    numpy

# ── Desktop user ─────────────────────────────────────────────────────────────
RUN useradd -m -s /bin/bash user \
    && echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ── Xorg dummy display configuration ─────────────────────────────────────────
RUN install -d -m 0755 /etc/X11/xorg.conf.d
COPY docker/xorg-dummy.conf /etc/X11/xorg.conf

# ── OSWorld control-plane code ───────────────────────────────────────────────
# Copy the full OSWorld directory to the expected location
COPY OSWorld /opt/osworld/app/OSWorld

# ── Run install_resetd.sh in build mode ──────────────────────────────────────
# This creates dirs, renders systemd unit files, and seeds baseline home.
# Services are NOT started; systemd will start them when the container boots.
RUN OSWORLD_SKIP_X_SOCKET_CHECK=1 \
    OSWORLD_SKIP_RUNTIME_SETUP=1 \
    OSWORLD_RESET_USER=user \
    OSWORLD_RESET_HOME=/home/user \
    OSWORLD_CONTROL_PLANE_ROOT=/opt/osworld \
    OSWORLD_RESET_BASELINE_MODE=minimal \
    OSWORLD_RESET_SESSION_ROOT=/tmp/osworld/session \
    PYTHON_BIN=/usr/bin/python3 \
    bash /opt/osworld/app/OSWorld/desktop_env/providers/aws/scripts/install_resetd.sh

# Mask tmp.mount — Docker already provides /tmp as tmpfs via --tmpfs /tmp.
# Without this, systemd remounts /tmp after the overlay service creates its
# session dirs, causing the overlay upper/work dirs to disappear.
RUN ln -sf /dev/null /etc/systemd/system/tmp.mount

# Override tmpfiles.d/tmp.conf: change D (wipe contents) to d (create only).
# systemd-tmpfiles-setup.service runs the D directive at boot and clears /tmp,
# destroying overlay session dirs created by container-init.sh.
RUN printf 'd /tmp 1777 root root -\n' > /etc/tmpfiles.d/tmp.conf

# Enable OSWorld services (create multi-user.target.wants symlinks without systemd running)
RUN install -d -m 0755 /etc/systemd/system/multi-user.target.wants && \
    for svc in osworld-home-overlay osworld-resetd osworld-graphical-session osworld-server; do \
        ln -sf "/etc/systemd/system/${svc}.service" \
               "/etc/systemd/system/multi-user.target.wants/${svc}.service"; \
    done

# ── Container init entrypoint ─────────────────────────────────────────────────
COPY docker/container-init.sh /usr/local/bin/container-init.sh
RUN chmod +x /usr/local/bin/container-init.sh

# Expose OSWorld server and reset daemon ports
EXPOSE 5000 5001

# systemd as PID 1; container-init runs first to finalize any runtime setup
CMD ["/usr/local/bin/container-init.sh"]
