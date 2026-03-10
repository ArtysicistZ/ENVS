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
RUN apt-get update && apt-get install -y \
    openbox \
    xinit \
    xvfb \
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

# ── Google Chrome ────────────────────────────────────────────────────────────
RUN curl -fsSL https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
        -o /tmp/chrome.deb \
    && apt-get install -y /tmp/chrome.deb \
    && rm /tmp/chrome.deb \
    && rm -rf /var/lib/apt/lists/*

# ── VS Code ──────────────────────────────────────────────────────────────────
RUN curl -fsSL "https://update.code.visualstudio.com/latest/linux-deb-x64/stable" \
        -o /tmp/vscode.deb \
    && apt-get install -y /tmp/vscode.deb \
    && rm /tmp/vscode.deb \
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
