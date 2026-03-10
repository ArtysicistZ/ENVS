#!/bin/bash
# Container entrypoint: hand off to systemd (PID 1).
# install_resetd.sh already ran during docker build and rendered all unit files.
# systemd will start the enabled OSWorld services on boot.
#
# If the baseline home was not fully seeded during build (e.g. first boot with
# a fresh image), the DockerProvider will call POST /prepare_baseline on the
# resetd HTTP endpoint after the container's services are healthy.

set -e

# Ensure /run/lock and cgroup mounts are ready (added by --tmpfs /run --tmpfs /run/lock)
install -d -m 1777 /run/lock 2>/dev/null || true

# Load overlay kernel module before systemd starts osworld-home-overlay.service
# Check if already loaded first (avoids modprobe dependency when module is pre-loaded)
if grep -q "^overlay " /proc/modules 2>/dev/null; then
    echo "overlay module already loaded" >&2
else
    MODPROBE=$(command -v modprobe 2>/dev/null || echo /sbin/modprobe)
    if [ -x "$MODPROBE" ]; then
        if ! "$MODPROBE" overlay 2>/tmp/modprobe-overlay.err; then
            err=$(cat /tmp/modprobe-overlay.err)
            if echo "$err" | grep -qiE "already in kernel|Module already"; then
                echo "overlay module already loaded" >&2
            else
                echo "WARN: modprobe overlay failed: $err — continuing anyway" >&2
            fi
        fi
    else
        echo "WARN: modprobe not found — overlay module may not be loaded" >&2
    fi
fi

# Pre-create overlay session dirs on /tmp (Docker tmpfs) before systemd starts.
# This prevents a race where the overlay service runs before dirs exist.
install -d -m 0755 /tmp/osworld/session/home-upper /tmp/osworld/session/home-work

# Hand off to systemd as PID 1
exec /sbin/init
