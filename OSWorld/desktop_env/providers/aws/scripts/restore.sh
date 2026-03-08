#!/bin/bash
# Restore VM to clean snapshot state.
# Kills task processes, restores /home/user, dconf, packages, users.
# Run as root (via sudo). Expects /home/user_clean to exist.

# Step 0: Kill task/agent-spawned processes while preserving desktop infrastructure.
#
# Three layers of protection (a process is protected if ANY layer matches):
#
#   1. Server tree (BFS from server PID) — protects the Flask server and this
#      very restore script's ancestor chain.
#
#   2. Boot PID snapshot (.pids_snapshot) — protects every process that was
#      running when the VM first booted.  Correctly distinguishes a task's
#      `python3 scraper.py` (new PID) from the desktop's `python3` (boot PID).
#
#   3. Desktop-infrastructure comm-name allowlist — safety net for the rare case
#      where a desktop process crashes mid-task and gets restarted with a new PID
#      by the session manager.  Only contains names that OSWorld tasks NEVER
#      launch (window managers, dbus, audio daemons, etc.).  Ambiguous names
#      like python3/bash/sh are intentionally excluded.

SERVER_PID=$(pgrep -u user -f 'server/main.py' | head -1)

# --- Layer 1: server process tree (BFS via pgrep -P) ---
SAFE_PIDS=""
if [ -n "$SERVER_PID" ]; then
  SAFE_PIDS=" $SERVER_PID "
  frontier="$SERVER_PID"
  while [ -n "$frontier" ]; do
    next_frontier=""
    for p in $frontier; do
      for child in $(pgrep -P "$p" 2>/dev/null); do
        SAFE_PIDS="${SAFE_PIDS}${child} "
        next_frontier="$next_frontier $child"
      done
    done
    frontier="$next_frontier"
  done
fi

# --- Layer 2: boot PID snapshot ---
BASELINE_PIDS=""
if [ -f /home/user_clean/.pids_snapshot ]; then
  BASELINE_PIDS=" $(tr '\n' ' ' < /home/user_clean/.pids_snapshot) "
fi

# --- Layer 3: desktop-infrastructure comm-name allowlist ---
# These are process names that:
#   a) belong to the desktop session / display server / audio stack, AND
#   b) no OSWorld task ever launches.
# ps -o comm= truncates to 15 chars; names below are already ≤15.
# Covers both XFCE (OSWorld default) and GNOME (some AMI variants).
DESKTOP_ALLOWLIST=" \
  xfwm4 xfce4-panel xfce4-session xfdesktop xfconfd xfce4-power-man \
  mutter gnome-shell gnome-session-b \
  dbus-daemon dbus-launch dbus-broker \
  pulseaudio pipewire pipewire-pulse wireplumber \
  at-spi-bus-laun at-spi2-registr \
  gvfsd gvfsd-fuse gvfs-udisks2-vo gvfsd-trash gvfsd-metadata \
  xdg-desktop-por xdg-document-po xdg-permission- \
  ibus-daemon ibus-x11 ibus-extension- ibus-memconf \
  gnome-keyring-d ssh-agent gpg-agent \
  polkitd polkit-gnome-au \
  nm-applet blueman-applet blueman-tray \
  xfce4-notifyd notification-dae \
  Xorg Xwayland Xvfb x11vnc \
  gsd-a11y-settin gsd-color gsd-datetime gsd-housekeepin \
  gsd-keyboard gsd-media-keys gsd-mouse gsd-power gsd-print-notif \
  gsd-rfkill gsd-screensaver gsd-sharing gsd-smartcard gsd-sound \
  gsd-wacom gsd-xsettings \
  tracker-miner-f tracker-store \
  evolution-calen evolution-addre evolution-sourc \
  tumblerd \
"

if [ -n "$BASELINE_PIDS" ]; then
  # Primary path: PID snapshot + allowlist + server tree
  for pid in $(ps -u user -o pid= 2>/dev/null); do
    pid=$(echo "$pid" | tr -d ' ')
    # Layer 1: skip server tree
    case "$SAFE_PIDS" in *" $pid "*) continue ;; esac
    # Layer 2: skip boot PIDs
    case "$BASELINE_PIDS" in *" $pid "*) continue ;; esac
    # Layer 3: skip desktop infrastructure by comm name
    COMM=$(ps -p "$pid" -o comm= 2>/dev/null) || continue
    case "$DESKTOP_ALLOWLIST" in *" $COMM "*) continue ;; esac
    # Passed all layers — this is a task-spawned process, kill it
    kill -9 "$pid" 2>/dev/null || true
  done
else
  # Fallback: no PID snapshot — use conservative app-name pattern.
  # This only runs if snapshot creation failed entirely.
  pkill -9 -f 'google-chrome|chrome|chromium|firefox|libreoffice|soffice|vlc|gedit|mousepad|thunar|nautilus|nemo|evince|eog|gimp|inkscape|code|kate|xed|socat' 2>/dev/null || true
fi
sleep 1

# Step 1: Kill dconf daemon BEFORE restoring settings (prevents stale in-memory cache)
pkill -9 dconf 2>/dev/null || true

# Step 2: Delete everything in /home/user except server/
find /home/user -mindepth 1 -maxdepth 1 ! -name server -exec rm -rf {} +
if [ $? -ne 0 ]; then
  echo "RESTORE_FAILED: could not clean /home/user"
  exit 1
fi

# Step 3: Restore from clean snapshot
cp -a /home/user_clean/. /home/user/
if [ $? -ne 0 ]; then
  echo "RESTORE_FAILED: cp -a from snapshot failed"
  exit 1
fi

# Step 4: Fix ownership
chown -R user:user /home/user
if [ $? -ne 0 ]; then
  echo "RESTORE_FAILED: chown failed"
  exit 1
fi

# Step 5: Restore dconf/gsettings (dconf daemon was killed in Step 1,
# so these writes go directly to the database file; any future access
# will auto-start a fresh daemon that reads the clean state)
su - user -c 'dconf reset -f /' 2>/dev/null || true
if [ -f /home/user_clean/.dconf_snapshot ]; then
  su - user -c 'dconf load / < /home/user_clean/.dconf_snapshot' 2>/dev/null || true
fi
# Kill dconf again in case it was auto-started by the above commands
pkill -9 dconf 2>/dev/null || true

# Step 6: Clear clipboard
su - user -c 'xsel -bc' 2>/dev/null || true

# Step 7: Clean /tmp (preserve X11/ICE sockets)
find /tmp -mindepth 1 -maxdepth 1 ! -name '.X*' ! -name '.ICE*' \
  -exec rm -rf {} + 2>/dev/null || true

# Step 8: Clear user crontab (tasks/agents can create cron jobs)
crontab -u user -r 2>/dev/null || true

# Step 9: Remove extra packages installed by tasks
# Use timeout to prevent dpkg purge from blocking the entire restore
if [ -f /home/user_clean/.dpkg_snapshot ]; then
  rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock 2>/dev/null || true
  dpkg --configure -a 2>/dev/null || true
  dpkg --get-selections > /tmp/_dpkg_current 2>/dev/null
  EXTRA_PKGS=$(comm -23 <(awk '{print $1}' /tmp/_dpkg_current | sort) \
                        <(awk '{print $1}' /home/user_clean/.dpkg_snapshot | sort))
  if [ -n "$EXTRA_PKGS" ]; then
    echo "$EXTRA_PKGS" | timeout 60 xargs -r dpkg --purge --force-depends 2>/dev/null || true
  fi
  rm -f /tmp/_dpkg_current
fi

# Step 10: Remove extra pip packages (with timeout)
if [ -f /home/user_clean/.pip_snapshot ]; then
  pip3 freeze > /tmp/_pip_current 2>/dev/null || true
  EXTRA_PIPS=$(comm -23 <(awk -F= '{print $1}' /tmp/_pip_current | sort) \
                        <(awk -F= '{print $1}' /home/user_clean/.pip_snapshot | sort))
  if [ -n "$EXTRA_PIPS" ]; then
    echo "$EXTRA_PIPS" | timeout 30 xargs -r pip3 uninstall -y 2>/dev/null || true
  fi
  rm -f /tmp/_pip_current
fi

# Step 11: Remove extra user accounts
if [ -f /home/user_clean/.users_snapshot ]; then
  comm -23 <(awk -F: '$3 >= 1000 && $1 != "nobody" {print $1}' /etc/passwd | sort) \
           <(sort /home/user_clean/.users_snapshot) \
    | while read u; do userdel -r "$u" 2>/dev/null || true; done
fi

# Step 12: Remove extra home directories
if [ -f /home/user_clean/.home_dirs_snapshot ]; then
  comm -23 <(ls -1 /home | sort) \
           <(sort /home/user_clean/.home_dirs_snapshot) \
    | while read d; do rm -rf "/home/$d" 2>/dev/null || true; done
fi

echo RESTORE_DONE
