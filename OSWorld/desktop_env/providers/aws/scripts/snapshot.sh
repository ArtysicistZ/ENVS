#!/bin/bash
# One-time snapshot of clean VM state.
# Creates /home/user_clean with baseline home dir, dconf, dpkg, pip, users, /home listing.
# Run as root (via sudo).
set -e

cp -a /home/user /home/user_clean
rm -rf /home/user_clean/server

# Snapshot dconf (gsettings)
su - user -c 'dconf dump /' > /home/user_clean/.dconf_snapshot 2>/dev/null || true

# Snapshot dpkg selections
dpkg --get-selections > /home/user_clean/.dpkg_snapshot 2>/dev/null || true

# Snapshot pip packages
pip3 freeze > /home/user_clean/.pip_snapshot 2>/dev/null || true

# Snapshot user accounts (UIDs >= 1000, excluding nobody)
awk -F: '$3 >= 1000 && $1 != "nobody" {print $1}' /etc/passwd > /home/user_clean/.users_snapshot 2>/dev/null || true

# Snapshot /home directory listing
ls -1 /home > /home/user_clean/.home_dirs_snapshot 2>/dev/null || true

# Snapshot baseline PIDs (desktop session infrastructure).
# During restore, only processes whose PID is NOT in this set (and not in the
# server tree) will be killed. This preserves dbus-daemon, window manager, panel,
# session manager, pulseaudio, etc. — even if they share comm names like 'bash'
# or 'python3' with task-launched processes.
ps -u user -o pid= | tr -d ' ' | sort -n > /home/user_clean/.pids_snapshot 2>/dev/null || true

# Write integrity marker — must be the LAST step.
# vm_reset.py checks for this file (not just directory existence) to verify
# the snapshot completed successfully and wasn't interrupted mid-copy.
touch /home/user_clean/.snapshot_complete

echo SNAPSHOT_DONE
