"""VM soft-reset logic: snapshot clean state on first boot, restore on each reset.

Handles: home directory, dconf/gsettings, dpkg packages, pip packages,
user accounts, /home directories, clipboard, /tmp, and process cleanup.
"""
import base64
import logging
import os
import requests

logger = logging.getLogger("desktopenv.providers.aws.vm_reset")

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")

SUDO_PASSWORD = "osworld-public-evaluation"

# Track which VMs already have a clean snapshot
_snapshot_ready: dict = {}  # ip -> True


def _load_script(name: str) -> str:
    path = os.path.join(_SCRIPTS_DIR, name)
    with open(path) as f:
        return f.read()


def _exec_in_vm(ip: str, cmd: str, port: int = 5000, timeout: int = 30) -> dict:
    """Execute a shell command inside the VM via its HTTP API."""
    url = f"http://{ip}:{port}/setup/execute"
    resp = requests.post(url, json={"command": cmd, "shell": True}, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"VM exec returned HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def _write_and_run_script(ip: str, script: str, port: int = 5000, timeout: int = 60) -> dict:
    """Write a bash script to the VM via base64 and execute it with sudo."""
    encoded = base64.b64encode(script.encode()).decode()
    write_cmd = f"echo {encoded} | base64 -d > /tmp/_arpo_script.sh && chmod +x /tmp/_arpo_script.sh"
    _exec_in_vm(ip, write_cmd, port, timeout=15)
    run_cmd = f"echo '{SUDO_PASSWORD}' | sudo -S bash /tmp/_arpo_script.sh"
    return _exec_in_vm(ip, run_cmd, port, timeout=timeout)


def snapshot_home(ip: str, port: int = 5000):
    """One-time: snapshot /home/user, dpkg, pip, user accounts, dconf.

    Idempotent — skips if /home/user_clean already exists on the VM.
    """
    if _snapshot_ready.get(ip):
        return

    # Check for the integrity marker file, not just directory existence.
    # If snapshot.sh was interrupted mid-copy, the directory exists but is incomplete.
    result = _exec_in_vm(ip, "test -f /home/user_clean/.snapshot_complete && echo EXISTS || echo MISSING", port)
    if "EXISTS" in result.get("output", ""):
        logger.info(f"Snapshot /home/user_clean already exists on {ip}; skipping creation.")
        _snapshot_ready[ip] = True
        return

    logger.info(f"Creating clean-state snapshot on {ip}...")
    script = _load_script("snapshot.sh")
    result = _write_and_run_script(ip, script, port, timeout=60)
    if "SNAPSHOT_DONE" in result.get("output", ""):
        logger.info(f"Clean-state snapshot created on {ip}.")
        _snapshot_ready[ip] = True
    else:
        logger.warning(f"Snapshot creation may have failed on {ip}: {result}")


def restore_home(ip: str, port: int = 5000):
    """Restore VM to clean snapshot state. Raises RuntimeError on failure."""
    script = _load_script("restore.sh")
    result = _write_and_run_script(ip, script, port, timeout=180)
    output = result.get("output", "")
    if "RESTORE_FAILED" in output:
        raise RuntimeError(f"Restore script reported failure on {ip}: {output}")
    if "RESTORE_DONE" not in output:
        raise RuntimeError(f"Home restore failed on {ip}: {result}")
    logger.info(f"Clean-state restore complete on {ip}.")


def soft_reset(ip: str, port: int = 5000):
    """Kill task processes and restore VM to clean snapshot state.

    Falls back to kill-only if no snapshot is available.
    Raises RuntimeError if restore fails (caller should fall back to full relaunch).
    """
    if _snapshot_ready.get(ip):
        restore_home(ip, port)
    else:
        # No snapshot — use conservative app-name pattern to avoid killing desktop infrastructure
        kill_cmd = (
            "pkill -9 -f 'google-chrome|chrome|chromium|firefox|libreoffice|soffice|vlc|gedit|"
            "mousepad|thunar|nautilus|nemo|evince|eog|gimp|inkscape|code|kate|xed|socat' "
            "2>/dev/null || true; sleep 1"
        )
        _exec_in_vm(ip, kill_cmd, port, timeout=15)
        logger.warning(f"No snapshot available for {ip}; kill-only soft reset (dirty state).")
