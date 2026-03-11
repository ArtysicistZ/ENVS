"""
DockerProvider — runs the full OSWorld stack directly inside a Docker container
(no QEMU). Each container runs Ubuntu 22.04 + systemd + X + OSWorld server + resetd.

Container pool: one container per slot_id.
- Container name: osworld-slot-{n}
- Docker network: osworld-net (bridge, created if absent)
- Reset: POST /reset to resetd at port 5001 (OverlayFS wipe, ~5-8s)
- Thread safety: per-slot RLock (P2)
- Timeouts: aligned with resetd's 120s hard limit (P3)

Host requirements for 64+ containers:
- fs.inotify.max_user_instances >= 4096 (each container's systemd needs ~10 instances)
- fs.inotify.max_user_watches >= 524288
Apply via: sudo sysctl -w fs.inotify.max_user_instances=8192
Persist via: echo 'fs.inotify.max_user_instances=8192' | sudo tee /etc/sysctl.d/99-osworld-containers.conf
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from typing import Optional

import requests

from desktop_env.providers.base import Provider, VMManager

logger = logging.getLogger("desktopenv.providers.docker.DockerProvider")
logger.setLevel(logging.INFO)

# --- Configuration constants ---
OSWORLD_IMAGE = os.environ.get("OSWORLD_DOCKER_IMAGE", "osworld:latest")
DOCKER_NETWORK = os.environ.get("OSWORLD_DOCKER_NETWORK", "osworld-net")

# Ports inside the container (no host port mapping)
CONTAINER_SERVER_PORT = 5000   # osworld-server (Flask)
CONTAINER_RESETD_PORT = 5001   # osworld-resetd (Flask)

# Per-container resource limits (tune via env vars for your host)
# Memory: desktop stack + Chrome + apps ≈ 3-4 GB per container.
# At 64 containers with 6 GB limit each → 384 GB max (the limit is per-container
# ceiling, not a reservation; actual RSS is typically 3-4 GB).
CONTAINER_MEM_LIMIT = os.environ.get("OSWORLD_CONTAINER_MEM_LIMIT", "6g")
CONTAINER_MEMSWAP_LIMIT = os.environ.get("OSWORLD_CONTAINER_MEMSWAP_LIMIT", "8g")
# CPU: fractional CPUs per container.  0 = no limit (Docker default).
CONTAINER_CPUS = float(os.environ.get("OSWORLD_CONTAINER_CPUS", "0"))
# /tmp tmpfs size per container.  Overlay session dirs need ~50 MB; 512 MB gives
# headroom for Chrome temp files.  Old default was 2 GB (128 GB at 64 slots).
CONTAINER_TMP_SIZE = os.environ.get("OSWORLD_CONTAINER_TMP_SIZE", "512m")

# Timeouts (P3: server-side reset is capped at 120s so client must be longer)
RESETD_RESET_TIMEOUT = 150          # HTTP timeout for POST /reset on resetd
RESETD_PREPARE_BASELINE_TIMEOUT = 60
CONTAINER_STARTUP_TIMEOUT = 240     # wait for container services to be ready
CONTAINER_STARTUP_POLL = 3          # poll interval in seconds
RESET_VERIFY_POLL = 3
RESET_VERIFY_TIMEOUT = 90           # wait for port 5000 to respond after reset

# Concurrency: max containers booting simultaneously.  Each container's systemd
# needs ~10 inotify instances, so we limit concurrency to avoid overwhelming the
# Docker daemon.  Set higher if the host has ample resources.
MAX_CONCURRENT_CREATES = int(os.environ.get("OSWORLD_MAX_CONCURRENT_CREATES", "8"))

# Minimum required inotify instances per container (systemd + GNOME + Chrome + services).
# Each container's systemd alone needs ~10 instances, but GNOME session components
# (gnome-settings-daemon, nautilus, etc.) and Chrome add 40-90 more.
_INOTIFY_PER_CONTAINER = 150
_MIN_INOTIFY_INSTANCES = 4096  # absolute minimum for any multi-container setup


def _slot_id_from_path(path_to_vm: str) -> int:
    """Extract integer slot_id from a path_to_vm string like 'slot_3'."""
    m = re.search(r"(\d+)$", path_to_vm)
    if m:
        return int(m.group(1))
    return 0


class DockerVMManager(VMManager):
    """Minimal manager: returns slot-scoped path_to_vm identifiers."""

    _slot_counter = 0
    _slot_lock = threading.Lock()

    def initialize_registry(self, **kwargs):
        pass

    def add_vm(self, vm_path, **kwargs):
        pass

    def delete_vm(self, vm_path, **kwargs):
        pass

    def occupy_vm(self, vm_path, pid, **kwargs):
        pass

    def list_free_vms(self, **kwargs):
        return []

    def check_and_clean(self, **kwargs):
        pass

    def get_vm_path(self, **kwargs) -> str:
        with self._slot_lock:
            slot = DockerVMManager._slot_counter
            DockerVMManager._slot_counter += 1
        return f"slot_{slot}"


def _check_host_inotify_limits(n_containers: int = 64) -> None:
    """Check and optionally fix inotify limits on the host.

    Each container's systemd needs ~10-15 inotify instances.  The default
    Linux limit (128) is too low for more than ~8 containers, causing
    systemd to crash with exit code 255 ('failed to create inotify fd').
    """
    try:
        with open("/proc/sys/fs/inotify/max_user_instances") as f:
            current = int(f.read().strip())
    except (OSError, ValueError):
        logger.warning("Cannot read fs.inotify.max_user_instances; skipping check")
        return

    needed = max(n_containers * _INOTIFY_PER_CONTAINER, _MIN_INOTIFY_INSTANCES)
    if current >= needed:
        logger.info(
            "inotify max_user_instances=%d (need %d for %d containers) — OK",
            current, needed, n_containers,
        )
        return

    logger.warning(
        "inotify max_user_instances=%d is TOO LOW for %d containers (need %d). "
        "Containers will crash with exit 255. Attempting to increase...",
        current, n_containers, needed,
    )
    target = max(needed, 65536)
    try:
        subprocess.run(
            ["sudo", "-n", "sysctl", "-w", f"fs.inotify.max_user_instances={target}"],
            check=True, capture_output=True, timeout=5,
        )
        logger.info("Increased fs.inotify.max_user_instances to %d", target)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        logger.error(
            "CANNOT auto-fix inotify limit. Fix manually:\n"
            "  sudo sysctl -w fs.inotify.max_user_instances=%d\n"
            "  echo 'fs.inotify.max_user_instances=%d' | sudo tee /etc/sysctl.d/99-osworld-containers.conf",
            target, target,
        )

    # Also check max_user_watches (each container watches hundreds of files)
    try:
        with open("/proc/sys/fs/inotify/max_user_watches") as f:
            watches = int(f.read().strip())
        watches_needed = n_containers * 8192  # ~8K watches per container (systemd + GNOME + apps)
        if watches < watches_needed:
            logger.warning(
                "inotify max_user_watches=%d is low for %d containers (need %d). Attempting to increase...",
                watches, n_containers, watches_needed,
            )
            watches_target = max(watches_needed, 1048576)
            try:
                subprocess.run(
                    ["sudo", "-n", "sysctl", "-w", f"fs.inotify.max_user_watches={watches_target}"],
                    check=True, capture_output=True, timeout=5,
                )
                logger.info("Increased fs.inotify.max_user_watches to %d", watches_target)
            except Exception:
                logger.error(
                    "CANNOT auto-fix max_user_watches. Fix manually:\n"
                    "  sudo sysctl -w fs.inotify.max_user_watches=%d", watches_target,
                )
    except (OSError, ValueError):
        pass


class DockerProvider(Provider):
    """Docker provider that runs the OSWorld GUI stack directly inside containers."""

    # Semaphore: limits concurrent container creation to avoid overwhelming
    # the Docker daemon.  Unlike the old Lock() which serialized ALL creation,
    # this allows up to MAX_CONCURRENT_CREATES containers to boot simultaneously.
    _container_create_sem = threading.Semaphore(MAX_CONCURRENT_CREATES)

    def __init__(self, region: str = None):
        super().__init__(region=region)
        self._import_docker()
        self._ensure_network()
        _check_host_inotify_limits()
        # Per-slot locks for thread safety (P2)
        self._slot_locks: dict[int, threading.RLock] = {}
        self._slot_locks_meta = threading.Lock()
        # Cache of container IPs: slot_id -> ip
        self._slot_ips: dict[int, str] = {}
        self._slot_ips_lock = threading.RLock()
        # Track which slots have had prepare_baseline called (idempotency for BUG #6)
        self._baseline_prepared: set[int] = set()
        self._baseline_prepared_lock = threading.Lock()

    def _import_docker(self):
        try:
            import docker as _docker
            self._docker = _docker
            self.client = _docker.from_env()
        except ImportError as e:
            raise ImportError(
                "DockerProvider requires the 'docker' Python package. "
                "Install it: pip install docker"
            ) from e

    def _get_slot_lock(self, slot_id: int) -> threading.RLock:
        with self._slot_locks_meta:
            if slot_id not in self._slot_locks:
                self._slot_locks[slot_id] = threading.RLock()
            return self._slot_locks[slot_id]

    def _ensure_network(self):
        """Create the Docker bridge network if it doesn't exist."""
        try:
            import docker as _docker
            client = _docker.from_env()
            try:
                client.networks.get(DOCKER_NETWORK)
            except _docker.errors.NotFound:
                client.networks.create(
                    DOCKER_NETWORK,
                    driver="bridge",
                    options={"com.docker.network.bridge.name": "br-osworld"},
                )
                logger.info("Created Docker network: %s", DOCKER_NETWORK)
        except Exception as e:
            logger.warning("Could not ensure Docker network %s: %s", DOCKER_NETWORK, e)

    def _container_name(self, slot_id: int) -> str:
        return f"osworld-slot-{slot_id}"

    # Maximum time (seconds) to wait inside the creation lock for a newly
    # created container's resetd to become healthy.  At 64 containers the
    # CPU load can exceed 1000 during mass boot; 300s gives enough headroom.
    # If this times out, the container is NOT removed — start_emulator has
    # its own wait_for_port that gives an additional chance.
    _CREATION_HEALTH_TIMEOUT = 300

    def _get_or_start_container(self, slot_id: int):
        """Return the container for this slot, starting it if it doesn't exist.

        Container creation is throttled by a semaphore (MAX_CONCURRENT_CREATES)
        to avoid overwhelming the Docker daemon.  Multiple containers can boot
        in parallel — inotify limits (not serialization) prevent exit-255 crashes.

        If an existing container is in a bad state (exited/dead/restarting),
        it is removed and recreated.
        """
        name = self._container_name(slot_id)

        # Fast path: container already exists and is running
        try:
            container = self.client.containers.get(name)
            container.reload()
            status = container.status
            if status == "running":
                return container
            if status in ("exited", "dead", "restarting"):
                logger.warning(
                    "[slot %d] Container %s in bad state '%s' (restart_count=%s); "
                    "will remove and recreate.",
                    slot_id, name, status,
                    container.attrs.get("RestartCount", "?"),
                )
                try:
                    container.stop(timeout=5)
                except Exception:
                    pass
                container.remove(force=True)
                # Fall through to create a new container
            else:
                # created/paused — try starting
                container.start()
                return container
        except self._docker.errors.NotFound:
            pass

        # ── Semaphore-throttled creation ──────────────────────────────
        run_kwargs = dict(
            image=OSWORLD_IMAGE,
            name=name,
            detach=True,
            privileged=True,
            cgroupns="host",
            network=DOCKER_NETWORK,
            tmpfs={
                "/run": "rw,nosuid,nodev,size=64m",
                "/run/lock": "rw,nosuid,nodev,size=16m",
                "/tmp": f"rw,nosuid,nodev,size={CONTAINER_TMP_SIZE}",
                "/var/log": "rw,nosuid,nodev,size=64m",
                "/var/tmp": "rw,nosuid,nodev,size=64m",
                "/var/cache": "rw,nosuid,nodev,size=128m",
            },
            volumes={
                "/sys/fs/cgroup": {"bind": "/sys/fs/cgroup", "mode": "rw"},
            },
            environment={
                "OSWORLD_SLOT_ID": str(slot_id),
            },
            shm_size="256m",
            restart_policy={"Name": "unless-stopped"},
        )
        if CONTAINER_MEM_LIMIT:
            run_kwargs["mem_limit"] = CONTAINER_MEM_LIMIT
        if CONTAINER_MEMSWAP_LIMIT:
            run_kwargs["memswap_limit"] = CONTAINER_MEMSWAP_LIMIT
        if CONTAINER_CPUS > 0:
            run_kwargs["nano_cpus"] = int(CONTAINER_CPUS * 1e9)

        with DockerProvider._container_create_sem:
            # Double-check: another thread may have created this container
            # while we waited for the semaphore.
            try:
                container = self.client.containers.get(name)
                container.reload()
                if container.status == "running":
                    logger.info("[slot %d] Container %s already created by another thread.", slot_id, name)
                    return container
                # Bad state — remove it
                try:
                    container.stop(timeout=5)
                except Exception:
                    pass
                container.remove(force=True)
            except self._docker.errors.NotFound:
                pass

            logger.info("[slot %d] Creating container %s ...", slot_id, name)
            container = self.client.containers.run(**run_kwargs)

            # Phase 1: wait for container status == "running"
            for _check in range(30):  # up to 30s
                time.sleep(1)
                container.reload()
                if container.status == "running":
                    break
            if container.status != "running":
                logger.error(
                    "[slot %d] Container %s failed to start (status=%s). Removing.",
                    slot_id, name, container.status,
                )
                try:
                    container.stop(timeout=5)
                except Exception:
                    pass
                container.remove(force=True)
                raise RuntimeError(
                    f"[slot {slot_id}] Container failed to start (status={container.status})"
                )

            # Phase 2: wait for resetd health check (confirms systemd + all
            # services are fully booted) before releasing the semaphore.
            ip = self._resolve_container_ip(slot_id)
            with self._slot_ips_lock:
                self._slot_ips[slot_id] = ip

            deadline = time.time() + self._CREATION_HEALTH_TIMEOUT
            healthy = False
            while time.time() < deadline:
                # Verify container is still running
                container.reload()
                if container.status != "running":
                    exit_code = container.attrs.get("State", {}).get("ExitCode", "?")
                    restart_count = container.attrs.get("RestartCount", "?")
                    msg = (
                        f"[slot {slot_id}] Container died during boot "
                        f"(status={container.status}, exit={exit_code}, restarts={restart_count})"
                    )
                    if exit_code == 255:
                        msg += (
                            ". Exit 255 = systemd failed to initialize "
                            "(likely inotify limit exhausted). "
                            "Fix: sudo sysctl -w fs.inotify.max_user_instances=8192"
                        )
                    logger.error(msg)
                    try:
                        container.stop(timeout=5)
                    except Exception:
                        pass
                    container.remove(force=True)
                    self._invalidate_cached_ip(slot_id)
                    raise RuntimeError(msg)
                # Check resetd health
                try:
                    r = requests.get(
                        f"http://{ip}:{CONTAINER_RESETD_PORT}/health",
                        timeout=3,
                    )
                    if r.status_code == 200:
                        healthy = True
                        break
                except Exception:
                    pass
                time.sleep(2)

            if not healthy:
                # Do NOT remove the container — it may still be booting under
                # heavy CPU load (64-container mass boot).  start_emulator has
                # its own _wait_for_port call that gives another chance.
                logger.warning(
                    "[slot %d] Container %s resetd not healthy within %ds during creation; "
                    "leaving container running for start_emulator retry.",
                    slot_id, name, self._CREATION_HEALTH_TIMEOUT,
                )

            logger.info(
                "[slot %d] Container %s healthy (id=%s, ip=%s).",
                slot_id, name, container.short_id, ip,
            )

        return container

    def _invalidate_cached_ip(self, slot_id: int):
        """Clear cached IP for a slot so the next lookup re-queries Docker."""
        with self._slot_ips_lock:
            old = self._slot_ips.pop(slot_id, None)
            if old:
                logger.info("[slot %d] Invalidated cached IP %s", slot_id, old)

    def _resolve_container_ip(self, slot_id: int) -> str:
        """Query Docker for the container's current IP (no cache)."""
        name = self._container_name(slot_id)
        try:
            container = self.client.containers.get(name)
        except self._docker.errors.NotFound:
            raise RuntimeError(f"Container {name} not found during IP resolution")
        ip = None
        for attempt in range(5):
            container.reload()
            nets = container.attrs.get("NetworkSettings", {}).get("Networks", {})
            if DOCKER_NETWORK in nets:
                ip = nets[DOCKER_NETWORK].get("IPAddress")
            if not ip:
                for net_info in nets.values():
                    ip = net_info.get("IPAddress")
                    if ip:
                        break
            if ip:
                break
            logger.debug("[slot %d] IP not yet assigned, retrying (%d/5)...", slot_id, attempt + 1)
            time.sleep(1)
        if not ip:
            raise RuntimeError(f"Could not determine IP for container {name}")
        return ip

    def _get_container_ip(self, slot_id: int) -> str:
        """Return cached container IP, resolving from Docker on first call."""
        with self._slot_ips_lock:
            if slot_id in self._slot_ips:
                return self._slot_ips[slot_id]
            ip = self._resolve_container_ip(slot_id)
            self._slot_ips[slot_id] = ip
            logger.info("[slot %d] Container IP: %s", slot_id, ip)
            return ip

    def _wait_for_port(self, ip: str, port: int, timeout: float, poll: float = 2.0,
                       slot_id: int | None = None) -> bool:
        """Wait until GET /health responds 200 on ip:port.

        If slot_id is provided, periodically checks whether the container is in a
        crash loop (exited/restarting).  If detected, removes and recreates it,
        refreshes the IP, and continues waiting.
        """
        deadline = time.time() + timeout
        recreate_count = 0
        max_recreates = 3
        last_crash_check = 0.0
        while time.time() < deadline:
            try:
                r = requests.get(f"http://{ip}:{port}/health", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass

            # Check for crash-loop every 15s (avoid hammering Docker API)
            now = time.time()
            if slot_id is not None and (now - last_crash_check) > 15 and recreate_count < max_recreates:
                last_crash_check = now
                try:
                    name = self._container_name(slot_id)
                    container = self.client.containers.get(name)
                    container.reload()
                    status = container.status
                    restart_count = container.attrs.get("RestartCount", 0)
                    if status in ("exited", "dead") or (status == "restarting" and restart_count >= 2):
                        recreate_count += 1
                        logger.warning(
                            "[slot %d] Container crash-loop detected (status=%s, restarts=%s); "
                            "recreating (attempt %d/%d)",
                            slot_id, status, restart_count, recreate_count, max_recreates,
                        )
                        try:
                            container.stop(timeout=5)
                        except Exception:
                            pass
                        container.remove(force=True)
                        self._invalidate_cached_ip(slot_id)
                        time.sleep(2)  # brief pause before recreating
                        container = self._get_or_start_container(slot_id)
                        ip = self._resolve_container_ip(slot_id)
                        with self._slot_ips_lock:
                            self._slot_ips[slot_id] = ip
                        logger.info("[slot %d] Recreated container, new IP: %s", slot_id, ip)
                except Exception as e:
                    logger.debug("[slot %d] Crash-loop check error: %s", slot_id, e)

            time.sleep(poll)
        return False

    def _call_resetd(self, ip: str, endpoint: str, timeout: float = RESETD_RESET_TIMEOUT) -> dict:
        r = requests.post(
            f"http://{ip}:{CONTAINER_RESETD_PORT}{endpoint}",
            json={},
            timeout=timeout,
        )
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            logger.warning("resetd %s returned non-JSON: %s", endpoint, r.text[:200])
            return {"status": "ok"}

    # ── Provider interface ────────────────────────────────────────────────────

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu", **kwargs):
        slot_id = _slot_id_from_path(path_to_vm)
        lock = self._get_slot_lock(slot_id)
        with lock:
            container = self._get_or_start_container(slot_id)
            ip = self._get_container_ip(slot_id)

            logger.info("[slot %d] Waiting for osworld-resetd (port %d) ...", slot_id, CONTAINER_RESETD_PORT)
            if not self._wait_for_port(ip, CONTAINER_RESETD_PORT, timeout=CONTAINER_STARTUP_TIMEOUT,
                                       poll=CONTAINER_STARTUP_POLL, slot_id=slot_id):
                raise RuntimeError(
                    f"[slot {slot_id}] resetd at {ip}:{CONTAINER_RESETD_PORT} did not come up within "
                    f"{CONTAINER_STARTUP_TIMEOUT}s"
                )
            # Refresh IP in case the container was recreated during the wait
            ip = self._get_container_ip(slot_id)
            logger.info("[slot %d] Waiting for osworld-server (port %d) ...", slot_id, CONTAINER_SERVER_PORT)
            if not self._wait_for_port(ip, CONTAINER_SERVER_PORT, timeout=CONTAINER_STARTUP_TIMEOUT,
                                       poll=CONTAINER_STARTUP_POLL, slot_id=slot_id):
                raise RuntimeError(
                    f"[slot {slot_id}] osworld-server at {ip}:{CONTAINER_SERVER_PORT} did not come up within "
                    f"{CONTAINER_STARTUP_TIMEOUT}s"
                )

            # Call prepare_baseline only on the first start for this slot (idempotent guard)
            with self._baseline_prepared_lock:
                need_baseline = slot_id not in self._baseline_prepared
            if need_baseline:
                try:
                    result = self._call_resetd(ip, "/prepare_baseline", timeout=RESETD_PREPARE_BASELINE_TIMEOUT)
                    logger.info("[slot %d] prepare_baseline: %s", slot_id, result.get("status"))
                    with self._baseline_prepared_lock:
                        self._baseline_prepared.add(slot_id)
                except Exception as e:
                    logger.warning("[slot %d] prepare_baseline failed (non-fatal): %s", slot_id, e)
            else:
                logger.info("[slot %d] prepare_baseline already done, skipping.", slot_id)

        logger.info("[slot %d] Container %s ready at %s", slot_id, self._container_name(slot_id), ip)

    def get_ip_address(self, path_to_vm: str) -> str:
        """Return 'ip:server_port:chromium_port:vnc_port:vlc_port' for DesktopEnv compatibility."""
        slot_id = _slot_id_from_path(path_to_vm)
        ip = self._get_container_ip(slot_id)
        # Chrome DevTools at 9222 (socat forwards 0.0.0.0:9222 → localhost:1337).
        # VNC (8006) is not used in Docker mode; 0 tells DesktopEnv to skip VNC.
        return f"{ip}:{CONTAINER_SERVER_PORT}:9222:0:8080"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        """No-op: the overlay FS serves as the snapshot mechanism."""
        pass

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> str:
        """Reset via overlay FS wipe through resetd HTTP endpoint.

        On connection failure the cached IP is invalidated and one retry is
        attempted with a fresh IP lookup (handles container restarts that change
        the bridge IP).
        """
        slot_id = _slot_id_from_path(path_to_vm)
        lock = self._get_slot_lock(slot_id)
        with lock:
            ip = self._get_container_ip(slot_id)

            logger.info("[slot %d] Requesting reset via resetd at %s:%d ...", slot_id, ip, CONTAINER_RESETD_PORT)
            try:
                result = self._call_resetd(ip, "/reset", timeout=RESETD_RESET_TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as e:
                # Container may have restarted with a new IP.  Invalidate cache
                # and retry once with a fresh IP.
                logger.warning("[slot %d] resetd connection failed (%s), refreshing IP ...", slot_id, e)
                self._invalidate_cached_ip(slot_id)
                ip = self._get_container_ip(slot_id)
                try:
                    result = self._call_resetd(ip, "/reset", timeout=RESETD_RESET_TIMEOUT)
                except Exception as e2:
                    logger.error("[slot %d] resetd /reset failed after IP refresh: %s", slot_id, e2)
                    raise RuntimeError(f"[slot {slot_id}] overlay reset failed: {e2}") from e2
            except Exception as e:
                logger.error("[slot %d] resetd /reset failed: %s", slot_id, e)
                raise RuntimeError(f"[slot {slot_id}] overlay reset failed: {e}") from e

            status = result.get("status", "unknown")
            logger.info("[slot %d] resetd /reset response: status=%s", slot_id, status)
            if status not in ("ok", "busy"):
                logger.warning("[slot %d] Unexpected reset status: %s", slot_id, result)

            # Wait for osworld-server (port 5000) to be healthy after reset
            logger.info("[slot %d] Waiting for osworld-server to recover (up to %ds) ...", slot_id, RESET_VERIFY_TIMEOUT)
            if not self._wait_for_port(ip, CONTAINER_SERVER_PORT, timeout=RESET_VERIFY_TIMEOUT, poll=RESET_VERIFY_POLL):
                raise RuntimeError(
                    f"[slot {slot_id}] osworld-server did not recover within {RESET_VERIFY_TIMEOUT}s after reset"
                )
            logger.info("[slot %d] Reset complete. osworld-server is up.", slot_id)

        return path_to_vm

    # ── Health check & recovery ─────────────────────────────────────────────

    def health_check_slot(self, slot_id: int) -> dict:
        """Check health of a specific container slot.

        Returns a dict with keys:
          slot_id, container, status, docker_status, ip, resetd, server, ts
        Status is one of: healthy, degraded, down, missing, error.
        """
        name = self._container_name(slot_id)
        result = {"slot_id": slot_id, "container": name, "status": "unknown", "ts": time.time()}

        try:
            container = self.client.containers.get(name)
            container.reload()
            result["docker_status"] = container.status
            result["restart_count"] = container.attrs.get("RestartCount", 0)

            if container.status != "running":
                result["status"] = "down"
                result["exit_code"] = container.attrs.get("State", {}).get("ExitCode")
                return result

            # Resolve IP
            try:
                ip = self._get_container_ip(slot_id)
                result["ip"] = ip
            except Exception as e:
                result["status"] = "degraded"
                result["ip_error"] = str(e)
                return result

            # Check resetd (port 5001)
            try:
                r = requests.get(f"http://{ip}:{CONTAINER_RESETD_PORT}/health", timeout=5)
                result["resetd"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
            except Exception:
                result["resetd"] = "unreachable"
                result["status"] = "degraded"
                return result

            # Check osworld-server (port 5000)
            try:
                r = requests.get(f"http://{ip}:{CONTAINER_SERVER_PORT}/health", timeout=5)
                result["server"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
            except Exception:
                result["server"] = "unreachable"
                result["status"] = "degraded"
                return result

            result["status"] = "healthy"
            return result

        except self._docker.errors.NotFound:
            result["status"] = "missing"
            return result
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            return result

    def recover_slot(self, slot_id: int) -> bool:
        """Remove a dead/unhealthy container and recreate it.

        Waits for both resetd and osworld-server to become healthy, then
        re-runs prepare_baseline.  Returns True on success.
        """
        name = self._container_name(slot_id)
        lock = self._get_slot_lock(slot_id)
        with lock:
            logger.info("[slot %d] Starting recovery of %s ...", slot_id, name)

            # 1. Remove dead container
            try:
                container = self.client.containers.get(name)
                try:
                    container.stop(timeout=5)
                except Exception:
                    pass
                container.remove(force=True)
                logger.info("[slot %d] Removed dead container %s", slot_id, name)
            except self._docker.errors.NotFound:
                pass
            except Exception as e:
                logger.warning("[slot %d] Error removing container: %s", slot_id, e)

            # 2. Clear cached state
            self._invalidate_cached_ip(slot_id)
            with self._baseline_prepared_lock:
                self._baseline_prepared.discard(slot_id)

            # 3. Recreate container
            try:
                container = self._get_or_start_container(slot_id)
                ip = self._get_container_ip(slot_id)

                # 4. Wait for services
                if not self._wait_for_port(ip, CONTAINER_RESETD_PORT,
                                           timeout=CONTAINER_STARTUP_TIMEOUT,
                                           poll=CONTAINER_STARTUP_POLL, slot_id=slot_id):
                    logger.error("[slot %d] Recovery: resetd did not come up", slot_id)
                    return False

                if not self._wait_for_port(ip, CONTAINER_SERVER_PORT,
                                           timeout=CONTAINER_STARTUP_TIMEOUT,
                                           poll=CONTAINER_STARTUP_POLL, slot_id=slot_id):
                    logger.error("[slot %d] Recovery: osworld-server did not come up", slot_id)
                    return False

                # 5. Prepare baseline
                try:
                    result = self._call_resetd(ip, "/prepare_baseline",
                                               timeout=RESETD_PREPARE_BASELINE_TIMEOUT)
                    logger.info("[slot %d] Recovery: prepare_baseline: %s", slot_id, result.get("status"))
                    with self._baseline_prepared_lock:
                        self._baseline_prepared.add(slot_id)
                except Exception as e:
                    logger.warning("[slot %d] Recovery: prepare_baseline failed (non-fatal): %s", slot_id, e)

                logger.info("[slot %d] Recovery complete — container healthy at %s", slot_id, ip)
                return True

            except Exception as e:
                logger.error("[slot %d] Recovery failed: %s", slot_id, e)
                return False

    def stop_emulator(self, path_to_vm: str):
        slot_id = _slot_id_from_path(path_to_vm)
        with self._slot_ips_lock:
            self._slot_ips.pop(slot_id, None)
        name = self._container_name(slot_id)
        try:
            container = self.client.containers.get(name)
            container.stop(timeout=10)
            container.remove()
            logger.info("[slot %d] Container %s stopped and removed.", slot_id, name)
        except self._docker.errors.NotFound:
            pass
        except Exception as e:
            logger.warning("[slot %d] Could not stop/remove %s: %s", slot_id, name, e)
