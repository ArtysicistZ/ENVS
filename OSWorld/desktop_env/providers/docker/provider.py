"""
DockerProvider — runs the full OSWorld stack directly inside a Docker container
(no QEMU). Each container runs Ubuntu 22.04 + systemd + X + OSWorld server + resetd.

Container pool: one container per slot_id.
- Container name: osworld-slot-{n}
- Docker network: osworld-net (bridge, created if absent)
- Reset: POST /reset to resetd at port 5001 (OverlayFS wipe, ~5-8s)
- Thread safety: per-slot RLock (P2)
- Timeouts: aligned with resetd's 120s hard limit (P3)
"""
from __future__ import annotations

import logging
import os
import re
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

# Timeouts (P3: server-side reset is capped at 120s so client must be longer)
RESETD_RESET_TIMEOUT = 150          # HTTP timeout for POST /reset on resetd
RESETD_PREPARE_BASELINE_TIMEOUT = 60
CONTAINER_STARTUP_TIMEOUT = 180     # wait for container services to be ready
CONTAINER_STARTUP_POLL = 3          # poll interval in seconds
RESET_VERIFY_POLL = 3
RESET_VERIFY_TIMEOUT = 90           # wait for port 5000 to respond after reset


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


class DockerProvider(Provider):
    """Docker provider that runs the OSWorld GUI stack directly inside containers."""

    def __init__(self, region: str = None):
        super().__init__(region=region)
        self._import_docker()
        self._ensure_network()
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

    def _get_or_start_container(self, slot_id: int):
        """Return the container for this slot, starting it if it doesn't exist."""
        name = self._container_name(slot_id)
        try:
            container = self.client.containers.get(name)
            if container.status != "running":
                container.start()
            return container
        except self._docker.errors.NotFound:
            pass

        logger.info("[slot %d] Starting new container %s ...", slot_id, name)
        container = self.client.containers.run(
            OSWORLD_IMAGE,
            name=name,
            detach=True,
            privileged=True,
            cgroupns="host",
            network=DOCKER_NETWORK,
            tmpfs={
                "/run": "rw,nosuid,nodev,size=64m",
                "/run/lock": "rw,nosuid,nodev,size=16m",
                "/tmp": "rw,nosuid,nodev,size=2g",
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
        logger.info("[slot %d] Container %s started (id=%s)", slot_id, name, container.short_id)
        return container

    def _get_container_ip(self, slot_id: int) -> str:
        # Hold the lock throughout to prevent concurrent threads from both doing
        # the expensive container.reload() call and potentially storing stale results.
        with self._slot_ips_lock:
            if slot_id in self._slot_ips:
                return self._slot_ips[slot_id]

            name = self._container_name(slot_id)
            container = self.client.containers.get(name)
            # Retry up to 5 times (1s apart) in case Docker bridge assignment is delayed
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

            self._slot_ips[slot_id] = ip
            logger.info("[slot %d] Container IP: %s", slot_id, ip)
            return ip

    def _wait_for_port(self, ip: str, port: int, timeout: float, poll: float = 2.0) -> bool:
        """Wait until GET /health responds 200 on ip:port."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"http://{ip}:{port}/health", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(poll)
        return False

    def _call_resetd(self, ip: str, endpoint: str, timeout: float = RESETD_RESET_TIMEOUT) -> dict:
        r = requests.post(
            f"http://{ip}:{CONTAINER_RESETD_PORT}{endpoint}",
            json={},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()

    # ── Provider interface ────────────────────────────────────────────────────

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu", **kwargs):
        slot_id = _slot_id_from_path(path_to_vm)
        lock = self._get_slot_lock(slot_id)
        with lock:
            container = self._get_or_start_container(slot_id)
            ip = self._get_container_ip(slot_id)

            logger.info("[slot %d] Waiting for osworld-resetd (port %d) ...", slot_id, CONTAINER_RESETD_PORT)
            if not self._wait_for_port(ip, CONTAINER_RESETD_PORT, timeout=CONTAINER_STARTUP_TIMEOUT, poll=CONTAINER_STARTUP_POLL):
                raise RuntimeError(
                    f"[slot {slot_id}] resetd at {ip}:{CONTAINER_RESETD_PORT} did not come up within "
                    f"{CONTAINER_STARTUP_TIMEOUT}s"
                )
            logger.info("[slot %d] Waiting for osworld-server (port %d) ...", slot_id, CONTAINER_SERVER_PORT)
            if not self._wait_for_port(ip, CONTAINER_SERVER_PORT, timeout=CONTAINER_STARTUP_TIMEOUT, poll=CONTAINER_STARTUP_POLL):
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
        # Chrome DevTools at 9222, VLC HTTP at 8080 — both accessible via container bridge IP.
        # VNC (8006) is not used in Docker mode; 0 tells DesktopEnv to skip VNC.
        return f"{ip}:{CONTAINER_SERVER_PORT}:9222:0:8080"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        """No-op: the overlay FS serves as the snapshot mechanism."""
        pass

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> str:
        """Reset via overlay FS wipe through resetd HTTP endpoint."""
        slot_id = _slot_id_from_path(path_to_vm)
        lock = self._get_slot_lock(slot_id)
        with lock:
            ip = self._get_container_ip(slot_id)

            logger.info("[slot %d] Requesting reset via resetd at %s:%d ...", slot_id, ip, CONTAINER_RESETD_PORT)
            try:
                result = self._call_resetd(ip, "/reset", timeout=RESETD_RESET_TIMEOUT)
                status = result.get("status", "unknown")
                logger.info("[slot %d] resetd /reset response: status=%s", slot_id, status)
                if status not in ("ok", "busy"):
                    logger.warning("[slot %d] Unexpected reset status: %s", slot_id, result)
            except Exception as e:
                logger.error("[slot %d] resetd /reset failed: %s", slot_id, e)
                raise RuntimeError(f"[slot {slot_id}] overlay reset failed: {e}") from e

            # Wait for osworld-server (port 5000) to be healthy after reset
            logger.info("[slot %d] Waiting for osworld-server to recover (up to %ds) ...", slot_id, RESET_VERIFY_TIMEOUT)
            if not self._wait_for_port(ip, CONTAINER_SERVER_PORT, timeout=RESET_VERIFY_TIMEOUT, poll=RESET_VERIFY_POLL):
                raise RuntimeError(
                    f"[slot {slot_id}] osworld-server did not recover within {RESET_VERIFY_TIMEOUT}s after reset"
                )
            logger.info("[slot %d] Reset complete. osworld-server is up.", slot_id)

        return path_to_vm

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
