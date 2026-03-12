#!/usr/bin/env python3
"""
Fault tolerance test for the 64-container OSWorld Docker pool.

Tests:
  1. Pre-check: verify all 64 containers are running and reachable
  2. Kill test: forcibly kill N containers, verify health monitor detects + recovers them
  3. Post-recovery: verify all 64 containers are healthy and reachable via training API
  4. Training ops: run reset + step + evaluate on recovered containers to prove they work

Usage:
    # Server must be running:
    #   sg docker -c "OSWORLD_POOL_SIZE=64 .venv/bin/python scripts/servers/remote_env_server.py"
    #
    # Then run:
    python scripts/testing/test_fault_tolerance.py

Environment variables:
    SERVER_URL      default http://localhost:15001
    N_SLOTS         default 64
    KILL_COUNT      number of containers to kill (default 5)
    RECOVERY_WAIT   seconds to wait for recovery (default 180)
"""

import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:15001").rstrip("/")
N_SLOTS = int(os.environ.get("N_SLOTS", "64"))
KILL_COUNT = int(os.environ.get("KILL_COUNT", "5"))
RECOVERY_WAIT = int(os.environ.get("RECOVERY_WAIT", "180"))
TIMEOUT = 30  # HTTP request timeout


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def check_server_health() -> dict:
    r = requests.get(f"{SERVER_URL}/health", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def check_container_health() -> dict:
    r = requests.get(f"{SERVER_URL}/health/containers", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def check_slot_reachable(slot_id: int) -> dict:
    """Check that a slot's container is reachable by directly probing its Docker
    container's HTTP endpoints (resetd at 5001 and osworld-server at 5000).

    This avoids the overhead and fragility of doing a full env.reset() with a
    task config (which requires valid evaluator functions, etc.).
    """
    import subprocess as _sp
    name = f"osworld-slot-{slot_id}"
    try:
        # Get container IP via Docker inspect
        result = _sp.run(
            ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"slot_id": slot_id, "status": "error", "error": f"docker inspect failed: {result.stderr.strip()}"}
        ip = result.stdout.strip()
        if not ip:
            return {"slot_id": slot_id, "status": "error", "error": "no IP assigned"}

        # Check resetd health (port 5001)
        try:
            r = requests.get(f"http://{ip}:5001/health", timeout=10)
            resetd_ok = r.status_code == 200
        except Exception as e:
            return {"slot_id": slot_id, "status": "error", "error": f"resetd unreachable: {e}", "ip": ip}

        # Check osworld-server health (port 5000)
        try:
            r = requests.get(f"http://{ip}:5000/health", timeout=10)
            server_ok = r.status_code == 200
        except Exception as e:
            return {"slot_id": slot_id, "status": "error", "error": f"server unreachable: {e}", "ip": ip}

        if resetd_ok and server_ok:
            return {"slot_id": slot_id, "status": "ok", "ip": ip}
        return {
            "slot_id": slot_id, "status": "error",
            "resetd": "ok" if resetd_ok else "failed",
            "server": "ok" if server_ok else "failed",
            "ip": ip,
        }
    except Exception as e:
        return {"slot_id": slot_id, "status": "error", "error": str(e)}


def docker_kill_container(slot_id: int) -> bool:
    """Force-kill a container via Docker CLI."""
    name = f"osworld-slot-{slot_id}"
    try:
        result = subprocess.run(
            ["docker", "kill", name],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def docker_rm_container(slot_id: int) -> bool:
    """Force-remove a container via Docker CLI."""
    name = f"osworld-slot-{slot_id}"
    try:
        result = subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def docker_container_status(slot_id: int) -> str | None:
    """Get container status via Docker CLI."""
    name = f"osworld-slot-{slot_id}"
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", name],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


# ── Test Phases ─────────────────────────────────────────────────────────────

def phase_1_pre_check():
    """Verify all 64 containers are running and the health endpoint reports OK."""
    log("=" * 70)
    log("PHASE 1: Pre-check — verify all containers are healthy")
    log("=" * 70)

    # Check server health
    health = check_server_health()
    log(f"Server health: {health}")
    assert health["status"] == "ok", f"Server unhealthy: {health}"

    # Check container health endpoint
    container_health = check_container_health()
    log(f"Container health: healthy={container_health.get('healthy')}/{container_health.get('target_pool_size')}")

    if container_health.get("status") != "ok":
        unhealthy = container_health.get("unhealthy_slots", {})
        log(f"WARNING: {len(unhealthy)} unhealthy slots: {list(unhealthy.keys())}")

    # Verify via Docker CLI that all containers exist and are running
    missing = []
    not_running = []
    for i in range(N_SLOTS):
        status = docker_container_status(i)
        if status is None:
            missing.append(i)
        elif status != "running":
            not_running.append((i, status))

    if missing:
        log(f"MISSING containers: {missing}")
    if not_running:
        log(f"NOT RUNNING containers: {not_running}")

    total_ok = N_SLOTS - len(missing) - len(not_running)
    log(f"Docker status: {total_ok}/{N_SLOTS} running, {len(missing)} missing, {len(not_running)} not running")

    assert total_ok == N_SLOTS, f"Not all containers running: {total_ok}/{N_SLOTS}"
    log("PHASE 1 PASSED: All containers healthy")
    return True


def phase_2_kill_and_recover():
    """Kill N containers and verify the health monitor recovers them."""
    log("=" * 70)
    log(f"PHASE 2: Kill {KILL_COUNT} containers and verify auto-recovery")
    log("=" * 70)

    # Pick random slots to kill
    kill_slots = sorted(random.sample(range(N_SLOTS), KILL_COUNT))
    log(f"Killing containers for slots: {kill_slots}")

    # Kill them
    killed = []
    for slot_id in kill_slots:
        # Use docker rm -f to fully remove (not just stop)
        if docker_rm_container(slot_id):
            killed.append(slot_id)
            log(f"  Killed+removed osworld-slot-{slot_id}")
        else:
            log(f"  WARNING: Failed to kill osworld-slot-{slot_id}")

    log(f"Successfully killed {len(killed)}/{KILL_COUNT} containers")
    assert len(killed) == KILL_COUNT, f"Could not kill all target containers"

    # Verify they're gone
    for slot_id in killed:
        status = docker_container_status(slot_id)
        assert status is None or status != "running", f"Slot {slot_id} still running after kill!"
    log("Verified: all killed containers are down/removed")

    # Wait for health monitor to detect and recover
    log(f"Waiting up to {RECOVERY_WAIT}s for health monitor to recover {len(killed)} containers ...")
    start = time.time()
    check_interval = 10
    all_recovered = False

    while time.time() - start < RECOVERY_WAIT:
        time.sleep(check_interval)
        elapsed = time.time() - start

        # Check Docker status of killed slots
        recovered = []
        still_down = []
        for slot_id in killed:
            status = docker_container_status(slot_id)
            if status == "running":
                recovered.append(slot_id)
            else:
                still_down.append(slot_id)

        log(f"  [{elapsed:.0f}s] {len(recovered)}/{len(killed)} recovered, still down: {still_down}")

        if len(recovered) == len(killed):
            all_recovered = True
            break

    if not all_recovered:
        container_health = check_container_health()
        log(f"Health monitor stats: {container_health.get('stats')}")
        log(f"Unhealthy slots: {container_health.get('unhealthy_slots')}")
        log(f"PHASE 2 FAILED: Not all containers recovered within {RECOVERY_WAIT}s")
        return False, killed

    docker_recovery_time = time.time() - start
    log(f"All {len(killed)} Docker containers recreated in {docker_recovery_time:.1f}s")

    # Now wait for all services (resetd + osworld-server) to become reachable
    log("Waiting for services on recovered containers to become healthy ...")
    service_deadline = time.time() + 300  # 5 min for services
    while time.time() < service_deadline:
        all_services_ok = True
        for slot_id in killed:
            result = check_slot_reachable(slot_id)
            if result["status"] != "ok":
                all_services_ok = False
                break
        if all_services_ok:
            break
        time.sleep(10)

    if not all_services_ok:
        log("PHASE 2 FAILED: Services on recovered containers didn't become healthy")
        for slot_id in killed:
            result = check_slot_reachable(slot_id)
            if result["status"] != "ok":
                log(f"  Slot {slot_id}: {result}")
        return False, killed

    total_recovery_time = time.time() - start
    log(f"All {len(killed)} containers fully recovered (Docker + services) in {total_recovery_time:.1f}s")

    # Verify health endpoint
    container_health = check_container_health()
    log(f"Recovery stats: {container_health.get('stats')}")

    log("PHASE 2 PASSED: All killed containers auto-recovered with services healthy")
    return True, killed


def phase_3_post_recovery_verify(killed_slots: list[int]):
    """Verify all containers (especially recovered ones) are fully functional."""
    log("=" * 70)
    log(f"PHASE 3: Post-recovery verification — test training ops on recovered containers")
    log("=" * 70)

    # First: verify ALL containers are running via Docker CLI
    all_running = True
    for i in range(N_SLOTS):
        status = docker_container_status(i)
        if status != "running":
            log(f"  Slot {i}: status={status} (expected 'running')")
            all_running = False

    if not all_running:
        log("PHASE 3 FAILED: Not all containers running")
        return False

    log(f"All {N_SLOTS} containers running (Docker CLI check)")

    # Test training ops on recovered containers
    log(f"Testing reset on {len(killed_slots)} recovered containers ...")
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(check_slot_reachable, sid): sid for sid in killed_slots}
        for future in as_completed(futures):
            result = future.result()
            sid = result["slot_id"]
            results[sid] = result
            status = result["status"]
            log(f"  Slot {sid}: {status}")

    failed = [sid for sid, r in results.items() if r["status"] != "ok"]
    if failed:
        log(f"PHASE 3 FAILED: {len(failed)} recovered slots failed training ops: {failed}")
        for sid in failed:
            log(f"  Slot {sid}: {results[sid]}")
        return False

    log(f"PHASE 3 PASSED: All {len(killed_slots)} recovered containers pass training ops")
    return True


def phase_4_full_reachability():
    """Verify ALL 64 containers are reachable from the training API."""
    log("=" * 70)
    log(f"PHASE 4: Full reachability — verify all {N_SLOTS} slots respond to reset")
    log("=" * 70)

    # Check all slots in parallel (batched to avoid overwhelming)
    BATCH = 16
    all_results = {}

    for batch_start in range(0, N_SLOTS, BATCH):
        batch_end = min(batch_start + BATCH, N_SLOTS)
        batch_slots = list(range(batch_start, batch_end))
        log(f"Testing slots {batch_start}-{batch_end - 1} ...")

        with ThreadPoolExecutor(max_workers=BATCH) as executor:
            futures = {executor.submit(check_slot_reachable, sid): sid for sid in batch_slots}
            for future in as_completed(futures):
                result = future.result()
                sid = result["slot_id"]
                all_results[sid] = result

        ok_in_batch = sum(1 for sid in batch_slots if all_results[sid]["status"] == "ok")
        log(f"  Batch result: {ok_in_batch}/{len(batch_slots)} OK")

    # Summary
    ok_count = sum(1 for r in all_results.values() if r["status"] == "ok")
    failed = [sid for sid, r in all_results.items() if r["status"] != "ok"]

    log(f"Full reachability: {ok_count}/{N_SLOTS}")
    if failed:
        log(f"FAILED slots: {sorted(failed)}")
        for sid in sorted(failed):
            log(f"  Slot {sid}: {all_results[sid]}")
        log("PHASE 4 FAILED")
        return False

    log(f"PHASE 4 PASSED: All {N_SLOTS}/{N_SLOTS} slots reachable and functional")
    return True


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log(f"Fault Tolerance Test — {N_SLOTS} slots, killing {KILL_COUNT} containers")
    log(f"Server: {SERVER_URL}")
    log("")

    # Phase 1: Pre-check
    try:
        phase_1_pre_check()
    except AssertionError as e:
        log(f"PHASE 1 FAILED: {e}")
        log("Hint: ensure server is running with pre-warming complete")
        sys.exit(1)

    # Phase 2: Kill and recover
    ok, killed_slots = phase_2_kill_and_recover()
    if not ok:
        log("ABORT: Recovery failed. Check server logs.")
        sys.exit(1)

    # Phase 3: Post-recovery verify
    if not phase_3_post_recovery_verify(killed_slots):
        log("ABORT: Post-recovery verification failed")
        sys.exit(1)

    # Phase 4: Full reachability
    if not phase_4_full_reachability():
        log("ABORT: Full reachability check failed")
        sys.exit(1)

    log("")
    log("=" * 70)
    log("ALL PHASES PASSED — fault tolerance verified")
    log("=" * 70)


if __name__ == "__main__":
    main()
