#!/usr/bin/env python3
"""
Multi-container scalability test for Docker provider.
Launches N containers in parallel, each doing: start → reset → screenshot → reset again.
Verifies all containers are independent and resets don't interfere.

Usage:
    # Start the server first:
    sg docker -c "export PROVIDER=docker && .venv/bin/python scripts/servers/remote_env_server.py" &

    # Run the test (default 4 slots; override with N_SLOTS):
    sg docker -c "N_SLOTS=4 .venv/bin/python scripts/testing/test_multi_container.py"
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OSWORLD_ROOT = REPO_ROOT / "OSWorld"
TASK_LIST = OSWORLD_ROOT / "evaluation_examples" / "test_success_uitars1.5_wo_impossible.json"
EXAMPLES_DIR = OSWORLD_ROOT / "evaluation_examples" / "examples"
SERVER_URL = os.environ.get("REMOTE_ENV_SERVER_URL", "http://localhost:15001")
N_SLOTS = int(os.environ.get("N_SLOTS", "4"))

RESET_TIMEOUT = 600  # 10 min (first reset includes container start + prepare_baseline)
STEP_TIMEOUT = 60


def load_tasks(n):
    """Load n task configs for testing."""
    with open(TASK_LIST) as f:
        task_map = json.load(f)
    tasks = []
    for app, task_ids in task_map.items():
        for tid in task_ids:
            config_path = EXAMPLES_DIR / app / f"{tid}.json"
            with open(config_path) as f:
                tasks.append(json.load(f))
            if len(tasks) >= n:
                return tasks
    return tasks


def reset_slot(slot_id, task_config):
    """POST /env/reset for a single slot. Returns (slot_id, success, duration, details)."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{SERVER_URL}/env/reset",
            json={"task_config": task_config, "slot_id": slot_id},
            timeout=RESET_TIMEOUT,
        )
        dur = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            has_obs = data.get("obs_messages") is not None
            return slot_id, True, dur, f"has_obs={has_obs}"
        return slot_id, False, dur, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return slot_id, False, time.time() - t0, f"ERROR: {e}"


def evaluate_slot(slot_id):
    """POST /env/evaluate for a single slot."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{SERVER_URL}/env/evaluate",
            json={"slot_id": slot_id},
            timeout=STEP_TIMEOUT,
        )
        dur = time.time() - t0
        if r.status_code == 200:
            return slot_id, True, dur, f"score={r.json()}"
        return slot_id, False, dur, f"HTTP {r.status_code}"
    except Exception as e:
        return slot_id, False, time.time() - t0, f"ERROR: {e}"


def check_server():
    """Verify server is up."""
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def check_disk_pressure():
    """Warn if SSD is getting full."""
    try:
        st = os.statvfs("/")
        free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        total_gb = (st.f_blocks * st.f_frsize) / (1024 ** 3)
        pct_used = (1 - st.f_bavail / st.f_blocks) * 100
        print(f"  Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total ({pct_used:.1f}% used)")
        if free_gb < 20:
            print(f"  WARNING: Only {free_gb:.1f} GB free — may run out during test!")
        return free_gb
    except Exception:
        return None


def main():
    print("=" * 70)
    print(f"Multi-Container Scalability Test  (N_SLOTS={N_SLOTS})")
    print("=" * 70)

    if not check_server():
        print("ERROR: Server not running at", SERVER_URL)
        print('Start: sg docker -c "export PROVIDER=docker && .venv/bin/python scripts/servers/remote_env_server.py"')
        return 1

    print("Server is up.")
    check_disk_pressure()

    tasks = load_tasks(N_SLOTS)
    if len(tasks) < N_SLOTS:
        print(f"WARNING: Only {len(tasks)} tasks available, need {N_SLOTS}")
        N = len(tasks)
    else:
        N = N_SLOTS
    print(f"Loaded {N} tasks for {N} slots.\n")

    # ── Phase 1: Parallel first reset (includes container start) ─────────
    print(f"Phase 1: Parallel first reset for {N} slots...")
    print("  (Each slot starts a new container — this takes 1-3 min)\n")
    t_start = time.time()
    results_1 = {}
    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = {
            pool.submit(reset_slot, i, tasks[i]): i
            for i in range(N)
        }
        for f in as_completed(futures):
            slot_id, ok, dur, details = f.result()
            status = "OK" if ok else "FAIL"
            results_1[slot_id] = ok
            print(f"  slot {slot_id:2d}: {status}  {dur:6.1f}s  {details}")

    phase1_time = time.time() - t_start
    ok_count = sum(1 for v in results_1.values() if v)
    print(f"\nPhase 1 done: {ok_count}/{N} succeeded in {phase1_time:.1f}s\n")

    if ok_count == 0:
        print("All slots failed. Aborting.")
        return 1

    # ── Phase 2: Parallel second reset (overlay FS reset, no container start)
    print(f"Phase 2: Parallel second reset (overlay wipe) for {ok_count} slots...")
    # Use different tasks for the second reset to verify state isolation
    tasks_2 = tasks[:]  # rotate: slot i gets task (i+1) % N
    t_start = time.time()
    results_2 = {}
    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = {}
        for i in range(N):
            if results_1.get(i):
                task_idx = (i + 1) % N
                futures[pool.submit(reset_slot, i, tasks_2[task_idx])] = i
        for f in as_completed(futures):
            slot_id, ok, dur, details = f.result()
            status = "OK" if ok else "FAIL"
            results_2[slot_id] = ok
            print(f"  slot {slot_id:2d}: {status}  {dur:6.1f}s  {details}")

    phase2_time = time.time() - t_start
    ok_count_2 = sum(1 for v in results_2.values() if v)
    print(f"\nPhase 2 done: {ok_count_2}/{ok_count} succeeded in {phase2_time:.1f}s\n")

    # ── Phase 3: Parallel evaluate ────────────────────────────────────────
    print(f"Phase 3: Parallel evaluate for {ok_count_2} slots...")
    t_start = time.time()
    results_3 = {}
    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = {}
        for i in range(N):
            if results_2.get(i):
                futures[pool.submit(evaluate_slot, i)] = i
        for f in as_completed(futures):
            slot_id, ok, dur, details = f.result()
            status = "OK" if ok else "FAIL"
            results_3[slot_id] = ok
            print(f"  slot {slot_id:2d}: {status}  {dur:6.1f}s  {details}")

    phase3_time = time.time() - t_start
    ok_count_3 = sum(1 for v in results_3.values() if v)
    print(f"\nPhase 3 done: {ok_count_3}/{ok_count_2} evaluated in {phase3_time:.1f}s\n")

    # ── Check disk after test ─────────────────────────────────────────────
    print("Post-test disk check:")
    check_disk_pressure()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Slots tested:        {N}")
    print(f"Phase 1 (first reset): {ok_count}/{N} OK in {phase1_time:.1f}s")
    print(f"Phase 2 (second reset): {ok_count_2}/{ok_count} OK in {phase2_time:.1f}s")
    print(f"Phase 3 (evaluate):    {ok_count_3}/{ok_count_2} OK in {phase3_time:.1f}s")
    total = phase1_time + phase2_time + phase3_time
    print(f"Total test time:     {total:.1f}s ({total/60:.1f} min)")

    if ok_count == N and ok_count_2 == ok_count and ok_count_3 == ok_count_2:
        print("\nAll phases PASSED.")
        return 0
    else:
        print("\nSome phases had failures. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
