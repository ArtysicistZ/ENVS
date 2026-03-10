#!/usr/bin/env python3
"""
Test all 128 OSWorld tasks through the DockerProvider container lifecycle.
Tests: container start → prepare_baseline → reset with task_config (×128, no restart).
Records timing for each task. Container stays running across all tasks.

Usage:
    sg docker -c "export PROVIDER=docker && .venv/bin/python scripts/test_128_tasks_docker.py"
"""

import json
import os
import sys
import time
import csv
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OSWORLD_ROOT = REPO_ROOT / "OSWorld"
TASK_LIST = OSWORLD_ROOT / "evaluation_examples" / "test_success_uitars1.5_wo_impossible.json"
EXAMPLES_DIR = OSWORLD_ROOT / "evaluation_examples" / "examples"
SERVER_URL = "http://localhost:15001"
RESULTS_FILE = REPO_ROOT / "scripts" / "testing" / "test_128_results.csv"

# Timeouts
RESET_TIMEOUT = 300  # 5 min max per reset
STEP_TIMEOUT = 30
HEALTH_TIMEOUT = 10


def wait_for_server(url, timeout=60):
    """Wait for the remote env server to be ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/docs", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def load_all_tasks():
    """Load all 128 task configs in order."""
    with open(TASK_LIST) as f:
        task_map = json.load(f)

    tasks = []
    for app, task_ids in task_map.items():
        for tid in task_ids:
            config_path = EXAMPLES_DIR / app / f"{tid}.json"
            with open(config_path) as f:
                config = json.load(f)
            tasks.append({
                "app": app,
                "task_id": tid,
                "config": config,
            })
    return tasks


def test_reset(task_config, slot_id=0):
    """Call POST /env/reset with a task config. Returns (success, duration, details)."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{SERVER_URL}/env/reset",
            json={"task_config": task_config, "slot_id": slot_id},
            timeout=RESET_TIMEOUT,
        )
        duration = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            obs_count = len(data.get("obs_messages", []))
            is_done = data.get("is_done", None)
            return True, duration, f"obs={obs_count}, is_done={is_done}"
        else:
            return False, duration, f"HTTP {r.status_code}: {r.text[:200]}"
    except requests.exceptions.Timeout:
        duration = time.time() - t0
        return False, duration, "TIMEOUT"
    except Exception as e:
        duration = time.time() - t0
        return False, duration, f"ERROR: {e}"


def test_step(action="WAIT", slot_id=0):
    """Call POST /env/step with a simple action. Returns (success, duration, details)."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{SERVER_URL}/env/step",
            json={"action": action, "slot_id": slot_id},
            timeout=STEP_TIMEOUT,
        )
        duration = time.time() - t0
        if r.status_code == 200:
            return True, duration, "step OK"
        else:
            return False, duration, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        duration = time.time() - t0
        return False, duration, f"ERROR: {e}"


def main():
    print("=" * 80)
    print("OSWorld 128-Task Docker Lifecycle Test")
    print("=" * 80)

    # Load tasks
    tasks = load_all_tasks()
    print(f"Loaded {len(tasks)} tasks")

    # Check if server is running
    print(f"Checking server at {SERVER_URL} ...")
    if not wait_for_server(SERVER_URL, timeout=10):
        print("ERROR: Server not running. Start it first:")
        print('  sg docker -c "export PROVIDER=docker && .venv/bin/python scripts/remote_env_server.py"')
        sys.exit(1)
    print("Server is ready.\n")

    # Results tracking
    results = []
    success_count = 0
    fail_count = 0
    total_reset_time = 0.0
    app_stats = {}  # app -> {success, fail, total_time}

    # Write CSV header
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_num", "app", "task_id", "reset_ok", "reset_time_s",
            "step_ok", "step_time_s", "cumulative_time_s", "details"
        ])

    test_start = time.time()

    for i, task in enumerate(tasks):
        app = task["app"]
        tid = task["task_id"]
        config = task["config"]

        print(f"[{i+1:3d}/128] {app}/{tid[:12]}... ", end="", flush=True)

        # Reset with this task
        reset_ok, reset_time, reset_details = test_reset(config)
        total_reset_time += reset_time

        # Quick step (WAIT action) to verify env responds
        step_ok, step_time, step_details = False, 0.0, "skipped"
        if reset_ok:
            step_ok, step_time, step_details = test_step("WAIT")

        cumulative = time.time() - test_start

        if reset_ok:
            success_count += 1
            status = "OK"
        else:
            fail_count += 1
            status = "FAIL"

        # Per-app stats
        if app not in app_stats:
            app_stats[app] = {"success": 0, "fail": 0, "total_time": 0.0}
        if reset_ok:
            app_stats[app]["success"] += 1
        else:
            app_stats[app]["fail"] += 1
        app_stats[app]["total_time"] += reset_time

        print(f"{status}  reset={reset_time:6.1f}s  step={step_time:4.1f}s  cum={cumulative:7.1f}s  {reset_details[:60]}")

        # Append to CSV
        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i + 1, app, tid, reset_ok, f"{reset_time:.2f}",
                step_ok, f"{step_time:.2f}", f"{cumulative:.2f}",
                f"{reset_details}; {step_details}"
            ])

    total_time = time.time() - test_start

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tasks:     {len(tasks)}")
    print(f"Successful:      {success_count}")
    print(f"Failed:          {fail_count}")
    print(f"Success rate:    {success_count/len(tasks)*100:.1f}%")
    print(f"Total time:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total reset time:{total_reset_time:.1f}s")
    print(f"Avg reset time:  {total_reset_time/len(tasks):.1f}s")
    if success_count > 0:
        print(f"Avg reset (OK):  {total_reset_time/success_count:.1f}s")
    print()

    # Per-app breakdown
    print(f"{'App':<22} {'OK':>4} {'Fail':>4} {'Total':>5} {'Avg(s)':>8}")
    print("-" * 50)
    for app in sorted(app_stats.keys()):
        s = app_stats[app]
        total_tasks = s["success"] + s["fail"]
        avg = s["total_time"] / total_tasks if total_tasks > 0 else 0
        print(f"{app:<22} {s['success']:>4} {s['fail']:>4} {total_tasks:>5} {avg:>8.1f}")

    print(f"\nResults saved to: {RESULTS_FILE}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
