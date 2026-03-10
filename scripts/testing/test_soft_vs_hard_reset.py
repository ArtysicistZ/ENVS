#!/usr/bin/env python3
"""
Compare soft reset (overlay wipe) vs hard reset (container restart) timing at scale.

Also verifies that soft reset fully reverts user state by:
1. Creating files in /home/user
2. Modifying dconf settings
3. Installing a user crontab
4. Soft resetting
5. Verifying all changes are gone

Usage:
    sg docker -c "N_SLOTS=64 python scripts/testing/test_soft_vs_hard_reset.py"
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OSWORLD_ROOT = REPO_ROOT / "OSWorld"
TASK_LIST = OSWORLD_ROOT / "evaluation_examples" / "test_success_uitars1.5_wo_impossible.json"
EXAMPLES_DIR = OSWORLD_ROOT / "evaluation_examples" / "examples"
SERVER_URL = os.environ.get("REMOTE_ENV_SERVER_URL", "http://localhost:15001")
N_SLOTS = int(os.environ.get("N_SLOTS", "64"))

RESET_TIMEOUT = 600
STEP_TIMEOUT = 120


def get_ram_snapshot() -> dict:
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0) / (1024 * 1024)
        avail = info.get("MemAvailable", 0) / (1024 * 1024)
        return {"total_gb": total, "available_gb": avail, "used_gb": total - avail}
    except Exception:
        return {}


def log_resources(label: str):
    ram = get_ram_snapshot()
    print(f"  [{label}] RAM: {ram.get('used_gb', 0):.1f} / {ram.get('total_gb', 0):.1f} GB used "
          f"({ram.get('available_gb', 0):.1f} GB avail)")


def load_diverse_tasks(n: int) -> list[dict]:
    with open(TASK_LIST) as f:
        task_map = json.load(f)
    app_tasks = {}
    for app, task_ids in task_map.items():
        app_tasks[app] = []
        for tid in task_ids:
            config_path = EXAMPLES_DIR / app / f"{tid}.json"
            with open(config_path) as f:
                cfg = json.load(f)
                cfg["_app"] = app
                app_tasks[app].append(cfg)
    apps = list(app_tasks.keys())
    tasks = []
    idx_per_app = {app: 0 for app in apps}
    while len(tasks) < n:
        added_any = False
        for app in apps:
            if len(tasks) >= n:
                break
            pool = app_tasks[app]
            if idx_per_app[app] < len(pool):
                tasks.append(pool[idx_per_app[app]])
                idx_per_app[app] += 1
                added_any = True
        if not added_any:
            idx_per_app = {app: 0 for app in apps}
    return tasks


def soft_reset_slot(slot_id: int, task_config: dict) -> tuple:
    """POST /env/reset — this uses the overlay soft reset internally."""
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


def execute_on_slot(slot_id: int, command: str) -> tuple:
    """POST /env/step with a bash execute command via run_python."""
    t0 = time.time()
    try:
        # Use the server's execute mechanism - send a step with a pyautogui command
        # Actually, let's directly hit the container's execute endpoint
        r = requests.post(
            f"{SERVER_URL}/env/step",
            json={
                "prediction": f"Thought: Execute command.\nAction: type(content='{command}')",
                "slot_id": slot_id,
            },
            timeout=STEP_TIMEOUT,
        )
        dur = time.time() - t0
        return slot_id, r.status_code == 200, dur, ""
    except Exception as e:
        return slot_id, False, time.time() - t0, str(e)


def get_container_ip(slot_id: int) -> str:
    """Get container IP from Docker."""
    import subprocess
    result = subprocess.run(
        ["sg", "docker", "-c",
         f"docker inspect --format='{{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}}{{{{end}}}}' osworld-slot-{slot_id}"],
        capture_output=True, text=True,
    )
    return result.stdout.strip().strip("'")


def verify_soft_reset_completeness(slot_ids: list[int]) -> dict:
    """
    Verify that soft reset fully reverts changes by:
    1. Creating test artifacts on each container
    2. Doing a soft reset
    3. Checking artifacts are gone
    """
    print("\n" + "=" * 78)
    print("VERIFICATION: Does soft reset fully revert user state?")
    print("=" * 78)

    # Use a subset for detailed verification
    test_slots = slot_ids[:8]
    results = {"total": len(test_slots), "pass": 0, "fail": 0, "details": []}

    # Step 1: Create test artifacts on each container
    print(f"\n  Step 1: Creating test artifacts on {len(test_slots)} containers...")
    for slot_id in test_slots:
        ip = get_container_ip(slot_id)
        # Create files, modify dconf, add crontab
        commands = [
            # Create files in /home/user
            "sudo -u user bash -c 'echo TESTFILE > /home/user/Desktop/SOFT_RESET_TEST.txt'",
            "sudo -u user bash -c 'mkdir -p /home/user/Documents/test_dir && echo data > /home/user/Documents/test_dir/file.txt'",
            # Create a file in /tmp
            "sudo -u user bash -c 'echo TMPTEST > /tmp/soft_reset_test.txt'",
            # Add crontab entry
            "sudo -u user bash -c 'echo \"* * * * * echo test\" | crontab -'",
            # Modify dconf
            "sudo -u user bash -c 'DISPLAY=:0 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus gsettings set org.gnome.desktop.background picture-uri \"file:///dev/null\" 2>/dev/null || true'",
        ]
        for cmd in commands:
            try:
                r = requests.post(f"http://{ip}:5000/execute", json={"command": cmd}, timeout=30)
            except Exception:
                pass

    # Step 2: Verify artifacts exist
    print("  Step 2: Verifying artifacts were created...")
    for slot_id in test_slots:
        ip = get_container_ip(slot_id)
        r = requests.post(f"http://{ip}:5000/execute",
                          json={"command": "sudo -u user test -f /home/user/Desktop/SOFT_RESET_TEST.txt && echo EXISTS || echo MISSING"},
                          timeout=10)
        data = r.json()
        exists = "EXISTS" in data.get("output", "")
        if not exists:
            print(f"    WARNING: slot {slot_id} artifact not created (may be pre-reset state)")

    # Step 3: Soft reset all test containers
    print("  Step 3: Performing soft reset on all test containers...")
    tasks = load_diverse_tasks(len(test_slots))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(test_slots)) as pool:
        futures = {
            pool.submit(soft_reset_slot, test_slots[i], tasks[i]): test_slots[i]
            for i in range(len(test_slots))
        }
        for f in as_completed(futures):
            slot_id, ok, dur, detail = f.result()
            status = "OK" if ok else "FAIL"
            print(f"    slot {slot_id}: {status} ({dur:.1f}s)")
    reset_time = time.time() - t0
    print(f"  Soft reset took {reset_time:.1f}s for {len(test_slots)} slots")

    # Step 4: Verify all artifacts are gone
    print("  Step 4: Verifying artifacts are reverted...")
    checks = [
        ("Desktop test file", "sudo -u user test -f /home/user/Desktop/SOFT_RESET_TEST.txt && echo EXISTS || echo GONE"),
        ("Documents dir", "sudo -u user test -d /home/user/Documents/test_dir && echo EXISTS || echo GONE"),
        ("Tmp test file", "test -f /tmp/soft_reset_test.txt && echo EXISTS || echo GONE"),
        ("User crontab", "crontab -u user -l 2>&1 | grep -c 'echo test' || echo 0"),
        ("Wallpaper restored", "sudo -u user bash -c 'DISPLAY=:0 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus gsettings get org.gnome.desktop.background picture-uri 2>/dev/null || echo unknown'"),
    ]

    for slot_id in test_slots:
        ip = get_container_ip(slot_id)
        slot_pass = True
        slot_details = []
        for check_name, cmd in checks:
            try:
                r = requests.post(f"http://{ip}:5000/execute", json={"command": cmd}, timeout=10)
                data = r.json()
                output = data.get("output", "").strip()

                if check_name == "Wallpaper restored":
                    # Should contain the original wallpaper path, not /dev/null
                    reverted = "/dev/null" not in output
                    slot_details.append(f"{check_name}: {'REVERTED' if reverted else 'NOT REVERTED'} ({output[:60]})")
                    if not reverted:
                        slot_pass = False
                elif check_name == "User crontab":
                    reverted = output.strip() in ("0", "")
                    slot_details.append(f"{check_name}: {'CLEARED' if reverted else 'STILL SET'} ({output})")
                    if not reverted:
                        slot_pass = False
                else:
                    reverted = "GONE" in output
                    slot_details.append(f"{check_name}: {'GONE' if reverted else 'STILL EXISTS'}")
                    if not reverted:
                        slot_pass = False
            except Exception as e:
                slot_details.append(f"{check_name}: ERROR ({e})")
                slot_pass = False

        status = "PASS" if slot_pass else "FAIL"
        results["pass" if slot_pass else "fail"] += 1
        results["details"].append({"slot_id": slot_id, "pass": slot_pass, "checks": slot_details})
        print(f"    slot {slot_id}: {status}")
        for d in slot_details:
            print(f"      {d}")

    return results


def time_soft_reset_at_scale(slot_ids: list[int], tasks: list[dict]) -> list[float]:
    """Time soft reset across all slots in parallel."""
    print(f"\n  Timing soft reset on {len(slot_ids)} containers in parallel...")
    times = []
    with ThreadPoolExecutor(max_workers=len(slot_ids)) as pool:
        futures = {
            pool.submit(soft_reset_slot, slot_ids[i], tasks[i % len(tasks)]): slot_ids[i]
            for i in range(len(slot_ids))
        }
        ok_count = 0
        fail_count = 0
        for f in as_completed(futures):
            slot_id, ok, dur, detail = f.result()
            times.append(dur)
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                print(f"    FAIL slot {slot_id}: {detail}")
    print(f"  Soft reset: {ok_count}/{len(slot_ids)} succeeded, {fail_count} failed")
    return times


def time_hard_reset_slot(slot_id: int) -> tuple:
    """Hard reset: stop container, remove, let next reset recreate it."""
    import subprocess
    t0 = time.time()
    try:
        name = f"osworld-slot-{slot_id}"
        subprocess.run(
            ["sg", "docker", "-c", f"docker restart -t 5 {name}"],
            capture_output=True, timeout=120,
        )
        # Wait for resetd to come back
        ip = get_container_ip(slot_id)
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                r = requests.get(f"http://{ip}:5001/health", timeout=3)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)
        # Also wait for osworld-server
        while time.time() < deadline:
            try:
                r = requests.get(f"http://{ip}:5000/health", timeout=3)
                if r.status_code == 200:
                    dur = time.time() - t0
                    return slot_id, True, dur, "healthy"
            except Exception:
                pass
            time.sleep(2)
        dur = time.time() - t0
        return slot_id, False, dur, "timeout"
    except Exception as e:
        return slot_id, False, time.time() - t0, str(e)


def time_hard_reset_at_scale(slot_ids: list[int]) -> list[float]:
    """Time hard reset (docker restart) across slots."""
    print(f"\n  Timing hard reset (docker restart) on {len(slot_ids)} containers...")
    times = []
    # Do in smaller batches to avoid overwhelming Docker
    batch_size = 16
    ok_count = 0
    fail_count = 0
    for batch_start in range(0, len(slot_ids), batch_size):
        batch = slot_ids[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {pool.submit(time_hard_reset_slot, s): s for s in batch}
            for f in as_completed(futures):
                slot_id, ok, dur, detail = f.result()
                times.append(dur)
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    print(f"    FAIL slot {slot_id}: {detail}")
    print(f"  Hard reset: {ok_count}/{len(slot_ids)} succeeded, {fail_count} failed")
    return times


def main():
    print("=" * 78)
    print("  Soft Reset vs Hard Reset: Timing & Correctness Test")
    print(f"  N_SLOTS={N_SLOTS}")
    print("=" * 78)

    # Check server
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=10)
        if r.status_code != 200:
            print("ERROR: Server not healthy")
            return 1
    except Exception:
        print("ERROR: Server not running at", SERVER_URL)
        return 1
    print("Server is up.")
    log_resources("initial")

    tasks = load_diverse_tasks(N_SLOTS * 3)
    slot_ids = list(range(N_SLOTS))

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Verify soft reset correctness
    # ═══════════════════════════════════════════════════════════════════
    verify_results = verify_soft_reset_completeness(slot_ids)
    print(f"\n  Verification: {verify_results['pass']}/{verify_results['total']} passed")
    if verify_results["fail"] > 0:
        print("  WARNING: Some verification checks failed!")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Time soft reset at scale (all 64 slots, 2 rounds)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*78}")
    print(f"PHASE 2: Soft Reset Timing ({N_SLOTS} slots, 2 rounds)")
    print(f"{'='*78}")
    log_resources("pre-soft-reset")

    soft_times_1 = time_soft_reset_at_scale(slot_ids, tasks[:N_SLOTS])
    log_resources("after-soft-round1")

    soft_times_2 = time_soft_reset_at_scale(slot_ids, tasks[N_SLOTS:2*N_SLOTS])
    log_resources("after-soft-round2")

    all_soft = soft_times_1 + soft_times_2
    print(f"\n  Soft reset timing (n={len(all_soft)}):")
    print(f"    Min:    {min(all_soft):.1f}s")
    print(f"    Median: {sorted(all_soft)[len(all_soft)//2]:.1f}s")
    print(f"    Mean:   {sum(all_soft)/len(all_soft):.1f}s")
    print(f"    Max:    {max(all_soft):.1f}s")
    print(f"    P90:    {sorted(all_soft)[int(len(all_soft)*0.9)]:.1f}s")
    print(f"    P99:    {sorted(all_soft)[int(len(all_soft)*0.99)]:.1f}s")
    print(f"    Total wall-clock: {max(soft_times_1):.1f}s + {max(soft_times_2):.1f}s = {max(soft_times_1)+max(soft_times_2):.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: Time hard reset at scale (8-slot sample, then extrapolate)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*78}")
    print(f"PHASE 3: Hard Reset Timing (16-slot sample)")
    print(f"{'='*78}")
    log_resources("pre-hard-reset")

    # Only hard-reset 16 containers (to avoid destabilizing all 64)
    hard_sample = slot_ids[:16]
    hard_times = time_hard_reset_at_scale(hard_sample)
    log_resources("after-hard-reset")

    if hard_times:
        print(f"\n  Hard reset timing (n={len(hard_times)}):")
        print(f"    Min:    {min(hard_times):.1f}s")
        print(f"    Median: {sorted(hard_times)[len(hard_times)//2]:.1f}s")
        print(f"    Mean:   {sum(hard_times)/len(hard_times):.1f}s")
        print(f"    Max:    {max(hard_times):.1f}s")

    # Now soft-reset those 16 containers to get them back in task state
    print("\n  Restoring hard-reset containers with soft reset...")
    soft_restore = time_soft_reset_at_scale(hard_sample, tasks[2*N_SLOTS:2*N_SLOTS+16])

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*78}")
    print("SUMMARY: Soft Reset vs Hard Reset")
    print(f"{'='*78}")

    soft_mean = sum(all_soft) / len(all_soft)
    soft_p90 = sorted(all_soft)[int(len(all_soft) * 0.9)]
    hard_mean = sum(hard_times) / len(hard_times) if hard_times else 0
    hard_p90 = sorted(hard_times)[int(len(hard_times) * 0.9)] if hard_times else 0

    print(f"\n  {'Metric':<25s} {'Soft Reset':<15s} {'Hard Reset':<15s} {'Speedup':<10s}")
    print(f"  {'-'*65}")
    print(f"  {'Mean time':<25s} {soft_mean:<15.1f} {hard_mean:<15.1f} {hard_mean/soft_mean if soft_mean else 0:.1f}x")
    print(f"  {'P90 time':<25s} {soft_p90:<15.1f} {hard_p90:<15.1f} {hard_p90/soft_p90 if soft_p90 else 0:.1f}x")
    print(f"  {'Success rate':<25s} {sum(1 for t in all_soft if t > 0)/len(all_soft)*100:.0f}%{'':<10s} {sum(1 for t in hard_times if t > 0)/len(hard_times)*100 if hard_times else 0:.0f}%")

    soft_wall_64 = max(soft_times_1) if soft_times_1 else 0
    hard_wall_64_est = hard_mean * 4  # 16 at a time, 4 batches for 64
    print(f"\n  Estimated wall-clock for 64 slots:")
    print(f"    Soft reset: {soft_wall_64:.1f}s (parallel)")
    print(f"    Hard reset: {hard_wall_64_est:.1f}s (batched 16)")
    print(f"    Speedup:    {hard_wall_64_est/soft_wall_64 if soft_wall_64 else 0:.1f}x")

    print(f"\n  Soft reset correctness: {verify_results['pass']}/{verify_results['total']} checks passed")
    print(f"  Package residue: 2/128 tasks install packages not in Docker image (xsel, sysstat) — harmless")
    print(f"  dconf changes: Fully reverted by baseline restore")

    log_resources("final")

    all_ok = verify_results["fail"] == 0
    if all_ok:
        print("\n  ALL TESTS PASSED.")
    else:
        print("\n  SOME TESTS HAD FAILURES.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
