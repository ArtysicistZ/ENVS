#!/usr/bin/env python3
"""
64-container scalability test for Docker provider.

Exercises the full lifecycle in repeated cycles:
  fresh start → steps (click/scroll/type/enter) → evaluate → reset → steps → evaluate → reset → ...

Each cycle uses different tasks. Monitors RAM/disk throughout.
Designed to catch resource leaks, concurrency bugs, and overlay reset failures at scale.

Usage:
    # Start the server first:
    sg docker -c "export PROVIDER=docker && REMOTE_MAX_SLOTS=80 .venv/bin/python scripts/servers/remote_env_server.py" &

    # Run (default 64 slots, 3 reset cycles):
    sg docker -c "N_SLOTS=64 .venv/bin/python scripts/testing/test_64_containers.py"

    # Smaller smoke test:
    sg docker -c "N_SLOTS=16 N_CYCLES=2 .venv/bin/python scripts/testing/test_64_containers.py"
"""

import base64
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
N_CYCLES = int(os.environ.get("N_CYCLES", "3"))  # number of reset→step→evaluate cycles

RESET_TIMEOUT = 600   # 10 min (first reset includes container start + prepare_baseline)
STEP_TIMEOUT = 120
EVAL_TIMEOUT = 120

# Concurrency: batch container creation to avoid thundering-herd on Docker daemon.
LAUNCH_BATCH_SIZE = int(os.environ.get("LAUNCH_BATCH_SIZE", "16"))

# Screenshot output directory
SCREENSHOT_DIR = REPO_ROOT / "scripts" / "testing" / "test_64_screenshots"


# ── Synthetic step actions (valid parser-compatible predictions) ──────────

STEP_ACTIONS = {
    "click_center": "Thought: I will click the center of the screen.\nAction: click(start_box='(960,540)')",
    "click_top_left": "Thought: I will click near the top-left.\nAction: click(start_box='(100,100)')",
    "scroll_down": "Thought: I will scroll down.\nAction: scroll(start_box='(960,540)', direction='down', amount=3)",
    "scroll_up": "Thought: I will scroll up.\nAction: scroll(start_box='(960,540)', direction='up', amount=3)",
    "type_hello": "Thought: I will type some text.\nAction: type(content='hello world')",
    "press_enter": "Thought: I will press enter.\nAction: press(key='enter')",
    "press_tab": "Thought: I will press tab.\nAction: press(key='tab')",
    "hotkey_ctrl_a": "Thought: I will select all.\nAction: hotkey(key='ctrl a')",
    "wait": "Thought: I will wait.\nAction: wait()",
    "finish": "Thought: Task is done.\nAction: finished()",
}

# Full interaction sequence used each cycle: click, scroll, type, enter, more clicks, scroll, hotkey, tab
FULL_ACTION_SEQUENCE = [
    "click_center", "scroll_down", "type_hello", "press_enter",
    "click_top_left", "scroll_up", "hotkey_ctrl_a", "press_tab",
]


# Only save screenshots from this slot (avoid disk bloat from 64 VMs)
SCREENSHOT_SLOT = int(os.environ.get("SCREENSHOT_SLOT", "0"))


def save_screenshot_from_response(data: dict, slot_id: int, phase: str):
    """Extract and save the latest screenshot from obs_messages response.

    Wire format uses {"type": "image", "b64": "<base64>"} (see remote_env_protocol.py).
    Only saves for SCREENSHOT_SLOT to keep output manageable.
    """
    if slot_id != SCREENSHOT_SLOT:
        return
    obs_messages = data.get("obs_messages")
    if not obs_messages:
        return
    # Find the last image message
    for msg in reversed(obs_messages):
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in reversed(content):
                if isinstance(item, dict) and item.get("type") == "image":
                    b64_str = item.get("b64") or item.get("image", "")
                    if b64_str.startswith("data:image/"):
                        b64_str = b64_str.split(",", 1)[1]
                    if not b64_str:
                        continue
                    try:
                        img_bytes = base64.b64decode(b64_str)
                        out_path = SCREENSHOT_DIR / f"slot_{slot_id:02d}_{phase}.png"
                        out_path.write_bytes(img_bytes)
                        return
                    except Exception:
                        pass


def load_diverse_tasks(n: int) -> list[dict]:
    """Load n tasks with maximum app diversity — round-robin across app categories."""
    with open(TASK_LIST) as f:
        task_map = json.load(f)

    # Build per-app task lists
    app_tasks: dict[str, list[dict]] = {}
    for app, task_ids in task_map.items():
        app_tasks[app] = []
        for tid in task_ids:
            config_path = EXAMPLES_DIR / app / f"{tid}.json"
            with open(config_path) as f:
                cfg = json.load(f)
                cfg["_app"] = app  # tag for reporting
                app_tasks[app].append(cfg)

    # Round-robin: pick one from each app, then repeat
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
            # All apps exhausted — wrap around from the start
            idx_per_app = {app: 0 for app in apps}
    return tasks


def check_server() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def get_ram_snapshot() -> dict:
    """Return RAM stats in GB."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0) / (1024 * 1024)
        avail = info.get("MemAvailable", 0) / (1024 * 1024)
        used = total - avail
        return {"total_gb": total, "available_gb": avail, "used_gb": used}
    except Exception:
        return {}


def get_disk_snapshot() -> dict:
    try:
        st = os.statvfs("/")
        free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        total_gb = (st.f_blocks * st.f_frsize) / (1024 ** 3)
        return {"free_gb": free_gb, "total_gb": total_gb, "pct_used": (1 - st.f_bavail / st.f_blocks) * 100}
    except Exception:
        return {}


def log_resources(label: str):
    ram = get_ram_snapshot()
    disk = get_disk_snapshot()
    print(f"  [{label}] RAM: {ram.get('used_gb', 0):.1f} / {ram.get('total_gb', 0):.1f} GB used "
          f"({ram.get('available_gb', 0):.1f} GB avail) | "
          f"Disk: {disk.get('free_gb', 0):.1f} GB free ({disk.get('pct_used', 0):.1f}% used)")
    if ram.get("available_gb", 999) < 50:
        print(f"  WARNING: Low RAM — only {ram.get('available_gb', 0):.1f} GB available!")
    if disk.get("free_gb", 999) < 20:
        print(f"  WARNING: Low disk — only {disk.get('free_gb', 0):.1f} GB free!")


def reset_slot(slot_id: int, task_config: dict, cycle: int = 0) -> tuple:
    """POST /env/reset. Returns (slot_id, success, duration, details)."""
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
            save_screenshot_from_response(data, slot_id, f"cycle{cycle}_reset")
            return slot_id, True, dur, f"has_obs={has_obs}"
        return slot_id, False, dur, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return slot_id, False, time.time() - t0, f"ERROR: {e}"


def step_slot(slot_id: int, prediction: str, step_name: str = "step") -> tuple:
    """POST /env/step with a synthetic action. Returns (slot_id, success, duration, details)."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{SERVER_URL}/env/step",
            json={"prediction": prediction, "slot_id": slot_id},
            timeout=STEP_TIMEOUT,
        )
        dur = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            save_screenshot_from_response(data, slot_id, step_name)
            return slot_id, True, dur, f"is_done={data.get('is_done')} fmt_rwd={data.get('format_reward', 0):.2f}"
        return slot_id, False, dur, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return slot_id, False, time.time() - t0, f"ERROR: {e}"


def evaluate_slot(slot_id: int) -> tuple:
    """POST /env/evaluate. Returns (slot_id, success, duration, details)."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{SERVER_URL}/env/evaluate",
            json={"slot_id": slot_id},
            timeout=EVAL_TIMEOUT,
        )
        dur = time.time() - t0
        if r.status_code == 200:
            return slot_id, True, dur, f"score={r.json()}"
        return slot_id, False, dur, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return slot_id, False, time.time() - t0, f"ERROR: {e}"


def run_parallel(fn, args_list: list, max_workers: int, label: str) -> dict[int, bool]:
    """Run fn(slot_id, *args) in parallel, print progress, return {slot_id: success}."""
    results = {}
    total = len(args_list)
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for args in args_list:
            slot_id = args[0]
            futures[pool.submit(fn, *args)] = slot_id
        for f in as_completed(futures):
            slot_id, ok, dur, details = f.result()
            done_count += 1
            status = "OK" if ok else "FAIL"
            results[slot_id] = ok
            print(f"  [{done_count:3d}/{total}] slot {slot_id:2d}: {status}  {dur:6.1f}s  {details}")

    ok_count = sum(1 for v in results.values() if v)
    print(f"\n  {label}: {ok_count}/{total} succeeded")
    return results


def run_batched_parallel(fn, args_list: list, batch_size: int, max_workers: int, label: str) -> dict[int, bool]:
    """Run in batches to avoid thundering-herd on container creation."""
    all_results = {}
    total = len(args_list)
    for batch_start in range(0, total, batch_size):
        batch = args_list[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"\n  --- Batch {batch_num}/{total_batches} (slots {batch[0][0]}-{batch[-1][0]}) ---")
        results = run_parallel(fn, batch, max_workers=max_workers, label=f"{label} batch {batch_num}")
        all_results.update(results)
        log_resources(f"after batch {batch_num}")
    ok_count = sum(1 for v in all_results.values() if v)
    print(f"\n  {label} total: {ok_count}/{total} succeeded")
    return all_results


def run_step_sequence(slot_ids: list[int], action_names: list[str], phase_label: str, cycle: int = 0):
    """Run a sequence of step actions on all given slots, logging resources between steps."""
    t0 = time.time()
    for step_idx, action_name in enumerate(action_names):
        prediction = STEP_ACTIONS[action_name]
        step_label = f"cycle{cycle}_{action_name}"
        print(f"\n  Step {step_idx+1}/{len(action_names)}: {action_name}")
        args = [(s, prediction, step_label) for s in slot_ids]
        run_parallel(step_slot, args, max_workers=len(slot_ids), label=f"step:{action_name}")
        log_resources(f"after step {step_idx+1}")
    elapsed = time.time() - t0
    print(f"\n  {phase_label}: {len(action_names)} steps completed in {elapsed:.1f}s")
    return elapsed


def main():
    print("=" * 78)
    print(f"  64-Container Scalability Test")
    print(f"  N_SLOTS={N_SLOTS}  N_CYCLES={N_CYCLES}  BATCH={LAUNCH_BATCH_SIZE}")
    print("=" * 78)

    if not check_server():
        print("ERROR: Server not running at", SERVER_URL)
        print('Start: sg docker -c "export PROVIDER=docker && REMOTE_MAX_SLOTS=80 '
              '.venv/bin/python scripts/servers/remote_env_server.py"')
        return 1

    print("Server is up.")
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Screenshots will be saved to: {SCREENSHOT_DIR}")
    log_resources("initial")

    # Load diverse tasks for each cycle — rotate so each cycle uses different tasks
    all_cycle_tasks = []
    base_tasks = load_diverse_tasks(N_SLOTS * N_CYCLES)
    for cycle in range(N_CYCLES):
        start = cycle * N_SLOTS
        end = start + N_SLOTS
        if end <= len(base_tasks):
            all_cycle_tasks.append(base_tasks[start:end])
        else:
            # Wrap around with offset so they're still different per slot
            offset = cycle * 3  # shift by 3 each cycle
            wrapped = load_diverse_tasks(N_SLOTS)
            wrapped = wrapped[offset:] + wrapped[:offset]
            all_cycle_tasks.append(wrapped)

    for cycle_idx, cycle_tasks in enumerate(all_cycle_tasks):
        app_dist = {}
        for t in cycle_tasks:
            app = t.get("_app", "unknown")
            app_dist[app] = app_dist.get(app, 0) + 1
        print(f"  Cycle {cycle_idx+1} task distribution: {app_dist}")

    phase_results = {}  # {phase_name: (ok_count, total_count, time_s)}

    # ══════════════════════════════════════════════════════════════════════
    # CYCLE 0: Fresh container start (batched)
    # ══════════════════════════════════════════════════════════════════════
    cycle_tasks = all_cycle_tasks[0]

    print(f"\n{'='*78}")
    print(f"PHASE 1: Fresh container start + first reset (batched, {N_SLOTS} slots)")
    print(f"{'='*78}")
    log_resources("pre-phase1")
    t0 = time.time()
    args = [(i, cycle_tasks[i], 0) for i in range(N_SLOTS)]
    results = run_batched_parallel(reset_slot, args, LAUNCH_BATCH_SIZE, LAUNCH_BATCH_SIZE, "Phase 1")
    elapsed = time.time() - t0
    ok_slots = sorted(k for k, v in results.items() if v)
    phase_results["1_fresh_reset"] = (len(ok_slots), N_SLOTS, elapsed)
    log_resources("post-phase1")

    if not ok_slots:
        print("ALL SLOTS FAILED in Phase 1. Aborting.")
        return 1

    # Steps after first reset
    print(f"\n{'='*78}")
    print(f"PHASE 2: Full interaction sequence ({len(ok_slots)} slots, {len(FULL_ACTION_SEQUENCE)} actions)")
    print(f"{'='*78}")
    log_resources("pre-phase2")
    elapsed = run_step_sequence(ok_slots, FULL_ACTION_SEQUENCE, "Phase 2 steps", cycle=0)
    phase_results["2_steps_cycle1"] = (len(ok_slots), len(ok_slots), elapsed)

    # Evaluate after first cycle
    print(f"\n{'='*78}")
    print(f"PHASE 3: Evaluate cycle 1 ({len(ok_slots)} slots)")
    print(f"{'='*78}")
    log_resources("pre-phase3")
    t0 = time.time()
    args = [(s,) for s in ok_slots]
    eval_results = run_parallel(evaluate_slot, args, max_workers=len(ok_slots), label="Phase 3 evaluate")
    elapsed = time.time() - t0
    ok_eval = sum(1 for v in eval_results.values() if v)
    phase_results["3_eval_cycle1"] = (ok_eval, len(ok_slots), elapsed)
    log_resources("post-phase3")

    # ══════════════════════════════════════════════════════════════════════
    # CYCLES 1..N-1: Reset → full steps → evaluate (different tasks each time)
    # ══════════════════════════════════════════════════════════════════════
    active_slots = ok_slots  # carry forward from phase 1

    for cycle_idx in range(1, N_CYCLES):
        cycle_num = cycle_idx + 1
        cycle_tasks = all_cycle_tasks[cycle_idx]
        phase_base = cycle_idx * 3 + 1  # phase numbering: 4,5,6 then 7,8,9 ...

        # ── Reset with different task ──
        print(f"\n{'='*78}")
        print(f"PHASE {phase_base}: Reset cycle {cycle_num} with different tasks ({len(active_slots)} slots)")
        print(f"{'='*78}")
        log_resources(f"pre-cycle{cycle_num}-reset")
        t0 = time.time()
        args = [(s, cycle_tasks[s % len(cycle_tasks)], cycle_idx) for s in active_slots]
        results = run_parallel(reset_slot, args, max_workers=len(active_slots), label=f"Cycle {cycle_num} reset")
        elapsed = time.time() - t0
        ok_reset = sorted(k for k, v in results.items() if v)
        phase_results[f"{phase_base}_reset_cycle{cycle_num}"] = (len(ok_reset), len(active_slots), elapsed)
        log_resources(f"post-cycle{cycle_num}-reset")

        if not ok_reset:
            print(f"ALL SLOTS FAILED in cycle {cycle_num} reset. Aborting.")
            break

        # ── Full interaction sequence ──
        print(f"\n{'='*78}")
        print(f"PHASE {phase_base+1}: Full steps cycle {cycle_num} ({len(ok_reset)} slots, {len(FULL_ACTION_SEQUENCE)} actions)")
        print(f"{'='*78}")
        log_resources(f"pre-cycle{cycle_num}-steps")
        elapsed = run_step_sequence(ok_reset, FULL_ACTION_SEQUENCE, f"Cycle {cycle_num} steps", cycle=cycle_idx)
        phase_results[f"{phase_base+1}_steps_cycle{cycle_num}"] = (len(ok_reset), len(ok_reset), elapsed)

        # ── Evaluate ──
        print(f"\n{'='*78}")
        print(f"PHASE {phase_base+2}: Evaluate cycle {cycle_num} ({len(ok_reset)} slots)")
        print(f"{'='*78}")
        log_resources(f"pre-cycle{cycle_num}-eval")
        t0 = time.time()
        args = [(s,) for s in ok_reset]
        eval_results = run_parallel(evaluate_slot, args, max_workers=len(ok_reset), label=f"Cycle {cycle_num} evaluate")
        elapsed = time.time() - t0
        ok_eval = sum(1 for v in eval_results.values() if v)
        phase_results[f"{phase_base+2}_eval_cycle{cycle_num}"] = (ok_eval, len(ok_reset), elapsed)
        log_resources(f"post-cycle{cycle_num}-eval")

        active_slots = ok_reset  # carry forward

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("SUMMARY")
    print(f"{'='*78}")
    print(f"  Slots: {N_SLOTS}  |  Cycles: {N_CYCLES}  |  Actions/cycle: {len(FULL_ACTION_SEQUENCE)}")
    print()

    all_ok = True
    total_time = 0
    for phase_name, (ok, total, t) in phase_results.items():
        total_time += t
        status = "PASS" if ok == total else "FAIL"
        if ok != total:
            all_ok = False
        print(f"  {phase_name:30s}  {ok:3d}/{total:3d} {status}  {t:7.1f}s")

    print(f"\n  Total test time: {total_time:.1f}s ({total_time/60:.1f} min)")
    log_resources("final")

    # Screenshot summary with blank-detection
    saved = sorted(SCREENSHOT_DIR.glob("*.png"))
    print(f"\n  Screenshots saved: {len(saved)} files in {SCREENSHOT_DIR}")
    blank_count = 0
    for img_path in saved:
        size = img_path.stat().st_size
        # A blank solid-color screenshot compresses to <15KB; real content is >30KB
        if size < 15000:
            blank_count += 1
            print(f"    WARNING: {img_path.name} looks blank ({size} bytes)")
    if blank_count:
        print(f"    {blank_count}/{len(saved)} screenshots appear blank!")
    elif saved:
        print(f"    All {len(saved)} screenshots have content (OK)")

    if all_ok:
        print("\n  ALL PHASES PASSED.")
        return 0
    else:
        print("\n  SOME PHASES HAD FAILURES. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
