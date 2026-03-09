#!/usr/bin/env python3
"""
Smoke-test every task in test_data/osworld_examples/tasks/:
  For each task: POST /env/reset  →  measure timing  →  soft reset (bare reset to clean state)
  →  measure soft reset timing.

Prints per-task: OK/FAIL  category/task_id  reset_time  soft_reset_time

Warns if any reset exceeds SLOW_RESET_THRESHOLD seconds.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

SERVER = os.environ.get("E2E_SERVER_URL", "http://localhost:15001")
# 3× of the slowest observed reset (Chrome ~30-35s → cap 90s).
# LibreOffice ~8-15s (fixed), GIMP ~12s, bare ~2s.
TASK_RESET_TIMEOUT = 90    # s — 3× of Chrome's ~30s
SOFT_RESET_TIMEOUT = 15    # s — 3× of ~5s bare reset
SLOW_RESET_THRESHOLD = 45  # warn if task reset exceeds this (1.5× Chrome)
INTER_TASK_PAUSE = 0.3     # s between tasks

BARE_TASK = {
    "id": "smoke-bare",
    "snapshot": "os",
    "instruction": "Bare reset",
    "source": "smoke",
    "config": [],
    "trajectory": "",
    "related_apps": ["os"],
    "evaluator": {
        "postconfig": [],
        "func": "exact_match",
        "expected": {"type": "rule", "rules": {"expected": "dummy"}},
        "result": {"type": "rule", "rules": "dummy"},
    },
    "proxy": False,
    "fixed_ip": False,
}


def do_reset(task_config: dict, timeout: int, label: str):
    payload = json.dumps({"task_config": task_config, "slot_id": 0}).encode()
    req = urllib.request.Request(
        f"{SERVER}/env/reset",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        elapsed = time.time() - t0
        return True, elapsed, None
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        return False, elapsed, f"HTTP {e.code}: {e.read().decode()[:120]}"
    except Exception as e:
        elapsed = time.time() - t0
        return False, elapsed, str(e)[:120]


def main():
    tasks_dir = Path(__file__).resolve().parent.parent / "test_data" / "osworld_examples" / "tasks"
    if not tasks_dir.exists():
        print(f"ERROR: {tasks_dir} not found", file=sys.stderr)
        sys.exit(1)

    passed, failed, slow = [], [], []
    total_reset_time = 0.0
    total_soft_reset_time = 0.0

    for cat in sorted(os.listdir(tasks_dir)):
        cat_dir = tasks_dir / cat
        if not cat_dir.is_dir():
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith(".json"):
                continue
            task_file = cat_dir / fname
            short = f"{cat}/{fname[:8]}"

            # Load task
            try:
                with open(task_file) as f:
                    task = json.load(f)
            except Exception as e:
                print(f" FAIL {short}: load error: {e}", flush=True)
                failed.append((short, str(e)))
                continue

            # --- Task reset ---
            ok, t_reset, err = do_reset(task, TASK_RESET_TIMEOUT, f"task {short}")
            total_reset_time += t_reset

            if not ok:
                print(f" FAIL {short}: task reset failed ({t_reset:.1f}s): {err}", flush=True)
                failed.append((short, err))
                # Still do a soft reset to clean up, with longer timeout
                do_reset(BARE_TASK, SOFT_RESET_TIMEOUT * 2, f"cleanup {short}")
                time.sleep(INTER_TASK_PAUSE)
                continue

            if t_reset > SLOW_RESET_THRESHOLD:
                slow.append((short, "task", t_reset))
                warn = f"  [SLOW task_reset={t_reset:.1f}s]"
            else:
                warn = ""

            # --- Soft reset (bare clean state) ---
            ok_sr, t_sr, err_sr = do_reset(BARE_TASK, SOFT_RESET_TIMEOUT, f"soft-reset {short}")
            total_soft_reset_time += t_sr

            if not ok_sr:
                print(f" FAIL {short}: soft reset failed ({t_sr:.1f}s): {err_sr}{warn}", flush=True)
                failed.append((short, f"soft_reset: {err_sr}"))
                time.sleep(INTER_TASK_PAUSE)
                continue

            if t_sr > SLOW_RESET_THRESHOLD:
                slow.append((short, "soft", t_sr))
                warn += f"  [SLOW soft_reset={t_sr:.1f}s]"

            status = "  OK" if not warn else "SLOW"
            print(f" {status} {short}  task={t_reset:.1f}s  soft={t_sr:.1f}s{warn}", flush=True)
            passed.append((short, t_reset, t_sr))
            time.sleep(INTER_TASK_PAUSE)

    n = len(passed) + len(failed)
    print(f"\n=== Results: {len(passed)}/{n} passed ===")
    print(f"    Avg task reset : {total_reset_time/max(n,1):.1f}s")
    print(f"    Avg soft reset : {total_soft_reset_time/max(n,1):.1f}s")

    if slow:
        print(f"\n  SLOW RESETS (>{SLOW_RESET_THRESHOLD}s):")
        for s, kind, t in slow:
            print(f"    {s}  [{kind}]  {t:.1f}s")

    if failed:
        print("\n  FAILURES:")
        for name, err in failed:
            print(f"    {name}: {err}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
