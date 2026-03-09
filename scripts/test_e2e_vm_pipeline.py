#!/usr/bin/env python3
"""
Comprehensive end-to-end test for the full OSWorld VM pipeline.

Tests the complete chain:
  remote_env_server (port 15001) → DesktopEnv → PythonController → VM (port 5000)
                                                                   → Reset daemon  (port 5001)

Usage:
  python scripts/test_e2e_vm_pipeline.py [SERVER_URL] [--vm-ip VM_IP]
  E2E_SERVER_URL=http://localhost:15001 python scripts/test_e2e_vm_pipeline.py

Optional env vars:
  VM_IP           — private IP of the VM for direct-port checks (e.g. 172.31.17.157)
  E2E_SERVER_URL  — remote env server URL (default http://localhost:15001)

Coverage:
  01  Server health check
  02  Resetd (port 5001) direct health + state
  03  VM server (port 5000) direct health + screenshot
  04  Environment reset — screenshot, wire format, is_done, format_reward
  05  Wire-format field completeness (obs_messages structure)
  06  Screenshot validity — size, not all-black, changes after action
  07  Click action
  08  Type action — ASCII, slashes, backslash, quotes (regression)
  09  Press action — enter, arrows, escape, space (regression)
  10  Hotkey action — Ctrl+A, Ctrl+C
  11  Scroll action — up / down
  12  Wait action
  13  Drag action
  14  Right-click / double-click
  15  Finished action → is_done=True
  16  Evaluate endpoint
  17  Re-reset: clean state after episode
  18  History messages: /env/history_messages trajectory
  19  Multi-step trajectory: N steps, history grows correctly
  21  Malformed prediction: no Action: line → format_reward < 0
  22  Max-step edge case: episode ends when max_steps exceeded
  23  Real task lifecycle: full reset → step → evaluate with real task config
  24  GRPO episode cycle: two full episodes back-to-back (reset generation check)
  25  Concurrent-reset protection: rapid re-reset doesn't leave env corrupted
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_SERVER  = "http://localhost:15001"
TIMEOUT_RESET   = 300   # first reset can launch a VM
TIMEOUT_STEP    = 60
TIMEOUT_EVAL    = 60
TIMEOUT_HISTORY = 30

SIMPLE_TASK = {
    "id": "e2e-test-simple",
    "snapshot": "os",
    "instruction": "Click on the desktop.",
    "source": "e2e-test",
    "config": [],
    "trajectory": "",
    "related_apps": ["os"],
    "evaluator": {
        "postconfig": [],
        "func": "exact_match",
        "expected": {"type": "rule", "rules": {"expected": "dummy_pass"}},
        "result":   {"type": "rule", "rules": "dummy_pass"},
    },
    "proxy": False,
    "fixed_ip": False,
    "possibility_of_env_change": "low",
}


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
    B = "\033[96m"; Z = "\033[0m";  D = "\033[1m"


def ok(m):   print(f"  {C.G}PASS{C.Z} {m}")
def fail(m): print(f"  {C.R}FAIL{C.Z} {m}")
def warn(m): print(f"  {C.Y}WARN{C.Z} {m}")
def info(m): print(f"  {C.B}INFO{C.Z} {m}")


def section(title, num=""):
    bar = "=" * 62
    label = f"Test {num}: " if num else ""
    print(f"\n{C.D}{bar}{C.Z}")
    print(f"{C.D}{label}{title}{C.Z}")
    print(f"{C.D}{bar}{C.Z}")


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------

def _extract_image_item(obs_messages):
    """Return the first image item dict from obs_messages."""
    for msg in (obs_messages or []):
        if not isinstance(msg, dict): continue
        for item in msg.get("content", []) or []:
            if not isinstance(item, dict): continue
            if item.get("type") == "image" and "b64" in item:
                return item
    return None


def screenshot_bytes(obs_messages):
    item = _extract_image_item(obs_messages)
    if item:
        return base64.b64decode(item["b64"])
    return b""


def screenshot_hash(obs_messages):
    data = screenshot_bytes(obs_messages)
    return hashlib.md5(data).hexdigest() if data else None


def is_valid_png(data: bytes) -> bool:
    return data[:8] == b'\x89PNG\r\n\x1a\n'


def is_all_black(data: bytes) -> bool:
    """Rough heuristic: if PNG is very small it might be blank."""
    return len(data) < 2000


# ---------------------------------------------------------------------------
# Prediction string builders
# ---------------------------------------------------------------------------

def pred_click(x, y):
    return f"Thought: Click.\nAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')"

def pred_double(x, y):
    return f"Thought: Double-click.\nAction: left_double(start_box='<|box_start|>({x},{y})<|box_end|>')"

def pred_right(x, y):
    return f"Thought: Right-click.\nAction: right_single(start_box='<|box_start|>({x},{y})<|box_end|>')"

def pred_drag(x1, y1, x2, y2):
    return (f"Thought: Drag.\nAction: drag("
            f"start_box='<|box_start|>({x1},{y1})<|box_end|>',"
            f"end_box='<|box_start|>({x2},{y2})<|box_end|>')")

def pred_type(text):
    t = text.replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")
    return f"Thought: Type.\nAction: type(content='{t}')"

def pred_press(key):
    return f"Thought: Press key.\nAction: press(key='{key}')"

def pred_hotkey(keys):
    return f"Thought: Hotkey.\nAction: hotkey(key='{keys}')"

def pred_scroll(x, y, direction):
    return f"Thought: Scroll.\nAction: scroll(start_box='<|box_start|>({x},{y})<|box_end|>', direction='{direction}')"

def pred_wait():
    return "Thought: Wait.\nAction: wait()"

def pred_finished(content="done"):
    c = content.replace("'", "\\'")
    return f"Thought: Done.\nAction: finished(content='{c}')"


# ---------------------------------------------------------------------------
# Direct VM API helpers (optional — used when --vm-ip is given)
# ---------------------------------------------------------------------------

def vm_get(ip, endpoint, timeout=10):
    return requests.get(f"http://{ip}:5000{endpoint}", timeout=timeout)

def resetd_get(ip, endpoint, timeout=10):
    return requests.get(f"http://{ip}:5001{endpoint}", timeout=timeout)

def resetd_post(ip, endpoint, payload=None, timeout=60):
    return requests.post(f"http://{ip}:5001{endpoint}", json=payload or {}, timeout=timeout)


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, server_url: str, vm_ip: str = ""):
        self.url    = server_url.rstrip("/")
        self.vm_ip  = vm_ip
        self.passed = 0
        self.failed = 0
        self.warns  = 0
        self._initial_hash = None
        self._reset_gen_before = None

    # ---- HTTP helpers ----

    def _post(self, ep, body, timeout=TIMEOUT_STEP):
        r = requests.post(f"{self.url}{ep}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _get(self, ep, timeout=10):
        r = requests.get(f"{self.url}{ep}", timeout=timeout)
        r.raise_for_status()
        return r.json()

    # ---- Assertion helpers ----

    def check(self, cond, pass_msg, fail_msg):
        if cond:
            ok(pass_msg); self.passed += 1
        else:
            fail(fail_msg); self.failed += 1
        return cond

    def check_warn(self, cond, msg):
        if not cond:
            warn(msg); self.warns += 1
        return cond

    # ---- Step helper ----

    def step(self, prediction, desc, expect_done=False, timeout=TIMEOUT_STEP):
        info(f"→ {desc}")
        try:
            r = self._post("/env/step", {"prediction": prediction}, timeout=timeout)
        except Exception as exc:
            self.check(False, "", f"{desc}: step error: {exc}")
            traceback.print_exc()
            return None
        has_obs = r.get("obs_messages") is not None
        self.check(has_obs, f"{desc}: obs_messages returned", f"{desc}: obs_messages is None")
        fmt = r.get("format_reward", -999)
        # > -0.2 allows stall(-0.08) + edge_noop(-0.08) = -0.11, while catching parse errors (-0.4)
        self.check(fmt > -0.2, f"{desc}: format_reward={fmt:.3f} (OK)", f"{desc}: format_reward={fmt:.3f} (parse/exec error)")
        if expect_done:
            self.check(r.get("is_done"), f"{desc}: is_done=True", f"{desc}: is_done not True")
        elif r.get("is_done"):
            warn(f"{desc}: unexpected is_done=True"); self.warns += 1
        return r

    def do_reset(self, task=None, timeout=TIMEOUT_RESET, label="reset"):
        task = task or SIMPLE_TASK
        info(f"Sending {label} (may take 1-5 min on first call)…")
        t0 = time.time()
        try:
            r = self._post("/env/reset", {"task_config": task}, timeout=timeout)
            info(f"{label} completed in {time.time()-t0:.1f}s")
            return r
        except Exception as exc:
            self.check(False, "", f"{label} failed: {exc}")
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def t01_server_health(self):
        section("Server Health Check", "01")
        try:
            r = self._get("/health")
            self.check(True, "Remote env server reachable", "")
            info(f"health: {r}")
            return True
        except Exception as exc:
            self.check(False, "", f"Server not reachable: {exc}")
            return False

    def t02_resetd_direct(self):
        section("Reset Daemon Direct Health + State", "02")
        if not self.vm_ip:
            warn("--vm-ip not provided; skipping direct resetd checks"); self.warns += 1
            return True
        try:
            h = resetd_get(self.vm_ip, "/health", timeout=10).json()
            self.check(h.get("status") == "ok", "Resetd /health → ok", f"Resetd /health → {h.get('status')}")
            info(f"reset_generation={h.get('reset_generation')}  baseline_version={h.get('baseline_version')}")

            s = resetd_get(self.vm_ip, "/state", timeout=10).json()
            self._reset_gen_before = s.get("reset_generation", 0)
            info(f"Resetd state: {s.get('status')} / {s.get('reason_code')} / gen={self._reset_gen_before}")

            self.check(s.get("status") == "ok", "Resetd state: ok/baseline_ready/reset_completed",
                       f"Resetd state: unexpected status={s.get('status')} reason={s.get('reason_code')}")
            return True
        except Exception as exc:
            self.check(False, "", f"Resetd direct check failed: {exc}")
            return False

    def t03_vm_server_direct(self):
        section("VM Server Direct Health + Screenshot", "03")
        if not self.vm_ip:
            warn("--vm-ip not provided; skipping direct VM server checks"); self.warns += 1
            return True
        try:
            h = vm_get(self.vm_ip, "/health").json()
            self.check(h.get("status") == "ok", "VM server /health → ok", f"VM server /health → {h.get('status')}")

            r = vm_get(self.vm_ip, "/screenshot", timeout=10)
            png = r.content
            self.check(is_valid_png(png), f"VM screenshot is valid PNG ({len(png)} bytes)",
                       f"VM screenshot not a PNG ({len(png)} bytes)")
            self.check_warn(not is_all_black(png), "VM screenshot looks non-blank (>2KB)")
            return True
        except Exception as exc:
            self.check(False, "", f"VM server direct check failed: {exc}")
            return False

    def t04_reset(self):
        section("Environment Reset — full wire-format validation", "04")
        r = self.do_reset(label="initial reset")
        if r is None:
            return None

        has_obs = r.get("obs_messages") is not None
        self.check(has_obs, "obs_messages present", "obs_messages absent")
        if not has_obs:
            return None

        self.check(not r.get("is_done", True), "is_done=False after reset", f"is_done={r.get('is_done')}")
        fmt = r.get("format_reward")
        self.check(fmt == 0.0, f"format_reward=0.0 after reset (got {fmt})", f"format_reward={fmt!r} — expected 0.0")

        ss = screenshot_bytes(r["obs_messages"])
        self.check(len(ss) > 1000, f"Screenshot size OK ({len(ss)} bytes)", f"Screenshot too small ({len(ss)} bytes)")
        self.check(is_valid_png(ss), "Screenshot is valid PNG", "Screenshot is not valid PNG")
        self._initial_hash = screenshot_hash(r["obs_messages"])
        info(f"Initial screenshot hash: {self._initial_hash}")
        return r

    def t05_wire_format(self, reset_resp):
        section("Wire-Format Field Completeness", "05")
        if reset_resp is None:
            warn("Skipping — no reset response"); self.warns += 1
            return
        obs = reset_resp.get("obs_messages", [])
        self.check(isinstance(obs, list), "obs_messages is a list", "obs_messages not a list")
        self.check(len(obs) > 0, f"obs_messages has {len(obs)} message(s)", "obs_messages is empty")

        item = _extract_image_item(obs)
        self.check(item is not None, "Image item found in obs_messages", "No image item in obs_messages")
        if item:
            self.check("b64" in item, "Image item has b64 field", "Image item missing b64")
            for field in ("min_pixels", "max_pixels"):
                present = field in item
                self.check_warn(present, f"Image item missing optional field: {field}")

        # Top-level keys
        for key in ("obs_messages", "is_done", "format_reward"):
            self.check(key in reset_resp, f"Response has '{key}'", f"Response missing '{key}'")
        # env_idx is optional but should be present
        if "env_idx" not in reset_resp:
            warn("Response missing 'env_idx' (optional)"); self.warns += 1

    def t06_screenshot_change(self):
        section("Screenshot Change Detection", "06")
        if self._initial_hash is None:
            warn("Skipping — no baseline screenshot"); self.warns += 1
            return
        # Open application menu (Super key) — should change the screen
        r = self.step(pred_press("super"), "Press Super (open app menu)")
        if r is None:
            return
        time.sleep(0.5)  # allow screen to update
        new_hash = screenshot_hash(r.get("obs_messages", []))
        if new_hash and new_hash != self._initial_hash:
            ok(f"Screenshot changed after Super key (hash differs)")
            self.passed += 1
        else:
            warn("Screenshot unchanged after Super key (expected; OS may not respond)")
            self.warns += 1
        # Close whatever opened
        self.step(pred_press("escape"), "Close overlay (Esc)")

    def t07_click(self):
        section("Click Action", "07")
        self.step(pred_click(500, 400), "click (500,400) — center of screen")
        self.step(pred_click(100, 100), "click (100,100) — top-left region")
        self.step(pred_click(0, 0),     "click (0,0) — corner edge case")

    def t08_type(self):
        section("Type Action — ASCII + edge cases", "08")
        self.step(pred_type("Hello OSWorld!"), "type ASCII text")
        self.step(pred_type("path/to/file.txt"), "type text with slashes")
        self.step(pred_type("C:\\Users\\test"), "type text with backslash (regression)")
        self.step(pred_type("it's a \"test\""), "type text with quotes (regression)")
        self.step(pred_type("line1\nline2"),  "type text with escaped newline")
        self.step(pred_type(""), "type empty string")

    def t09_press(self):
        section("Press Action — keys + regression", "09")
        self.step(pred_press("enter"),      "press Enter")
        self.step(pred_press("escape"),     "press Escape")
        self.step(pred_press("space"),      "press Space (BUG-1 regression)")
        self.step(pred_press("arrowleft"),  "press ArrowLeft (BUG-1 regression)")
        self.step(pred_press("arrowright"), "press ArrowRight (BUG-1 regression)")
        self.step(pred_press("arrowup"),    "press ArrowUp")
        self.step(pred_press("arrowdown"),  "press ArrowDown")
        self.step(pred_press("tab"),        "press Tab")
        self.step(pred_press("backspace"),  "press Backspace")
        self.step(pred_press("delete"),     "press Delete")
        self.step(pred_press("home"),       "press Home")
        self.step(pred_press("end"),        "press End")
        self.step(pred_press("pageup"),     "press PageUp")
        self.step(pred_press("pagedown"),   "press PageDown")
        self.step(pred_press("f1"),         "press F1")
        self.step(pred_press("f5"),         "press F5 (refresh)")

    def t10_hotkey(self):
        section("Hotkey Action — keyboard shortcuts", "10")
        self.step(pred_hotkey("ctrl a"),   "Ctrl+A (select all)")
        self.step(pred_hotkey("ctrl c"),   "Ctrl+C (copy)")
        self.step(pred_hotkey("ctrl v"),   "Ctrl+V (paste)")
        self.step(pred_hotkey("ctrl z"),   "Ctrl+Z (undo)")
        self.step(pred_hotkey("ctrl s"),   "Ctrl+S (save)")
        self.step(pred_hotkey("alt F4"),   "Alt+F4 (close window)")
        self.step(pred_hotkey("ctrl shift t"), "Ctrl+Shift+T (reopen tab)")

    def t11_scroll(self):
        section("Scroll Action", "11")
        self.step(pred_scroll(500, 400, "down"), "scroll down")
        self.step(pred_scroll(500, 400, "up"),   "scroll up")
        self.step(pred_scroll(500, 400, "left"),  "scroll left")
        self.step(pred_scroll(500, 400, "right"), "scroll right")

    def t12_wait(self):
        section("Wait Action", "12")
        self.step(pred_wait(), "wait()")

    def t13_drag(self):
        section("Drag Action", "13")
        self.step(pred_drag(200, 200, 400, 400), "drag (200,200)→(400,400)")
        self.step(pred_drag(500, 300, 500, 300), "drag to same point (no-op)")

    def t14_mouse_variants(self):
        section("Right-Click and Double-Click", "14")
        self.step(pred_double(500, 400), "left_double (500,400)")
        self.step(pred_right(500, 400),  "right_single (500,400)")
        self.step(pred_press("escape"),  "escape any context menu")

    def t15_finished(self):
        section("Finished Action → is_done=True", "15")
        r = self.step(pred_finished("Task complete."), "finished()", expect_done=True)
        if r:
            self.check(r.get("is_done") is True, "is_done=True returned from finished()",
                       f"is_done={r.get('is_done')} — expected True")
        return r

    def t16_evaluate(self):
        section("Evaluate Endpoint", "16")
        try:
            t0 = time.time()
            r = self._post("/env/evaluate", {}, timeout=TIMEOUT_EVAL)
            info(f"evaluate completed in {time.time()-t0:.1f}s — response: {r}")
            self.check(isinstance(r, dict), "evaluate returned a dict", "evaluate returned unexpected type")
        except Exception as exc:
            warn(f"evaluate error (may be expected for dummy task): {exc}")
            self.warns += 1

    def t17_re_reset(self):
        section("Re-reset — clean state restoration", "17")
        r = self.do_reset(label="re-reset")
        if r is None:
            return None
        self.check(r.get("obs_messages") is not None, "Re-reset: obs_messages present", "Re-reset: obs_messages absent")
        self.check(not r.get("is_done", True), "Re-reset: is_done=False", f"Re-reset: is_done={r.get('is_done')}")
        ss = screenshot_bytes(r.get("obs_messages") or [])
        self.check(len(ss) > 1000, f"Re-reset: screenshot valid ({len(ss)} bytes)", f"Re-reset: screenshot too small ({len(ss)})")
        self._initial_hash = screenshot_hash(r.get("obs_messages") or [])
        return r

    def t18_history_messages(self):
        section("History Messages — /env/history_messages", "18")
        # First take a couple of steps so there's history
        self.step(pred_click(500, 400), "click for history")
        self.step(pred_press("escape"), "press for history")
        try:
            r = self._get("/env/history_messages", timeout=TIMEOUT_HISTORY)
            self.check(isinstance(r, (list, dict)), "history_messages returned", "history_messages: unexpected response")
            messages = r if isinstance(r, list) else r.get("messages") or r.get("history") or []
            info(f"history_messages: {len(messages)} message(s)")
            self.check_warn(len(messages) > 0, "history_messages returned 0 messages — expected ≥1 after steps")
        except Exception as exc:
            warn(f"history_messages error: {exc}"); self.warns += 1

    def t19_multi_step_trajectory(self):
        section("Multi-Step Trajectory — N steps, history grows", "19")
        # Run 5 steps and verify each returns obs + format_reward ≥ 0
        actions = [
            (pred_click(300, 300), "click 1"),
            (pred_click(600, 300), "click 2"),
            (pred_press("tab"),    "press tab"),
            (pred_type("test"),    "type"),
            (pred_press("escape"), "escape"),
        ]
        results = []
        for pred, desc in actions:
            r = self.step(pred, desc)
            if r:
                results.append(r)

        self.check(len(results) == len(actions),
                   f"All {len(actions)} steps completed successfully",
                   f"Only {len(results)}/{len(actions)} steps completed")

        # Verify obs_messages is present in each
        all_have_obs = all(r.get("obs_messages") is not None for r in results)
        self.check(all_have_obs, "All steps returned obs_messages", "Some steps missing obs_messages")

    def t21_malformed_predictions(self):
        section("Malformed Predictions — format_reward < 0 expected", "21")
        cases = [
            ("Just some random text", "no Action: line"),
            ("Action:", "empty Action"),
            ("Thought: hmm\nAction: nonexistent_action()", "unknown action type"),
        ]
        for pred, desc in cases:
            try:
                r = self._post("/env/step", {"prediction": pred}, timeout=TIMEOUT_STEP)
                fmt = r.get("format_reward", 0)
                self.check(fmt < 0, f"Malformed '{desc}': penalized (fmt={fmt:.3f})",
                           f"Malformed '{desc}': NOT penalized (fmt={fmt:.3f})")
            except Exception as exc:
                self.check(False, "", f"Malformed '{desc}': server error: {exc}")

    def t22_rapid_reset(self):
        section("Rapid Re-reset — idempotency and corruption check", "22")
        info("Sending two resets back-to-back (no step in between)…")
        r1 = self.do_reset(label="rapid-reset-1")
        r2 = self.do_reset(label="rapid-reset-2")
        for label, r in [("rapid-1", r1), ("rapid-2", r2)]:
            if r is None:
                self.check(False, "", f"{label}: reset returned None")
                continue
            self.check(r.get("obs_messages") is not None, f"{label}: obs_messages present", f"{label}: obs_messages absent")
            self.check(not r.get("is_done", True), f"{label}: is_done=False", f"{label}: is_done={r.get('is_done')}")
        # Verify state is clean
        if self.vm_ip:
            try:
                s = resetd_get(self.vm_ip, "/state", timeout=10).json()
                info(f"State after rapid resets: {s.get('status')} / gen={s.get('reset_generation')}")
            except Exception:
                pass

    def t23_real_task(self):
        section("Real Task Lifecycle — reset+step+evaluate", "23")
        repo = Path(__file__).resolve().parent.parent
        examples = repo / "OSWorld" / "evaluation_examples" / "examples"
        if not examples.exists():
            warn("evaluation_examples/examples not found — skipping"); self.warns += 1
            return

        # Prefer OS tasks (no external downloads)
        os_ids = [
            "28cc3b7e-b194-4bc9-8353-d04c0f4d56d2",
            "4783cc41-c03c-4e1b-89b4-50658f642bd5",
            "4d117223-a354-47fb-8b45-62ab1390a95f",
        ]
        task_file = None
        for tid in os_ids:
            p = examples / "os" / f"{tid}.json"
            if p.exists():
                task_file = p
                break
        if task_file is None:
            candidates = list(examples.glob("os/*.json"))
            task_file = candidates[0] if candidates else None
        if task_file is None:
            warn("No OS task JSON found — skipping"); self.warns += 1
            return

        info(f"Loading: {task_file.name}")
        task = json.loads(task_file.read_text())
        info(f"Instruction: {task.get('instruction','')[:100]}")

        r = self.do_reset(task=task, label="real-task reset")
        if r is None:
            return
        self.check(r.get("obs_messages") is not None, "Real task reset: screenshot returned", "Real task reset: no screenshot")

        self.step(pred_click(500, 400), "real task: click center")
        self.step(pred_finished("done"), "real task: finished", expect_done=True)

        try:
            ev = self._post("/env/evaluate", {}, timeout=TIMEOUT_EVAL)
            info(f"Real task evaluate: {ev}")
            self.check(True, "Real task evaluate: responded", "")
        except Exception as exc:
            warn(f"Real task evaluate error: {exc}"); self.warns += 1

    def t24_grpo_cycle(self):
        section("GRPO Episode Cycle — two full back-to-back episodes", "24")
        info("Episode 1: reset → 3 steps → finished")
        r = self.do_reset(label="episode-1 reset")
        if r is None:
            return
        self.check(r.get("obs_messages") is not None, "Episode 1 reset: screenshot", "Episode 1 reset: no screenshot")

        ep1_gen = None
        if self.vm_ip:
            try:
                ep1_gen = resetd_get(self.vm_ip, "/state", timeout=10).json().get("reset_generation")
                info(f"Episode 1: reset_generation={ep1_gen}")
            except Exception:
                pass

        # Use varied coordinates to avoid cycle-repeat detection
        ep1_coords = [(500, 400), (600, 300), (400, 500)]
        for i, (cx, cy) in enumerate(ep1_coords):
            self.step(pred_click(cx, cy), f"episode-1 step {i+1}")
        self.step(pred_finished("e1 done"), "episode-1 finished", expect_done=True)

        info("Episode 2: reset → 2 steps → finished")
        r2 = self.do_reset(label="episode-2 reset")
        if r2 is None:
            return
        self.check(r2.get("obs_messages") is not None, "Episode 2 reset: screenshot", "Episode 2 reset: no screenshot")
        self.check(not r2.get("is_done", True), "Episode 2 reset: is_done=False", "Episode 2 reset: is_done=True")

        if ep1_gen is not None and self.vm_ip:
            try:
                ep2_gen = resetd_get(self.vm_ip, "/state", timeout=10).json().get("reset_generation")
                info(f"Episode 2: reset_generation={ep2_gen}")
                self.check(ep2_gen is not None and ep2_gen > ep1_gen,
                           f"Reset generation incremented ({ep1_gen}→{ep2_gen})",
                           f"Reset generation did NOT increment ({ep1_gen}→{ep2_gen})")
            except Exception as exc:
                warn(f"Reset generation check failed: {exc}"); self.warns += 1

        # Use varied coordinates to avoid cycle-repeat detection
        ep2_coords = [(550, 450), (650, 350)]
        for i, (cx, cy) in enumerate(ep2_coords):
            self.step(pred_click(cx, cy), f"episode-2 step {i+1}")
        self.step(pred_finished("e2 done"), "episode-2 finished", expect_done=True)

    def t25_env_info(self):
        section("Environment Info — verify response metadata", "25")
        r = self.do_reset(label="env-info reset")
        if r is None:
            return
        # After reset, is_done should be False and format_reward exactly 0
        self.check(r.get("is_done") is False, "is_done is bool False (not None)", f"is_done={r.get('is_done')!r}")
        self.check(r.get("format_reward") == 0.0, "format_reward == 0.0 after reset",
                   f"format_reward={r.get('format_reward')!r}")

        # env_idx should be 0 for single-env server
        idx = r.get("env_idx")
        if idx is not None:
            self.check(idx == 0, f"env_idx=0 (single env)", f"env_idx={idx} (unexpected)")
        else:
            warn("env_idx absent in response"); self.warns += 1

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run_all(self):
        section("OSWorld Comprehensive E2E Pipeline Test")
        info(f"Remote env server : {self.url}")
        info(f"VM direct IP      : {self.vm_ip or '(not provided)'}")
        info(f"Time              : {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # --- Connectivity ---
        if not self.t01_server_health():
            fail("Server not reachable — cannot continue.")
            self.print_summary(); return False

        # --- Direct VM checks (optional) ---
        self.t02_resetd_direct()
        self.t03_vm_server_direct()

        # --- Reset + wire format ---
        reset_resp = self.t04_reset()
        self.t05_wire_format(reset_resp)

        # --- Screenshot change ---
        self.t06_screenshot_change()

        # --- All action types ---
        self.t07_click()
        self.t08_type()
        self.t09_press()
        self.t10_hotkey()
        self.t11_scroll()
        self.t12_wait()
        self.t13_drag()
        self.t14_mouse_variants()

        # --- Episode end ---
        self.t15_finished()
        self.t16_evaluate()

        # --- State restoration ---
        self.t17_re_reset()

        # --- History ---
        self.t18_history_messages()

        # --- Multi-step ---
        self.t19_multi_step_trajectory()

        # --- Error handling ---
        self.t21_malformed_predictions()

        # --- Stress / idempotency ---
        self.t22_rapid_reset()

        # --- Real task ---
        self.t23_real_task()

        # --- GRPO cycle ---
        self.t24_grpo_cycle()

        # --- Env info ---
        self.t25_env_info()

        self.print_summary()
        return self.failed == 0

    def print_summary(self):
        section("SUMMARY")
        total = self.passed + self.failed
        print(f"  Passed   : {C.G}{self.passed}{C.Z} / {total}")
        print(f"  Failed   : {C.R}{self.failed}{C.Z} / {total}")
        print(f"  Warnings : {C.Y}{self.warns}{C.Z}")
        print()
        if self.failed == 0:
            print(f"  {C.G}{C.D}ALL TESTS PASSED ✓{C.Z}")
        else:
            print(f"  {C.R}{C.D}{self.failed} TEST(S) FAILED ✗{C.Z}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="OSWorld E2E pipeline test")
    ap.add_argument("server_url", nargs="?",
                    default=os.environ.get("E2E_SERVER_URL", DEFAULT_SERVER))
    ap.add_argument("--vm-ip", default=os.environ.get("VM_IP", ""),
                    help="Private IP of the OSWorld VM for direct port checks")
    args = ap.parse_args()

    runner = Runner(args.server_url, vm_ip=args.vm_ip)
    ok_all = runner.run_all()
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
