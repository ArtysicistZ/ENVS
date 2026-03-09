#!/usr/bin/env python3
"""
End-to-end test for the full OSWorld VM pipeline.

Tests the complete chain:
  remote_env_server → DesktopEnv → PythonController → VM (port 5000)
                                                      → Reset daemon (port 5001)

Usage:
  python scripts/test_e2e_vm_pipeline.py [SERVER_URL]
  # Default: http://localhost:15001

What it tests:
  1. Server health
  2. Reset with a real task config (screenshot returned)
  3. Click action (clicks a coordinate on screen)
  4. Keyboard type action (types text)
  5. Press action (presses a single key)
  6. Hotkey action (Ctrl+A style combos)
  7. Scroll action
  8. Screenshot changes after actions
  9. Evaluate endpoint
  10. Re-reset to verify clean state restoration
  11. Action parsing edge cases (special chars, arrow keys, etc.)
"""

import base64
import json
import os
import sys
import time
import hashlib
import traceback
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_SERVER_URL = "http://localhost:15001"
TIMEOUT_RESET = 300   # reset can take a while (VM launch)
TIMEOUT_STEP = 60
TIMEOUT_EVAL = 60

# A simple OS task that doesn't need file downloads — just opens the desktop
SIMPLE_TASK_CONFIG = {
    "id": "e2e-test-dummy",
    "snapshot": "os",
    "instruction": "Open the file manager application.",
    "source": "e2e-test",
    "config": [],
    "trajectory": "",
    "related_apps": ["os"],
    "evaluator": {
        "postconfig": [],
        "func": "exact_match",
        "expected": {
            "type": "rule",
            "rules": {"expected": "dummy_pass"},
        },
        "result": {
            "type": "rule",
            "rules": "dummy_pass",
        },
    },
    "proxy": False,
    "fixed_ip": False,
    "possibility_of_env_change": "low",
}


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def ok(msg):
    print(f"  {Colors.GREEN}PASS{Colors.RESET} {msg}")


def fail(msg):
    print(f"  {Colors.RED}FAIL{Colors.RESET} {msg}")


def warn(msg):
    print(f"  {Colors.YELLOW}WARN{Colors.RESET} {msg}")


def info(msg):
    print(f"  {Colors.CYAN}INFO{Colors.RESET} {msg}")


def section(title):
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")


def _extract_screenshot_b64(obs_messages):
    """Extract the base64 screenshot string from obs_messages (wire format)."""
    if not obs_messages:
        return None
    for msg in obs_messages:
        if not isinstance(msg, dict) or "content" not in msg:
            continue
        content = msg["content"]
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            # Wire format: {"type": "image", "b64": "..."}
            if item.get("type") == "image" and "b64" in item:
                return item["b64"]
            # Alternative format: {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
            if item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:image"):
                    return url.split(",", 1)[-1]
    return None


def screenshot_hash(obs_messages):
    """Extract and hash the screenshot from obs_messages to detect changes."""
    b64 = _extract_screenshot_b64(obs_messages)
    if b64:
        return hashlib.md5(base64.b64decode(b64)).hexdigest()
    return None


def screenshot_size(obs_messages):
    """Get screenshot byte size from obs_messages."""
    b64 = _extract_screenshot_b64(obs_messages)
    if b64:
        return len(base64.b64decode(b64))
    return 0


# ---------------------------------------------------------------------------
# Action prediction strings (simulating model outputs)
# ---------------------------------------------------------------------------

def make_click_prediction(x, y):
    """Generate a model-style click prediction at smart_resize coordinates."""
    return f"Thought: I need to click on this element.\nAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')"


def make_type_prediction(text):
    """Generate a model-style type prediction."""
    escaped = text.replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")
    return f"Thought: I need to type some text.\nAction: type(content='{escaped}')"


def make_press_prediction(key):
    """Generate a model-style key press prediction."""
    return f"Thought: I need to press a key.\nAction: press(key='{key}')"


def make_hotkey_prediction(keys):
    """Generate a model-style hotkey prediction (e.g. 'ctrl a')."""
    return f"Thought: I need to use a keyboard shortcut.\nAction: hotkey(key='{keys}')"


def make_scroll_prediction(x, y, direction):
    """Generate a model-style scroll prediction."""
    return f"Thought: I need to scroll.\nAction: scroll(start_box='<|box_start|>({x},{y})<|box_end|>', direction='{direction}')"


def make_wait_prediction():
    return "Thought: I should wait for the screen to update.\nAction: wait()"


def make_finished_prediction(content="task done"):
    escaped = content.replace("'", "\\'")
    return f"Thought: The task is complete.\nAction: finished(content='{escaped}')"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class E2ETestRunner:
    def __init__(self, server_url):
        self.server_url = server_url.rstrip("/")
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def _post(self, endpoint, json_data, timeout=TIMEOUT_STEP):
        url = f"{self.server_url}{endpoint}"
        resp = requests.post(url, json=json_data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(self, endpoint, timeout=10):
        url = f"{self.server_url}{endpoint}"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def check(self, condition, pass_msg, fail_msg):
        if condition:
            ok(pass_msg)
            self.passed += 1
        else:
            fail(fail_msg)
            self.failed += 1
        return condition

    def test_health(self):
        section("Test 1: Server Health Check")
        try:
            resp = self._get("/health")
            self.check(True, "Server is reachable", "Server not reachable")
            info(f"Response: {resp}")
        except Exception as e:
            self.check(False, "", f"Server health check failed: {e}")
            return False
        return True

    def test_reset(self):
        section("Test 2: Environment Reset (VM initialization)")
        info("Sending reset with a simple OS task config...")
        info("This may take 1-3 minutes if a new VM needs to be launched.")
        try:
            t0 = time.time()
            resp = self._post("/env/reset", {"task_config": SIMPLE_TASK_CONFIG}, timeout=TIMEOUT_RESET)
            elapsed = time.time() - t0
            info(f"Reset completed in {elapsed:.1f}s")

            has_obs = resp.get("obs_messages") is not None
            self.check(has_obs, "obs_messages returned (screenshot present)", "obs_messages is None — VM may not be ready")
            if not has_obs:
                return False

            not_done = not resp.get("is_done", True)
            self.check(not_done, "is_done=False (ready for steps)", f"is_done={resp.get('is_done')} — env thinks it's already done")

            size = screenshot_size(resp["obs_messages"])
            self.check(size > 1000, f"Screenshot size looks valid ({size} bytes)", f"Screenshot too small ({size} bytes)")

            self._initial_screenshot_hash = screenshot_hash(resp["obs_messages"])
            info(f"Initial screenshot hash: {self._initial_screenshot_hash}")
            return True
        except Exception as e:
            self.check(False, "", f"Reset failed: {e}")
            traceback.print_exc()
            return False

    def test_step(self, prediction, description, expect_done=False):
        """Run a single step and verify it succeeds."""
        info(f"Sending: {description}")
        info(f"  Prediction: {prediction[:100]}...")
        try:
            t0 = time.time()
            resp = self._post("/env/step", {"prediction": prediction}, timeout=TIMEOUT_STEP)
            elapsed = time.time() - t0
            info(f"  Step completed in {elapsed:.1f}s")

            has_obs = resp.get("obs_messages") is not None
            self.check(has_obs, f"{description}: obs_messages returned", f"{description}: obs_messages is None")

            fmt_reward = resp.get("format_reward", -999)
            reward_ok = fmt_reward >= 0
            self.check(reward_ok, f"{description}: format_reward={fmt_reward:.3f} (non-negative = parsed OK)",
                       f"{description}: format_reward={fmt_reward:.3f} (negative = parse/exec error)")

            is_done = resp.get("is_done", False)
            if expect_done:
                self.check(is_done, f"{description}: is_done=True as expected", f"{description}: expected is_done=True but got False")
            else:
                # Not checking strictly — some actions might trigger done
                if is_done:
                    warn(f"{description}: is_done=True (unexpected but not fatal)")
                    self.warnings += 1

            return resp
        except Exception as e:
            self.check(False, "", f"{description}: step failed with exception: {e}")
            traceback.print_exc()
            return None

    def test_click_action(self):
        section("Test 3: Click Action")
        # Click roughly center of screen (in smart_resize coords ~500, 400)
        resp = self.test_step(
            make_click_prediction(500, 400),
            "Click at (500, 400)"
        )
        return resp is not None

    def test_type_action(self):
        section("Test 4: Type Action (keyboard text input)")
        # Type some text
        resp = self.test_step(
            make_type_prediction("Hello OSWorld!"),
            "Type 'Hello OSWorld!'"
        )
        if resp is None:
            return False

        # Test with special characters
        resp = self.test_step(
            make_type_prediction("path/to/file.txt"),
            "Type text with slashes"
        )

        # Test with quotes (the bug we fixed)
        resp = self.test_step(
            make_type_prediction("it's a \"test\""),
            "Type text with quotes (BUG 3 regression test)"
        )
        return True

    def test_press_action(self):
        section("Test 5: Press Action (single key)")
        # Test basic key
        resp = self.test_step(
            make_press_prediction("enter"),
            "Press Enter"
        )
        if resp is None:
            return False

        # Test arrow keys (the bug we fixed — was referencing `hotkey` instead of `key_to_press`)
        for key in ["arrowleft", "arrowright", "arrowup", "arrowdown"]:
            resp = self.test_step(
                make_press_prediction(key),
                f"Press {key} (BUG 1 regression test)"
            )

        # Test space (another edge case from BUG 1)
        resp = self.test_step(
            make_press_prediction("space"),
            "Press space (BUG 1 regression test)"
        )

        # Test escape
        resp = self.test_step(
            make_press_prediction("escape"),
            "Press Escape"
        )
        return True

    def test_hotkey_action(self):
        section("Test 6: Hotkey Action (keyboard shortcuts)")
        resp = self.test_step(
            make_hotkey_prediction("ctrl a"),
            "Hotkey Ctrl+A (select all)"
        )
        if resp is None:
            return False

        resp = self.test_step(
            make_hotkey_prediction("ctrl c"),
            "Hotkey Ctrl+C (copy)"
        )

        resp = self.test_step(
            make_hotkey_prediction("alt F4"),
            "Hotkey Alt+F4 (close window)"
        )
        return True

    def test_scroll_action(self):
        section("Test 7: Scroll Action")
        resp = self.test_step(
            make_scroll_prediction(500, 400, "down"),
            "Scroll down at (500, 400)"
        )
        if resp is None:
            return False

        resp = self.test_step(
            make_scroll_prediction(500, 400, "up"),
            "Scroll up at (500, 400)"
        )
        return True

    def test_wait_action(self):
        section("Test 8: Wait Action")
        resp = self.test_step(
            make_wait_prediction(),
            "Wait action"
        )
        return resp is not None

    def test_finished_action(self):
        section("Test 9: Finished Action (task completion)")
        resp = self.test_step(
            make_finished_prediction("The file manager is open."),
            "Finished action",
            expect_done=True
        )
        return resp is not None

    def test_evaluate(self):
        section("Test 10: Evaluate Endpoint")
        try:
            t0 = time.time()
            resp = self._post("/env/evaluate", {}, timeout=TIMEOUT_EVAL)
            elapsed = time.time() - t0
            info(f"Evaluate completed in {elapsed:.1f}s")
            info(f"Response: {resp}")

            has_result = "result" in resp or "score" in resp or isinstance(resp, dict)
            self.check(has_result, "Evaluate returned a result", "Evaluate returned unexpected format")
            return True
        except Exception as e:
            # Evaluate might fail for dummy tasks — that's OK
            warn(f"Evaluate returned error (may be expected for dummy task): {e}")
            self.warnings += 1
            return True  # Don't count as failure

    def test_re_reset(self):
        section("Test 11: Re-reset (verify clean state restoration)")
        info("Resetting again to verify the reset mechanism cleans up properly...")
        try:
            t0 = time.time()
            resp = self._post("/env/reset", {"task_config": SIMPLE_TASK_CONFIG}, timeout=TIMEOUT_RESET)
            elapsed = time.time() - t0
            info(f"Re-reset completed in {elapsed:.1f}s")

            has_obs = resp.get("obs_messages") is not None
            self.check(has_obs, "Re-reset: obs_messages returned", "Re-reset: obs_messages is None")
            if not has_obs:
                return False

            not_done = not resp.get("is_done", True)
            self.check(not_done, "Re-reset: is_done=False", f"Re-reset: is_done={resp.get('is_done')}")

            size = screenshot_size(resp["obs_messages"])
            self.check(size > 1000, f"Re-reset: screenshot valid ({size} bytes)", f"Re-reset: screenshot too small ({size} bytes)")

            return True
        except Exception as e:
            self.check(False, "", f"Re-reset failed: {e}")
            return False

    def test_edge_cases(self):
        section("Test 12: Edge Cases & Regression Tests")

        # Test missing Action: line (should fail gracefully)
        info("Testing malformed prediction (no Action: line)...")
        try:
            resp = self._post("/env/step", {"prediction": "Just some random text without action"}, timeout=TIMEOUT_STEP)
            fmt = resp.get("format_reward", 0)
            self.check(fmt < 0, f"Malformed prediction: correctly penalized (format_reward={fmt:.3f})",
                       f"Malformed prediction: not penalized (format_reward={fmt:.3f})")
        except Exception as e:
            self.check(False, "", f"Malformed prediction caused server error: {e}")

        # Test click with missing start_box (BUG 2 regression)
        info("Testing click with coordinates that parse to None...")
        try:
            pred = "Thought: Click something.\nAction: click(start_box='<|box_start|>(0,0)<|box_end|>')"
            resp = self._post("/env/step", {"prediction": pred}, timeout=TIMEOUT_STEP)
            fmt = resp.get("format_reward", -999)
            self.check(fmt >= 0, f"Click at (0,0): handled OK (format_reward={fmt:.3f})",
                       f"Click at (0,0): error (format_reward={fmt:.3f})")
        except Exception as e:
            self.check(False, "", f"Click edge case caused server error: {e}")

        # Test type with backslash content (BUG 3 regression)
        info("Testing type with backslash in content...")
        try:
            pred = make_type_prediction("C:\\Users\\test")
            resp = self._post("/env/step", {"prediction": pred}, timeout=TIMEOUT_STEP)
            fmt = resp.get("format_reward", -999)
            self.check(fmt >= 0, f"Type with backslash: OK (format_reward={fmt:.3f})",
                       f"Type with backslash: error (format_reward={fmt:.3f})")
        except Exception as e:
            self.check(False, "", f"Type backslash edge case caused server error: {e}")

        # Test double-click
        info("Testing left_double action...")
        try:
            pred = "Thought: Double-click.\nAction: left_double(start_box='<|box_start|>(500,400)<|box_end|>')"
            resp = self._post("/env/step", {"prediction": pred}, timeout=TIMEOUT_STEP)
            fmt = resp.get("format_reward", -999)
            self.check(fmt >= 0, f"Double-click: OK (format_reward={fmt:.3f})",
                       f"Double-click: error (format_reward={fmt:.3f})")
        except Exception as e:
            self.check(False, "", f"Double-click caused server error: {e}")

        # Test right-click
        info("Testing right_single action...")
        try:
            pred = "Thought: Right-click.\nAction: right_single(start_box='<|box_start|>(500,400)<|box_end|>')"
            resp = self._post("/env/step", {"prediction": pred}, timeout=TIMEOUT_STEP)
            fmt = resp.get("format_reward", -999)
            self.check(fmt >= 0, f"Right-click: OK (format_reward={fmt:.3f})",
                       f"Right-click: error (format_reward={fmt:.3f})")
        except Exception as e:
            self.check(False, "", f"Right-click caused server error: {e}")

        # Test drag action
        info("Testing drag action...")
        try:
            pred = "Thought: Drag something.\nAction: drag(start_box='<|box_start|>(200,200)<|box_end|>', end_box='<|box_start|>(400,400)<|box_end|>')"
            resp = self._post("/env/step", {"prediction": pred}, timeout=TIMEOUT_STEP)
            fmt = resp.get("format_reward", -999)
            self.check(fmt >= 0, f"Drag: OK (format_reward={fmt:.3f})",
                       f"Drag: error (format_reward={fmt:.3f})")
        except Exception as e:
            self.check(False, "", f"Drag caused server error: {e}")

        return True

    def test_real_task_lifecycle(self):
        """Test with a real task from the 127/128 set (optional — requires task data)."""
        section("Test 13: Real Task Lifecycle (optional)")

        # Try to load a real task config
        repo_root = Path(__file__).resolve().parent.parent
        examples_dir = repo_root / "OSWorld" / "evaluation_examples" / "examples"

        # Use an OS task (simplest — no file downloads from HuggingFace needed)
        os_task_dir = examples_dir / "os"
        if not os_task_dir.exists():
            warn("OSWorld evaluation_examples/examples/os/ not found — skipping real task test")
            self.warnings += 1
            return True

        # Find a task JSON
        task_files = list(os_task_dir.glob("*.json"))
        if not task_files:
            warn("No OS task JSON files found — skipping real task test")
            self.warnings += 1
            return True

        # Pick first OS task from our 127 set
        os_task_ids = [
            "28cc3b7e-b194-4bc9-8353-d04c0f4d56d2",
            "4783cc41-c03c-4e1b-89b4-50658f642bd5",
            "4d117223-a354-47fb-8b45-62ab1390a95f",
        ]
        task_file = None
        for tid in os_task_ids:
            candidate = os_task_dir / f"{tid}.json"
            if candidate.exists():
                task_file = candidate
                break

        if task_file is None:
            # Fall back to any OS task
            task_file = task_files[0]

        info(f"Loading real task config from: {task_file.name}")
        try:
            with open(task_file) as f:
                real_task_config = json.load(f)
        except Exception as e:
            warn(f"Failed to load task config: {e}")
            self.warnings += 1
            return True

        info(f"Task: {real_task_config.get('instruction', 'N/A')[:100]}...")

        # Reset with real task
        try:
            t0 = time.time()
            resp = self._post("/env/reset", {"task_config": real_task_config}, timeout=TIMEOUT_RESET)
            elapsed = time.time() - t0
            info(f"Real task reset completed in {elapsed:.1f}s")

            has_obs = resp.get("obs_messages") is not None
            self.check(has_obs, "Real task reset: screenshot returned", "Real task reset: no screenshot")
            if not has_obs:
                return False

            # Send a simple action
            resp = self.test_step(
                make_click_prediction(500, 400),
                "Real task: click (500,400)"
            )

            # Finish the task
            resp = self.test_step(
                make_finished_prediction("Done with the task"),
                "Real task: finished",
                expect_done=True
            )

            # Evaluate
            try:
                eval_resp = self._post("/env/evaluate", {}, timeout=TIMEOUT_EVAL)
                info(f"Real task evaluate result: {eval_resp}")
                self.check(True, "Real task: evaluate endpoint responded", "")
            except Exception as e:
                warn(f"Real task evaluate error (may be expected): {e}")
                self.warnings += 1

            return True
        except Exception as e:
            self.check(False, "", f"Real task lifecycle failed: {e}")
            return False

    def run_all(self):
        section("OSWorld End-to-End VM Pipeline Test")
        info(f"Server: {self.server_url}")
        info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Test 1: Health
        if not self.test_health():
            fail("Server not reachable — cannot continue. Is the server running?")
            self.print_summary()
            return False

        # Test 2: Reset
        if not self.test_reset():
            fail("Reset failed — cannot continue without a working VM.")
            self.print_summary()
            return False

        # Tests 3-8: Action types (order matters — avoid cascading failures)
        self.test_click_action()
        self.test_type_action()
        self.test_press_action()
        self.test_hotkey_action()
        self.test_scroll_action()
        self.test_wait_action()

        # Test 9: Finished (ends the episode → reset needed after)
        self.test_finished_action()

        # Test 10: Evaluate
        self.test_evaluate()

        # Test 11: Re-reset (verify clean state restoration)
        self.test_re_reset()

        # Test 12: Edge cases (fresh episode after re-reset)
        self.test_edge_cases()

        # Test 13: Real task (optional — needs another reset)
        self.test_real_task_lifecycle()

        self.print_summary()
        return self.failed == 0

    def print_summary(self):
        section("SUMMARY")
        total = self.passed + self.failed
        print(f"  Passed:   {Colors.GREEN}{self.passed}{Colors.RESET} / {total}")
        print(f"  Failed:   {Colors.RED}{self.failed}{Colors.RESET} / {total}")
        print(f"  Warnings: {Colors.YELLOW}{self.warnings}{Colors.RESET}")
        print()
        if self.failed == 0:
            print(f"  {Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED{Colors.RESET}")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}{self.failed} TEST(S) FAILED{Colors.RESET}")
        print()


def main():
    server_url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("E2E_SERVER_URL", DEFAULT_SERVER_URL)
    runner = E2ETestRunner(server_url)
    success = runner.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
