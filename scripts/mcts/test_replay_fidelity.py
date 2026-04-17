#!/usr/bin/env python3
"""
Comprehensive VM Replay Fidelity & Speed Test.

Tests whether replaying a sequence of actions on a fresh VM produces the
same visual state as executing those actions step-by-step on the original VM.

Tests performed:
  1. Step-by-step execution on VM-A vs full replay on VM-B → per-step screenshot comparison
  2. Replay speed benchmarks with different pause intervals (0.5s, 1.0s, 2.0s)
  3. Partial replay fidelity (replay first N steps, compare)
  4. Replay-then-continue: replay N steps, execute 1 more, compare to VM-A at step N+1

Usage:
    python scripts/mcts/test_replay_fidelity.py \
        --server-url http://10.100.4.7:15001 \
        --task-id 7a4deb26-d57d-4ea9-9a73-630f66a7b568 \
        --vm-a 0 --vm-b 1

    # Use a trajectory from file (skip live execution on VM-A):
    python scripts/mcts/test_replay_fidelity.py \
        --server-url http://10.100.4.7:15001 \
        --task-id 7a4deb26-d57d-4ea9-9a73-630f66a7b568 \
        --trajectory-file docs/research/mcts_smoke_test_success.jsonl \
        --trajectory-index 2 \
        --vm-a 0 --vm-b 1

    # Multi-VM replay (test replay on 3 VMs simultaneously):
    python scripts/mcts/test_replay_fidelity.py \
        --server-url http://10.100.4.7:15001 \
        --task-id 7a4deb26-d57d-4ea9-9a73-630f66a7b568 \
        --trajectory-file docs/research/mcts_smoke_test_success.jsonl \
        --trajectory-index 2 \
        --vm-a 0 --vm-b 1 --extra-vms 2 3 4
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "OSWorld"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("replay_fidelity")


# ---------------------------------------------------------------------------
# Image comparison utilities
# ---------------------------------------------------------------------------

def decode_screenshot(b64_str: str) -> "Image.Image":
    """Decode a base64 JPEG/PNG string to a PIL Image."""
    from PIL import Image
    # Handle data URI prefix
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def screenshot_similarity(img_a: "Image.Image", img_b: "Image.Image") -> Dict[str, float]:
    """Compute multiple similarity metrics between two screenshots.

    Returns dict with:
      - pixel_match_pct: percentage of exactly-matching pixels
      - mse: mean squared error (lower=better)
      - ssim: structural similarity index (higher=better, max=1.0)
      - psnr: peak signal-to-noise ratio in dB (higher=better)
    """
    import numpy as np

    # Resize to same dimensions if needed
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size)

    a = np.array(img_a, dtype=np.float64)
    b = np.array(img_b, dtype=np.float64)

    # Pixel-exact match percentage
    exact_match = np.all(a == b, axis=-1)  # per-pixel RGB match
    pixel_match_pct = float(exact_match.mean() * 100)

    # MSE
    diff = a - b
    mse = float(np.mean(diff ** 2))

    # PSNR
    if mse < 1e-10:
        psnr = float("inf")
    else:
        psnr = float(10 * np.log10(255.0 ** 2 / mse))

    # SSIM (simplified, per-channel average)
    ssim_val = _compute_ssim(a, b)

    return {
        "pixel_match_pct": round(pixel_match_pct, 2),
        "mse": round(mse, 4),
        "ssim": round(ssim_val, 6),
        "psnr": round(psnr, 2),
    }


def _compute_ssim(a, b, window_size=11):
    """Compute SSIM between two numpy arrays (H, W, C)."""
    import numpy as np

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Average across channels
    ssim_channels = []
    for c in range(a.shape[2]):
        ac, bc = a[:, :, c], b[:, :, c]

        mu_a = _uniform_filter(ac, window_size)
        mu_b = _uniform_filter(bc, window_size)

        sigma_a_sq = _uniform_filter(ac ** 2, window_size) - mu_a ** 2
        sigma_b_sq = _uniform_filter(bc ** 2, window_size) - mu_b ** 2
        sigma_ab = _uniform_filter(ac * bc, window_size) - mu_a * mu_b

        numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a_sq + sigma_b_sq + C2)

        ssim_map = numerator / denominator
        ssim_channels.append(float(ssim_map.mean()))

    return float(sum(ssim_channels) / len(ssim_channels))


def _uniform_filter(arr, size):
    """Simple uniform (box) filter using cumulative sum for speed."""
    import numpy as np
    from scipy.ndimage import uniform_filter
    return uniform_filter(arr, size=size, mode="reflect")


def save_comparison_image(img_a, img_b, path, title=""):
    """Save a side-by-side comparison image with diff overlay."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    # Resize to same size
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size)

    w, h = img_a.size

    # Create diff image (amplified)
    a_arr = np.array(img_a, dtype=np.float64)
    b_arr = np.array(img_b, dtype=np.float64)
    diff = np.abs(a_arr - b_arr)
    # Amplify differences for visibility (5x)
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
    diff_img = Image.fromarray(diff_amplified)

    # Create side-by-side: [VM-A | VM-B | Diff×5]
    margin = 10
    label_h = 30
    canvas_w = w * 3 + margin * 4
    canvas_h = h + margin * 2 + label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))

    # Paste images
    canvas.paste(img_a, (margin, label_h + margin))
    canvas.paste(img_b, (w + margin * 2, label_h + margin))
    canvas.paste(diff_img, (w * 2 + margin * 3, label_h + margin))

    # Add labels
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    labels = [
        (margin, f"VM-A (original)"),
        (w + margin * 2, f"VM-B (replay)"),
        (w * 2 + margin * 3, f"Diff ×5"),
    ]
    for x, label in labels:
        draw.text((x, 5), label, fill=(255, 255, 255), font=font)

    if title:
        draw.text((canvas_w // 2 - 100, 5), title, fill=(255, 200, 0), font=font)

    canvas.save(path)


# ---------------------------------------------------------------------------
# Env client helpers (non-Ray, synchronous for this test)
# ---------------------------------------------------------------------------

@dataclass
class ReplayTestResult:
    """Result of a single replay comparison."""
    step: int
    similarity: Dict[str, float]
    vm_a_time: float  # seconds to execute on VM-A
    vm_b_time: float  # seconds to replay on VM-B
    action: str


@dataclass
class ReplayBenchmark:
    """Timing result for a full replay at a given pause interval."""
    pause_sec: float
    n_steps: int
    total_time: float
    per_step_time: float
    success: bool


class EnvClient:
    """Synchronous env client for testing (no Ray)."""

    RESET_TIMEOUT = 300
    STEP_TIMEOUT = 60

    def __init__(self, server_url: str, slot_id: int):
        self.server_url = server_url.rstrip("/")
        self.slot_id = slot_id

    def _post(self, path: str, body: dict, timeout: float = 60) -> dict:
        import requests
        url = f"{self.server_url}{path}"
        r = requests.post(url, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def reset(self, task_config: dict) -> dict:
        return self._post("/env/reset", {"task_config": task_config, "slot_id": self.slot_id},
                          timeout=self.RESET_TIMEOUT)

    def step(self, prediction: str) -> dict:
        return self._post("/env/step", {"prediction": prediction, "slot_id": self.slot_id},
                          timeout=self.STEP_TIMEOUT)

    def get_screenshot(self) -> Optional[str]:
        """Get current screenshot as base64."""
        resp = self._post("/env/history_messages", {"slot_id": self.slot_id}, timeout=30)
        messages = resp.get("messages") or resp.get("history_messages", [])
        for msg in reversed(messages):
            content = msg.get("content", [])
            if isinstance(content, list):
                for c in reversed(content):
                    if isinstance(c, dict) and c.get("type") == "image":
                        b64 = c.get("b64", "")
                        if b64:
                            return b64
                        img_str = c.get("image", "")
                        if img_str.startswith("data:image"):
                            return img_str.split(",", 1)[1]
        return None

    def evaluate(self) -> dict:
        return self._post("/env/evaluate", {"slot_id": self.slot_id}, timeout=350)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def load_task_config(task_id: str) -> dict:
    """Load task config from OSWorld examples."""
    task_file = os.path.join(PROJ_ROOT,
                             "OSWorld/evaluation_examples/test_all_300tasks_noproxy_softreset_clean.json")
    with open(task_file) as f:
        raw = json.load(f)

    for domain, task_ids in raw.items():
        if task_id in task_ids:
            cfg_path = os.path.join(PROJ_ROOT, "OSWorld/evaluation_examples/examples",
                                    domain, task_id + ".json")
            with open(cfg_path) as f:
                tc = json.load(f)
            tc["domain"] = domain
            tc["id"] = task_id
            tc["task_id"] = task_id
            return tc

    raise ValueError(f"Task {task_id} not found in task file")


def load_trajectory_actions(trajectory_file: str, trajectory_index: int) -> List[str]:
    """Load action strings from a trajectory JSONL file."""
    path = os.path.join(PROJ_ROOT, trajectory_file)
    with open(path) as f:
        for i, line in enumerate(f):
            if i == trajectory_index:
                d = json.loads(line)
                actions = [s["action"] for s in d.get("steps", [])]
                logger.info("Loaded %d actions from trajectory %d (task=%s, eval=%.1f)",
                            len(actions), i, d.get("task_id", "?"), d.get("eval_result", 0))
                return actions
    raise ValueError(f"Trajectory index {trajectory_index} not found in {trajectory_file}")


def test_stepwise_vs_replay(
    task_config: dict,
    actions: List[str],
    client_a: EnvClient,
    client_b: EnvClient,
    output_dir: str,
    action_pause: float = 1.0,
) -> List[ReplayTestResult]:
    """Test 1: Execute step-by-step on VM-A, replay on VM-B, compare per-step screenshots.

    Protocol:
      1. Reset both VMs to the same task
      2. On VM-A: execute actions one at a time, screenshot after each
      3. Reset VM-B again, then replay ALL actions at once, screenshot after each
      4. Compare screenshots at each step
    """
    logger.info("=" * 70)
    logger.info("TEST 1: Step-by-step vs Replay — per-step comparison")
    logger.info("  Actions: %d, Pause: %.1fs", len(actions), action_pause)
    logger.info("  VM-A (slot %d): step-by-step execution", client_a.slot_id)
    logger.info("  VM-B (slot %d): sequential replay", client_b.slot_id)
    logger.info("=" * 70)

    # Filter out terminal actions (finished/fail) — they don't change VM state
    exec_actions = []
    for a in actions:
        if "Action:" in a:
            act_str = a.split("Action:")[-1].strip().split("\n")[0].strip()
            import re
            m = re.match(r"(\w+)\(", act_str)
            if m and m.group(1) in ("finished", "fail"):
                logger.info("  Skipping terminal action: %s", m.group(1))
                continue
        exec_actions.append(a)

    n_actions = len(exec_actions)
    logger.info("  Executable actions (non-terminal): %d", n_actions)

    # ---- Phase 1: Reset both VMs ----
    logger.info("\n[Phase 1] Resetting both VMs...")
    t0 = time.time()
    client_a.reset(task_config)
    client_b.reset(task_config)
    reset_time = time.time() - t0
    logger.info("  Reset done in %.1fs", reset_time)

    # Take initial screenshots (after reset, before any action)
    time.sleep(2.0)  # wait for desktop to stabilize
    ss_a_init = client_a.get_screenshot()
    ss_b_init = client_b.get_screenshot()
    if ss_a_init and ss_b_init:
        img_a_init = decode_screenshot(ss_a_init)
        img_b_init = decode_screenshot(ss_b_init)
        sim_init = screenshot_similarity(img_a_init, img_b_init)
        logger.info("  Initial state similarity (post-reset): %s", sim_init)
        save_comparison_image(img_a_init, img_b_init,
                              os.path.join(output_dir, "step_00_initial.png"),
                              title="Initial (post-reset)")
    else:
        logger.warning("  Could not get initial screenshots")

    # ---- Phase 2: Execute on VM-A step-by-step, collecting screenshots ----
    logger.info("\n[Phase 2] Executing %d actions on VM-A step-by-step...", n_actions)
    screenshots_a = []
    step_times_a = []

    for i, action in enumerate(exec_actions):
        act_preview = action.split("Action:")[-1].strip()[:60] if "Action:" in action else action[:60]
        logger.info("  VM-A step %d/%d: %s", i + 1, n_actions, act_preview)

        t_step = time.time()
        client_a.step(action)
        step_time = time.time() - t_step
        step_times_a.append(step_time)

        # Wait for UI to settle, then screenshot
        time.sleep(action_pause)
        ss = client_a.get_screenshot()
        screenshots_a.append(ss)
        logger.info("    executed in %.2fs, screenshot: %s",
                     step_time, "OK" if ss else "FAILED")

    total_a_time = sum(step_times_a) + action_pause * n_actions
    logger.info("  VM-A total: %.1fs (%.2fs avg/step + %.1fs pause)",
                total_a_time, sum(step_times_a) / max(1, n_actions), action_pause)

    # ---- Phase 3: Reset VM-B and replay all actions ----
    logger.info("\n[Phase 3] Resetting VM-B and replaying %d actions...", n_actions)
    client_b.reset(task_config)
    time.sleep(2.0)  # wait for desktop to stabilize

    screenshots_b = []
    step_times_b = []

    for i, action in enumerate(exec_actions):
        act_preview = action.split("Action:")[-1].strip()[:60] if "Action:" in action else action[:60]
        logger.info("  VM-B replay step %d/%d: %s", i + 1, n_actions, act_preview)

        t_step = time.time()
        client_b.step(action)
        step_time = time.time() - t_step
        step_times_b.append(step_time)

        time.sleep(action_pause)
        ss = client_b.get_screenshot()
        screenshots_b.append(ss)
        logger.info("    replayed in %.2fs, screenshot: %s",
                     step_time, "OK" if ss else "FAILED")

    total_b_time = sum(step_times_b) + action_pause * n_actions
    logger.info("  VM-B total: %.1fs (%.2fs avg/step + %.1fs pause)",
                total_b_time, sum(step_times_b) / max(1, n_actions), action_pause)

    # ---- Phase 4: Compare screenshots ----
    logger.info("\n[Phase 4] Comparing screenshots...")
    results = []

    for i in range(n_actions):
        ss_a = screenshots_a[i]
        ss_b = screenshots_b[i]

        if ss_a and ss_b:
            img_a = decode_screenshot(ss_a)
            img_b = decode_screenshot(ss_b)
            sim = screenshot_similarity(img_a, img_b)

            act_preview = exec_actions[i].split("Action:")[-1].strip()[:60] if "Action:" in exec_actions[i] else "?"
            logger.info("  Step %d: pixel=%.1f%% SSIM=%.4f MSE=%.1f PSNR=%.1f — %s",
                        i + 1, sim["pixel_match_pct"], sim["ssim"], sim["mse"], sim["psnr"],
                        act_preview)

            # Save comparison image
            save_comparison_image(img_a, img_b,
                                  os.path.join(output_dir, f"step_{i+1:02d}_compare.png"),
                                  title=f"Step {i+1}")

            results.append(ReplayTestResult(
                step=i + 1,
                similarity=sim,
                vm_a_time=step_times_a[i],
                vm_b_time=step_times_b[i],
                action=exec_actions[i].split("Action:")[-1].strip()[:100] if "Action:" in exec_actions[i] else "?",
            ))
        else:
            logger.warning("  Step %d: missing screenshot (A=%s B=%s)",
                           i + 1, "OK" if ss_a else "NONE", "OK" if ss_b else "NONE")
            results.append(ReplayTestResult(
                step=i + 1,
                similarity={"pixel_match_pct": 0, "mse": -1, "ssim": 0, "psnr": 0},
                vm_a_time=step_times_a[i],
                vm_b_time=step_times_b[i],
                action="?",
            ))

    return results


def test_replay_speed_benchmarks(
    task_config: dict,
    actions: List[str],
    client: EnvClient,
    pause_intervals: List[float] = [0.5, 1.0, 2.0],
) -> List[ReplayBenchmark]:
    """Test 2: Benchmark replay speed with different pause intervals."""
    import re

    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Replay Speed Benchmarks")
    logger.info("  Pause intervals: %s", pause_intervals)
    logger.info("=" * 70)

    # Filter terminal actions
    exec_actions = []
    for a in actions:
        if "Action:" in a:
            act_str = a.split("Action:")[-1].strip().split("\n")[0].strip()
            m = re.match(r"(\w+)\(", act_str)
            if m and m.group(1) in ("finished", "fail"):
                continue
        exec_actions.append(a)

    benchmarks = []

    for pause in pause_intervals:
        logger.info("\n--- Pause=%.1fs ---", pause)

        # Reset VM
        t0 = time.time()
        client.reset(task_config)
        reset_time = time.time() - t0
        time.sleep(2.0)
        logger.info("  Reset: %.1fs", reset_time)

        # Replay all actions with this pause
        t_start = time.time()
        steps_ok = 0
        for i, action in enumerate(exec_actions):
            try:
                client.step(action)
                steps_ok += 1
            except Exception as e:
                logger.error("  Step %d failed: %s", i, e)
                break
            if pause > 0 and i < len(exec_actions) - 1:
                time.sleep(pause)
        total_time = time.time() - t_start

        bm = ReplayBenchmark(
            pause_sec=pause,
            n_steps=steps_ok,
            total_time=round(total_time, 2),
            per_step_time=round(total_time / max(1, steps_ok), 2),
            success=(steps_ok == len(exec_actions)),
        )
        benchmarks.append(bm)
        logger.info("  Result: %d/%d steps, %.1fs total, %.2fs/step",
                     steps_ok, len(exec_actions), total_time, bm.per_step_time)

    return benchmarks


def test_partial_replay(
    task_config: dict,
    actions: List[str],
    client_a: EnvClient,
    client_b: EnvClient,
    output_dir: str,
    checkpoints: Optional[List[int]] = None,
    action_pause: float = 1.0,
) -> List[Dict[str, Any]]:
    """Test 3: Replay first N steps on VM-B, compare to VM-A at step N.

    This validates that partial replays reach the correct intermediate state.
    """
    import re

    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Partial Replay Fidelity")
    logger.info("=" * 70)

    exec_actions = []
    for a in actions:
        if "Action:" in a:
            act_str = a.split("Action:")[-1].strip().split("\n")[0].strip()
            m = re.match(r"(\w+)\(", act_str)
            if m and m.group(1) in ("finished", "fail"):
                continue
        exec_actions.append(a)

    if checkpoints is None:
        # Default: check at 1/3, 2/3, and full
        n = len(exec_actions)
        checkpoints = sorted(set([max(1, n // 3), max(1, 2 * n // 3), n]))

    logger.info("  Checkpoints: %s (of %d actions)", checkpoints, len(exec_actions))

    # Execute all actions on VM-A, save screenshots at checkpoints
    logger.info("\n[Phase 1] Full execution on VM-A, saving checkpoint screenshots...")
    client_a.reset(task_config)
    time.sleep(2.0)

    checkpoint_screenshots_a = {}
    for i, action in enumerate(exec_actions):
        client_a.step(action)
        time.sleep(action_pause)
        step_num = i + 1
        if step_num in checkpoints:
            ss = client_a.get_screenshot()
            checkpoint_screenshots_a[step_num] = ss
            logger.info("  VM-A checkpoint at step %d: screenshot %s",
                        step_num, "OK" if ss else "FAILED")

    # For each checkpoint, reset VM-B and replay up to that point
    results = []
    for cp in checkpoints:
        logger.info("\n[Checkpoint %d] Resetting VM-B and replaying %d steps...", cp, cp)
        client_b.reset(task_config)
        time.sleep(2.0)

        t0 = time.time()
        for i in range(cp):
            client_b.step(exec_actions[i])
            if i < cp - 1:
                time.sleep(action_pause)
        time.sleep(action_pause)  # final settle
        replay_time = time.time() - t0

        ss_b = client_b.get_screenshot()

        ss_a = checkpoint_screenshots_a.get(cp)
        if ss_a and ss_b:
            img_a = decode_screenshot(ss_a)
            img_b = decode_screenshot(ss_b)
            sim = screenshot_similarity(img_a, img_b)
            logger.info("  Checkpoint %d: pixel=%.1f%% SSIM=%.4f MSE=%.1f replay=%.1fs",
                        cp, sim["pixel_match_pct"], sim["ssim"], sim["mse"], replay_time)

            save_comparison_image(img_a, img_b,
                                  os.path.join(output_dir, f"partial_cp{cp:02d}_compare.png"),
                                  title=f"Partial replay: {cp} steps")

            results.append({
                "checkpoint_step": cp,
                "similarity": sim,
                "replay_time": round(replay_time, 2),
            })
        else:
            logger.warning("  Checkpoint %d: missing screenshot", cp)
            results.append({
                "checkpoint_step": cp,
                "similarity": None,
                "replay_time": round(replay_time, 2),
            })

    return results


def test_replay_then_continue(
    task_config: dict,
    actions: List[str],
    client_a: EnvClient,
    client_b: EnvClient,
    output_dir: str,
    replay_until: int = 4,
    action_pause: float = 1.0,
) -> Dict[str, Any]:
    """Test 4: Replay N steps, then execute one more action — verify both VMs match.

    This is the critical MCTS workflow: replay parent's history, then branch with a new action.
    """
    import re

    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Replay-then-Continue (MCTS branch simulation)")
    logger.info("  Replay %d steps, then execute step %d", replay_until, replay_until + 1)
    logger.info("=" * 70)

    exec_actions = []
    for a in actions:
        if "Action:" in a:
            act_str = a.split("Action:")[-1].strip().split("\n")[0].strip()
            m = re.match(r"(\w+)\(", act_str)
            if m and m.group(1) in ("finished", "fail"):
                continue
        exec_actions.append(a)

    if replay_until >= len(exec_actions):
        replay_until = len(exec_actions) - 1
        logger.warning("  Adjusted replay_until to %d (max available)", replay_until)

    if replay_until < 1:
        logger.error("  Not enough actions to test replay-then-continue")
        return {"error": "not enough actions"}

    continue_action = exec_actions[replay_until]

    # Execute all replay_until+1 actions on VM-A
    logger.info("\n[Phase 1] VM-A: executing %d steps normally...", replay_until + 1)
    client_a.reset(task_config)
    time.sleep(2.0)
    for i in range(replay_until + 1):
        client_a.step(exec_actions[i])
        time.sleep(action_pause)

    ss_a = client_a.get_screenshot()
    logger.info("  VM-A at step %d: screenshot %s", replay_until + 1, "OK" if ss_a else "FAILED")

    # On VM-B: replay first replay_until steps, then execute the continue_action
    logger.info("\n[Phase 2] VM-B: replay %d steps + execute step %d...", replay_until, replay_until + 1)
    client_b.reset(task_config)
    time.sleep(2.0)

    t0 = time.time()
    for i in range(replay_until):
        client_b.step(exec_actions[i])
        time.sleep(action_pause)
    replay_time = time.time() - t0

    # Now execute the "branch" action
    t_branch = time.time()
    client_b.step(continue_action)
    branch_time = time.time() - t_branch
    time.sleep(action_pause)

    ss_b = client_b.get_screenshot()
    logger.info("  VM-B at step %d: screenshot %s (replay=%.1fs, branch=%.2fs)",
                replay_until + 1, "OK" if ss_b else "FAILED", replay_time, branch_time)

    result = {
        "replay_steps": replay_until,
        "continue_action": continue_action.split("Action:")[-1].strip()[:100] if "Action:" in continue_action else "?",
        "replay_time": round(replay_time, 2),
        "branch_time": round(branch_time, 2),
    }

    if ss_a and ss_b:
        img_a = decode_screenshot(ss_a)
        img_b = decode_screenshot(ss_b)
        sim = screenshot_similarity(img_a, img_b)
        result["similarity"] = sim
        logger.info("  Comparison: pixel=%.1f%% SSIM=%.4f MSE=%.1f PSNR=%.1f",
                    sim["pixel_match_pct"], sim["ssim"], sim["mse"], sim["psnr"])

        save_comparison_image(img_a, img_b,
                              os.path.join(output_dir, f"replay_continue_step{replay_until+1:02d}.png"),
                              title=f"Replay {replay_until} + Execute 1")
    else:
        result["similarity"] = None
        logger.warning("  Missing screenshots for comparison")

    return result


def test_multi_vm_replay(
    task_config: dict,
    actions: List[str],
    clients: List[EnvClient],
    output_dir: str,
    action_pause: float = 1.0,
) -> Dict[str, Any]:
    """Bonus: Replay on multiple VMs simultaneously, compare all to each other.

    Tests whether replay is deterministic across many VMs.
    """
    import re
    import itertools

    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Multi-VM Replay Consistency (%d VMs)", len(clients))
    logger.info("=" * 70)

    exec_actions = []
    for a in actions:
        if "Action:" in a:
            act_str = a.split("Action:")[-1].strip().split("\n")[0].strip()
            m = re.match(r"(\w+)\(", act_str)
            if m and m.group(1) in ("finished", "fail"):
                continue
        exec_actions.append(a)

    # Reset all VMs
    logger.info("[Phase 1] Resetting %d VMs...", len(clients))
    for c in clients:
        c.reset(task_config)
    time.sleep(3.0)

    # Execute on all VMs
    logger.info("[Phase 2] Executing %d actions on all VMs...", len(exec_actions))
    for i, action in enumerate(exec_actions):
        for c in clients:
            c.step(action)
        time.sleep(action_pause)

    # Screenshot all
    time.sleep(1.0)
    screenshots = {}
    for c in clients:
        ss = c.get_screenshot()
        screenshots[c.slot_id] = ss
        logger.info("  VM %d: screenshot %s", c.slot_id, "OK" if ss else "FAILED")

    # Pairwise comparison
    results = {"n_vms": len(clients), "pairwise": []}
    valid_ids = [sid for sid, ss in screenshots.items() if ss]

    for sid_a, sid_b in itertools.combinations(valid_ids, 2):
        img_a = decode_screenshot(screenshots[sid_a])
        img_b = decode_screenshot(screenshots[sid_b])
        sim = screenshot_similarity(img_a, img_b)
        results["pairwise"].append({
            "vm_a": sid_a, "vm_b": sid_b, **sim
        })
        logger.info("  VM %d vs VM %d: pixel=%.1f%% SSIM=%.4f",
                    sid_a, sid_b, sim["pixel_match_pct"], sim["ssim"])

        save_comparison_image(img_a, img_b,
                              os.path.join(output_dir, f"multi_vm{sid_a}_vs_vm{sid_b}.png"),
                              title=f"VM {sid_a} vs VM {sid_b}")

    # Summary stats
    if results["pairwise"]:
        import numpy as np
        ssims = [p["ssim"] for p in results["pairwise"]]
        pixels = [p["pixel_match_pct"] for p in results["pairwise"]]
        results["summary"] = {
            "mean_ssim": round(float(np.mean(ssims)), 4),
            "min_ssim": round(float(np.min(ssims)), 4),
            "mean_pixel_match": round(float(np.mean(pixels)), 2),
            "min_pixel_match": round(float(np.min(pixels)), 2),
        }
        logger.info("  Summary: mean_SSIM=%.4f min_SSIM=%.4f mean_pixel=%.1f%%",
                    results["summary"]["mean_ssim"], results["summary"]["min_ssim"],
                    results["summary"]["mean_pixel_match"])

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VM Replay Fidelity & Speed Test")
    parser.add_argument("--server-url", type=str, default="http://10.100.4.7:15001")
    parser.add_argument("--task-id", type=str, default="7a4deb26-d57d-4ea9-9a73-630f66a7b568")
    parser.add_argument("--vm-a", type=int, default=0, help="Slot ID for VM-A (original)")
    parser.add_argument("--vm-b", type=int, default=1, help="Slot ID for VM-B (replay)")
    parser.add_argument("--extra-vms", type=int, nargs="*", default=[],
                        help="Extra VM slot IDs for multi-VM test")
    parser.add_argument("--trajectory-file", type=str, default=None,
                        help="JSONL file with pre-recorded trajectories")
    parser.add_argument("--trajectory-index", type=int, default=2,
                        help="Index of trajectory in the JSONL file")
    parser.add_argument("--action-pause", type=float, default=1.0,
                        help="Pause after each action (seconds)")
    parser.add_argument("--output-dir", type=str, default="docs/research/replay_fidelity",
                        help="Output directory for comparison images and results")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip speed benchmark test (test 2)")
    parser.add_argument("--skip-partial", action="store_true",
                        help="Skip partial replay test (test 3)")
    parser.add_argument("--skip-continue", action="store_true",
                        help="Skip replay-then-continue test (test 4)")
    parser.add_argument("--skip-multi", action="store_true",
                        help="Skip multi-VM test (test 5)")
    args = parser.parse_args()

    output_dir = os.path.join(PROJ_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load task config ----
    logger.info("Loading task config for %s...", args.task_id)
    task_config = load_task_config(args.task_id)
    logger.info("Task: [%s] %s", task_config.get("domain", "?"),
                task_config.get("instruction", "")[:80])

    # ---- Load actions ----
    if args.trajectory_file:
        actions = load_trajectory_actions(args.trajectory_file, args.trajectory_index)
    else:
        logger.error("--trajectory-file is required (use a pre-recorded trajectory)")
        logger.info("Example: --trajectory-file docs/research/mcts_smoke_test_success.jsonl --trajectory-index 2")
        sys.exit(1)

    logger.info("Actions to replay (%d total):", len(actions))
    for i, a in enumerate(actions):
        if "Action:" in a:
            act_part = a.split("Action:")[-1].strip().split("\n")[0]
            logger.info("  [%d] %s", i, act_part)

    # ---- Create clients ----
    client_a = EnvClient(args.server_url, args.vm_a)
    client_b = EnvClient(args.server_url, args.vm_b)
    logger.info("VM-A: slot %d, VM-B: slot %d", args.vm_a, args.vm_b)

    all_results = {
        "task_id": args.task_id,
        "n_actions": len(actions),
        "action_pause": args.action_pause,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    # ---- Test 1: Step-by-step vs Replay ----
    try:
        results_1 = test_stepwise_vs_replay(
            task_config, actions, client_a, client_b, output_dir,
            action_pause=args.action_pause,
        )
        all_results["test1_stepwise_vs_replay"] = [
            {"step": r.step, "similarity": r.similarity,
             "vm_a_time": r.vm_a_time, "vm_b_time": r.vm_b_time, "action": r.action}
            for r in results_1
        ]

        # Summary
        if results_1:
            import numpy as np
            ssims = [r.similarity["ssim"] for r in results_1]
            pixels = [r.similarity["pixel_match_pct"] for r in results_1]
            logger.info("\n--- Test 1 Summary ---")
            logger.info("  SSIM:  mean=%.4f min=%.4f max=%.4f",
                        np.mean(ssims), np.min(ssims), np.max(ssims))
            logger.info("  Pixel: mean=%.1f%% min=%.1f%% max=%.1f%%",
                        np.mean(pixels), np.min(pixels), np.max(pixels))
            all_results["test1_summary"] = {
                "mean_ssim": round(float(np.mean(ssims)), 4),
                "min_ssim": round(float(np.min(ssims)), 4),
                "mean_pixel_match": round(float(np.mean(pixels)), 2),
                "min_pixel_match": round(float(np.min(pixels)), 2),
            }
    except Exception as e:
        logger.error("Test 1 failed: %s", e, exc_info=True)
        all_results["test1_error"] = str(e)

    # ---- Test 2: Speed Benchmarks ----
    if not args.skip_benchmark:
        try:
            benchmarks = test_replay_speed_benchmarks(
                task_config, actions, client_b,
                pause_intervals=[0.5, 1.0, 2.0],
            )
            all_results["test2_speed_benchmarks"] = [
                {"pause_sec": b.pause_sec, "n_steps": b.n_steps,
                 "total_time": b.total_time, "per_step_time": b.per_step_time,
                 "success": b.success}
                for b in benchmarks
            ]

            logger.info("\n--- Test 2 Summary ---")
            for b in benchmarks:
                logger.info("  pause=%.1fs: %.1fs total, %.2fs/step, %s",
                            b.pause_sec, b.total_time, b.per_step_time,
                            "OK" if b.success else "FAIL")
        except Exception as e:
            logger.error("Test 2 failed: %s", e, exc_info=True)
            all_results["test2_error"] = str(e)

    # ---- Test 3: Partial Replay ----
    if not args.skip_partial:
        try:
            partial_results = test_partial_replay(
                task_config, actions, client_a, client_b, output_dir,
                action_pause=args.action_pause,
            )
            all_results["test3_partial_replay"] = partial_results

            logger.info("\n--- Test 3 Summary ---")
            for pr in partial_results:
                if pr.get("similarity"):
                    logger.info("  Checkpoint %d: SSIM=%.4f pixel=%.1f%% time=%.1fs",
                                pr["checkpoint_step"], pr["similarity"]["ssim"],
                                pr["similarity"]["pixel_match_pct"], pr["replay_time"])
        except Exception as e:
            logger.error("Test 3 failed: %s", e, exc_info=True)
            all_results["test3_error"] = str(e)

    # ---- Test 4: Replay-then-Continue ----
    if not args.skip_continue:
        try:
            continue_result = test_replay_then_continue(
                task_config, actions, client_a, client_b, output_dir,
                replay_until=4,
                action_pause=args.action_pause,
            )
            all_results["test4_replay_then_continue"] = continue_result

            logger.info("\n--- Test 4 Summary ---")
            if continue_result.get("similarity"):
                logger.info("  Replay %d + Execute 1: SSIM=%.4f pixel=%.1f%%",
                            continue_result["replay_steps"],
                            continue_result["similarity"]["ssim"],
                            continue_result["similarity"]["pixel_match_pct"])
        except Exception as e:
            logger.error("Test 4 failed: %s", e, exc_info=True)
            all_results["test4_error"] = str(e)

    # ---- Test 5: Multi-VM (if extra VMs provided) ----
    if args.extra_vms and not args.skip_multi:
        try:
            all_vm_ids = [args.vm_a, args.vm_b] + args.extra_vms
            multi_clients = [EnvClient(args.server_url, sid) for sid in all_vm_ids]
            multi_result = test_multi_vm_replay(
                task_config, actions, multi_clients, output_dir,
                action_pause=args.action_pause,
            )
            all_results["test5_multi_vm"] = multi_result
        except Exception as e:
            logger.error("Test 5 failed: %s", e, exc_info=True)
            all_results["test5_error"] = str(e)

    # ---- Save results ----
    result_path = os.path.join(output_dir, "replay_fidelity_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("\nResults saved to %s", result_path)

    # ---- Final verdict ----
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VERDICT")
    logger.info("=" * 70)

    if "test1_summary" in all_results:
        s = all_results["test1_summary"]
        ssim_ok = s["min_ssim"] > 0.90
        pixel_ok = s["min_pixel_match"] > 80
        verdict = "PASS" if ssim_ok and pixel_ok else "MARGINAL" if ssim_ok else "FAIL"
        logger.info("  Test 1 (step-by-step vs replay): %s", verdict)
        logger.info("    min_SSIM=%.4f (threshold: >0.90)", s["min_ssim"])
        logger.info("    min_pixel=%.1f%% (threshold: >80%%)", s["min_pixel_match"])

    if "test4_replay_then_continue" in all_results:
        r = all_results["test4_replay_then_continue"]
        if r.get("similarity"):
            ssim = r["similarity"]["ssim"]
            verdict = "PASS" if ssim > 0.90 else "MARGINAL" if ssim > 0.80 else "FAIL"
            logger.info("  Test 4 (replay-then-continue): %s (SSIM=%.4f)", verdict, ssim)

    if "test5_multi_vm" in all_results and "summary" in all_results["test5_multi_vm"]:
        s = all_results["test5_multi_vm"]["summary"]
        verdict = "PASS" if s["min_ssim"] > 0.90 else "MARGINAL" if s["min_ssim"] > 0.80 else "FAIL"
        logger.info("  Test 5 (multi-VM consistency): %s (min_SSIM=%.4f)", verdict, s["min_ssim"])

    logger.info("\nDone. Comparison images saved to %s/", output_dir)


if __name__ == "__main__":
    main()
