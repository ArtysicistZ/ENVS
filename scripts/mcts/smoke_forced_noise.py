#!/usr/bin/env python3
"""Forced-noise smoke test: run the same chrome task with a specific
foreign noise element (System Monitor) that fires at a fixed step.
Compare agent reasoning across runs to isolate the effect of the
"shared computer" prompt.

Bypasses RuntimeNoiseSampler by injecting noise_meta directly on
task_config — env.reset() will use it verbatim and set up the scheduler.
"""
import argparse
import json
import os
import sys
import time

# Put project root on sys.path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "OSWorld"))

import ray

from verl.mcts.env_client import MCTSEnvClient
from verl.mcts.vllm_pool import VLLMPool
from verl.mcts.config import MCTSConfig
from verl.mcts.tree import TreeNode, BranchBudget
from verl.mcts.orchestrator import MCTSOrchestrator
from verl.trainer.gui_agent import add_box_token


SYSTEM_MONITOR_CMD = r"""
cat > /tmp/coworker_notes.txt << 'EOF'
Meeting notes — Q2 planning

- Revisit the onboarding doc for new engineers
- Reach out to Alex about the monitoring dashboard
- Review Jamie's PR on the auth refactor
- Schedule 1:1 with Priya next week
- Double-check the staging deploy before Friday
- Reminder: submit expenses by end of month
EOF
gedit /tmp/coworker_notes.txt >/dev/null 2>&1 & disown
sleep 1.2
wmctrl -r 'coworker_notes' -b add,maximized_vert,maximized_horz 2>/dev/null
wmctrl -a 'coworker_notes' 2>/dev/null
sleep 0.3
"""


def build_forced_noise_meta(fire_step: int = 4) -> dict:
    """Construct a noise_meta that fires System Monitor at a specific step.
    env.reset() reads this verbatim (see _build_noise_meta in desktop_env.py)."""
    element = {
        "id": "forced_system_monitor",
        "category": "human_task_session",
        "recovery_cost": 1,
        "once": True,
        "command": SYSTEM_MONITOR_CMD,
        "fire_step": fire_step,
    }
    return {
        "trigger_mode": "deterministic_schedule_v4_forced",
        "success_rate_used": 0.5,
        "observed_sr_used": 0.5,
        "fires_count": 1,
        "fire_steps": [fire_step],
        "total_recovery_cost": 1,
        "elements": [element],
    }


def load_task(task_id: str, domain: str = "chrome") -> dict:
    p = os.path.join(PROJ_ROOT, "OSWorld", "evaluation_examples", "examples",
                     domain, f"{task_id}.json")
    tc = json.load(open(p))
    tc["id"] = task_id
    tc["domain"] = domain
    return tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="2ad9387a-65d8-4e33-ad5b-7580065a27ca")
    ap.add_argument("--domain", default="chrome")
    ap.add_argument("--fire_step", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--url", default="http://10.100.4.6:15001")
    ap.add_argument("--slot_id", type=int, default=0)
    ap.add_argument("--out", default="/tmp/smoke_forced_output.json")
    args = ap.parse_args()

    tc = load_task(args.task_id, args.domain)
    tc["enable_noise"] = True
    tc["noise_mode"] = "runtime_library"
    # Direct noise_meta — bypass sampler, fire System Monitor at fire_step.
    tc["noise_meta"] = build_forced_noise_meta(args.fire_step)

    print(f"Task: {tc['instruction'][:140]}")
    print(f"Forcing noise: System Monitor at step {args.fire_step}")

    ray.init(ignore_reinit_error=True)

    env = ray.remote(num_cpus=0, num_gpus=0)(MCTSEnvClient).remote(
        worker_idx=0,
        remote_server_url=args.url,
        slot_id=args.slot_id,
    )

    print("Loading vLLM (8 GPUs)...")
    vllm_pool = VLLMPool(
        n_gpus=8,
        model_path=os.path.join(PROJ_ROOT, "checkpoints/mcts_sft_v2.1/beta05_2e-6/epoch_1"),
        gpu_memory_utilization=0.85,
        max_model_len=12000,
        limit_images=3,
    )
    from transformers import AutoProcessor, AutoTokenizer
    model_path = os.path.join(PROJ_ROOT, "checkpoints/mcts_sft_v2.1/beta05_2e-6/epoch_1")
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    cfg = MCTSConfig(
        max_steps=args.max_steps,
        limit_images=3,
        generation_max_tokens=512,
        probe_temperature=1.0,
        total_probe_budget=1,
        k_max=1,
    )
    orch = MCTSOrchestrator(cfg, vllm_pool, processor, tokenizer, task_sr_map={})

    print("Resetting env with forced noise...")
    ray.get(env.reset.remote(tc), timeout=300)

    init_ss = ray.get(env.get_obs_screenshot.remote())
    node = TreeNode(node_id="smoke_0", vm_slot_id=0, depth=0,
                    budget=BranchBudget(0), instruction=tc["instruction"],
                    current_screenshot_b64=init_ss)

    log = {"task_id": args.task_id, "fire_step": args.fire_step, "steps": []}

    for step in range(args.max_steps):
        msgs = orch._build_messages_for_node(node)
        vllm_input = orch._prepare_vllm_input(msgs)
        if vllm_input is None:
            print(f"[step {step}] failed to build input")
            break
        outs = orch._generate_batch_raw([vllm_input])
        action_text = outs[0]

        # Execute
        result = ray.get(env.step.remote(action_text))
        noise_burden = result.get("noise_burden", {}) or {}
        done = result.get("is_done", False)

        # Save the screenshot the agent saw this step
        ss_dir = os.path.dirname(args.out) or "/tmp"
        ss_stem = os.path.splitext(os.path.basename(args.out))[0]
        import base64 as _b64
        if node.current_screenshot_b64:
            with open(f"{ss_dir}/{ss_stem}_step{step:02d}.png", "wb") as f:
                f.write(_b64.b64decode(node.current_screenshot_b64))

        entry = {
            "step": step,
            "action_text": action_text,
            "noise_burden": {
                "events_fired_this_step": noise_burden.get("events_fired_this_step", 0),
                "step_recovery_cost": noise_burden.get("step_recovery_cost", 0),
                "events": noise_burden.get("events", []),
            },
            "is_done": done,
        }
        log["steps"].append(entry)
        print(f"\n===== STEP {step} =====")
        print(f"[noise] fired={entry['noise_burden']['events_fired_this_step']}, "
              f"cost={entry['noise_burden']['step_recovery_cost']}")
        print(action_text[:800])
        if done:
            print("(env marked done)")
            break

        # Update node for next step
        node.record_action(add_box_token(action_text))
        next_ss = ray.get(env.get_obs_screenshot.remote())
        node.current_screenshot_b64 = next_ss

    # Evaluate
    try:
        ev = ray.get(env.evaluate.remote(), timeout=120)
        log["eval"] = ev
        print(f"\n[eval] {ev}")
    except Exception as e:
        log["eval"] = {"error": str(e)}

    with open(args.out, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved log to {args.out}")


if __name__ == "__main__":
    main()
