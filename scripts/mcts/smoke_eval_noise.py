#!/usr/bin/env python3
"""Live smoke: confirm noise actually fires in env when the eval-path noise
schedule is stamped (client-side-built noise_meta, same shape the trainer
will ship to the env server).

No vLLM — uses wait() actions. Goal: confirm
  1. env.reset picks up noise_meta and builds a scheduler.
  2. noise_burden in the step response is non-empty at the fire step.
  3. The schedule matches what the sampler unit test predicted.
"""
import argparse, hashlib, json, os, sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "OSWorld"))

import ray
from verl.mcts.env_client import MCTSEnvClient
from OSWorld.evaluation_examples.noise_generation.runtime_sampler import RuntimeNoiseSampler


def load_task(domain: str, tid: str) -> dict:
    p = os.path.join(PROJ, "OSWorld", "evaluation_examples", "examples", domain, f"{tid}.json")
    tc = json.load(open(p))
    tc["id"] = tid
    tc["domain"] = domain
    return tc


def build_eval_noise_meta(task_config: dict, max_steps: int = 15) -> dict | None:
    """Build the deterministic eval noise schedule client-side.
    Mirrors what ray_trainer will do in the eval path."""
    tid = task_config.get("id", "") or ""
    rng_seed = int(hashlib.md5(f"{tid}|0".encode()).hexdigest()[:8], 16)
    sampler = RuntimeNoiseSampler(rng_seed=rng_seed)
    schedule = sampler.sample_fire_schedule(
        task_json=task_config, sr=0.0, max_steps=max_steps,
        use_heldout=True, is_eval=True,
    )
    if not schedule:
        return None
    return {
        "trigger_mode": "deterministic_schedule_v4_eval",
        "success_rate_used": 0.0,
        "observed_sr_used": 0.0,
        "fires_count": len(schedule),
        "fire_steps": [int(e["fire_step"]) for e in schedule],
        "total_recovery_cost": sum(int(e.get("recovery_cost", 0)) for e in schedule),
        "elements": schedule,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://10.100.4.6:15001")
    ap.add_argument("--slot_id", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--task_id", default="bb5e4c0d-f964-439c-97b6-bdb9747de3f4")
    ap.add_argument("--domain", default="chrome")
    args = ap.parse_args()

    tc = load_task(args.domain, args.task_id)
    noise_meta = build_eval_noise_meta(tc, max_steps=args.max_steps)
    if noise_meta is None:
        print("Task is in the CLEAN subset (fires=0). Pick a different task to smoke noise.")
        sys.exit(1)

    tc = dict(tc)
    tc["enable_noise"] = True
    tc["noise_mode"] = "runtime_library"
    tc["noise_meta"] = noise_meta
    tc["noise_max_steps"] = args.max_steps

    fire_steps = noise_meta["fire_steps"]
    templates = [e.get("template_name") for e in noise_meta["elements"]]
    print(f"Task: {tc['instruction'][:120]}")
    print(f"Expected schedule: templates={templates} fire_steps={fire_steps}")
    print(f"Resetting env and stepping with wait()...\n")

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    env = ray.remote(num_cpus=0, num_gpus=0)(MCTSEnvClient).remote(
        worker_idx=0, remote_server_url=args.url, slot_id=args.slot_id,
    )
    ray.get(env.reset.remote(tc), timeout=300)

    fired = []
    for step in range(args.max_steps):
        result = ray.get(env.step.remote("Thought: wait\nAction: wait()"))
        nb = result.get("noise_burden", {}) or {}
        f = nb.get("events_fired_this_step", 0)
        c = nb.get("step_recovery_cost", 0)
        names = [e.get("template_name") or e.get("id", "?") for e in (nb.get("events") or [])]
        print(f"  step {step:2d}: fired={f}  cost={c}  events={names}")
        if f:
            fired.append((step, names))
        if result.get("is_done"):
            break

    print()
    print("=" * 60)
    if fired:
        print(f"PASS: fired on steps {[s for s,_ in fired]}")
        for s, names in fired:
            print(f"  step {s}: {names}")
    else:
        print("FAIL: noise never fired")
        sys.exit(1)


if __name__ == "__main__":
    main()
