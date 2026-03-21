#!/usr/bin/env python3
"""
MCTS VM-usage profiling: run all 86 doable tasks with 80 VMs to measure
how many VMs each task actually activates.

This data decides whether to run 1 task × 80 VMs or 2 tasks × 40 VMs.

Usage:
    python scripts/mcts/run_mcts_profiling.py \
        --max-steps 15 --vms-per-task 80

Three env servers (80 VMs total):
    10.100.4.7:15001  slots 0-31  (32 VMs)
    10.100.4.6:15001  slots 0-15  (16 VMs)
    10.100.4.8:15001  slots 0-31  (32 VMs)
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "OSWorld"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mcts_profiling")

# ---- Default server layout: 80 VMs across 3 clusters ----
DEFAULT_SERVERS = [
    ("http://10.100.4.7:15001", 32),
    ("http://10.100.4.6:15001", 16),
    ("http://10.100.4.8:15001", 32),
]


def load_doable_task_ids() -> set:
    """Load the 86 doable task IDs from combined trajectories."""
    path = os.path.join(PROJ_ROOT, "checkpoints/arpo-inference/all_trajectories_combined.jsonl")
    doable = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("eval_result", 0) > 0:
                doable.add(d["task_id"])
    return doable


def load_task_configs(task_file: str, doable_ids: set) -> List[dict]:
    """Load task configs for doable tasks from the 300-task file."""
    with open(task_file) as f:
        raw = json.load(f)
    base_path = os.path.dirname(task_file)
    tasks = []
    for domain, task_ids in raw.items():
        for tid in task_ids:
            if tid not in doable_ids:
                continue
            cfg_path = os.path.join(base_path, "examples", domain, tid + ".json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f2:
                    tc = json.load(f2)
                tc["domain"] = domain
                tc["id"] = tid
                tc["task_id"] = tid
                tasks.append(tc)
    return tasks


def create_workers(servers: List[Tuple[str, int]], ray_module):
    """Create MCTSEnvClient Ray actors distributed across servers.

    Returns (workers, total_vms) where workers[i] is assigned to the correct
    server URL and slot_id, same pattern as ray_trainer._create_envs().
    """
    from verl.mcts.env_client import MCTSEnvClient
    RemoteClient = ray_module.remote(num_cpus=0, num_gpus=0)(MCTSEnvClient)

    workers = []
    worker_idx = 0
    for url, count in servers:
        for slot in range(count):
            w = RemoteClient.remote(
                worker_idx=worker_idx,
                remote_server_url=url,
                slot_id=slot,
            )
            workers.append(w)
            worker_idx += 1
    total = sum(c for _, c in servers)
    logger.info("Created %d MCTSEnvClient actors across %d servers", total, len(servers))
    for url, count in servers:
        logger.info("  %s: %d VMs (slots 0-%d)", url, count, count - 1)
    return workers, total


def main():
    parser = argparse.ArgumentParser(description="MCTS VM-usage profiling across 86 doable tasks")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--vms-per-task", type=int, default=80,
                        help="Max VMs per task (default 80 = all)")
    parser.add_argument("--tp", type=int, default=8, help="Number of GPUs (1 vLLM engine each)")
    parser.add_argument("--model-path", type=str, default="ByteDance-Seed/UI-TARS-1.5-7B")
    parser.add_argument("--task-file", type=str,
                        default="OSWorld/evaluation_examples/test_all_300tasks_noproxy_softreset_clean.json")
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--output", type=str, default="docs/research/mcts_vm_profiling.json")
    parser.add_argument("--n-tasks", type=int, default=0,
                        help="Limit to N tasks (0 = all 86)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Filter by domain (e.g. gimp)")
    parser.add_argument("--task-ids", nargs="+", default=None,
                        help="Specific task IDs to run")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks already in output file")
    args = parser.parse_args()

    # ---- Load tasks ----
    task_file = os.path.join(PROJ_ROOT, args.task_file)
    doable_ids = load_doable_task_ids()
    all_tasks = load_task_configs(task_file, doable_ids)
    logger.info("Loaded %d doable tasks", len(all_tasks))

    # Filter
    if args.domain:
        all_tasks = [t for t in all_tasks if t.get("domain") == args.domain]
        logger.info("Filtered to %d tasks in domain '%s'", len(all_tasks), args.domain)
    if args.task_ids:
        # Use specific task IDs (e.g. spread by difficulty)
        id_set = set(args.task_ids)
        all_tasks = [t for t in all_tasks if t["task_id"] in id_set]
        # Preserve order of args.task_ids
        id_order = {tid: i for i, tid in enumerate(args.task_ids)}
        all_tasks.sort(key=lambda t: id_order.get(t["task_id"], 999))
        logger.info("Selected %d specific tasks", len(all_tasks))
    elif args.n_tasks > 0:
        all_tasks = all_tasks[:args.n_tasks]
        logger.info("Limited to %d tasks", len(all_tasks))

    # Resume support
    completed_ids = set()
    output_path = os.path.join(PROJ_ROOT, args.output)
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            prev = json.load(f)
        for r in prev.get("results", []):
            completed_ids.add(r["task_id"])
        logger.info("Resuming: skipping %d already-completed tasks", len(completed_ids))
        all_tasks = [t for t in all_tasks if t["task_id"] not in completed_ids]

    logger.info("Will profile %d tasks with %d VMs each, %d steps",
                len(all_tasks), args.vms_per_task, args.max_steps)

    # ---- Init Ray ----
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # ---- Create workers ----
    workers, total_vms = create_workers(DEFAULT_SERVERS, ray)
    assert total_vms >= args.vms_per_task, \
        f"Need {args.vms_per_task} VMs but only {total_vms} available"

    # Use first vms_per_task workers
    task_workers = workers[:args.vms_per_task]

    # ---- Load processor/tokenizer ----
    logger.info("Loading processor/tokenizer from %s...", args.model_path)
    from transformers import AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # ---- Create VLLMPool ----
    if args.tp == 1:
        from vllm import LLM
        vllm_pool = LLM(
            model=args.model_path, tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_mem, max_model_len=32768,
            limit_mm_per_prompt={"image": 3}, trust_remote_code=True, dtype="bfloat16",
        )
    else:
        from verl.mcts.vllm_pool import VLLMPool
        vllm_pool = VLLMPool(
            n_gpus=args.tp, model_path=args.model_path,
            gpu_memory_utilization=args.gpu_mem, max_model_len=12000, limit_images=3,
        )
    logger.info("vLLM ready (%d GPUs).", args.tp)

    # ---- Create orchestrator ----
    from verl.mcts.config import MCTSConfig
    from verl.mcts.orchestrator import MCTSOrchestrator

    config = MCTSConfig(
        vms_per_task=args.vms_per_task,
        max_active_vms=args.vms_per_task,  # unlimit — use all
        max_steps=args.max_steps,
        tensor_parallel_size=args.tp,
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_mem,
        remote_server_urls=[url for url, _ in DEFAULT_SERVERS],
    )
    orchestrator = MCTSOrchestrator(config, vllm_pool, processor, tokenizer)

    # ---- Run all tasks ----
    results = []
    # Load previous results for resume
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            prev = json.load(f)
        results = prev.get("results", [])

    all_trajectories = []

    for task_idx, task_config in enumerate(all_tasks):
        tid = task_config["task_id"]
        domain = task_config.get("domain", "?")
        instruction = task_config.get("instruction", "")[:80]

        logger.info("\n" + "=" * 70)
        logger.info("Task %d/%d: [%s] %s — %s",
                     task_idx + 1, len(all_tasks), domain, tid[:11], instruction)
        logger.info("=" * 70)

        # Reset all VMs for this task
        t0 = time.time()
        logger.info("Resetting %d VMs...", len(task_workers))
        reset_futures = [w.reset.remote(task_config) for w in task_workers]
        try:
            ray.get(reset_futures, timeout=300)
        except Exception as e:
            logger.error("Reset failed for task %s: %s", tid, e)
            results.append({
                "task_id": tid, "domain": domain,
                "error": str(e), "vms_activated": 0,
            })
            continue
        reset_time = time.time() - t0
        logger.info("Reset done in %.1fs", reset_time)

        # Run MCTS
        t1 = time.time()
        tree, trajectories = orchestrator.run_task(task_config, task_workers)
        mcts_time = time.time() - t1

        # Collect stats
        all_nodes = tree.all_nodes()
        n_activated = len(all_nodes)
        n_success = sum(1 for n in all_nodes if (n.eval_score or 0) > 0)
        n_branches = sum(1 for n in all_nodes if n.parent is not None)
        max_depth = max((n.depth for n in all_nodes), default=0)
        steps_per_node = [len(n.action_history) for n in all_nodes]
        branch_steps = sorted(set(n.depth for n in all_nodes if n.parent is not None))

        result = {
            "task_id": tid,
            "domain": domain,
            "instruction": instruction,
            "vms_activated": n_activated,
            "vms_available": len(task_workers),
            "branches_spawned": n_branches,
            "branch_at_steps": branch_steps,
            "max_depth": max_depth,
            "successful_trajectories": n_success,
            "total_trajectories": len(trajectories),
            "mcts_time_sec": round(mcts_time, 1),
            "reset_time_sec": round(reset_time, 1),
            "steps_per_node": steps_per_node,
            "tree_summary": tree.summary(),
        }
        results.append(result)
        all_trajectories.extend(trajectories)

        logger.info("  VMs activated: %d/%d", n_activated, len(task_workers))
        logger.info("  Branches: %d at steps %s", n_branches, branch_steps)
        logger.info("  Successful: %d/%d trajectories", n_success, len(trajectories))
        logger.info("  Time: %.1fs MCTS + %.1fs reset", mcts_time, reset_time)

        # ---- Save incrementally (survive crashes) ----
        _save_results(output_path, results, args, all_tasks)

    # ---- Final summary ----
    _print_summary(results)
    _save_results(output_path, results, args, all_tasks)

    # Save successful trajectories
    successful = [t for t in all_trajectories if t.get("eval_result", 0) > 0]
    if successful:
        success_path = output_path.replace(".json", "_success.jsonl")
        from verl.utils.trajectory_io import TrajectoryWriter
        with TrajectoryWriter(success_path) as writer:
            for traj in successful:
                writer.write(traj)
        logger.info("Saved %d successful trajectories to %s", len(successful), success_path)

    logger.info("Done. Results: %s", output_path)


def _save_results(output_path, results, args, all_tasks):
    """Save results incrementally."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Compute summary stats
    activated = [r["vms_activated"] for r in results if "error" not in r]
    summary = {}
    if activated:
        import statistics
        summary = {
            "tasks_completed": len(results),
            "tasks_total": len(all_tasks),
            "vms_per_task": args.vms_per_task,
            "vm_usage_min": min(activated),
            "vm_usage_max": max(activated),
            "vm_usage_mean": round(statistics.mean(activated), 1),
            "vm_usage_median": round(statistics.median(activated), 1),
            "vm_usage_p90": round(sorted(activated)[int(len(activated) * 0.9)] if activated else 0, 1),
            "tasks_using_over_40": sum(1 for v in activated if v > 40),
            "tasks_using_over_20": sum(1 for v in activated if v > 20),
            "total_successful": sum(r.get("successful_trajectories", 0) for r in results),
        }

    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "vms_per_task": args.vms_per_task,
                "max_steps": args.max_steps,
                "tp": args.tp,
            },
            "summary": summary,
            "results": results,
        }, f, indent=2, ensure_ascii=False)


def _print_summary(results):
    """Print final summary statistics."""
    activated = [r["vms_activated"] for r in results if "error" not in r]
    if not activated:
        logger.info("No completed tasks.")
        return

    import statistics
    logger.info("\n" + "=" * 70)
    logger.info("PROFILING SUMMARY (%d tasks)", len(activated))
    logger.info("=" * 70)
    logger.info("  VM usage: min=%d, max=%d, mean=%.1f, median=%.1f",
                min(activated), max(activated),
                statistics.mean(activated), statistics.median(activated))
    logger.info("  Tasks using >40 VMs: %d/%d", sum(1 for v in activated if v > 40), len(activated))
    logger.info("  Tasks using >20 VMs: %d/%d", sum(1 for v in activated if v > 20), len(activated))
    logger.info("  Total successful trajectories: %d",
                sum(r.get("successful_trajectories", 0) for r in results))

    # Per-domain breakdown
    by_domain = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_domain[r["domain"]].append(r["vms_activated"])
    logger.info("\n  Per-domain VM usage:")
    for domain in sorted(by_domain.keys()):
        vals = by_domain[domain]
        logger.info("    %s: n=%d, mean=%.1f, max=%d",
                    domain, len(vals), statistics.mean(vals), max(vals))


if __name__ == "__main__":
    main()
