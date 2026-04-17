#!/usr/bin/env python3
"""
Production MCTS trajectory collection.

Usage:
    python scripts/mcts/run_mcts_collection.py --config configs/mcts_collection_86tasks.yaml
    python scripts/mcts/run_mcts_collection.py --config configs/mcts_collection_86tasks.yaml --resume
"""

import json
import logging
import os
import sys
import time

import yaml

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "OSWorld"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mcts_collection")


def load_doable_task_ids():
    """Load doable task IDs from available sources."""
    doable = set()

    # Source 1: MCTS v1 successful trajectories (84 tasks)
    mcts_path = os.path.join(PROJ_ROOT, "checkpoints/mcts_trajectories/combined/mcts_success.jsonl")
    if os.path.exists(mcts_path):
        with open(mcts_path) as f:
            for line in f:
                if line.strip():
                    doable.add(json.loads(line)["task_id"])

    # Source 2: base model n=8 eval (tasks with at least 1 success)
    eval_path = os.path.join(PROJ_ROOT, "checkpoints/arpo-inference/base_model_306tasks_n8/eval_results_at_0.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)
        for tid, v in eval_results.items():
            if isinstance(v, dict) and v.get("n_success", 0) > 0:
                doable.add(tid)

    # Fallback: legacy file
    legacy_path = os.path.join(PROJ_ROOT, "checkpoints/arpo-inference/all_trajectories_combined.jsonl")
    if os.path.exists(legacy_path):
        with open(legacy_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("eval_result", 0) > 0:
                    doable.add(d["task_id"])

    return doable


def load_task_configs(task_file):
    """Load task configs from a task list JSON file.

    The file maps domain -> [task_id, ...]. Each task's full config
    is loaded from OSWorld/evaluation_examples/examples/{domain}/{tid}.json.
    """
    with open(task_file) as f:
        raw = json.load(f)
    base_path = os.path.dirname(task_file)
    tasks = []
    for domain, task_ids in raw.items():
        for tid in task_ids:
            cfg_path = os.path.join(base_path, "examples", domain, tid + ".json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f2:
                    tc = json.load(f2)
                tc["domain"] = domain
                tc["id"] = tid
                tc["task_id"] = tid
                tasks.append(tc)
            else:
                logger.warning("Task config not found: %s", cfg_path)
    return tasks


def create_workers(urls, counts, ray_module):
    from verl.mcts.env_client import MCTSEnvClient
    RemoteClient = ray_module.remote(num_cpus=0, num_gpus=0)(MCTSEnvClient)
    workers = []
    worker_idx = 0
    for url, count in zip(urls, counts):
        for slot in range(count):
            w = RemoteClient.remote(worker_idx=worker_idx, remote_server_url=url, slot_id=slot)
            workers.append(w)
            worker_idx += 1
    total = sum(counts)
    logger.info("Created %d MCTSEnvClient actors across %d servers", total, len(urls))
    for url, count in zip(urls, counts):
        logger.info("  %s: %d VMs", url, count)
    return workers, total


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load tasks (task file already contains only the target tasks)
    task_file = os.path.join(PROJ_ROOT, cfg["task_file"])
    all_tasks = load_task_configs(task_file)
    logger.info("Loaded %d doable tasks", len(all_tasks))

    # Output & resume
    output_dir = os.path.join(PROJ_ROOT, cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "collection_results.json")
    results = []
    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            prev = json.load(f)
        results = prev.get("results", [])
        completed_ids = {r["task_id"] for r in results}
        all_tasks = [t for t in all_tasks if t["task_id"] not in completed_ids]
        logger.info("Resuming: skipping %d completed tasks", len(completed_ids))

    logger.info("Will collect %d tasks, %d VMs each (max_active=%d), %d steps",
                len(all_tasks), cfg["vms_per_task"], cfg["max_active_vms"], cfg["max_steps"])

    # Init Ray
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create workers
    workers, total_vms = create_workers(
        cfg["remote_server_urls"], cfg["remote_server_env_counts"], ray)

    # Load processor/tokenizer
    from transformers import AutoProcessor, AutoTokenizer
    logger.info("Loading processor/tokenizer...")
    processor = AutoProcessor.from_pretrained(cfg["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])

    # Create VLLMPool
    from verl.mcts.vllm_pool import VLLMPool
    vllm_pool = VLLMPool(
        n_gpus=cfg["tp"],
        model_path=cfg["model_path"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        max_model_len=cfg["max_model_len"],
        limit_images=cfg["limit_images"],
    )
    logger.info("vLLM ready (%d GPUs).", cfg["tp"])

    # Create MCTSConfig from yaml
    from verl.mcts.config import MCTSConfig
    from verl.mcts.orchestrator import MCTSOrchestrator

    config = MCTSConfig(
        vms_per_task=cfg["vms_per_task"],
        max_active_vms=cfg["max_active_vms"],
        probe_temperature=cfg["probe_temperature"],
        total_probe_budget=cfg["total_probe_budget"],
        k_max=cfg["k_max"],
        spatial_grid_size=cfg["spatial_grid_size"],
        min_cluster_size=cfg["min_cluster_size"],
        max_branch_per_explorer=cfg["max_branch_per_explorer"],
        child_branch_budget=cfg["child_branch_budget"],
        max_branches_per_step=cfg["max_branches_per_step"],
        never_branch_after=cfg["never_branch_after"],
        late_step_threshold=cfg["late_step_threshold"],
        replay_pause_sec=cfg["replay_pause_sec"],
        stuck_repeat_limit=cfg["stuck_repeat_limit"],
        stuck_wait_limit=cfg["stuck_wait_limit"],
        max_steps=cfg["max_steps"],
        model_path=cfg["model_path"],
        tensor_parallel_size=cfg["tp"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        max_model_len=cfg["max_model_len"],
        limit_images=cfg["limit_images"],
        generation_max_tokens=cfg["generation_max_tokens"],
        remote_server_urls=cfg["remote_server_urls"],
        output_dir=cfg["output_dir"],
        save_full_tree=cfg.get("save_full_tree", False),
        # Noise config (v3)
        enable_noise=cfg.get("enable_noise", False),
        noise_mode=cfg.get("noise_mode", "runtime_library"),
        noise_branch_probability=cfg.get("noise_branch_probability", 0.8),
        noise_min_fire_step=cfg.get("noise_min_fire_step", 3),
        noise_min_task_buffer=cfg.get("noise_min_task_buffer", 4),
        noise_sr_file=cfg.get("noise_sr_file", ""),
    )

    # Load per-task clean SR for noise fire count decisions
    task_sr_map: dict = {}
    sr_file = cfg.get("noise_sr_file", "")
    if sr_file:
        sr_path = os.path.join(PROJ_ROOT, sr_file)
        if os.path.exists(sr_path):
            with open(sr_path) as f:
                sr_data = json.load(f)
            # Handle both formats: {task_id: {success_rate: ...}} and {results: [...]}
            if isinstance(sr_data, dict) and "results" not in sr_data:
                for tid, info in sr_data.items():
                    if isinstance(info, dict):
                        task_sr_map[tid] = float(info.get("success_rate", 0))
                    else:
                        task_sr_map[tid] = float(info)
            elif isinstance(sr_data, dict) and "results" in sr_data:
                for r in sr_data["results"]:
                    task_sr_map[r["task_id"]] = float(r.get("success_rate", 0))
            logger.info("Loaded clean SR for %d tasks from %s", len(task_sr_map), sr_path)
        else:
            logger.warning("noise_sr_file not found: %s", sr_path)

    if config.enable_noise:
        logger.info("Noisy MCTS enabled: branch_prob=%.2f, min_fire_step=%d, min_buffer=%d",
                     config.noise_branch_probability, config.noise_min_fire_step,
                     config.noise_min_task_buffer)

    orchestrator = MCTSOrchestrator(config, vllm_pool, processor, tokenizer,
                                     task_sr_map=task_sr_map)

    # Run tasks
    task_workers = workers[:cfg["vms_per_task"]]

    for task_idx, task_config in enumerate(all_tasks):
        tid = task_config["task_id"]
        domain = task_config.get("domain", "?")
        instruction = task_config.get("instruction", "")[:80]

        logger.info("\n" + "=" * 70)
        logger.info("Task %d/%d: [%s] %s — %s",
                     task_idx + 1, len(all_tasks), domain, tid[:11], instruction)
        logger.info("=" * 70)

        # Reset all VMs
        t0 = time.time()
        logger.info("Resetting %d VMs...", len(task_workers))
        reset_futures = [w.reset.remote(task_config) for w in task_workers]
        try:
            ray.get(reset_futures, timeout=300)
        except Exception as e:
            logger.error("Reset failed: %s", e)
            results.append({"task_id": tid, "domain": domain, "error": str(e),
                            "vms_activated": 0})
            _save(results_path, results, cfg)
            continue
        reset_time = time.time() - t0
        logger.info("Reset done in %.1fs", reset_time)

        # Run MCTS
        t1 = time.time()
        tree, trajectories = orchestrator.run_task(task_config, task_workers)
        mcts_time = time.time() - t1

        # Stats
        all_nodes = tree.all_nodes()
        n_activated = len(all_nodes)
        n_success = sum(1 for n in all_nodes if (n.eval_score or 0) > 0)

        result = {
            "task_id": tid,
            "domain": domain,
            "instruction": instruction,
            "vms_activated": n_activated,
            "successful_trajectories": n_success,
            "total_trajectories": len(trajectories),
            "mcts_time_sec": round(mcts_time, 1),
            "reset_time_sec": round(reset_time, 1),
        }
        results.append(result)
        logger.info("  VMs: %d, Success: %d/%d, Time: %.1fs",
                    n_activated, n_success, len(trajectories), mcts_time)

        # Save incrementally
        _save(results_path, results, cfg)

        # Save full tree (v2)
        if cfg.get("save_full_tree", False):
            from verl.mcts.tree_io import save_mcts_tree
            trees_dir = os.path.join(output_dir, "trees")
            os.makedirs(trees_dir, exist_ok=True)
            tree_path = os.path.join(trees_dir, f"{tid}.json")
            save_mcts_tree(tree, task_config, tree_path, limit_images=cfg["limit_images"])
            logger.info("  Saved full tree to %s (%d nodes, %d successful)",
                        tree_path, n_activated, n_success)

        # Append successful trajectories (always — backwards compatible)
        successful = [t for t in trajectories if t.get("eval_result", 0) > 0]
        if successful:
            success_path = os.path.join(output_dir, "mcts_success.jsonl")
            from verl.utils.trajectory_io import TrajectoryWriter
            with TrajectoryWriter(success_path) as writer:
                for traj in successful:
                    writer.write(traj)
            logger.info("  Appended %d successful to %s", len(successful), success_path)

    # Final summary
    total_success = sum(r.get("successful_trajectories", 0) for r in results)
    logger.info("\n" + "=" * 70)
    logger.info("DONE: %d tasks, %d total successful trajectories", len(results), total_success)
    logger.info("=" * 70)


def _save(path, results, cfg):
    with open(path, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
