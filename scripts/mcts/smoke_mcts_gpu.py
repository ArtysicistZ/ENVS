#!/usr/bin/env python3
"""
GPU smoke test for the MCTS trajectory collection system.

Validates the full pipeline on 1-2 tasks with 5 steps:
1. Pre-setup 40 VMs for one task
2. Run MCTS exploration with dynamic VM spawning
3. Save trajectories and verify SFT format compatibility

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/mcts/smoke_mcts_gpu.py \
        --server-url http://10.100.4.7:15001 \
        --n-tasks 1 --max-steps 5

    # Full 8-GPU run:
    python scripts/mcts/smoke_mcts_gpu.py \
        --server-url http://10.100.4.7:15001 \
        --n-tasks 2 --max-steps 10 --tp 8 --vms-per-task 20
"""

import argparse
import json
import logging
import os
import sys
import time

# Add project root to path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "OSWorld"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mcts_smoke")


def main():
    parser = argparse.ArgumentParser(description="MCTS GPU smoke test")
    parser.add_argument("--server-url", type=str, default="http://10.100.4.7:15001")
    parser.add_argument("--n-tasks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--vms-per-task", type=int, default=10,
                        help="VMs per task (use fewer for quick smoke test)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--model-path", type=str, default="ByteDance-Seed/UI-TARS-1.5-7B")
    parser.add_argument("--task-file", type=str,
                        default="OSWorld/evaluation_examples/test_all_300tasks_noproxy_softreset_clean.json")
    parser.add_argument("--output", type=str, default="docs/research/mcts_smoke_test.json")
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--domain", type=str, default=None,
                        help="Specific domain to test (e.g. libreoffice_impress)")
    args = parser.parse_args()

    # ---- Load tasks ----
    # Task file is {domain: [task_id, ...]}. Individual configs at examples/{domain}/{id}.json
    task_file = os.path.join(PROJ_ROOT, args.task_file)
    logger.info("Loading tasks from %s", task_file)
    with open(task_file) as f:
        raw = json.load(f)
    base_path = os.path.dirname(task_file)
    all_tasks = []
    for domain, task_ids in raw.items():
        for tid in task_ids:
            cfg_path = os.path.join(base_path, "examples", domain, tid + ".json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f2:
                    tc = json.load(f2)
                tc["domain"] = domain
                tc["id"] = tid
                tc["task_id"] = tid
                all_tasks.append(tc)
    logger.info("Loaded %d tasks", len(all_tasks))

    # Select tasks
    from collections import defaultdict
    by_domain = defaultdict(list)
    for t in all_tasks:
        by_domain[t["domain"]].append(t)
    selected = []
    if args.domain and args.domain in by_domain:
        selected = by_domain[args.domain][:args.n_tasks]
    else:
        for domain in sorted(by_domain.keys()):
            if len(selected) >= args.n_tasks:
                break
            selected.append(by_domain[domain][0])

    logger.info("Selected %d tasks:", len(selected))
    for t in selected:
        logger.info("  [%s] %s: %s", t.get("domain", "?"), t["id"][:11],
                     t.get("instruction", "")[:80])

    # ---- Initialize Ray ----
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # ---- Create MCTSEnvClient actors (lightweight, no tokenizer loading) ----
    from verl.mcts.env_client import MCTSEnvClient

    total_vms = args.vms_per_task * len(selected)
    logger.info("Creating %d MCTSEnvClient actors...", total_vms)

    RemoteMCTSEnvClient = ray.remote(num_cpus=0, num_gpus=0)(MCTSEnvClient)
    workers = []
    for i in range(total_vms):
        w = RemoteMCTSEnvClient.remote(
            worker_idx=i,
            remote_server_url=args.server_url,
            slot_id=i,
        )
        workers.append(w)

    # ---- Load processor/tokenizer (CPU-only, safe before GPU init) ----
    logger.info("Loading processor/tokenizer from %s...", args.model_path)
    from transformers import AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # ---- Create vLLM engine(s) ----
    if args.tp == 1:
        # Single GPU: direct LLM instance in the driver process
        logger.info("Loading vLLM model on 1 GPU...")
        from vllm import LLM
        vllm_pool = LLM(
            model=args.model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 3},
            trust_remote_code=True,
            dtype="bfloat16",
        )
    else:
        # Multi-GPU: 1 vLLM engine per GPU, distributed via Ray
        logger.info("Creating VLLMPool with %d GPUs (TP=1 each)...", args.tp)
        from verl.mcts.vllm_pool import VLLMPool
        vllm_pool = VLLMPool(
            n_gpus=args.tp,
            model_path=args.model_path,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=12000,
            limit_images=3,
        )
    logger.info("vLLM ready.")

    # ---- Create MCTS config ----
    from verl.mcts.config import MCTSConfig
    from verl.mcts.orchestrator import MCTSOrchestrator

    config = MCTSConfig(
        vms_per_task=args.vms_per_task,
        max_steps=args.max_steps,
        tensor_parallel_size=args.tp,
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_mem,
        remote_server_urls=[args.server_url],
        max_branch_per_explorer=3,   # reduced for smoke test
        child_branch_budget=1,       # reduced for smoke test
    )

    orchestrator = MCTSOrchestrator(config, vllm_pool, processor, tokenizer)

    # ---- Run MCTS for each task ----
    all_trajectories = []

    for task_idx, task_config in enumerate(selected):
        task_workers = workers[task_idx * args.vms_per_task:(task_idx + 1) * args.vms_per_task]

        logger.info("\n" + "=" * 70)
        logger.info("Task %d/%d: [%s] %s",
                     task_idx + 1, len(selected),
                     task_config.get("domain", "?"),
                     task_config.get("instruction", "")[:80])
        logger.info("=" * 70)

        # Reset + setup all VMs for this task
        logger.info("Resetting %d VMs...", len(task_workers))
        t0 = time.time()
        reset_futures = [w.reset.remote(task_config) for w in task_workers]
        ray.get(reset_futures, timeout=300)
        logger.info("Reset done in %.1fs", time.time() - t0)

        # Run MCTS
        t1 = time.time()
        tree, trajectories = orchestrator.run_task(task_config, task_workers)
        elapsed = time.time() - t1

        logger.info("MCTS completed in %.1fs", elapsed)
        logger.info("Tree: %s", tree.summary())

        # Print tree structure
        for node in tree.all_nodes():
            status = "SUCCESS" if (node.eval_score or 0) > 0 else "FAIL" if node.done else "ACTIVE"
            logger.info("  %s: depth=%d steps=%d eval=%.2f %s",
                         node.node_id, node.depth, len(node.action_history),
                         node.eval_score or 0, status)

        all_trajectories.extend(trajectories)

    # ---- Save trajectories ----
    successful = [t for t in all_trajectories if t.get("eval_result", 0) > 0]
    logger.info("\nTotal trajectories: %d, Successful: %d", len(all_trajectories), len(successful))

    output_path = os.path.join(PROJ_ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save successful trajectories in JSONL format (SFT-compatible)
    success_path = output_path.replace(".json", "_success.jsonl")
    if successful:
        from verl.utils.trajectory_io import TrajectoryWriter
        with TrajectoryWriter(success_path) as writer:
            for traj in successful:
                writer.write(traj)
        logger.info("Saved %d successful trajectories to %s", len(successful), success_path)
    else:
        logger.info("No successful trajectories to save.")

    # Save all trajectories (including failures) for analysis
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "n_tasks": args.n_tasks,
                "max_steps": args.max_steps,
                "vms_per_task": args.vms_per_task,
                "tp": args.tp,
            },
            "trajectories": all_trajectories,
            "summary": {
                "total": len(all_trajectories),
                "successful": len(successful),
                "tasks": len(selected),
            },
        }, f, indent=2, ensure_ascii=False)
    logger.info("Saved all trajectories to %s", output_path)

    # ---- Verify SFT compatibility ----
    if successful:
        logger.info("\n--- SFT Compatibility Check ---")
        from verl.utils.trajectory_io import make_sft_conversations
        traj = successful[0]
        sft_examples = make_sft_conversations(traj, system_prompt="""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## User Instruction
{instruction}""")
        logger.info("SFT examples from 1 trajectory: %d", len(sft_examples))
        for ex in sft_examples[:3]:
            n_msgs = len(ex["messages"])
            label_preview = ex["label"][:80]
            logger.info("  Step %d: %d messages, label=%s...", ex["step"], n_msgs, label_preview)
        logger.info("SFT compatibility: OK")
    else:
        logger.info("No successful trajectories — cannot verify SFT compatibility")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
