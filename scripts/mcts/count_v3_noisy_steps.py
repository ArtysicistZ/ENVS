#!/usr/bin/env python3
"""Instantiate MCTSSFTDatasetV2 against the v3_noisy metadata just to count
unique KEEP steps (N) — used to compute the matched learning rate for the
v3-noisy SFT run. No model load, no training.
"""
import argparse
import json
import os
import sys

# Repo root on sys.path so we can import the training script's Dataset class.
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

from train_mcts_sft_v2 import MCTSSFTDatasetV2


class _NullTokenizer:
    pad_token_id = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree_dir", default="checkpoints/mcts_trajectories_v3_noisy/trees")
    ap.add_argument("--task_index", default="checkpoints/mcts_trajectories_v3_noisy/task_index.json")
    ap.add_argument("--mask_path", default="checkpoints/mcts_trajectories_v3_noisy/step_masks_v2.json")
    ap.add_argument("--sr_path", default="checkpoints/mcts_trajectories_v3_noisy/mcts_success.jsonl")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--max_step_ratio", type=float, default=2.0)
    ap.add_argument("--limit_images", type=int, default=3)
    ap.add_argument("--v21_keep_steps", type=int, default=20903)
    ap.add_argument("--v21_lr", type=float, default=2e-6)
    ap.add_argument("--global_batch", type=int, default=32)
    args = ap.parse_args()

    ds = MCTSSFTDatasetV2(
        tree_dir=args.tree_dir,
        task_index_path=args.task_index,
        mask_path=args.mask_path,
        sr_path=args.sr_path,
        tokenizer=_NullTokenizer(),
        processor=None,
        limit_images=args.limit_images,
        beta=args.beta,
        max_step_ratio=args.max_step_ratio,
    )

    N = len(ds)
    v21_N = args.v21_keep_steps
    new_lr = args.v21_lr * (v21_N / max(1, N))
    updates_v21 = (v21_N + args.global_batch - 1) // args.global_batch
    updates_new = (N + args.global_batch - 1) // args.global_batch
    grad_budget_v21 = args.v21_lr * updates_v21
    grad_budget_new = new_lr * updates_new

    print()
    print("=" * 60)
    print(f"v3_noisy unique KEEP steps (N):   {N}")
    print(f"v2.1 reference KEEP steps:        {v21_N}")
    print(f"ratio v21/new:                    {v21_N/N:.4f}")
    print()
    print(f"v2.1  LR={args.v21_lr:.3e}  updates={updates_v21}  LR*upd={grad_budget_v21:.6f}")
    print(f"v3    LR={new_lr:.3e}  updates={updates_new}  LR*upd={grad_budget_new:.6f}")
    print(f"total-gradient match check: {grad_budget_new / grad_budget_v21:.4f}  (want ~1.000)")
    print("=" * 60)

    # Write the LR out for config-templating
    out = {"N": N, "v21_N": v21_N, "new_lr": new_lr, "updates_new": updates_new}
    with open("checkpoints/mcts_trajectories_v3_noisy/_v3_lr_calc.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: checkpoints/mcts_trajectories_v3_noisy/_v3_lr_calc.json")


if __name__ == "__main__":
    main()
