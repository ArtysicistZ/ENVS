#!/usr/bin/env python3
"""Build v2.1-compatible metadata for v3 noisy trees so train_mcts_sft_v2.py
can consume them without code changes.

Writes under <trees_parent>/:
  - task_index.json     {task_id: ["<task_id>.json"]}
  - step_masks_v2.json  {}  (empty -> MCTSSFTDatasetV2 defaults to KEEP)
  - mcts_success.jsonl  {"task_id": ..., "sr": float}  per line

SR source: MCTS-SFT v2.1's own n=8 eval
(checkpoints/mcts_sft_v2.1/beta05_2e-6/eval_n8/eval_results_at_0.json).
Using the same SR file as v2.1 keeps the beta weighting identical across runs.
"""
import argparse
import glob
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trees_dir",
        default="checkpoints/mcts_trajectories_v3_noisy/trees",
        help="dir containing <task_id>.json tree files",
    )
    ap.add_argument(
        "--sr_json",
        default="checkpoints/mcts_sft_v2.1/beta05_2e-6/eval_n8/eval_results_at_0.json",
        help="source SR file: {task_id: {success_rate: float, ...}}",
    )
    args = ap.parse_args()

    trees_dir = Path(args.trees_dir).resolve()
    parent = trees_dir.parent
    tree_files = sorted(trees_dir.glob("*.json"))
    if not tree_files:
        raise SystemExit(f"no trees at {trees_dir}")

    # task_index: tree filename == task_id + ".json"
    task_index = {f.stem: [f.name] for f in tree_files}
    (parent / "task_index.json").write_text(json.dumps(task_index, indent=2))
    print(f"wrote task_index.json  ({len(task_index)} tasks)")

    # empty mask file -> MCTSSFTDatasetV2 defaults to KEEP when entry missing
    (parent / "step_masks_v2.json").write_text("{}")
    print("wrote step_masks_v2.json  (empty, all-KEEP)")

    # mcts_success.jsonl from v2.1 SR file
    sr_src = json.load(open(args.sr_json))
    out_path = parent / "mcts_success.jsonl"
    covered = 0
    missing = []
    with open(out_path, "w") as f:
        for tid in task_index:
            rec = sr_src.get(tid)
            if rec is None:
                missing.append(tid)
                sr = 0.0
            else:
                sr = float(rec.get("success_rate", 0.0))
                covered += 1
            f.write(json.dumps({"task_id": tid, "sr": sr}) + "\n")
    print(f"wrote mcts_success.jsonl  ({covered}/{len(task_index)} tasks had SR, "
          f"{len(missing)} defaulted to 0.0)")
    if missing:
        print(f"  missing SR for: {missing[:5]}{'...' if len(missing) > 5 else ''}")


if __name__ == "__main__":
    main()
