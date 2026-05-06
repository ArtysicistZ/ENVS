#!/usr/bin/env python3
"""Build v2.1-compatible metadata for the combined v3-noisy dataset.

Combines three collection rounds into a single trees dir for
train_mcts_sft_v2.py:
  - round 1:  checkpoints/mcts_trajectories_v3_noisy/trees
  - round 2a: checkpoints/mcts_trajectories_v3_noisy_round2a/trees
  - round 2b: checkpoints/mcts_trajectories_v3_noisy_round2b/trees

Layout written under --out_dir (default mcts_trajectories_v3_noisy_combined/):
  trees/<task_id>_<round>.json  (symlinks to source trees)
  task_index.json               {task_id: [tree_filenames...]}
  step_masks_v2.json            {}  (empty -> MCTSSFTDatasetV2 defaults KEEP)
  mcts_success.jsonl            SR sourced from v2.1 eval_n8 (same rollout policy)
"""
import argparse
import json
import os
from pathlib import Path


ROUND_DIRS = [
    ("r1",  "checkpoints/mcts_trajectories_v3_noisy/trees"),
    ("r2a", "checkpoints/mcts_trajectories_v3_noisy_round2a/trees"),
    ("r2b", "checkpoints/mcts_trajectories_v3_noisy_round2b/trees"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--round_dirs",
        nargs="*",
        default=None,
        help="Override source round dirs as 'label:path' pairs",
    )
    ap.add_argument(
        "--out_dir",
        default="checkpoints/mcts_trajectories_v3_noisy_combined",
        help="combined output dir (trees/ + metadata)",
    )
    ap.add_argument(
        "--sr_json",
        default="checkpoints/mcts_sft_v2.1/beta05_2e-6/eval_n8/eval_results_at_0.json",
        help="per-task SR source (same policy used for noisy rollouts)",
    )
    args = ap.parse_args()

    if args.round_dirs:
        rounds = [tuple(s.split(":", 1)) for s in args.round_dirs]
    else:
        rounds = ROUND_DIRS

    out_dir = Path(args.out_dir).resolve()
    trees_out = out_dir / "trees"
    trees_out.mkdir(parents=True, exist_ok=True)

    # Build task_index, symlinking <task_id>_<round>.json per source tree.
    task_index = {}
    per_round_counts = {}
    for label, src in rounds:
        src_dir = Path(src).resolve()
        if not src_dir.is_dir():
            print(f"skip {label}: {src_dir} missing")
            per_round_counts[label] = 0
            continue
        n = 0
        for tree_file in sorted(src_dir.glob("*.json")):
            tid = tree_file.stem
            link_name = f"{tid}_{label}.json"
            link_path = trees_out / link_name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(tree_file)
            task_index.setdefault(tid, []).append(link_name)
            n += 1
        per_round_counts[label] = n
        print(f"linked {label}: {n} trees from {src_dir}")

    for tid in task_index:
        task_index[tid].sort()

    (out_dir / "task_index.json").write_text(json.dumps(task_index, indent=2))
    print(f"wrote task_index.json  ({len(task_index)} tasks, "
          f"{sum(len(v) for v in task_index.values())} total trees)")

    (out_dir / "step_masks_v2.json").write_text("{}")
    print("wrote step_masks_v2.json  (empty, all-KEEP)")

    sr_src = json.load(open(args.sr_json))
    covered, missing = 0, []
    with (out_dir / "mcts_success.jsonl").open("w") as f:
        for tid in task_index:
            rec = sr_src.get(tid)
            if rec is None:
                missing.append(tid)
                sr = 0.0
            else:
                sr = float(rec.get("success_rate", 0.0))
                covered += 1
            f.write(json.dumps({"task_id": tid, "sr": sr}) + "\n")
    print(f"wrote mcts_success.jsonl  ({covered}/{len(task_index)} had SR, "
          f"{len(missing)} defaulted to 0.0)")
    if missing:
        print(f"  missing SR: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    print()
    print(f"combined root: {out_dir}")
    print(f"  per-round trees: " + ", ".join(
        f"{lbl}={n}" for lbl, n in per_round_counts.items()))


if __name__ == "__main__":
    main()
