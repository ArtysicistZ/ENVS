#!/usr/bin/env python3
"""Build a per-task fraction-subsampled CLEAN-MCTS dataset.

For each task, randomly keep `round(n_leaves * fraction)` of its successful
leaves; mark all other successful leaves as `eval_score=0` (drops them from
SFT). Empty mask file (all KEEP), preserving the v2.1_no_mask convention.

Use for the data-quantity ablation sweep (30 / 45 / 60 % of v2 raw, paired
against the existing v2_clean_aligned at ~75 % and v2.1_no_mask at 100 %).

Output layout (mirrors prepare_v2_clean_aligned.py):
  trees/<task_id>_<round>.json   (copy of v2 trees with eval_score=0 on
                                   leaves that fell out of the subsample)
  task_index.json                — keys = trainable tasks, values = list
                                   of v2 tree filenames per task
  step_masks_v2.json             — empty {} (all KEEP)
  mcts_success.jsonl             — SR copied from v2 combined_all

Usage:
  python scripts/mcts/prepare_v2_subsample_fraction.py \\
      --fraction 0.30 \\
      --out_dir checkpoints/mcts_trajectories_v2_subsample_30pct
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def count_succ_leaves(tree_dir: Path, idx: dict) -> dict:
    out = defaultdict(int)
    for tid, files in idx.items():
        for fn in files:
            tree = json.loads((tree_dir / fn).read_text())
            for n in tree["nodes"].values():
                if (n.get("eval_score") or 0) > 0:
                    out[tid] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_dir", default="/mnt/kevinzyz/arpo_local/checkpoints/mcts_trajectories_v2/combined_all")
    ap.add_argument("--out_dir", required=True, help="output dir, e.g. checkpoints/mcts_trajectories_v2_subsample_30pct")
    ap.add_argument("--fraction", type=float, required=True, help="per-task leaf retention fraction, e.g. 0.30")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert 0.0 < args.fraction <= 1.0, f"fraction must be in (0, 1], got {args.fraction}"

    v2_dir = Path(args.v2_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trees").mkdir(exist_ok=True)

    v2_idx = json.loads((v2_dir / "task_index.json").read_text())

    print(f"Counting successful leaves in v2 (clean) trees at {v2_dir} ...")
    v2_count = count_succ_leaves(v2_dir / "trees", v2_idx)

    tasks = sorted(v2_count.keys())
    total_v2 = sum(v2_count.values())
    print(f"v2 raw leaves on these tasks: {total_v2} across {len(tasks)} tasks")

    # Per-task target: round(n * fraction), with min 1 if n>=1 (preserve task coverage)
    targets = {}
    total_target = 0
    for t in tasks:
        n = v2_count[t]
        keep = max(1, int(round(n * args.fraction))) if n > 0 else 0
        keep = min(keep, n)
        targets[t] = keep
        total_target += keep
    print(f"target leaves at fraction={args.fraction:.2f}: {total_target} (~{100*total_target/total_v2:.1f}% of raw)")

    rng_master = random.Random(args.seed)
    selected = {}
    for t in tasks:
        all_leaves = []
        for fn in v2_idx.get(t, []):
            tree = json.loads((v2_dir / "trees" / fn).read_text())
            for nid, n in tree["nodes"].items():
                if (n.get("eval_score") or 0) > 0:
                    all_leaves.append((fn, nid))
        cap = targets[t]
        if len(all_leaves) <= cap:
            kept = set(all_leaves)
        else:
            rng = random.Random(rng_master.random())
            kept = set(rng.sample(all_leaves, cap))
        selected[t] = kept

    out_idx = {}
    n_dropped = 0
    n_kept = 0
    for t in tasks:
        out_idx[t] = list(v2_idx.get(t, []))
        for fn in v2_idx.get(t, []):
            tree = json.loads((v2_dir / "trees" / fn).read_text())
            keep_set = {leaf for (f, leaf) in selected[t] if f == fn}
            for nid, n in tree["nodes"].items():
                if (n.get("eval_score") or 0) > 0:
                    if nid in keep_set:
                        n_kept += 1
                    else:
                        n["eval_score"] = 0
                        n_dropped += 1
            (out / "trees" / fn).write_text(json.dumps(tree))

    print(f"Wrote {len(out_idx)} task entries; kept={n_kept} leaves, dropped={n_dropped}")

    (out / "task_index.json").write_text(json.dumps(out_idx, indent=2))
    (out / "step_masks_v2.json").write_text("{}")

    src_sr = v2_dir / "mcts_success.jsonl"
    if src_sr.exists():
        (out / "mcts_success.jsonl").write_text(src_sr.read_text())
    else:
        print("warning: source mcts_success.jsonl missing")

    print(f"Done. Output: {out}")
    print(f"Per-task target sample (first 10):")
    for t in tasks[:10]:
        print(f"  {t}: v2={v2_count[t]} target={targets[t]}")


if __name__ == "__main__":
    main()
