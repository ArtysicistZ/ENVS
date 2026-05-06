#!/usr/bin/env python3
"""Build an aligned CLEAN-MCTS dataset where each task has at most
   cap[task] = min(v2_raw_leaves[task], v3_noisy_leaves[task])
successful leaves. Random sample from v2 (clean) trees with fixed seed.

This is the clean-side counterpart to `prepare_v3_noisy_aligned.py`.
Together they form the aligned clean-vs-noisy MCTS-SFT ablation:
both sides have the same per-task leaf budget and no LLM mask, so the
only difference is collection-environment (clean vs noisy) and rollout
policy (base UI-TARS vs MCTS-SFT v2.1).

Output layout (mirrors prepare_v3_noisy_aligned.py):
  trees/<task_id>_<round>.json   (copy of v2 trees with eval_score=0 on
                                   leaves we are not keeping; ancestor
                                   nodes preserved for kept leaves)
  task_index.json                — keys = 86 trainable tasks, values = list
                                   of v2 tree filenames per task
  step_masks_v2.json             — empty {} (all KEEP)
  mcts_success.jsonl             — SR copied from v2 combined_all
                                   (per-task SR derived from same v2.1
                                   evaluator the original v2.1 SFT used)

Usage:
  python scripts/mcts/prepare_v2_clean_aligned.py
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
    ap.add_argument("--v2_dir", default="/mnt/kevinzyz/arpo_main/checkpoints/mcts_trajectories_v2/combined_all")
    ap.add_argument("--v3_dir", default="/mnt/kevinzyz/arpo_local/checkpoints/mcts_trajectories_v3_noisy_combined")
    ap.add_argument("--out_dir", default="/mnt/kevinzyz/arpo_local/checkpoints/mcts_trajectories_v2_clean_aligned")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v2_dir = Path(args.v2_dir)
    v3_dir = Path(args.v3_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trees").mkdir(exist_ok=True)

    v2_idx = json.loads((v2_dir / "task_index.json").read_text())
    v3_idx = json.loads((v3_dir / "task_index.json").read_text())

    print("Counting successful leaves in v2 (clean) and v3 (noisy)...")
    v2_count = count_succ_leaves(v2_dir / "trees", v2_idx)
    v3_count = count_succ_leaves(v3_dir / "trees", v3_idx)

    tasks = sorted(set(v2_count.keys()) | set(v3_count.keys()))
    caps = {t: min(v2_count.get(t, 0), v3_count.get(t, 0)) for t in tasks}
    total_v2 = sum(v2_count.get(t, 0) for t in tasks)
    total_v3 = sum(v3_count.get(t, 0) for t in tasks)
    total_cap = sum(caps.values())
    print(f"v2 raw leaves on these tasks: {total_v2}")
    print(f"v3 leaves on these tasks:     {total_v3}")
    print(f"aligned cap (sum of mins):    {total_cap}")

    rng_master = random.Random(args.seed)
    selected: dict = {}
    for t in tasks:
        all_leaves = []
        for fn in v2_idx.get(t, []):
            tree = json.loads((v2_dir / "trees" / fn).read_text())
            for nid, n in tree["nodes"].items():
                if (n.get("eval_score") or 0) > 0:
                    all_leaves.append((fn, nid))
        cap = caps[t]
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
    print(f"Per-task cap sample (first 10):")
    for t in tasks[:10]:
        print(f"  {t}: v2={v2_count.get(t,0)} v3={v3_count.get(t,0)} cap={caps[t]}")


if __name__ == "__main__":
    main()
