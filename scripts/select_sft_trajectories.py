"""
Select a balanced subset of SFT trajectories based on task difficulty.

Harder tasks keep more trajectories to maximize learning signal where the
model needs it most. Within each over-cap task, selects the most diverse
trajectories by action sequence similarity (greedy maximin).

Difficulty tiers and caps:
    Easy   (SR > 50%):     cap  6  — model already does well
    Medium (SR 25-50%):    cap  8  — moderate representation
    Hard   (SR 12.5-25%):  cap 10  — more room to learn
    V.Hard (SR < 12.5%):   cap 15  — every trajectory is precious

Usage:
    python scripts/select_sft_trajectories.py \
        --input checkpoints/arpo-inference/sft_trajectories_audited.jsonl \
        --output checkpoints/arpo-inference/sft_trajectories_selected.jsonl

    # Custom caps:
    python scripts/select_sft_trajectories.py \
        --caps 0.50:6 0.25:8 0.125:10 0:15
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict


def load_eval_results(eval_glob):
    """Aggregate per-task success rates across all eval result files."""
    task_stats = defaultdict(lambda: {"n_success": 0, "n_attempts": 0})

    eval_files = sorted(glob.glob(eval_glob))
    if not eval_files:
        raise FileNotFoundError(f"No eval result files found matching: {eval_glob}")

    for path in eval_files:
        with open(path) as f:
            results = json.load(f)
        for task_id, v in results.items():
            task_stats[task_id]["n_success"] += v.get("n_success", 0)
            task_stats[task_id]["n_attempts"] += v.get("n_attempts", 0)

    for stats in task_stats.values():
        n = stats["n_attempts"]
        stats["success_rate"] = stats["n_success"] / n if n > 0 else 0.0

    print(f"Loaded {len(eval_files)} eval files, {len(task_stats)} tasks with results")
    return dict(task_stats)


def parse_caps(caps_str_list):
    """Parse cap specs like ['0.50:6', '0.25:8', '0.125:10', '0:15'].

    Returns list of (threshold, cap) sorted descending by threshold.
    A task with SR > threshold gets that cap. First match wins.
    """
    tiers = []
    for spec in caps_str_list:
        thresh, cap = spec.split(":")
        tiers.append((float(thresh), int(cap)))
    tiers.sort(key=lambda x: -x[0])  # descending by threshold
    return tiers


def get_cap(success_rate, tiers):
    """Return trajectory cap for a given success rate."""
    for thresh, cap in tiers:
        if success_rate > thresh:
            return cap
    # Below all thresholds — use last tier's cap
    return tiers[-1][1] if tiers else 999


def extract_action_types(trajectory):
    """Extract ordered action types from trajectory steps."""
    action_types = []
    for step in trajectory.get("steps", []):
        action_text = step.get("action", "")
        m = re.search(r"Action:\s*(\w+)\(", action_text)
        action_types.append(m.group(1) if m else "unknown")
    return tuple(action_types)


def action_bigrams(action_types):
    """Get set of consecutive action type pairs."""
    if len(action_types) <= 1:
        return {(action_types[0],)} if action_types else set()
    return set(zip(action_types, action_types[1:]))


def diversity_distance(seq_a, seq_b):
    """Combined diversity measure between two trajectories.

    70% action bigram Jaccard distance + 30% normalized step count difference.
    """
    bg_a = action_bigrams(seq_a)
    bg_b = action_bigrams(seq_b)

    if not bg_a and not bg_b:
        jd = 0.0
    else:
        intersection = len(bg_a & bg_b)
        union = len(bg_a | bg_b)
        jd = 1.0 - (intersection / union) if union > 0 else 0.0

    step_diff = abs(len(seq_a) - len(seq_b)) / max(len(seq_a), len(seq_b), 1)

    return 0.7 * jd + 0.3 * step_diff


def select_diverse(trajectories, k):
    """Greedy maximin diversity selection.

    Starts with median-length trajectory, then iteratively adds the
    trajectory most different from all already-selected ones.
    """
    n = len(trajectories)
    if n <= k:
        return list(range(n))

    sequences = [extract_action_types(t) for t in trajectories]
    step_counts = [len(s) for s in sequences]

    # Precompute pairwise distances
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = diversity_distance(sequences[i], sequences[j])
            dist[i][j] = d
            dist[j][i] = d

    # Start with median step-count trajectory
    sorted_by_steps = sorted(range(n), key=lambda i: step_counts[i])
    selected = [sorted_by_steps[n // 2]]
    remaining = set(range(n)) - set(selected)

    while len(selected) < k:
        best_idx = -1
        best_min_dist = -1.0

        for i in remaining:
            min_dist = min(dist[i][j] for j in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        selected.append(best_idx)
        remaining.discard(best_idx)

    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description="Select balanced SFT trajectory subset")
    parser.add_argument(
        "--input",
        default="checkpoints/arpo-inference/sft_trajectories_audited.jsonl",
        help="Input audited trajectory JSONL",
    )
    parser.add_argument(
        "--eval_glob",
        default="checkpoints/arpo-inference/*/eval_results_at_0.json",
        help="Glob pattern for eval result files",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/arpo-inference/sft_trajectories_selected.jsonl",
        help="Output selected trajectory JSONL",
    )
    parser.add_argument(
        "--metadata",
        default="checkpoints/arpo-inference/sft_selection_metadata.json",
        help="Output selection metadata JSON",
    )
    parser.add_argument(
        "--caps",
        nargs="+",
        default=["0.50:6", "0.25:8", "0.125:10", "0:15"],
        help="Difficulty tier caps as 'threshold:cap' pairs (e.g., '0.50:6 0.25:8 0.125:10 0:15')",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print selection plan without writing files",
    )
    args = parser.parse_args()

    tiers = parse_caps(args.caps)
    print(f"Difficulty tiers: {tiers}")

    # Load eval results for difficulty estimation
    task_stats = load_eval_results(args.eval_glob)

    # Load trajectories
    trajectories = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    print(f"Loaded {len(trajectories)} trajectories from {args.input}")

    # Group by task
    task_trajs = defaultdict(list)
    for i, t in enumerate(trajectories):
        task_trajs[t["task_id"]].append((i, t))

    # Select per task
    selected_indices = []
    selection_details = {}
    tier_labels = {0.50: "easy (>50%)", 0.25: "medium (25-50%)", 0.125: "hard (12.5-25%)", 0: "v.hard (<12.5%)"}

    for task_id in sorted(task_trajs.keys()):
        items = task_trajs[task_id]
        sr = task_stats.get(task_id, {}).get("success_rate", 0.0)
        cap = get_cap(sr, tiers)
        n_available = len(items)

        if n_available <= cap:
            local_indices = list(range(n_available))
        else:
            task_trajectories = [t for _, t in items]
            local_indices = select_diverse(task_trajectories, cap)

        selected_global = [items[i][0] for i in local_indices]
        selected_indices.extend(selected_global)

        # Determine tier label
        tier_label = "v.hard (<12.5%)"
        for thresh, _ in tiers:
            if sr > thresh:
                tier_label = tier_labels.get(thresh, f">{thresh}")
                break

        selection_details[task_id] = {
            "success_rate": round(sr, 4),
            "tier": tier_label,
            "cap": cap,
            "n_available": n_available,
            "n_selected": len(local_indices),
            "trimmed": n_available > cap,
        }

    selected_indices.sort()

    # Tier summary
    tier_summary = defaultdict(lambda: {"n_tasks": 0, "n_available": 0, "n_selected": 0, "n_trimmed": 0})
    for details in selection_details.values():
        tier = details["tier"]
        tier_summary[tier]["n_tasks"] += 1
        tier_summary[tier]["n_available"] += details["n_available"]
        tier_summary[tier]["n_selected"] += details["n_selected"]
        if details["trimmed"]:
            tier_summary[tier]["n_trimmed"] += 1

    # Print summary
    print(f"\n{'Tier':<20} {'Tasks':>5} {'Available':>9} {'Selected':>8} {'Trimmed':>7}")
    print("-" * 55)
    for tier in ["easy (>50%)", "medium (25-50%)", "hard (12.5-25%)", "v.hard (<12.5%)"]:
        s = tier_summary.get(tier, {"n_tasks": 0, "n_available": 0, "n_selected": 0, "n_trimmed": 0})
        print(f"{tier:<20} {s['n_tasks']:>5} {s['n_available']:>9} {s['n_selected']:>8} {s['n_trimmed']:>7}")
    total_sel = sum(s["n_selected"] for s in tier_summary.values())
    print(f"{'TOTAL':<20} {len(task_trajs):>5} {len(trajectories):>9} {total_sel:>8}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        # Print per-task details for trimmed tasks
        print(f"\nTrimmed tasks:")
        for task_id, d in sorted(selection_details.items(), key=lambda x: x[1]["success_rate"]):
            if d["trimmed"]:
                print(f"  {task_id[:12]}  SR={d['success_rate']:.2%}  {d['n_available']} -> {d['n_selected']}")
        return

    # Write selected trajectories
    with open(args.output, "w") as f:
        for idx in selected_indices:
            f.write(json.dumps(trajectories[idx]) + "\n")
    print(f"\nWrote {len(selected_indices)} trajectories to {args.output}")

    # Write metadata
    metadata = {
        "source": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "eval_glob": args.eval_glob,
        "caps_spec": args.caps,
        "tiers": [{"threshold": t, "cap": c} for t, c in tiers],
        "total_input": len(trajectories),
        "total_selected": len(selected_indices),
        "n_tasks": len(task_trajs),
        "tier_summary": {k: dict(v) for k, v in tier_summary.items()},
        "per_task": selection_details,
    }
    with open(args.metadata, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {args.metadata}")


if __name__ == "__main__":
    main()
