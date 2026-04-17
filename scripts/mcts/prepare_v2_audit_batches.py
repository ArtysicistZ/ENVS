"""Prepare audit batches for v2 step mask auditing.

Reads all trees from combined_all/, identifies nodes on successful paths,
extracts their ancestor context + own_steps (action text only, no screenshots),
and creates ~200 batch files for parallel agent auditing.

Each agent reads its batch, classifies each own_step as KEEP or REMOVE,
and returns per-node masks.

Usage:
    python scripts/mcts/prepare_v2_audit_batches.py
"""

import json
import glob
import os
from collections import defaultdict


COMBINED_DIR = "checkpoints/mcts_trajectories_v2/combined_all"
BATCH_DIR = os.path.join(COMBINED_DIR, "audit_batches")
N_AGENTS = 200
TARGET_STEPS_PER_AGENT = 180  # soft target


def get_nodes_on_success_paths(tree_data):
    """Return set of node_ids on paths to successful leaves."""
    nodes = tree_data["nodes"]
    succ = {nid for nid, n in nodes.items()
            if n.get("eval_score") and n["eval_score"] > 0}
    on_path = set()
    for lid in succ:
        nid = lid
        while nid is not None:
            if nid in on_path:
                break
            on_path.add(nid)
            nid = nodes[nid].get("parent_id")
    return on_path


def get_ancestor_context(tree_data, node_id):
    """Get action text for all ancestor steps (context, not to be labeled)."""
    nodes = tree_data["nodes"]
    # Walk up to root
    chain = []
    cur = node_id
    while cur is not None:
        chain.append(cur)
        cur = nodes[cur].get("parent_id")
    chain.reverse()
    # Remove the target node itself — we only want ancestors
    chain = chain[:-1]

    context_actions = []
    for i, nid in enumerate(chain):
        n = nodes[nid]
        own_steps = n.get("own_steps", [])
        # If not the last ancestor, slice to what the next node saw
        if i < len(chain) - 1:
            next_nid = chain[i + 1]
            cut = nodes[next_nid].get("parent_steps_at_branch", len(own_steps))
            own_steps = own_steps[:cut]
        else:
            # Last ancestor before target — slice to what target node saw
            target = nodes[node_id]
            cut = target.get("parent_steps_at_branch", len(own_steps))
            own_steps = own_steps[:cut]

        for s in own_steps:
            context_actions.append(s.get("action", ""))
    return context_actions


def main():
    os.makedirs(BATCH_DIR, exist_ok=True)

    tree_dir = os.path.join(COMBINED_DIR, "trees")
    files = sorted(glob.glob(os.path.join(tree_dir, "*.json")))
    print(f"Loading {len(files)} tree files...")

    # Collect all audit items
    audit_items = []
    for f in files:
        fname = os.path.basename(f)
        parts = fname.replace(".json", "").rsplit("_", 1)
        round_label = parts[1] if len(parts) > 1 else "base"

        with open(os.path.realpath(f)) as fh:
            tree_data = json.load(fh)

        task_id = tree_data["task_id"]
        instruction = tree_data.get("instruction", "")
        nodes = tree_data["nodes"]
        on_path = get_nodes_on_success_paths(tree_data)

        for nid in on_path:
            n = nodes[nid]
            own_steps = n.get("own_steps", [])
            if not own_steps:
                continue

            # Extract action text for own_steps (no screenshots)
            own_actions = [s.get("action", "") for s in own_steps]

            # Get ancestor context
            context_actions = get_ancestor_context(tree_data, nid)

            audit_items.append({
                "key": f"{task_id}:{round_label}:{nid}",
                "task_id": task_id,
                "round": round_label,
                "node_id": nid,
                "instruction": instruction,
                "depth": n.get("depth", 0),
                "n_context_steps": len(context_actions),
                "context_actions": context_actions,
                "n_own_steps": len(own_actions),
                "own_actions": own_actions,
            })

    print(f"Total audit items: {len(audit_items)}")
    print(f"Total own_steps to label: {sum(it['n_own_steps'] for it in audit_items)}")

    # Sort by n_own_steps descending for balanced batching
    audit_items.sort(key=lambda x: -x["n_own_steps"])

    # Greedy bin-packing into N_AGENTS batches
    batches = [{"items": [], "total_steps": 0} for _ in range(N_AGENTS)]

    for item in audit_items:
        # Put in the batch with fewest steps so far
        min_batch = min(batches, key=lambda b: b["total_steps"])
        min_batch["items"].append(item)
        min_batch["total_steps"] += item["n_own_steps"]

    # Write batches
    for i, batch in enumerate(batches):
        batch_data = {
            "batch_id": f"v2_audit_{i:03d}",
            "n_nodes": len(batch["items"]),
            "n_steps_to_label": batch["total_steps"],
            "items": batch["items"],
        }
        path = os.path.join(BATCH_DIR, f"v2_audit_{i:03d}.json")
        with open(path, "w") as f:
            json.dump(batch_data, f, ensure_ascii=False)

    # Summary
    steps_per_batch = [b["total_steps"] for b in batches]
    nodes_per_batch = [len(b["items"]) for b in batches]
    print(f"\nCreated {N_AGENTS} batch files in {BATCH_DIR}/")
    print(f"Steps per batch: min={min(steps_per_batch)}, max={max(steps_per_batch)}, "
          f"avg={sum(steps_per_batch)/len(steps_per_batch):.0f}")
    print(f"Nodes per batch: min={min(nodes_per_batch)}, max={max(nodes_per_batch)}, "
          f"avg={sum(nodes_per_batch)/len(nodes_per_batch):.0f}")


if __name__ == "__main__":
    main()
