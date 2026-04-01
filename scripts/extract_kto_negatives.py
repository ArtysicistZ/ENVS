"""Extract Phase 1 KTO negatives from MCTS trees.

At each contrastive branch point (one child succeeds, another fails),
the failed child's first action is a definitively wrong action — we know
a better action exists (the successful sibling's action).

Output: kto_negatives_phase1.jsonl with one entry per negative step.
Each entry is an episode dict compatible with expand_episode().

Usage:
    python scripts/extract_kto_negatives.py \
        --tree_dir checkpoints/mcts_trajectories_v2/combined_all/trees \
        --task_index_path checkpoints/mcts_trajectories_v2/combined_all/task_index.json \
        --output_dir checkpoints/mcts_kto
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from verl.mcts.tree_io import load_mcts_tree, reconstruct_steps, _get_ancestor_chain


def extract_branch_negatives(tree_data, tree_filename):
    """Extract negative step examples from contrastive branch points.

    For each branch point where at least one child succeeds and one fails,
    yield one negative example per failed child: the shared prefix + the
    failed child's first (wrong) action.

    Returns list of dicts with:
        - task_id, instruction, tree_filename
        - parent_node_id, fail_node_id, winner_node_id
        - steps: shared prefix steps (context)
        - neg_action: the wrong action (screenshot + action text)
        - winner_q, loser_q
    """
    nodes = tree_data["nodes"]
    task_id = tree_data["task_id"]
    instruction = tree_data["instruction"]
    limit_images = tree_data.get("limit_images", 3)

    negatives = []

    for node_id, node in nodes.items():
        children_ids = [c for c in node.get("children_ids", []) if c in nodes]
        if len(children_ids) < 2:
            continue

        # Separate successful and failed children
        success_children = []
        fail_children = []
        for cid in children_ids:
            child = nodes[cid]
            q = child.get("q_value") or child.get("eval_score") or 0
            if q > 0:
                success_children.append(child)
            elif child.get("eval_score") is not None and child["eval_score"] == 0:
                fail_children.append(child)
            elif q == 0 and child.get("done"):
                fail_children.append(child)

        if not success_children or not fail_children:
            continue

        # Best successful child (for reference)
        best_winner = max(success_children,
                          key=lambda c: c.get("q_value") or c.get("eval_score") or 0)

        # Shared prefix: parent's path up to branch point
        # All children saw the same prefix (parent's steps up to parent_steps_at_branch)
        # Use the winner to reconstruct the prefix
        winner_full = reconstruct_steps(tree_data, best_winner["node_id"])
        prefix_len = len(winner_full) - len(best_winner.get("own_steps", []))
        prefix_steps = winner_full[:prefix_len]

        for fail_child in fail_children:
            fail_own = fail_child.get("own_steps", [])
            if not fail_own:
                continue

            # The negative example: prefix + failed child's FIRST action
            # This is the action at the exact state where a sibling succeeded
            neg_step = fail_own[0]

            # Build episode: prefix as context, neg_step as the step to train on
            all_steps = prefix_steps + [neg_step]

            negatives.append({
                "task_id": task_id,
                "instruction": instruction,
                "limit_images": limit_images,
                "tree_filename": tree_filename,
                "parent_node_id": node_id,
                "fail_node_id": fail_child["node_id"],
                "winner_node_id": best_winner["node_id"],
                "winner_q": best_winner.get("q_value") or best_winner.get("eval_score") or 0,
                "loser_q": fail_child.get("q_value") or fail_child.get("eval_score") or 0,
                "steps": all_steps,  # prefix + wrong action
                "neg_step_idx": len(prefix_steps),  # index of the wrong step
                "source": "branch_negative",
            })

    return negatives


def main():
    parser = argparse.ArgumentParser(description="Extract Phase 1 KTO negatives")
    parser.add_argument("--tree_dir", default="checkpoints/mcts_trajectories_v2/combined_all/trees")
    parser.add_argument("--task_index_path", default="checkpoints/mcts_trajectories_v2/combined_all/task_index.json")
    parser.add_argument("--output_dir", default="checkpoints/mcts_kto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.task_index_path) as f:
        task_index = json.load(f)

    all_negatives = []
    n_trees = 0

    for task_id, tree_files in task_index.items():
        for tree_filename in tree_files:
            tree_path = os.path.join(args.tree_dir, tree_filename)
            real_path = os.path.realpath(tree_path)
            if not os.path.exists(real_path):
                continue

            tree_data = load_mcts_tree(real_path)
            negs = extract_branch_negatives(tree_data, tree_filename)
            all_negatives.extend(negs)
            n_trees += 1

    # Save negatives
    output_path = os.path.join(args.output_dir, "kto_negatives_phase1.jsonl")
    with open(output_path, "w") as f:
        for neg in all_negatives:
            f.write(json.dumps(neg, ensure_ascii=False) + "\n")

    # Stats
    tasks_with_negs = len(set(n["task_id"] for n in all_negatives))
    print(f"Extracted {len(all_negatives)} Phase 1 negatives from {n_trees} trees")
    print(f"Tasks with negatives: {tasks_with_negs}")
    print(f"Saved to: {output_path}")

    # Save stats
    stats = {
        "n_negatives": len(all_negatives),
        "n_trees": n_trees,
        "n_tasks": tasks_with_negs,
        "output_path": output_path,
    }
    with open(os.path.join(args.output_dir, "phase1_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
