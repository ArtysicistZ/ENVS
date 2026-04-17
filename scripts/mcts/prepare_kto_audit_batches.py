"""Prepare Phase 2 KTO audit batches from failed trajectory steps.

For each failed node whose parent is on a success path, extract own_steps[1:3]
(first 2 steps after the divergent action). Package into batches for agent audit
using the SAME format and criteria as the v2 KEEP/REMOVE audit.

Agents see: instruction + context actions + actions to classify.
They do NOT know if the trajectory succeeded or failed.

Usage:
    python scripts/mcts/prepare_kto_audit_batches.py \
        --output_dir checkpoints/mcts_kto/audit_batches
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from verl.mcts.tree_io import load_mcts_tree, _get_ancestor_chain


MAX_STEPS_AFTER_DIVERGE = 2
N_BATCHES = 60


def extract_candidates(tree_data, tree_filename):
    """Extract candidate steps for audit from failed nodes near success paths."""
    nodes = tree_data["nodes"]
    task_id = tree_data["task_id"]
    instruction = tree_data["instruction"]

    # Find nodes on any success path
    succ_nodes = {nid for nid, n in nodes.items()
                  if n.get("eval_score") and n["eval_score"] > 0}
    on_success_path = set()
    for snid in succ_nodes:
        cur = snid
        while cur is not None:
            on_success_path.add(cur)
            cur = nodes[cur]["parent_id"]

    candidates = []

    for nid, node in nodes.items():
        # Only failed nodes whose parent is on a success path
        if not (node.get("eval_score") is not None and node["eval_score"] == 0):
            continue
        parent_id = node["parent_id"]
        if not parent_id or parent_id not in on_success_path:
            continue

        own_steps = node.get("own_steps", [])
        if len(own_steps) < 2:
            continue  # Need at least step 0 (divergent) + step 1 (candidate)

        # Build context: ancestor actions up to this node
        chain = _get_ancestor_chain(tree_data, nid)
        context_actions = []
        for i, anc_id in enumerate(chain[:-1]):  # exclude the node itself
            anc = nodes[anc_id]
            anc_own = anc.get("own_steps", [])
            if i < len(chain) - 2:
                # Slice at branch point
                next_id = chain[i + 1]
                cut = nodes[next_id].get("parent_steps_at_branch", len(anc_own))
                for s in anc_own[:cut]:
                    context_actions.append(s["action"])
            else:
                for s in anc_own:
                    context_actions.append(s["action"])

        # own_steps[0] is the divergent action (already Phase 1 negative)
        # Include it in context, then classify own_steps[1:3]
        context_actions.append(own_steps[0]["action"])

        round_label = tree_filename.replace(task_id + "_", "").replace(".json", "")
        actions_to_classify = []
        step_indices = []
        for j in range(1, min(len(own_steps), 1 + MAX_STEPS_AFTER_DIVERGE)):
            actions_to_classify.append(own_steps[j]["action"])
            step_indices.append(j)

        if not actions_to_classify:
            continue

        candidates.append({
            "key": f"{task_id}:{round_label}:{nid}",
            "task_id": task_id,
            "round": round_label,
            "node_id": nid,
            "instruction": instruction,
            "depth": node.get("depth", 0),
            "n_context_steps": len(context_actions),
            "context_actions": context_actions,
            "n_own_steps": len(actions_to_classify),
            "own_actions": actions_to_classify,
            "step_indices": step_indices,  # indices within own_steps
        })

    return candidates


def pack_batches(candidates, n_batches):
    """Greedy bin-packing into balanced batches by step count."""
    batches = [[] for _ in range(n_batches)]
    batch_steps = [0] * n_batches

    # Sort candidates by step count descending for better packing
    candidates.sort(key=lambda c: c["n_own_steps"], reverse=True)

    for cand in candidates:
        # Put in the lightest batch
        min_idx = min(range(n_batches), key=lambda i: batch_steps[i])
        batches[min_idx].append(cand)
        batch_steps[min_idx] += cand["n_own_steps"]

    return batches, batch_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_dir", default="checkpoints/mcts_trajectories_v2/combined_all/trees")
    parser.add_argument("--task_index_path", default="checkpoints/mcts_trajectories_v2/combined_all/task_index.json")
    parser.add_argument("--output_dir", default="checkpoints/mcts_kto/audit_batches")
    parser.add_argument("--n_batches", type=int, default=N_BATCHES)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.task_index_path) as f:
        task_index = json.load(f)

    # Extract all candidates
    all_candidates = []
    n_trees = 0
    for task_id, tree_files in task_index.items():
        for tf in tree_files:
            rp = os.path.realpath(os.path.join(args.tree_dir, tf))
            if not os.path.exists(rp):
                continue
            tree_data = load_mcts_tree(rp)
            cands = extract_candidates(tree_data, tf)
            all_candidates.extend(cands)
            n_trees += 1

    total_steps = sum(c["n_own_steps"] for c in all_candidates)
    print(f"Extracted {len(all_candidates)} nodes, {total_steps} steps from {n_trees} trees")

    # Pack into batches
    batches, batch_steps = pack_batches(all_candidates, args.n_batches)

    # Write batch files
    for i, (batch_items, n_steps) in enumerate(zip(batches, batch_steps)):
        batch_data = {
            "batch_id": f"kto_audit_{i:03d}",
            "n_nodes": len(batch_items),
            "n_steps_to_label": n_steps,
            "items": batch_items,
        }
        path = os.path.join(args.output_dir, f"kto_audit_{i:03d}.json")
        with open(path, "w") as f:
            json.dump(batch_data, f, ensure_ascii=False)

    # Summary
    print(f"\nCreated {args.n_batches} batches in {args.output_dir}")
    print(f"  Avg nodes/batch: {len(all_candidates) / args.n_batches:.0f}")
    print(f"  Avg steps/batch: {total_steps / args.n_batches:.0f}")
    print(f"  Min steps: {min(batch_steps)}, Max steps: {max(batch_steps)}")

    # Write audit instructions (same criteria as v2 KEEP/REMOVE)
    instructions_path = os.path.join(args.output_dir, "audit_instructions.md")
    with open(instructions_path, "w") as f:
        f.write("""# Step Classification Audit — KTO Phase 2

You are auditing action steps from a GUI agent performing OSWorld tasks.
For each step, determine whether the action is clearly WRONG.

## Input Format
Each item has:
- **instruction**: The task the agent was given
- **context_actions**: Previous actions the agent took (for context)
- **own_actions**: The action(s) to classify

## Classification Rules

For each action in `own_actions`, assign:

### WRONG (1)
- The reasoning in the Thought is clearly confused or contradicts the task
- The action targets the wrong application, menu, or UI element
- The action repeats an approach that just failed without trying something new
- The action is completely irrelevant to the task
- The action gives up or does random clicks without clear purpose

### CORRECT or AMBIGUOUS (0)
- The action is productive toward completing the task
- The action is a reasonable attempt given the task instruction
- The action recognizes a previous mistake and attempts recovery
- The action is a valid exploratory step (checking menus, reading UI)
- You cannot confidently determine whether the action is wrong

## Important
- Judge each action based ONLY on the task instruction and action text
- If you are NOT confident the action is wrong, mark it 0
- Only mark 1 when you are sure the action is clearly wrong
- Focus on the REASONING in the Thought section — confused reasoning is the strongest signal

## Output Format
For each item, output a JSON line:
```json
{"key": "<item key>", "mask": [1, 0, ...]}
```
where mask[i] corresponds to own_actions[i]. 1=WRONG, 0=CORRECT/AMBIGUOUS.
""")

    print(f"  Instructions: {instructions_path}")


if __name__ == "__main__":
    main()
