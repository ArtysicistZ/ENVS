"""Extract REMOVE steps from v2 SFT audit as KTO negatives.

These are steps on successful trajectories that were classified as wrong
by the 200-agent KEEP/REMOVE audit. They share the same context as
positive KEEP steps but are wrong actions.

Output: kto_negatives_remove.jsonl with one entry per REMOVE step.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from verl.mcts.tree_io import load_mcts_tree, reconstruct_steps, _get_ancestor_chain

# Import map_steps_to_nodes from v2
import importlib.util
_v2_spec = importlib.util.spec_from_file_location(
    "train_mcts_sft_v2",
    os.path.join(os.path.dirname(__file__), "train_mcts_sft_v2.py"),
)
_v2_mod = importlib.util.module_from_spec(_v2_spec)
_v2_spec.loader.exec_module(_v2_mod)
map_steps_to_nodes = _v2_mod.map_steps_to_nodes


def main():
    tree_dir = "checkpoints/mcts_trajectories_v2/combined_all/trees"
    task_index_path = "checkpoints/mcts_trajectories_v2/combined_all/task_index.json"
    mask_path = "checkpoints/mcts_trajectories_v2/combined_all/step_masks_v2.json"
    output_path = "checkpoints/mcts_kto/kto_negatives_remove.jsonl"

    with open(task_index_path) as f:
        task_index = json.load(f)
    with open(mask_path) as f:
        masks = json.load(f)

    # Deduplicate: same logic as MCTSSFTDatasetV2 but collect REMOVE steps
    seen_steps = set()
    negatives = []
    n_total = 0
    n_deduped = 0

    for task_id, tree_files in task_index.items():
        for tree_filename in tree_files:
            tree_path = os.path.join(tree_dir, tree_filename)
            real_path = os.path.realpath(tree_path)
            if not os.path.exists(real_path):
                continue
            tree_data = load_mcts_tree(real_path)
            round_label = tree_filename.replace(task_id + "_", "").replace(".json", "")
            nodes = tree_data["nodes"]
            instruction = tree_data.get("instruction", "")

            for leaf_id, leaf in nodes.items():
                if not (leaf.get("eval_score") and leaf["eval_score"] > 0):
                    continue

                traj_steps = reconstruct_steps(tree_data, leaf_id)
                step_sources = map_steps_to_nodes(tree_data, leaf_id)

                for k, (node_id, node_step_idx) in enumerate(step_sources):
                    step_id = (tree_filename, node_id, node_step_idx)
                    n_total += 1

                    if step_id in seen_steps:
                        n_deduped += 1
                        continue
                    seen_steps.add(step_id)

                    # Check mask — we want REMOVE (0) steps
                    mask_key = f"{task_id}:{round_label}:{node_id}"
                    node_mask = masks.get(mask_key)
                    if node_mask is None or node_step_idx >= len(node_mask):
                        continue  # No mask → skip (default was KEEP)
                    if node_mask[node_step_idx] != 0:
                        continue  # KEEP step → skip

                    # This is a REMOVE step — build the episode up to this step
                    # The context is all steps before k, the negative action is step k
                    all_steps = traj_steps[:k + 1]

                    negatives.append({
                        "task_id": task_id,
                        "instruction": instruction,
                        "limit_images": tree_data.get("limit_images", 3),
                        "tree_filename": tree_filename,
                        "node_id": node_id,
                        "node_step_idx": node_step_idx,
                        "steps": all_steps,
                        "neg_step_idx": k,
                        "source": "sft_remove",
                    })

    with open(output_path, "w") as f:
        for neg in negatives:
            f.write(json.dumps(neg, ensure_ascii=False) + "\n")

    tasks_with_negs = len(set(n["task_id"] for n in negatives))
    print(f"Extracted {len(negatives)} REMOVE negatives (deduped from {n_total} total)")
    print(f"Deduped: {n_deduped}")
    print(f"Tasks with negatives: {tasks_with_negs}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
