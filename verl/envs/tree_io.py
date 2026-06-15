"""Full ENVS tree serialization for v2 collection.

Saves the complete tree as a proper tree structure where each node stores
ONLY its own steps (actions + screenshots executed after branching from parent).
No duplication — each screenshot is stored exactly once.

To reconstruct the full root-to-node path for any node, walk up the parent
chain and concatenate each ancestor's steps in order.

Usage:
    from verl.envs.tree_io import save_envs_tree, load_envs_tree, reconstruct_trajectory

    # After ENVS collection:
    save_envs_tree(tree, task_config, "envs_trees/task_abc.json")

    # Later, to reconstruct any node's full trajectory:
    tree_data = load_envs_tree("envs_trees/task_abc.json")
    traj = reconstruct_trajectory(tree_data, "node_005")
    # traj has standard SFT fields: task_id, instruction, eval_result, limit_images, steps
"""

import json
import os
import random as _random
from typing import Any, Dict, List, Optional, Tuple

from verl.envs.tree import ENVSTree, TreeNode


# ================================================================
# Saving
# ================================================================

def save_envs_tree(
    tree: ENVSTree,
    task_config: Dict[str, Any],
    output_path: str,
    limit_images: int = 3,
) -> Dict[str, Any]:
    """Serialize the full ENVS tree to JSON.

    Each node stores ONLY its own steps — no duplication of parent data.
    The tree structure is preserved via parent_id / children_ids.
    """
    task_id = task_config.get("id", "unknown")
    instruction = task_config.get("instruction", "")

    # Back-propagate Q-values
    _propagate_q_values(tree)

    # Serialize nodes — each stores only its own steps
    nodes_data = {}
    for node in tree.all_nodes():
        if len(node.action_history) == 0 and node.parent is not None:
            continue
        nodes_data[node.node_id] = _serialize_node(node)

    # Tree summary
    root_q = tree.roots[0].q_value if tree.roots and tree.roots[0].q_value is not None else 0.0

    tree_data = {
        "task_id": task_id,
        "instruction": instruction,
        "limit_images": limit_images,
        "root_ids": [r.node_id for r in tree.roots if r.node_id in nodes_data],
        "tree_summary": {
            "n_nodes": len(nodes_data),
            "n_successful": sum(1 for n in tree.all_nodes()
                               if n.eval_score is not None and n.eval_score > 0),
            "n_failed": sum(1 for n in tree.all_nodes()
                           if n.eval_score is not None and n.eval_score == 0),
            "root_q": root_q,
        },
        "nodes": nodes_data,
    }

    # Write atomically
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(tree_data, f, ensure_ascii=False)
    os.replace(tmp_path, output_path)

    return tree_data


def _serialize_node(node: TreeNode) -> Dict[str, Any]:
    """Serialize one node. Stores ONLY this node's own steps (not parent's).

    own_steps = zip(screenshot_history, action_history)
    These are the steps executed AFTER this node branched from its parent.
    For root nodes, this is ALL steps.
    """
    # This node's OWN steps only (not inherited from parent)
    own_screenshots = node.screenshot_history
    own_actions = node.action_history
    n_own = min(len(own_screenshots), len(own_actions))

    own_steps = []
    for i in range(n_own):
        own_steps.append({
            "screenshot_b64": own_screenshots[i],
            "action": own_actions[i],
        })

    return {
        "node_id": node.node_id,
        "parent_id": node.parent.node_id if node.parent else None,
        "children_ids": [c.node_id for c in node.children],
        "depth": node.depth,
        "vm_slot_id": node.vm_slot_id,
        "eval_score": node.eval_score,
        "q_value": node.q_value if hasattr(node, "q_value") else None,
        "done": node.done,
        "is_terminal": node.is_terminal(),
        "own_steps": own_steps,
        "n_own_steps": n_own,
        # How many of the parent's own_steps existed when this child branched.
        # Used to slice the correct prefix from parent during reconstruction.
        # For root: 0 (no parent).
        "parent_steps_at_branch": getattr(node, "parent_steps_at_branch", 0),

        # Noise metadata (v3 noisy ENVS — empty/defaults for clean collection)
        "noise_enabled": getattr(node, "noise_enabled", False),
        "noise_seed": getattr(node, "noise_seed", 0),
        "noise_fire_count": getattr(node, "noise_fire_count", 0),
        "noise_fire_steps": list(getattr(node, "noise_fire_steps", [])),
        "noise_events_fired": list(getattr(node, "noise_events_fired", [])),
        "noise_recovery_events": list(getattr(node, "noise_recovery_events", [])),
        "noise_total_recovery_cost": getattr(node, "noise_total_recovery_cost", 0),
    }


def _propagate_q_values(tree: ENVSTree) -> None:
    """Back-propagate Q-values from leaves to root.

    Q(leaf) = eval_score (1.0 for success, 0.0 for failure)
    Q(internal) = mean(Q(child) for evaluated children)
    """
    for node in reversed(tree.all_nodes()):
        if node.children:
            child_qs = [c.q_value for c in node.children
                       if hasattr(c, "q_value") and c.q_value is not None]
            if child_qs:
                node.q_value = sum(child_qs) / len(child_qs)
            else:
                node.q_value = node.eval_score
        else:
            node.q_value = node.eval_score if node.eval_score is not None else 0.0


# ================================================================
# Loading
# ================================================================

def load_envs_tree(path: str) -> Dict[str, Any]:
    """Load a saved ENVS tree from JSON."""
    with open(path) as f:
        return json.load(f)


# ================================================================
# Reconstruction: node → full root-to-node trajectory
# ================================================================

def _get_ancestor_chain(tree_data: Dict[str, Any], node_id: str) -> List[str]:
    """Walk from node to root, return list of node_ids from root to node."""
    nodes = tree_data["nodes"]
    chain = []
    current_id = node_id
    while current_id is not None:
        chain.append(current_id)
        current_id = nodes[current_id]["parent_id"]
    chain.reverse()
    return chain


def reconstruct_steps(tree_data: Dict[str, Any], node_id: str) -> List[Dict[str, str]]:
    """Reconstruct the full root-to-node step sequence.

    Walks up the parent chain, concatenates each ancestor's own_steps in order.
    The result is the LOGICAL path the model saw during inference.

    Key subtlety: a parent node may continue executing AFTER spawning a child.
    The parent's own_steps grows beyond what the child saw. Each child records
    `parent_steps_at_branch` — how many of the parent's own_steps existed at
    branch time. We slice the parent's own_steps to that count.

    Example:
        root executes [a0, a1], branches child at step 2, then continues to [a0, a1, a2_root]
        child has parent_steps_at_branch=2, own_steps=[a2_child]
        Reconstruction: root.own_steps[:2] + child.own_steps = [a0, a1, a2_child]  ← correct
        NOT: root.own_steps[:3] + child.own_steps = [a0, a1, a2_root, a2_child]    ← wrong
    """
    chain = _get_ancestor_chain(tree_data, node_id)
    nodes = tree_data["nodes"]

    steps = []
    for i, nid in enumerate(chain):
        node = nodes[nid]
        own_steps = node["own_steps"]

        if i < len(chain) - 1:
            # This is an ancestor — the NEXT node in the chain is its child.
            # Slice own_steps to what the child saw at branch time.
            child_nid = chain[i + 1]
            child_node = nodes[child_nid]
            n_steps_at_branch = child_node.get("parent_steps_at_branch", len(own_steps))
            steps.extend(own_steps[:n_steps_at_branch])
        else:
            # This is the target node — include all its own steps
            steps.extend(own_steps)

    return steps


def reconstruct_trajectory(
    tree_data: Dict[str, Any],
    node_id: str,
) -> Dict[str, Any]:
    """Reconstruct a full SFT-compatible trajectory for one node.

    Returns a dict with standard fields: task_id, instruction, eval_result,
    limit_images, steps — directly compatible with expand_episode().
    """
    nodes = tree_data["nodes"]
    node = nodes[node_id]
    steps = reconstruct_steps(tree_data, node_id)

    return {
        "task_id": tree_data["task_id"],
        "instruction": tree_data["instruction"],
        "eval_result": node["eval_score"] if node["eval_score"] is not None else 0.0,
        "limit_images": tree_data["limit_images"],
        "steps": steps,
        # ENVS metadata
        "node_id": node_id,
        "parent_id": node["parent_id"],
        "q_value": node.get("q_value"),
        "n_steps": len(steps),
    }


def tree_to_trajectories(tree_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert all nodes in a saved tree to flat SFT-compatible trajectories.

    Each node becomes one trajectory with the full root-to-node path
    reconstructed from the tree structure. Compatible with expand_episode().
    """
    return [
        reconstruct_trajectory(tree_data, node_id)
        for node_id in tree_data["nodes"]
    ]


# ================================================================
# Training signal extraction
# ================================================================

def get_sibling_pairs(tree_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract (winner, loser) sibling pairs for DPO training.

    At each branch point where children have different Q-values:
    - chosen: full root-to-winner trajectory
    - rejected: full root-to-loser trajectory
    - They share a common prefix (root to parent) and diverge at the branch step

    Both trajectories are reconstructed from the tree — no duplication in storage.
    """
    nodes = tree_data["nodes"]

    pairs = []
    for node_id, node in nodes.items():
        children_ids = node.get("children_ids", [])
        if len(children_ids) < 2:
            continue

        # Get children with Q-values
        children = []
        for cid in children_ids:
            if cid in nodes and nodes[cid].get("q_value") is not None:
                children.append(nodes[cid])
        if len(children) < 2:
            continue

        # Sort by Q descending
        children.sort(key=lambda c: c.get("q_value", 0) or 0, reverse=True)
        winner = children[0]

        for loser in children[1:]:
            q_diff = (winner.get("q_value", 0) or 0) - (loser.get("q_value", 0) or 0)
            if q_diff <= 0:
                continue

            # Reconstruct full trajectories on demand
            winner_steps = reconstruct_steps(tree_data, winner["node_id"])
            loser_steps = reconstruct_steps(tree_data, loser["node_id"])

            # Shared prefix = what children saw of the parent at branch time
            # This is the winner's full path minus the winner's own steps
            shared_len = len(winner_steps) - winner["n_own_steps"]

            pairs.append({
                "task_id": tree_data["task_id"],
                "instruction": tree_data["instruction"],
                "limit_images": tree_data["limit_images"],
                "parent_node_id": node_id,
                "winner_node_id": winner["node_id"],
                "loser_node_id": loser["node_id"],
                "winner_q": winner.get("q_value"),
                "loser_q": loser.get("q_value"),
                "q_difference": q_diff,
                "shared_prefix_length": shared_len,
                "winner_steps": winner_steps,
                "loser_steps": loser_steps,
            })

    return pairs


def get_revision_trajectories(
    tree_data: Dict[str, Any],
    revision_templates: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Construct revision trajectories from failed→successful sibling splices.

    For each branch point with both successful and failed children:
    - Take the failed child's divergent action as context (not in loss)
    - Insert a revision signal (in loss)
    - Continue with successful child's actions (in loss)

    The shared parent prefix is context (not in loss).

    Returns list of revision trajectory dicts with loss_mask per step.
    """
    if revision_templates is None:
        revision_templates = [
            "Thought: I realize my previous approach was incorrect. Let me reconsider and take a different action.\nAction: ",
            "Thought: The last action didn't achieve what I intended. I need to try a different approach.\nAction: ",
            "Thought: Looking at the current state, I see that my previous action was wrong. Let me correct this.\nAction: ",
        ]

    nodes = tree_data["nodes"]
    rng = _random.Random(seed)

    revisions = []
    for node_id, node in nodes.items():
        children_ids = node.get("children_ids", [])
        if len(children_ids) < 2:
            continue

        children = [nodes[cid] for cid in children_ids if cid in nodes]
        successful = [c for c in children if (c.get("eval_score") or 0) > 0]
        failed = [c for c in children if (c.get("eval_score") or 0) == 0 and c.get("done")]

        if not successful or not failed:
            continue

        for fail_child in failed:
            succ_child = rng.choice(successful)

            # Shared prefix = what children saw of the parent at branch time
            # = the successful child's full path minus its own steps
            succ_full = reconstruct_steps(tree_data, succ_child["node_id"])
            parent_steps = succ_full[:len(succ_full) - succ_child["n_own_steps"]]

            # Failed child's own first action (the wrong action at the branch point)
            fail_own_steps = fail_child["own_steps"]
            if not fail_own_steps:
                continue

            # Successful child's own steps (the correct continuation)
            succ_own_steps = succ_child["own_steps"]
            if not succ_own_steps:
                continue

            # Build revision trajectory
            revision_steps = []
            loss_mask = []

            # 1. Shared prefix (parent's path) — CONTEXT only
            for s in parent_steps:
                revision_steps.append(s)
                loss_mask.append(0)

            # 2. Failed action at branch point — CONTEXT only
            revision_steps.append(fail_own_steps[0])
            loss_mask.append(0)

            # 3. Revision signal + successful first action — IN LOSS
            revision_signal = rng.choice(revision_templates)
            first_succ_action = succ_own_steps[0]["action"]
            # Combine revision thought with successful action
            if "Action:" in first_succ_action:
                action_part = first_succ_action.split("Action:")[-1].strip()
                revision_action = revision_signal + action_part
            else:
                revision_action = revision_signal + first_succ_action

            revision_steps.append({
                "screenshot_b64": succ_own_steps[0].get("screenshot_b64", ""),
                "action": revision_action,
            })
            loss_mask.append(1)

            # 4. Remaining successful steps — IN LOSS
            for s in succ_own_steps[1:]:
                revision_steps.append(s)
                loss_mask.append(1)

            revisions.append({
                "task_id": tree_data["task_id"],
                "instruction": tree_data["instruction"],
                "limit_images": tree_data["limit_images"],
                "eval_result": succ_child.get("eval_score", 1.0),
                "steps": revision_steps,
                "loss_mask": loss_mask,
                "revision_step": len(parent_steps) + 1,  # index of the revision signal
                "parent_node_id": node_id,
                "fail_node_id": fail_child["node_id"],
                "success_node_id": succ_child["node_id"],
            })

    return revisions
