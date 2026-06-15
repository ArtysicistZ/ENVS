"""ENVS-specific trajectory format.

Produces a STRICT SUPERSET of the standard trajectory_io format:
the existing SFT pipeline (trajectory_sft.py, expand_episode, select_sft_trajectories.py)
reads only task_id, instruction, eval_result, limit_images, steps — and ignores unknown fields.
So ENVS trajectories are directly compatible with the SFT pipeline.
"""

from typing import Any, Dict, List, Optional

from verl.envs.tree import TreeNode


def make_envs_trajectory(
    node: TreeNode,
    task_config: Dict[str, Any],
    limit_images: int = 3,
) -> Dict[str, Any]:
    """Convert a TreeNode into the compact JSONL trajectory format.

    This format is compatible with verl.utils.trajectory_io and can be
    fed directly into the SFT pipeline (train_sft.py, trajectory_sft.py).

    Only call this for nodes that have actually executed at least 1 step.
    """
    # Use PHYSICAL action sequence (what was actually executed on this VM)
    # not the logical tree path (which includes parent's post-branch actions)
    screenshots = node.get_full_screenshot_history()
    actions = node.get_physical_action_sequence()

    # Build steps list (parallel screenshots and actions)
    n_steps = min(len(screenshots), len(actions))
    steps = []
    for i in range(n_steps):
        steps.append({
            "screenshot_b64": screenshots[i],
            "action": actions[i],
        })

    # Build branch path string for metadata
    branch_path = _build_branch_path(node)

    # Find divergence step (where this node's path first differs from parent)
    diverged_at = node.depth if node.parent is not None else 0

    trajectory = {
        # Standard fields (SFT-compatible)
        "task_id": task_config.get("id", "unknown"),
        "instruction": node.instruction or task_config.get("instruction", ""),
        "eval_result": node.eval_score if node.eval_score is not None else 0.0,
        "limit_images": limit_images,
        "steps": steps,

        # ENVS-specific metadata (ignored by existing SFT pipeline)
        "branch_path": branch_path,
        "tree_depth": _tree_depth(node),
        "diverged_at_step": diverged_at,
        "parent_vm_idx": node.parent.vm_slot_id if node.parent else None,
        "is_hindsight": getattr(node, "is_hindsight", False),
        "vm_idx": node.vm_slot_id,
        "node_id": node.node_id,
        "n_steps_executed": n_steps,

        # Noise metadata (v3 noisy ENVS — empty for clean collection)
        "noise_enabled": node.noise_enabled,
        "noise_seed": node.noise_seed,
        "noise_fire_count": node.noise_fire_count,
        "noise_fire_steps": list(node.noise_fire_steps),
        "noise_events_fired": list(node.noise_events_fired),
        "noise_recovery_events": list(node.noise_recovery_events),
        "noise_total_recovery_cost": node.noise_total_recovery_cost,
        "trajectory_tag": _classify_trajectory(node),
    }

    return trajectory


def _classify_trajectory(node: TreeNode) -> str:
    """Classify trajectory for downstream training (SFT/KTO tag)."""
    success = (node.eval_score or 0.0) > 0.5
    has_noise = node.noise_enabled and node.noise_fire_count > 0
    has_recovery = len(node.noise_recovery_events) > 0

    if not has_noise:
        return "clean_success" if success else "clean_failure"
    if success and has_recovery:
        return "recovery_success"
    if success:
        return "noisy_success"
    return "noisy_failure"


def _build_branch_path(node: TreeNode) -> str:
    """Build a human-readable branch path string."""
    parts = []
    current = node
    while current is not None:
        if current.branch_action:
            # Extract action type from the branch action
            action_type = _extract_action_type(current.branch_action)
            parts.append(f"{action_type}(s{current.depth})")
        elif current.parent is None:
            parts.append("root")
        current = current.parent
    parts.reverse()
    return "->".join(parts)


def _tree_depth(node: TreeNode) -> int:
    """Count how many branch points are between root and this node."""
    depth = 0
    current = node
    while current.parent is not None:
        depth += 1
        current = current.parent
    return depth


def _extract_action_type(action_text: str) -> str:
    """Extract the action type (click, hotkey, etc.) from model output."""
    import re
    if "Action:" not in action_text:
        return "unknown"
    action_str = action_text.split("Action:")[-1].strip().split("\n")[0].strip()
    m = re.match(r"(\w+)\(", action_str)
    return m.group(1) if m else "unknown"


def filter_successful_trajectories(
    trajectories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter to only successful trajectories (eval_result > 0)."""
    return [t for t in trajectories if t.get("eval_result", 0) > 0]
