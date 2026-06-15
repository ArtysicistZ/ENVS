"""ENVS tree data structures."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BranchBudget:
    """Tracks remaining branch budget for a tree node."""

    branches_remaining: int = 5

    def can_branch(self) -> bool:
        return self.branches_remaining > 0

    def use(self) -> None:
        assert self.branches_remaining > 0, "No branch budget remaining"
        self.branches_remaining -= 1


@dataclass
class TreeNode:
    """A node in the ENVS tree.

    Each node corresponds to one VM executing a trajectory. The node tracks
    the VM's action history, screenshots, and branching metadata.
    """

    node_id: str
    vm_slot_id: int                         # env server slot ID
    depth: int                              # step index where this node was created
    budget: BranchBudget = field(default_factory=BranchBudget)
    parent: Optional[TreeNode] = None
    children: List[TreeNode] = field(default_factory=list)

    # Actions replayed on this VM at spawn time (the physical prefix)
    # For root nodes: empty. For children: the parent's physical sequence at branch time.
    replay_prefix: List[str] = field(default_factory=list)
    # Snapshot of parent's logical action/screenshot history at branch time.
    # Frozen at spawn — does NOT grow when the parent continues stepping.
    parent_action_snapshot: List[str] = field(default_factory=list)
    parent_screenshot_snapshot: List[str] = field(default_factory=list)
    # Action history: actions executed by THIS node's VM (after replay)
    action_history: List[str] = field(default_factory=list)
    # Screenshots: base64 JPEG for each step (parallel to action_history)
    screenshot_history: List[str] = field(default_factory=list)
    # The current step's screenshot (before taking an action)
    current_screenshot_b64: Optional[str] = None

    # Probing results for the current step
    candidates: List[str] = field(default_factory=list)
    candidates_logprobs: Optional[List] = None

    # The action assigned to execute at the current step
    action: Optional[str] = None
    # Whether this node was just spawned (uses branch_action instead of probing)
    just_spawned: bool = False
    branch_action: Optional[str] = None

    # State
    done: bool = False
    eval_score: Optional[float] = None
    q_value: Optional[float] = None   # back-propagated success rate from descendants
    prev_high: bool = False   # for late-step deferral gate

    # Number of steps the parent had in action_history when this child was spawned.
    # Used by tree_io to reconstruct the correct prefix from the parent's own_steps.
    # For root nodes: 0 (no parent).
    parent_steps_at_branch: int = 0

    # Metadata
    instruction: str = ""

    # Noise (v3: per-branch random noise)
    noise_seed: int = 0                     # unique RNG seed for this branch's noise schedule
    noise_enabled: bool = False             # whether this branch has noise
    noise_fire_count: int = 0               # scheduled number of fires
    noise_fire_steps: List[int] = field(default_factory=list)  # step indices where noise fires
    noise_events_fired: List[dict] = field(default_factory=list)  # actual fire records from env
    noise_recovery_events: List[dict] = field(default_factory=list)  # detected recoveries
    noise_total_recovery_cost: int = 0      # cumulative recovery cost of all fires

    def record_action(self, action_text: str, screenshot_b64: Optional[str] = None) -> None:
        """Record an executed action and its preceding screenshot.

        Always keeps screenshot_history aligned with action_history (same length).
        """
        self.action_history.append(action_text)
        if screenshot_b64 is not None:
            self.screenshot_history.append(screenshot_b64)
        elif self.current_screenshot_b64 is not None:
            self.screenshot_history.append(self.current_screenshot_b64)
        else:
            self.screenshot_history.append("")  # placeholder to keep alignment

    def get_action_history(self) -> List[str]:
        """Get the full LOGICAL action history (for prompt building).

        Returns the path from root to this node: parent snapshot (frozen at
        branch time) + this node's own actions. Does NOT walk the live parent
        chain — the parent may have continued stepping after spawning us.
        """
        return list(self.parent_action_snapshot) + list(self.action_history)

    def get_physical_action_sequence(self) -> List[str]:
        """Get the PHYSICAL action sequence that was executed on this VM.

        This is what should be replayed on a new VM to reproduce this node's state.
        = replay_prefix (actions replayed at spawn) + action_history (own actions)
        """
        return list(self.replay_prefix) + list(self.action_history)

    def get_full_screenshot_history(self) -> List[str]:
        """Get the full screenshot history from root to this node."""
        return list(self.parent_screenshot_snapshot) + list(self.screenshot_history)

    def get_majority_action(self) -> Optional[str]:
        """Get the most common action from candidates (plurality vote)."""
        if not self.candidates:
            return None
        # Use the full action text (not just type) for majority
        from collections import Counter
        counter = Counter(self.candidates)
        return counter.most_common(1)[0][0]

    def current_step(self) -> int:
        """The step index this node is currently at."""
        return len(self.get_action_history())

    def is_terminal(self) -> bool:
        """Check if the last action was terminal (finished/fail)."""
        if not self.action_history:
            return False
        last = self.action_history[-1]
        if "Action:" not in last:
            return False
        action_str = last.split("Action:")[-1].strip().split("\n")[0].strip()
        m = re.match(r"(\w+)\(", action_str)
        if m and m.group(1) in ("finished", "fail"):
            return True
        return False

    def is_stuck(self, repeat_limit: int = 3, wait_limit: int = 2) -> bool:
        """Check if the node is stuck (repeating same action or consecutive waits)."""
        history = self.action_history
        if not history:
            return False

        # Check consecutive wait()
        def _extract_type(text):
            if "Action:" not in text:
                return "UNKNOWN"
            action_str = text.split("Action:")[-1].strip().split("\n")[0].strip()
            m = re.match(r"(\w+)\(", action_str)
            return m.group(1) if m else "UNKNOWN"

        types = [_extract_type(a) for a in history]

        # Consecutive waits
        consecutive_waits = 0
        for t in reversed(types):
            if t == "wait":
                consecutive_waits += 1
            else:
                break
        if consecutive_waits >= wait_limit:
            return True

        # Same action repeated
        if len(history) >= repeat_limit:
            last_n = history[-repeat_limit:]
            if all(a == last_n[0] for a in last_n):
                return True

        return False


class ENVSTree:
    """Manages the full ENVS tree for one task."""

    def __init__(self):
        self.roots: List[TreeNode] = []
        self._all_nodes: List[TreeNode] = []

    def add_root(self, node: TreeNode) -> None:
        self.roots.append(node)
        self._all_nodes.append(node)

    def add_child(self, parent: TreeNode, child: TreeNode) -> None:
        child.parent = parent
        parent.children.append(child)
        self._all_nodes.append(child)

    def all_nodes(self) -> List[TreeNode]:
        return list(self._all_nodes)

    def leaves(self) -> List[TreeNode]:
        """Return all leaf nodes (nodes with no children)."""
        return [n for n in self._all_nodes if not n.children]

    def active_nodes(self) -> List[TreeNode]:
        """Return nodes that are not done."""
        return [n for n in self._all_nodes if not n.done]

    def get_unexplored_siblings(self, node: TreeNode) -> List[Dict[str, Any]]:
        """Find unexplored sibling actions near a node for Phase 2 hindsight.

        Walks up from the node to find branch points where alternative clusters
        were detected but not explored.

        Returns list of {"action": str, "branch_step": int, "node": TreeNode}
        sorted by depth ascending (shallowest first = most valuable).
        """
        siblings = []
        current = node
        while current.parent is not None:
            parent = current.parent
            # Check if parent had unexplored candidates
            if hasattr(parent, '_unexplored_actions') and parent._unexplored_actions:
                for action_text, step_idx in parent._unexplored_actions:
                    siblings.append({
                        "action": action_text,
                        "branch_step": step_idx,
                        "node": parent,
                    })
            current = parent
        # Sort by branch_step ascending (shallowest = most valuable)
        siblings.sort(key=lambda x: x["branch_step"])
        return siblings

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the tree structure."""
        return {
            "n_roots": len(self.roots),
            "n_total_nodes": len(self._all_nodes),
            "n_leaves": len(self.leaves()),
            "n_active": len(self.active_nodes()),
            "n_done": sum(1 for n in self._all_nodes if n.done),
            "n_successful": sum(1 for n in self._all_nodes
                                if n.eval_score is not None and n.eval_score > 0),
            "max_depth": max((n.current_step() for n in self._all_nodes), default=0),
        }
