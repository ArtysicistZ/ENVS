"""Test that save_mcts_tree stores each node once and reconstructs correctly.

Key test: parent continues executing AFTER branching children.
The children must see only the parent's prefix at branch time, NOT the
parent's post-branch actions. This is the critical bug that was caught.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.mcts.tree import MCTSTree, TreeNode, BranchBudget
from verl.mcts.tree_io import (
    save_mcts_tree,
    load_mcts_tree,
    reconstruct_steps,
    reconstruct_trajectory,
    tree_to_trajectories,
    get_sibling_pairs,
    get_revision_trajectories,
    _propagate_q_values,
)
from verl.utils.trajectory_sft import expand_episode


def make_test_tree():
    """Simulate the real orchestrator flow:

    root (node_001): executes [a0, a1], branches at step 2, THEN continues [a2_root]
      ├── child_A (node_002): executes [a2_A, a3_A], eval=1.0 (success)
      │     └── grandchild (node_004): executes [a4_gc, a5_gc], eval=1.0
      └── child_B (node_003): executes [a2_B, a3_B], eval=0.0 (failure)

    Root continues after branching (a2_root). Children must NOT see a2_root.
    """
    tree = MCTSTree()

    root = TreeNode(
        node_id="node_001", vm_slot_id=0, depth=0,
        budget=BranchBudget(5), instruction="Test task",
    )
    root.current_screenshot_b64 = "ss0"
    root.record_action("Thought: a0\nAction: click(start_box='(100,200)')", "ss0")
    root.current_screenshot_b64 = "ss1"
    root.record_action("Thought: a1\nAction: type(content='hello')", "ss1")
    root.current_screenshot_b64 = "ss2"
    tree.add_root(root)

    # Branch at step 2 — children snapshot parent with 2 actions
    child_a = TreeNode(
        node_id="node_002", vm_slot_id=1, depth=2,
        budget=BranchBudget(2), instruction="Test task",
        branch_action="Thought: a2_A\nAction: click(start_box='(300,400)')",
    )
    child_a.parent_action_snapshot = list(root.get_action_history())
    child_a.parent_screenshot_snapshot = list(root.get_full_screenshot_history())
    child_a.parent_steps_at_branch = len(root.action_history)  # 2
    child_a.replay_prefix = list(root.get_physical_action_sequence())
    child_a.current_screenshot_b64 = "ss2_A"
    tree.add_child(root, child_a)

    child_b = TreeNode(
        node_id="node_003", vm_slot_id=2, depth=2,
        budget=BranchBudget(2), instruction="Test task",
        branch_action="Thought: a2_B\nAction: scroll(start_box='(500,600)', direction='down')",
    )
    child_b.parent_action_snapshot = list(root.get_action_history())
    child_b.parent_screenshot_snapshot = list(root.get_full_screenshot_history())
    child_b.parent_steps_at_branch = len(root.action_history)  # 2
    child_b.replay_prefix = list(root.get_physical_action_sequence())
    child_b.current_screenshot_b64 = "ss2_B"
    tree.add_child(root, child_b)

    # ROOT CONTINUES after branching (this is the critical case)
    root.record_action("Thought: a2_root\nAction: click(start_box='(700,800)')", "ss2")
    root.current_screenshot_b64 = "ss3_root"
    root.done = True
    root.eval_score = 0.0

    # Children execute their actions
    child_a.record_action("Thought: a2_A\nAction: click(start_box='(300,400)')", "ss2_A")
    child_a.current_screenshot_b64 = "ss3_A"
    child_a.record_action("Thought: a3_A\nAction: finished(content='done')", "ss3_A")
    child_a.done = True
    child_a.eval_score = 1.0

    child_b.record_action("Thought: a2_B\nAction: scroll(start_box='(500,600)', direction='down')", "ss2_B")
    child_b.current_screenshot_b64 = "ss3_B"
    child_b.record_action("Thought: a3_B\nAction: fail()", "ss3_B")
    child_b.done = True
    child_b.eval_score = 0.0

    # Grandchild branches from child_a at step 4
    gc = TreeNode(
        node_id="node_004", vm_slot_id=3, depth=4,
        budget=BranchBudget(1), instruction="Test task",
        branch_action="Thought: a4_gc\nAction: hotkey(key='ctrl s')",
    )
    gc.parent_action_snapshot = list(child_a.get_action_history())  # [a0, a1, a2_A, a3_A]
    gc.parent_screenshot_snapshot = list(child_a.get_full_screenshot_history())
    gc.parent_steps_at_branch = len(child_a.action_history)  # 2
    gc.replay_prefix = list(child_a.get_physical_action_sequence())
    gc.current_screenshot_b64 = "ss4_gc"
    gc.record_action("Thought: a4_gc\nAction: hotkey(key='ctrl s')", "ss4_gc")
    gc.current_screenshot_b64 = "ss5_gc"
    gc.record_action("Thought: a5_gc\nAction: finished(content='saved')", "ss5_gc")
    gc.done = True
    gc.eval_score = 1.0
    tree.add_child(child_a, gc)

    return tree


def _save_and_load(tree):
    task_config = {"id": "test", "instruction": "Test task"}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    save_mcts_tree(tree, task_config, tmp)
    return load_mcts_tree(tmp), tmp


def test_no_duplication():
    """Each node stores only its own steps."""
    tree = make_test_tree()
    tree_data, tmp = _save_and_load(tree)
    try:
        nodes = tree_data["nodes"]
        assert nodes["node_001"]["n_own_steps"] == 3  # a0, a1, a2_root (root continued)
        assert nodes["node_002"]["n_own_steps"] == 2  # a2_A, a3_A
        assert nodes["node_003"]["n_own_steps"] == 2  # a2_B, a3_B
        assert nodes["node_004"]["n_own_steps"] == 2  # a4_gc, a5_gc
        print("PASS: No duplication")
    finally:
        os.unlink(tmp)


def test_parent_continues_after_branch():
    """THE critical test: children must NOT see parent's post-branch actions."""
    tree = make_test_tree()
    tree_data, tmp = _save_and_load(tree)
    try:
        # Root has 3 own_steps: [a0, a1, a2_root]
        # But children branched when root had only 2 steps
        # parent_steps_at_branch = 2

        # Child A's reconstructed path should be: [a0, a1, a2_A, a3_A]
        # NOT [a0, a1, a2_root, a2_A, a3_A]
        ca_steps = reconstruct_steps(tree_data, "node_002")
        assert len(ca_steps) == 4, f"Child A should have 4 steps, got {len(ca_steps)}"
        assert "a0" in ca_steps[0]["action"]
        assert "a1" in ca_steps[1]["action"]
        assert "a2_A" in ca_steps[2]["action"]  # NOT a2_root!
        assert "a3_A" in ca_steps[3]["action"]

        # Child B's reconstructed path: [a0, a1, a2_B, a3_B]
        cb_steps = reconstruct_steps(tree_data, "node_003")
        assert len(cb_steps) == 4
        assert "a2_B" in cb_steps[2]["action"]

        # Root's own path: [a0, a1, a2_root]
        root_steps = reconstruct_steps(tree_data, "node_001")
        assert len(root_steps) == 3
        assert "a2_root" in root_steps[2]["action"]

        # Grandchild: [a0, a1, a2_A, a3_A, a4_gc, a5_gc]
        gc_steps = reconstruct_steps(tree_data, "node_004")
        assert len(gc_steps) == 6
        assert "a2_A" in gc_steps[2]["action"]  # from child_a, not root
        assert "a4_gc" in gc_steps[4]["action"]

        print("PASS: Parent post-branch actions correctly excluded from children")
    finally:
        os.unlink(tmp)


def test_reconstruct_matches_logical():
    """Reconstructed steps must match each node's logical history exactly."""
    tree = make_test_tree()
    tree_data, tmp = _save_and_load(tree)
    try:
        node_objects = {n.node_id: n for n in tree.all_nodes()}

        for node_id in tree_data["nodes"]:
            steps = reconstruct_steps(tree_data, node_id)
            node = node_objects[node_id]

            expected_actions = node.get_action_history()
            expected_screenshots = node.get_full_screenshot_history()
            n = min(len(expected_actions), len(expected_screenshots))

            assert len(steps) == n, \
                f"{node_id}: got {len(steps)} steps, expected {n}"
            for i in range(n):
                assert steps[i]["action"] == expected_actions[i], \
                    f"{node_id} step {i}: action mismatch\n  got:      {steps[i]['action'][:50]}\n  expected: {expected_actions[i][:50]}"
                assert steps[i]["screenshot_b64"] == expected_screenshots[i], \
                    f"{node_id} step {i}: screenshot mismatch"

        print("PASS: Reconstruct matches logical history for all nodes")
    finally:
        os.unlink(tmp)


def test_prompt_reconstruction():
    """expand_episode on reconstructed data produces correct prompts."""
    tree = make_test_tree()
    tree_data, tmp = _save_and_load(tree)
    try:
        for node_id in tree_data["nodes"]:
            traj = reconstruct_trajectory(tree_data, node_id)
            examples = expand_episode(traj, train_all_steps=True)
            n_steps = len(traj["steps"])
            assert len(examples) == n_steps

            last = examples[-1]
            assert last["label"] == traj["steps"][-1]["action"]

            n_images = sum(
                1 for m in last["messages"]
                if isinstance(m.get("content"), list)
                and any(c.get("type") == "image" for c in m["content"] if isinstance(c, dict))
            )
            assert n_images == min(n_steps, 3)

        print("PASS: Prompt reconstruction via expand_episode")
    finally:
        os.unlink(tmp)


def test_q_propagation():
    """Q-values propagate correctly."""
    tree = make_test_tree()
    _propagate_q_values(tree)
    nodes = {n.node_id: n for n in tree.all_nodes()}

    assert nodes["node_004"].q_value == 1.0   # grandchild leaf
    assert nodes["node_002"].q_value == 1.0   # child_a has child gc (Q=1.0)
    assert nodes["node_003"].q_value == 0.0   # child_b leaf
    assert nodes["node_001"].q_value == 0.5   # root: mean(1.0, 0.0)

    print("PASS: Q-value propagation")


def test_space_efficiency():
    """Tree stores fewer steps than flat format."""
    tree = make_test_tree()
    tree_data, tmp = _save_and_load(tree)
    try:
        total_own = sum(n["n_own_steps"] for n in tree_data["nodes"].values())
        total_flat = sum(len(reconstruct_steps(tree_data, nid))
                        for nid in tree_data["nodes"])

        # Own: root=3 + childA=2 + childB=2 + gc=2 = 9
        # Flat: root=3 + childA=4 + childB=4 + gc=6 = 17
        assert total_own == 9
        assert total_flat == 17
        assert total_own < total_flat

        print(f"PASS: Space efficiency — tree={total_own} vs flat={total_flat} ({total_flat/total_own:.1f}x savings)")
    finally:
        os.unlink(tmp)


def test_sibling_pairs():
    """DPO pairs share prefix and diverge correctly."""
    tree = make_test_tree()
    tree_data, tmp = _save_and_load(tree)
    try:
        pairs = get_sibling_pairs(tree_data)
        assert len(pairs) >= 1

        pair = pairs[0]
        assert pair["winner_q"] > pair["loser_q"]

        w = pair["winner_steps"]
        l = pair["loser_steps"]
        prefix_len = pair["shared_prefix_length"]

        # Shared prefix = root's first 2 steps (before branching)
        assert prefix_len == 2
        for i in range(prefix_len):
            assert w[i]["action"] == l[i]["action"]

        # Diverge at branch point
        assert w[prefix_len]["action"] != l[prefix_len]["action"]

        # Winner should NOT contain root's post-branch action (a2_root)
        assert "a2_root" not in w[prefix_len]["action"]

        print("PASS: Sibling pairs")
    finally:
        os.unlink(tmp)


def test_revision_trajectories():
    """Revision trajectories have correct loss masks."""
    tree = make_test_tree()
    tree_data, tmp = _save_and_load(tree)
    try:
        revisions = get_revision_trajectories(tree_data)
        assert len(revisions) >= 1

        rev = revisions[0]
        assert len(rev["loss_mask"]) == len(rev["steps"])

        n_ctx = sum(1 for m in rev["loss_mask"] if m == 0)
        n_loss = sum(1 for m in rev["loss_mask"] if m == 1)
        assert n_ctx > 0, "Need context steps"
        assert n_loss > 0, "Need loss steps"

        # Prefix (2 root steps) + failed action (1) = 3 context
        assert n_ctx == 3, f"Expected 3 context, got {n_ctx}"

        # Verify expand_episode compatibility
        examples = expand_episode(rev, train_all_steps=True)
        assert len(examples) == len(rev["steps"])

        print(f"PASS: Revision trajectories ({n_ctx} context, {n_loss} loss)")
    finally:
        os.unlink(tmp)


def test_roundtrip():
    """Save → load → reconstruct → expand_episode works end to end."""
    tree = make_test_tree()
    tree_data, tmp = _save_and_load(tree)
    try:
        loaded = load_mcts_tree(tmp)
        trajs = tree_to_trajectories(loaded)

        for traj in trajs:
            assert "task_id" in traj
            assert "steps" in traj
            examples = expand_episode(traj, train_all_steps=True)
            assert len(examples) == len(traj["steps"])

        print("PASS: Full roundtrip")
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    test_no_duplication()
    test_parent_continues_after_branch()
    test_reconstruct_matches_logical()
    test_prompt_reconstruction()
    test_q_propagation()
    test_space_efficiency()
    test_sibling_pairs()
    test_revision_trajectories()
    test_roundtrip()
    print("\nAll tests passed!")
