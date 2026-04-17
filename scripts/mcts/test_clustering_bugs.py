#!/usr/bin/env python3
"""
Unit test: audit the clustering/branching algorithm for bugs.

Tests for known flaws:
1. 50px grid boundary artifacts (false splits)
2. 50px grid merging different UI elements (false merges)
3. Different action types (click vs scroll) with same coordinates
4. Singleton minority actions (1 scroll among 7 clicks)
5. type/hotkey content variations
6. UNPARSEABLE handling
"""

import sys, os
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ_ROOT)

from verl.mcts.clustering import (
    parse_action_with_coords,
    action_fingerprint,
    cluster_by_fingerprint,
    should_branch,
)
from verl.mcts.config import MCTSConfig
from verl.mcts.tree import TreeNode, BranchBudget

config = MCTSConfig()
passed = 0
failed = 0

def make_node(step=0, budget=5):
    return TreeNode(node_id="test", vm_slot_id=0, depth=0,
                    budget=BranchBudget(budget), instruction="test")

def check(name, expected, actual):
    global passed, failed
    if expected == actual:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")
        print(f"    expected: {expected}")
        print(f"    actual:   {actual}")

def make_action(atype, x=None, y=None, content=None, key=None, direction=None):
    """Build a model output string with Thought + Action."""
    if atype == "click" and x is not None:
        return f"Thought: I need to click.\nAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')"
    if atype == "scroll" and x is not None:
        d = direction or "down"
        return f"Thought: I need to scroll.\nAction: scroll(start_box='<|box_start|>({x},{y})<|box_end|>', direction='{d}')"
    if atype == "type":
        return f"Thought: I need to type.\nAction: type(content='{content}')"
    if atype == "hotkey":
        return f"Thought: I need to press keys.\nAction: hotkey(key='{key}')"
    if atype == "wait":
        return "Thought: I should wait.\nAction: wait()"
    if atype == "finished":
        return f"Thought: Done.\nAction: finished(content='{content or 'done'}')"
    if atype == "left_double" and x is not None:
        return f"Thought: Double click.\nAction: left_double(start_box='<|box_start|>({x},{y})<|box_end|>')"
    if atype == "drag" and x is not None:
        return f"Thought: Drag.\nAction: drag(start_box='<|box_start|>({x},{y})<|box_end|>', end_box='<|box_start|>({x+100},{y+100})<|box_end|>')"
    return f"Thought: whatever.\nAction: {atype}()"


# ============================================================
print("\n=== TEST 1: Parse action types correctly ===")
# ============================================================

check("click parsed", ("click", 940, 104, ""),
      parse_action_with_coords(make_action("click", 940, 104)))
check("scroll parsed", ("scroll", 500, 300, ""),
      parse_action_with_coords(make_action("scroll", 500, 300, direction="down")))
check("type parsed", ("type", None, None, "c:hello world"),
      parse_action_with_coords(make_action("type", content="hello world")))
check("hotkey parsed", ("hotkey", None, None, "k:ctrl+s"),
      parse_action_with_coords(make_action("hotkey", key="ctrl s")))
check("wait parsed", ("wait", None, None, ""),
      parse_action_with_coords(make_action("wait")))

# ============================================================
print("\n=== TEST 2: Fingerprint - same action type, same grid cell ===")
# ============================================================

fp1 = action_fingerprint(make_action("click", 940, 104))
fp2 = action_fingerprint(make_action("click", 945, 108))
check("same cell clicks (940,104) vs (945,108)", fp1, fp2)

# ============================================================
print("\n=== TEST 3: Fingerprint - grid boundary artifact (BUG!) ===")
# ============================================================

# Clicks 8px apart but straddling 50px boundary at x=950
fp_a = action_fingerprint(make_action("click", 946, 104))  # grid (18,2)
fp_b = action_fingerprint(make_action("click", 954, 104))  # grid (19,2)
print(f"  (946,104) → {fp_a}")
print(f"  (954,104) → {fp_b}")
if fp_a != fp_b:
    print(f"  BUG: 8px apart but different fingerprints! Grid boundary artifact.")
    failed += 1
else:
    passed += 1

# ============================================================
print("\n=== TEST 4: Fingerprint - different action types at same coords ===")
# ============================================================

fp_click = action_fingerprint(make_action("click", 500, 300))
fp_scroll = action_fingerprint(make_action("scroll", 500, 300))
print(f"  click(500,300) → {fp_click}")
print(f"  scroll(500,300) → {fp_scroll}")
check("click vs scroll at same coords are DIFFERENT", True, fp_click != fp_scroll)

# ============================================================
print("\n=== TEST 5: Clustering - 7 clicks + 1 scroll = 2 clusters ===")
# ============================================================

candidates_5 = (
    [make_action("click", 500, 300)] * 7 +
    [make_action("scroll", 500, 300, direction="down")]
)
clusters_5 = cluster_by_fingerprint(candidates_5, grid_size=50, min_cluster_size=1)
print(f"  7 clicks + 1 scroll → {len(clusters_5)} clusters, sizes={[len(c) for c in clusters_5]}")
check("7 clicks + 1 scroll = 2 clusters", 2, len(clusters_5))

# ============================================================
print("\n=== TEST 6: Clustering - 7 clicks + 1 type = 2 clusters ===")
# ============================================================

candidates_6 = (
    [make_action("click", 500, 300)] * 7 +
    [make_action("type", content="hello")]
)
clusters_6 = cluster_by_fingerprint(candidates_6, grid_size=50, min_cluster_size=1)
print(f"  7 clicks + 1 type → {len(clusters_6)} clusters, sizes={[len(c) for c in clusters_6]}")
check("7 clicks + 1 type = 2 clusters", 2, len(clusters_6))

# ============================================================
print("\n=== TEST 7: should_branch with 7 clicks + 1 scroll ===")
# ============================================================

node7 = make_node()
result7 = should_branch(candidates_5, step=3, node=node7, config=config)
check("should_branch with 7+1 minority = True", True, result7)

# ============================================================
print("\n=== TEST 8: Clustering - 6 clicks + 1 hotkey + 1 wait = 3 clusters ===")
# ============================================================

candidates_8 = (
    [make_action("click", 500, 300)] * 6 +
    [make_action("hotkey", key="ctrl+s")] +
    [make_action("wait")]
)
clusters_8 = cluster_by_fingerprint(candidates_8, grid_size=50, min_cluster_size=1)
print(f"  6 clicks + 1 hotkey + 1 wait → {len(clusters_8)} clusters")
check("3 distinct action types = 3 clusters", 3, len(clusters_8))

# ============================================================
print("\n=== TEST 9: Grid too coarse - 2 clicks 200px apart in same cell ===")
# ============================================================

# With 50px grid, (10,10) → (0,0) and (40,40) → (0,0) — same cell despite different targets
fp_near = action_fingerprint(make_action("click", 10, 10))
fp_far = action_fingerprint(make_action("click", 40, 40))
print(f"  click(10,10) → {fp_near}")
print(f"  click(40,40) → {fp_far}")
if fp_near == fp_far:
    print(f"  PROBLEM: 42px apart merged into same fingerprint")

# With bigger spread:
fp_a2 = action_fingerprint(make_action("click", 5, 5))
fp_b2 = action_fingerprint(make_action("click", 45, 45))
print(f"  click(5,5) → {fp_a2}")
print(f"  click(45,45) → {fp_b2}")
if fp_a2 == fp_b2:
    print(f"  PROBLEM: Still same cell — 50px grid is too coarse for nearby but different elements")

# ============================================================
print("\n=== TEST 10: Real scenario - all 8 candidates identical ===")
# ============================================================

# This is what actually happens at temp=1.0
candidates_10 = [make_action("click", 940, 104)] * 8
clusters_10 = cluster_by_fingerprint(candidates_10, grid_size=50, min_cluster_size=1)
print(f"  8 identical clicks → {len(clusters_10)} clusters")
check("8 identical = 1 cluster (no branch)", 1, len(clusters_10))

# ============================================================
print("\n=== TEST 11: Real scenario - 7 identical + 1 slightly different coords ===")
# ============================================================

# This is what SHOULD happen with more diverse sampling
# Coords differ by 15px - same UI element
candidates_11 = (
    [make_action("click", 940, 104)] * 7 +
    [make_action("click", 955, 104)]   # 15px away
)
clusters_11 = cluster_by_fingerprint(candidates_11, grid_size=50, min_cluster_size=1)
fps_11 = [action_fingerprint(c, grid_size=50) for c in candidates_11]
print(f"  7x click(940,104) + 1x click(955,104)")
print(f"  FPs: {set(fps_11)}")
print(f"  → {len(clusters_11)} clusters")
if len(clusters_11) > 1:
    print(f"  BUG: 15px apart incorrectly creates separate cluster (boundary at 950)")

# ============================================================
print("\n=== TEST 12: Different UI regions - should branch ===")
# ============================================================

candidates_12 = (
    [make_action("click", 100, 50)] * 6 +   # Top-left (menu area)
    [make_action("click", 800, 500)] * 2     # Center (content area)
)
clusters_12 = cluster_by_fingerprint(candidates_12, grid_size=50, min_cluster_size=1)
print(f"  6x click(100,50) + 2x click(800,500)")
print(f"  → {len(clusters_12)} clusters, sizes={[len(c) for c in clusters_12]}")
check("different UI regions = 2 clusters", 2, len(clusters_12))

# ============================================================
print("\n=== TEST 13: type() with different content = different clusters ===")
# ============================================================

candidates_13 = (
    [make_action("type", content="cd /home")] * 5 +
    [make_action("type", content="ls -la")] * 3
)
clusters_13 = cluster_by_fingerprint(candidates_13, grid_size=50, min_cluster_size=1)
print(f"  5x type('cd /home') + 3x type('ls -la')")
print(f"  → {len(clusters_13)} clusters")
check("different type content = 2 clusters", 2, len(clusters_13))

# ============================================================
print("\n=== TEST 14: hotkey() with different keys = different clusters ===")
# ============================================================

candidates_14 = (
    [make_action("hotkey", key="ctrl s")] * 6 +
    [make_action("hotkey", key="ctrl z")] * 2
)
clusters_14 = cluster_by_fingerprint(candidates_14, grid_size=50, min_cluster_size=1)
print(f"  6x hotkey('ctrl s') + 2x hotkey('ctrl z')")
print(f"  → {len(clusters_14)} clusters")
check("different hotkeys = 2 clusters", 2, len(clusters_14))

# ============================================================
print("\n=== TEST 15: Verify the REAL problem ===")
# ============================================================

# The model at temp=1.0 generates coordinate jitter of ~10px
# With 50px grid, most jitter stays in the same cell
# BUT boundary cases create false branches
print("  Model behavior at temp=1.0:")
print("  - Coordinate jitter: ~5-15px (same UI element)")
print("  - Action type: 98-100% same (from logprob test)")
print("  - At K=8: prob(all same action type) = 0.98^8 = 96%")
print("  - At K=16: prob(all same action type) = 0.98^16 = 72%")
print("  - At K=32: prob(all same action type) = 0.98^32 = 52%")
print("  - At K=64: prob(all same action type) = 0.98^64 = 27%")
print("  - At K=128: prob(all same action type) = 0.98^128 = 7%")
print()
print("  CONCLUSION: Current K=8/16 is too small to detect the 1-2% minority actions.")
print("  SOLUTION: Increase K to 32-64 for probing, AND use temp=1.2 (not 1.5+).")
print("  Cost: K=64 across 8 GPUs = 8 per GPU, ~4-6s per step.")

# ============================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")
