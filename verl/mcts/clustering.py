"""Action clustering and branching decision logic for MCTS.

Uses action fingerprinting (50px coordinate grid + content key) instead of
spatial distance clustering. Singletons count as clusters (min_cluster_size=1).

See docs/MCTS/ACTION_LEVEL_MCTS_SFT.md for the design.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# Action parsing
# ============================================================

def parse_action_with_coords(text: str) -> Tuple[str, Optional[int], Optional[int], str]:
    """Parse model output into (action_type, x_pixel, y_pixel, content_key)."""
    text = text.strip()
    if "Action:" not in text:
        return ("UNPARSEABLE", None, None, "")
    action_str = text.split("Action:")[-1].strip().split("\n")[0].strip()
    m = re.match(r"(\w+)\(", action_str)
    if not m:
        return ("UNPARSEABLE", None, None, "")
    action_type = m.group(1)

    coord_match = re.search(r"start_box=['\"].*?\((\d+),(\d+)\).*?['\"]", action_str)
    if coord_match:
        x, y = int(coord_match.group(1)), int(coord_match.group(2))
        return (action_type, x, y, "")

    content_match = re.search(r"content='(.*?)'", action_str)
    if content_match:
        content = _normalize_content(content_match.group(1)[:30])
        return (action_type, None, None, f"c:{content}")
    key_match = re.search(r"key='(.*?)'", action_str)
    if key_match:
        key = _normalize_hotkey(key_match.group(1))
        return (action_type, None, None, f"k:{key}")
    direction_match = re.search(r"direction='(.*?)'", action_str)
    if direction_match:
        return (action_type, None, None, f"d:{direction_match.group(1)}")
    return (action_type, None, None, "")


def _normalize_content(content: str) -> str:
    return " ".join(content.split())


def _normalize_hotkey(key: str) -> str:
    key = " ".join(key.split())
    parts = [p.strip() for p in re.split(r"[+ ]", key) if p.strip()]
    modifiers = sorted(p for p in parts if p in ("ctrl", "alt", "shift", "super", "meta"))
    non_modifiers = [p for p in parts if p not in ("ctrl", "alt", "shift", "super", "meta")]
    return "+".join(modifiers + non_modifiers)


# ============================================================
# Action fingerprinting (replaces spatial distance clustering)
# ============================================================

def action_fingerprint(text: str, grid_size: int = 50) -> str:
    """Compute a discrete fingerprint for an action.

    Spatial actions: quantize coordinates to a grid.
    Non-spatial actions: use content/key string.

    Two actions with the same fingerprint are "the same strategy."
    Two with different fingerprints are candidates for branching.
    """
    atype, x, y, ckey = parse_action_with_coords(text)

    if atype == "UNPARSEABLE":
        return f"UNPARSEABLE:{text[:50]}"

    if x is not None and y is not None:
        gx, gy = x // grid_size, y // grid_size
        return f"{atype}@({gx},{gy})"

    if ckey:
        return f"{atype}:{ckey}"

    return atype


def cluster_by_fingerprint(
    candidates: List[str],
    grid_size: int = 50,
    min_cluster_size: int = 1,
) -> List[List[int]]:
    """Cluster candidates by action fingerprint.

    Returns list of clusters (each is a list of candidate indices),
    sorted by size descending. Only includes clusters with >= min_cluster_size.
    """
    fps = [action_fingerprint(c, grid_size) for c in candidates]
    groups = defaultdict(list)
    for i, fp in enumerate(fps):
        groups[fp].append(i)

    clusters = [indices for indices in groups.values() if len(indices) >= min_cluster_size]
    clusters.sort(key=len, reverse=True)
    return clusters


def get_significant_clusters(
    candidates: List[str],
    dist_threshold: int = 50,
    min_cluster_size: int = 1,
) -> List[List[int]]:
    """Return clusters with >= min_cluster_size members.

    Uses fingerprint-based clustering (not spatial distance).
    `dist_threshold` is repurposed as `grid_size` for the fingerprint.
    """
    return cluster_by_fingerprint(candidates, grid_size=dist_threshold, min_cluster_size=min_cluster_size)


def representative(candidates: List[str], cluster: List[int]) -> str:
    """Pick the most common candidate from a cluster."""
    if not cluster:
        return candidates[0] if candidates else ""
    # Return the most frequent candidate text in the cluster
    texts = [candidates[i] for i in cluster]
    counter = Counter(texts)
    return counter.most_common(1)[0][0]


def compute_type_entropy(candidates: List[str]) -> float:
    """Compute Shannon entropy over action types (bits)."""
    types = [parse_action_with_coords(c)[0] for c in candidates]
    counter = Counter(types)
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


# ============================================================
# Branching decision
# ============================================================

def should_branch(
    candidates: List[str],
    step: int,
    node,  # TreeNode
    config,  # MCTSConfig
) -> bool:
    """Branching decision with simplified gates.

    Gate 1: >= 2 distinct fingerprint clusters
    Gate 2: Late-step deferral (step >= threshold: require 2 consecutive)
    Gate 3: Branch budget remaining
    Gate 4: Step cutoff (never_branch_after)
    """
    node_id = getattr(node, 'node_id', '?')

    # Compute fingerprint clusters
    clusters = cluster_by_fingerprint(
        candidates,
        grid_size=config.spatial_grid_size,
        min_cluster_size=config.min_cluster_size,
    )

    parsed_types = [parse_action_with_coords(c)[0] for c in candidates]
    type_counts = Counter(parsed_types)
    fps = [action_fingerprint(c, config.spatial_grid_size) for c in candidates]
    fp_counts = Counter(fps)

    logger.info("should_branch [%s step=%d]: %d candidates, types=%s, %d fingerprints, clusters=%s",
                node_id, step, len(candidates), dict(type_counts),
                len(fp_counts), [len(c) for c in clusters])

    # Gate 1: >= 2 distinct clusters
    if len(clusters) < 2:
        logger.info("should_branch [%s step=%d]: REJECTED Gate 1 (need >=2 clusters, got %d)",
                     node_id, step, len(clusters))
        return False

    # Gate 2: Late-step deferral
    if step >= config.late_step_threshold:
        if not node.prev_high:
            logger.info("should_branch [%s step=%d]: REJECTED Gate 2 (late deferral, first time)",
                         node_id, step)
            node.prev_high = True
            return False
        # 2nd consecutive — pass
    else:
        node.prev_high = False  # reset when below threshold

    # Gate 3: Budget
    if not node.budget.can_branch():
        logger.info("should_branch [%s step=%d]: REJECTED Gate 3 (budget exhausted)", node_id, step)
        return False

    # Gate 4: Step cutoff
    if step >= config.never_branch_after:
        logger.info("should_branch [%s step=%d]: REJECTED Gate 4 (step >= %d)",
                     node_id, step, config.never_branch_after)
        return False

    logger.info("should_branch [%s step=%d]: ALL GATES PASSED — %d clusters, branching!",
                node_id, step, len(clusters))
    return True


# Keep old function name for compatibility
def hierarchical_cluster(candidates, dist_threshold=50, min_cluster_size=1):
    """Backward-compatible wrapper."""
    return cluster_by_fingerprint(candidates, grid_size=dist_threshold, min_cluster_size=min_cluster_size)
