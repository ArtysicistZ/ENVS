"""
difficulty.py

Maps MCTS rollout success rate to a difficulty tier and a noise budget.
Noise budget is INVERSELY scaled to difficulty: harder tasks get less noise
(so they remain solvable and preserve training signal), easier tasks get more
noise (to stress-test robustness).

Authoritative source of success rates:
    checkpoints/mcts_trajectories_v2/combined_all/collection_results.json
    (field `.results[]`, each with task_id, domain, success_rate, total_runs,
    successful_trajectories)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


TIER_VERY_HARD = "very_hard"
TIER_HARD = "hard"
TIER_MEDIUM = "medium"
TIER_EASY = "easy"
TIER_VERY_EASY = "very_easy"

TIERS = [TIER_VERY_HARD, TIER_HARD, TIER_MEDIUM, TIER_EASY, TIER_VERY_EASY]


_TIER_BUDGETS: Dict[str, Dict] = {
    TIER_VERY_HARD: {
        "min_elements": 1,
        "max_elements": 1,
        "min_categories": 1,
        "max_categories": 1,
        "description": (
            "MINIMAL noise budget: add EXACTLY 1 noise element from 1 category "
            "(a single `notify-send` notification). This task is near-impossible "
            "(success rate <= 5%); any additional noise risks driving the success "
            "rate to 0 and destroying training signal. Do NOT use active disruptions "
            "(no panel toggles, no view changes, no window occlusion)."
        ),
    },
    TIER_HARD: {
        "min_elements": 2,
        "max_elements": 3,
        "min_categories": 1,
        "max_categories": 2,
        "description": (
            "LIGHT noise budget: add 2-3 noise elements across 1-2 categories. "
            "Use 1-2 passive elements (notification, background app) plus at most "
            "1 MILD active disruption (window_geometry only — e.g., shrink the "
            "window or move it slightly off-screen). No view mode changes, no "
            "panel toggles, no dialogs."
        ),
    },
    TIER_MEDIUM: {
        "min_elements": 4,
        "max_elements": 5,
        "min_categories": 2,
        "max_categories": 3,
        "description": (
            "MODERATE noise budget: add 4-5 noise elements across 2-3 categories. "
            "Mix passive noise (notifications, background apps, filesystem clutter) "
            "with 1-2 active disruptions from: window_geometry, panel_toggle, or "
            "view_mode_change. Active disruptions should be moderate — e.g., a "
            "sidebar opens or zoom changes to 125%. The task MUST remain tractable."
        ),
    },
    TIER_EASY: {
        "min_elements": 6,
        "max_elements": 8,
        "min_categories": 3,
        "max_categories": 5,
        "description": (
            "SUBSTANTIAL noise budget: add 6-8 noise elements across 3-5 categories. "
            "Use a MIX of passive noise AND 2-3 active disruptions. Active categories "
            "now include: window_occlusion, scroll_displacement, dialog_popup, and "
            "focus_stealing, in addition to window_geometry/panel_toggle/view_mode. "
            "The goal is a genuinely messy desktop that forces the agent to adapt."
        ),
    },
    TIER_VERY_EASY: {
        "min_elements": 8,
        "max_elements": 12,
        "min_categories": 4,
        "max_categories": 7,
        "description": (
            "MAXIMUM noise budget: add 8-12 noise elements across 4+ categories, "
            "using EVERY applicable noise category (both passive and active). Include "
            "at least 3-5 active disruptions: window geometry changes, panel toggles, "
            "view mode changes, window occlusion, scroll displacement, dialog popups, "
            "and focus stealing. This is the full 'messy desk' experience."
        ),
    },
}


def rate_to_tier(rate: float) -> str:
    """
    Bucket a success rate [0, 1] into one of 5 difficulty tiers.

    Inverse difficulty: rate -> tier -> noise budget, where LOW rate (hard task)
    gets the SMALLEST budget.
    """
    if rate <= 0.05:
        return TIER_VERY_HARD
    if rate <= 0.15:
        return TIER_HARD
    if rate <= 0.40:
        return TIER_MEDIUM
    if rate <= 0.75:
        return TIER_EASY
    return TIER_VERY_EASY


def tier_to_budget(tier: str) -> Dict:
    """Return the noise budget dict for a given tier."""
    if tier not in _TIER_BUDGETS:
        raise ValueError(f"Unknown tier: {tier}. Must be one of {TIERS}.")
    return _TIER_BUDGETS[tier]


def load_success_rates(collection_path: Path) -> Dict[str, Dict]:
    """
    Load `collection_results.json` and return a dict keyed by task_id.

    Returns: {task_id: {domain, success_rate, total_runs, successful_trajectories}}
    """
    with collection_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError(
            f"{collection_path} does not contain a top-level `results` list"
        )

    out: Dict[str, Dict] = {}
    for entry in results:
        tid = entry.get("task_id")
        if not tid:
            continue
        out[tid] = {
            "domain": entry.get("domain"),
            "success_rate": float(entry.get("success_rate", 0.0)),
            "total_runs": int(entry.get("total_runs", 0)),
            "successful_trajectories": int(entry.get("successful_trajectories", 0)),
        }
    return out
