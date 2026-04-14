"""
difficulty.py — v3 tier budgets (used by RuntimeNoiseSampler).

Maps MCTS rollout success rate to a difficulty tier and a per-task
cumulative `recovery_cost` ceiling. Noise intensity is INVERSELY scaled
to difficulty: harder tasks get less noise (so they remain solvable and
preserve training signal), easier tasks get more.

v3 budgets (robustness, not difficulty — see plan file):
  - very_hard  (SR ≤ 5%):  cap 0  — cost-0 ambient only, no interruptions
  - hard       (≤ 15%):    cap 1
  - medium     (≤ 40%):    cap 3
  - easy       (≤ 75%):    cap 4
  - very_easy  (> 75%):    cap 6

Authoritative source of success rates:
    checkpoints/mcts_trajectories_v2/combined_all/collection_results.json
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


# v3 cost-cap budgets. Each dict gives the per-task cumulative
# `recovery_cost` ceiling and a human description.
_TIER_BUDGETS: Dict[str, Dict] = {
    TIER_VERY_HARD: {
        "max_recovery_cost": 0,
        "description": (
            "NO active noise. Only cost-0 ambient elements may fire (passive "
            "notifications, visual flickers, state drift). The task is already "
            "near-impossible (SR ≤ 5%); any interruption risks destroying "
            "training signal."
        ),
    },
    TIER_HARD: {
        "max_recovery_cost": 1,
        "description": (
            "LIGHT noise. Unlimited cost-0 ambient + at most one cost-1 "
            "interruption (modal or focus-steal). Task success is rare; keep "
            "the gradient signal alive."
        ),
    },
    TIER_MEDIUM: {
        "max_recovery_cost": 3,
        "description": (
            "MODERATE noise. Ambient + up to 3 cost-units of interruption "
            "(e.g., one cost-2 occlusion + one cost-1 modal, or three cost-1 "
            "events). Task remains tractable."
        ),
    },
    TIER_EASY: {
        "max_recovery_cost": 4,
        "description": (
            "SUBSTANTIAL noise. Ambient + up to 4 cost-units (various mixes: "
            "occlusion + modal + focus-steal, or several cost-1 events). "
            "Genuinely cluttered desktop."
        ),
    },
    TIER_VERY_EASY: {
        "max_recovery_cost": 6,
        "description": (
            "MAX noise. Ambient + up to 6 cost-units. Full 'messy desk' "
            "experience including compositional multi-element firings."
        ),
    },
}


# v3 mapping — tier name → numeric curriculum tier used by the sampler.
# Curriculum tier values match `runtime_sampler.TIER_COST_CAP` keys.
TIER_NAME_TO_INT: Dict[str, int] = {
    TIER_VERY_HARD: 0,
    TIER_HARD:      1,
    TIER_MEDIUM:    3,
    TIER_EASY:      4,
    TIER_VERY_EASY: 5,
}


def rate_to_tier(rate: float) -> str:
    """
    Bucket a success rate [0, 1] into one of 5 difficulty tiers.

    Inverse difficulty: LOW rate (hard task) → SMALLEST budget.
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


def tier_to_int(tier: str) -> int:
    """Map tier name → numeric tier for the curriculum/sampler."""
    return TIER_NAME_TO_INT[tier]


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
