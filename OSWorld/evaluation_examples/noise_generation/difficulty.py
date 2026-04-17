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
from typing import Dict, Tuple


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

# Higher = more fragile / easier to derail, so static noise should be gentler.
DOMAIN_FRAGILITY_PRIOR: Dict[str, float] = {
    "chrome": 0.35,
    "gimp": 0.45,
    "libreoffice_calc": 0.55,
    "libreoffice_impress": 0.60,
    "libreoffice_writer": 0.58,
    "multi_apps": 0.85,
    "os": 0.65,
    "thunderbird": 0.62,
    "vlc": 0.30,
    "vs_code": 0.55,
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


def domain_fragility_prior(domain: str | None) -> float:
    return DOMAIN_FRAGILITY_PRIOR.get(str(domain or "").lower(), 0.50)


def estimate_horizon_proxy(task_json: Dict) -> float:
    """Best-effort static horizon proxy in [0, 1].

    We do not always have trajectory-derived mean step count at task-load time,
    so this uses cheap task metadata as a proxy:
    - more related apps => longer horizon
    - multi-app tasks => longer horizon
    - instructions with sequencing/export/save cues => slightly longer horizon
    """
    related_apps = task_json.get("related_apps") or []
    app_count = len(related_apps) if isinstance(related_apps, list) else 0
    instruction = str(task_json.get("instruction") or "").lower()
    domain = str(task_json.get("domain") or "").lower()

    score = min(app_count / 3.0, 1.0) * 0.45
    if domain == "multi_apps":
        score += 0.35

    cue_hits = 0
    for cue in (" then ", " after ", " export ", " save ", " rename ", " format ", " open ", " create "):
        if cue in f" {instruction} ":
            cue_hits += 1
    score += min(cue_hits / 4.0, 1.0) * 0.20
    return max(0.0, min(1.0, score))


def estimate_ui_fragility(task_json: Dict) -> float:
    """Heuristic UI fragility in [0, 1].

    This captures how easily a task is derailed by modals, focus-steal, or
    layout shifts even if the clean task is not especially hard.
    """
    instruction = str(task_json.get("instruction") or "").lower()
    domain = str(task_json.get("domain") or "").lower()

    score = 0.0
    if domain in {"multi_apps", "os", "thunderbird"}:
        score += 0.25
    if domain in {"libreoffice_writer", "libreoffice_impress", "libreoffice_calc", "vs_code"}:
        score += 0.20

    fragile_cues = (
        "dialog", "popup", "window", "tab", "export", "settings", "preference",
        "sidebar", "panel", "focus", "notification", "banner", "cookie",
    )
    cue_hits = sum(1 for cue in fragile_cues if cue in instruction)
    score += min(cue_hits / 5.0, 1.0) * 0.35

    evaluator = task_json.get("evaluator") or {}
    if isinstance(evaluator, dict) and isinstance(evaluator.get("func"), list):
        score += 0.15

    return max(0.0, min(1.0, score))


def compute_static_noise_seed(
    success_rate: float,
    task_json: Dict,
    explicit_tier: int | None = None,
) -> Dict[str, float | int]:
    """Seed runtime noise from a broader static prior.

    Success rate remains the anchor, but horizon/domain/UI fragility reduce the
    allowed starting tier for brittle tasks. This keeps the current inverse-SR
    intuition while making the controller less one-dimensional.
    """
    if explicit_tier is not None:
        tier = max(0, min(5, int(explicit_tier)))
        return {
            "tier": tier,
            "success_rate": float(success_rate),
            "horizon_proxy": estimate_horizon_proxy(task_json),
            "domain_fragility": domain_fragility_prior(task_json.get("domain")),
            "ui_fragility": estimate_ui_fragility(task_json),
            "fragility_penalty": 0.0,
        }

    base_tier = tier_to_int(rate_to_tier(float(success_rate)))
    horizon = estimate_horizon_proxy(task_json)
    domain_fragility = domain_fragility_prior(task_json.get("domain"))
    ui_fragility = estimate_ui_fragility(task_json)

    fragility_penalty = (
        0.35 * horizon
        + 0.35 * domain_fragility
        + 0.30 * ui_fragility
    )

    # Convert the penalty into at most a two-tier reduction.
    tier_delta = int(round(2.0 * fragility_penalty))
    tier = max(0, min(5, base_tier - tier_delta))

    return {
        "tier": tier,
        "base_tier": base_tier,
        "success_rate": float(success_rate),
        "horizon_proxy": horizon,
        "domain_fragility": domain_fragility,
        "ui_fragility": ui_fragility,
        "fragility_penalty": fragility_penalty,
    }


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
