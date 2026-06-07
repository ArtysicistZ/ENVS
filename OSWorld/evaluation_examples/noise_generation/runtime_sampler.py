"""
runtime_sampler.py — Universal random noise sampling from the 151-template library.

Design goal: no per-task noise JSON files. Every rollout draws a fresh
tier-appropriate subset from the template library, producing maximum
diversity without per-task authoring.

Architecture (Procgen-inspired): the 151-template library IS the noise
distribution. `sample_for_task()` returns a list of materialized element
dicts ready for `NoiseScheduler`, respecting:

  - task's `related_apps` (for target-touching templates, resolved via
    `templates.resolve_app_title(...)`)
  - curriculum tier (filters `tier_group ≤ current_tier`)
  - cost cap derived from tier (enforces cumulative `recovery_cost ≤ cap`)
  - optional per-task `avoid_categories` overlay
  - train/held-out catalog split (for Procgen-style OOD measurement)

Key constant: `HELDOUT_TEMPLATE_NAMES` — the 24 template names reserved
from training. Chosen to span categories (visual variants, compositional,
recovery variants, target-touching, persistent-overlay) so the held-out
distribution exercises kinds the training distribution doesn't cover.
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, List, Optional, Set

# Ensure sibling `templates` module is importable regardless of how this file
# is loaded (package import from the repo root OR direct import after the
# noise_generation dir is on sys.path).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import templates as _T  # noqa: E402


# ---------------------------------------------------------------------------
# Held-out OOD split (Procgen-style)
# ---------------------------------------------------------------------------
# 24 template names (~16% of the library) reserved from training. Touches
# every major category so the held-out distribution tests generalization
# across noise kinds rather than just a subset of one category.
HELDOUT_TEMPLATE_NAMES: Set[str] = {
    # Visual variants (diverse rendering)
    "modal_tkinter_custom_banner",
    "modal_browser_alert",
    "modal_browser_dialog_element",
    "notif_tkinter_corner_toast",
    "notif_terminal_flash",
    "overlay_tkinter_floating_widget",
    "overlay_browser_cookie_real",
    "focus_steal_scrolling_terminal",
    # Compositional
    "comp_cookie_plus_popup",
    "comp_ad_banner_plus_chat",
    "comp_decoy_nothing",
    # Recovery-path variants (each is a distinct recovery kind)
    "recovery_escape_only",
    "recovery_drag_window_required",
    "recovery_resize_required",
    "recovery_scroll_to_dismiss",
    "recovery_double_click",
    "recovery_type_to_close",
    "recovery_drag_to_corner",
    # App/browser/OS event subclass
    "app_license_key_prompt",
    "app_export_dialog",
    "browser_install_extension_banner",
    "browser_clear_browsing_data",
    "os_wifi_connection_dialog",
    "os_firewall_block_notice",
}


# ---------------------------------------------------------------------------
# Tier cost caps (v3 — robustness, not difficulty)
# ---------------------------------------------------------------------------
# Maps curriculum tier to the cumulative `recovery_cost` budget per task.
# Tiers 0..5 are used by the adaptive curriculum:
#   0 = no noise (pre-training baseline)
#   1 = cost cap 1 (ambient + occasional cost-1)
#   2 = cost cap 2
#   3 = cost cap 3 (medium default)
#   4 = cost cap 4
#   5 = cost cap 6 (max intensity)
TIER_COST_CAP: Dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 6,
}


def tier_for_success_rate(sr: float) -> int:
    """MCTS success rate -> starting curriculum tier.
    These mirror the v3 budgets from the plan (very_hard=0, ... very_easy=5)."""
    if sr <= 0.05:
        return 0  # very_hard — no noise; task is already near-impossible
    if sr <= 0.15:
        return 1  # hard — one cost-1 max
    if sr <= 0.40:
        return 3  # medium
    if sr <= 0.75:
        return 4  # easy
    return 5      # very_easy


def fires_for_sr(sr: float, rng: random.Random) -> int:
    """Map per-task success rate to a deterministic fire-count budget.

    This is the v4 difficulty knob — replaces probabilistic per-step dice
    rolls. The count is sampled from `rng` so that all `n` rollouts of the
    same `(task_id, training_step)` group get the same count (CRN with Fix A).

    Buckets (v4 spec):
      SR ≥ 0.50 → 3-5 fires per rollout (easy task can absorb noise)
      SR 0.25-0.50 → 1-3 fires
      SR 0.10-0.25 → exactly 1 fire
      SR < 0.10 → 0 fires (very_hard task: protect training signal)
    """
    if sr < 0.10:
        return 0
    if sr < 0.25:
        return 1
    if sr < 0.50:
        return rng.randint(1, 3)
    return rng.randint(3, 5)


def fires_for_sr_mcts(sr: float) -> int:
    """MCTS collection fire count — adaptive to task difficulty.

    Unlike ``fires_for_sr`` (used in ARPO training with CRN), this function
    is deterministic (no rng) because each MCTS branch gets its own seed.

    Buckets:
      SR < 0.15  → 0 fires (very hard task: noise would only produce failures)
      SR 0.15–0.60 → 1 fire (recoverable single disruption)
      SR > 0.60  → 2 fires (easy task can handle sequential disruptions)
    """
    if sr < 0.15:
        return 0
    if sr <= 0.60:
        return 1
    return 2


def fires_for_task_eval(task_id: str) -> int:
    """Deterministic fire count for noisy evaluation: always 1 per task.

    Every task in the 300-task eval gets exactly one held-out noise fire.
    We dropped the prior 80/20 noisy/clean hash-partition because a 56-task
    within-run clean control has almost no statistical power at n=1, and
    external clean baselines (e.g. sft_v1/eval_baseline_greedy) serve that
    role far better. The `task_id` argument is kept for call-site
    compatibility and for future per-task customization.
    """
    return 1 if task_id else 0


def feasibility_constrained_fire_steps(
    count: int,
    max_steps: int,
    element_costs: List[int],
    rng: random.Random,
    min_fire_step: int = 3,
    min_task_buffer: int = 4,
) -> List[int]:
    """Place ``count`` fire steps such that recovery is always feasible.

    Each fire must leave enough remaining steps for recovery AND task work:
      fire_step <= max_steps - recovery_cost - min_task_buffer

    For 2 fires the windows are non-overlapping: fire 1 in the first half,
    fire 2 in the second half (after fire 1's recovery window).

    Returns sorted fire-step indices, possibly fewer than ``count`` if
    feasibility cannot be satisfied for all elements.
    """
    if count <= 0 or max_steps <= 0 or not element_costs:
        return []

    steps: List[int] = []

    if count == 1:
        cost = element_costs[0]
        hi = max_steps - cost - min_task_buffer
        if hi < min_fire_step:
            return []  # element too costly for this horizon
        steps.append(rng.randint(min_fire_step, hi))

    elif count >= 2:
        # Fire 1: first half of the rollout
        cost1 = element_costs[0]
        mid = max_steps // 2
        hi1 = min(mid, max_steps - cost1 - min_task_buffer)
        if hi1 < min_fire_step:
            return []  # first element infeasible
        step1 = rng.randint(min_fire_step, hi1)
        steps.append(step1)

        # Fire 2: second half, after fire 1's recovery window
        cost2 = element_costs[1] if len(element_costs) > 1 else element_costs[0]
        lo2 = step1 + cost1 + 2  # +2 for perception + execution gap
        hi2 = max_steps - cost2 - min_task_buffer
        if lo2 > hi2:
            # Second fire infeasible — return only the first
            return steps
        steps.append(rng.randint(lo2, hi2))

    return sorted(steps)


def bucket_spaced_fire_steps(count: int, max_steps: int, rng: random.Random) -> List[int]:
    """Pick `count` fire-step indices in `[0, max_steps)` with bucket-based
    spacing to prevent harmful clumping (e.g. 5 fires all in steps 0-4).

    Algorithm: split `[0, max_steps)` into `count` equal buckets, pick one
    random step from each bucket. Within a bucket the choice is uniform.
    Result is sorted ascending.
    """
    if count <= 0 or max_steps <= 0:
        return []
    if count >= max_steps:
        return list(range(max_steps))
    bucket = max_steps / count
    steps: List[int] = []
    for b in range(count):
        lo = int(b * bucket)
        hi = min(int((b + 1) * bucket), max_steps) - 1
        if hi < lo:
            hi = lo
        steps.append(rng.randint(lo, hi))
    return sorted(set(steps))


# ---------------------------------------------------------------------------
# Per-task overrides (optional)
# ---------------------------------------------------------------------------

_OVERRIDES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "noise_overrides.json",
)


def _load_overrides() -> Dict[str, Dict]:
    """Load optional per-task override file. Returns {} if missing/malformed."""
    if not os.path.exists(_OVERRIDES_PATH):
        return {}
    try:
        with open(_OVERRIDES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# RuntimeNoiseSampler
# ---------------------------------------------------------------------------

class RuntimeNoiseSampler:
    """Sample noise elements at rollout start, no per-task JSON files required.

    Usage:
        sampler = RuntimeNoiseSampler()
        elements = sampler.sample_for_task(
            task_json=<clean_task_dict>,
            tier=3,
            max_recovery_cost=3,
        )
        # elements is a list of dicts; hand to NoiseScheduler.
    """

    def __init__(
        self,
        catalog: Optional[List[Dict]] = None,
        heldout_names: Optional[Set[str]] = None,
        rng_seed: Optional[int] = None,
    ):
        """
        Args:
            catalog: list of template dicts. Defaults to templates.TEMPLATE_CATALOG.
            heldout_names: set of template names reserved from training.
                Defaults to HELDOUT_TEMPLATE_NAMES.
            rng_seed: seed for sampling RNG. None → use fresh random state.
        """
        full = catalog if catalog is not None else _T.TEMPLATE_CATALOG
        held = heldout_names if heldout_names is not None else HELDOUT_TEMPLATE_NAMES
        self.train_catalog = [e for e in full if e["name"] not in held]
        self.eval_catalog = [e for e in full if e["name"] in held]
        self.rng = random.Random(rng_seed)
        self._overrides = _load_overrides()

    def sample_for_task(
        self,
        task_json: Dict,
        tier: int,
        max_recovery_cost: Optional[int] = None,
        avoid_categories: Optional[Set[str]] = None,
        use_heldout: bool = False,
    ) -> List[Dict]:
        """Sample an element list for one rollout.

        Args:
            task_json: the clean OSWorld task dict. Must have `id` and may have
                `related_apps`. Not modified.
            tier: current curriculum tier for this task (0..5).
            max_recovery_cost: override the tier's default cost cap if given.
            avoid_categories: additional categories to exclude (merges with
                any in the overrides file for this task_id).
            use_heldout: if True, sample from the held-out catalog (for OOD
                eval). Default False (training).

        Returns:
            List of materialized element dicts. Each has keys:
              id, category, recovery_cost, once, command
            Ready to hand to NoiseScheduler. May be empty (e.g. tier 0, or
            task has no compatible templates).
        """
        if tier <= 0:
            return []  # tier 0 = no noise

        if max_recovery_cost is None:
            max_recovery_cost = TIER_COST_CAP.get(tier, 0)

        # Resolve target-app WM title from the task's existing related_apps
        # field. If the task has no single identifiable target (e.g.
        # multi_apps), target-touching templates are excluded.
        target_title = _T.resolve_app_title(task_json.get("related_apps", []))

        # Merge override avoid_categories for this task.
        task_id = task_json.get("id")
        effective_avoid: Set[str] = set(avoid_categories or set())
        if task_id and task_id in self._overrides:
            extra = self._overrides[task_id].get("avoid_categories", [])
            effective_avoid.update(extra)

        catalog = self.eval_catalog if use_heldout else self.train_catalog

        # Filter: tier gate.
        eligible = [e for e in catalog if e.get("tier_group", 2) <= max(tier, 3)]
        # If no single target app, skip target-touching templates.
        if target_title is None:
            eligible = [e for e in eligible if not e.get("needs_target")]
        # Apply avoid_categories.
        if effective_avoid:
            eligible = [e for e in eligible if e.get("category") not in effective_avoid]

        if not eligible:
            return []

        # Split by cost.
        cost0 = [e for e in eligible if e.get("cost", 0) == 0]
        cost_pos = [e for e in eligible if e.get("cost", 0) > 0]

        selected: List[Dict] = []

        # Always include up to 2 cost-0 ambient elements if available.
        if cost0:
            k = min(2, len(cost0))
            selected.extend(self.rng.sample(cost0, k=k))

        # Greedy-fit cost-positive elements under the cap.
        self.rng.shuffle(cost_pos)
        cost_used = 0
        for e in cost_pos:
            c = e.get("cost", 0)
            if cost_used + c <= max_recovery_cost:
                selected.append(e)
                cost_used += c
                if cost_used >= max_recovery_cost:
                    break

        return [self._materialize(e, target_title) for e in selected]

    def sample_fire_schedule(
        self,
        task_json: Dict,
        sr: float,
        max_steps: int,
        avoid_categories: Optional[Set[str]] = None,
        use_heldout: bool = False,
        is_eval: bool = False,
    ) -> List[Dict]:
        """v4 deterministic fire schedule.

        Returns a list of materialized noise elements, each with an extra
        `fire_step` field giving the agent step index at which it should fire.
        Count is determined by `fires_for_sr(sr)` (or `fires_for_task_eval` when
        ``is_eval=True``); fire-step indices are bucket-spaced uniformly within
        `[0, max_steps)` so the rollout is not front-loaded with a clump of noise.

        Determinism: all randomness (fire steps, element selection) goes
        through `self.rng`. With Fix A's `(task_id, training_step)` seed, all
        n rollouts in a GRPO group get the SAME schedule. When ``is_eval=True``,
        the count is a pure function of task_id (SR-independent) and the seed
        is expected to be fixed (noise_step_seed=0), making the schedule
        identical across every eval run.

        Hard-task protection (training only): SR < 0.10 returns [].
        Eval: ~20% of tasks (by md5-hash partition) return []; the rest fire once.
        """
        if is_eval:
            count = fires_for_task_eval(task_json.get("id", ""))
        else:
            count = fires_for_sr(sr, self.rng)
        if count <= 0 or max_steps <= 0:
            return []

        target_title = _T.resolve_app_title(task_json.get("related_apps", []))
        task_id = task_json.get("id")
        effective_avoid: Set[str] = set(avoid_categories or set())
        if task_id and task_id in self._overrides:
            effective_avoid.update(self._overrides[task_id].get("avoid_categories", []))

        catalog = self.eval_catalog if use_heldout else self.train_catalog
        eligible = list(catalog)
        if target_title is None:
            eligible = [e for e in eligible if not e.get("needs_target")]
        if effective_avoid:
            eligible = [e for e in eligible if e.get("category") not in effective_avoid]
        if not eligible:
            return []

        # Cap count by what the eligible pool can support (no replacement).
        count = min(count, len(eligible))
        chosen_templates = self.rng.sample(eligible, k=count)
        fire_steps = bucket_spaced_fire_steps(count, max_steps, self.rng)
        # Pad fire_steps if dedup shrunk the list (rare).
        while len(fire_steps) < count:
            fire_steps.append(min(max_steps - 1, fire_steps[-1] + 1 if fire_steps else 0))
        fire_steps = sorted(fire_steps)[:count]

        out: List[Dict] = []
        for step_idx, tmpl in zip(fire_steps, chosen_templates):
            mat = self._materialize(tmpl, target_title)
            mat["fire_step"] = int(step_idx)
            out.append(mat)
        return out

    def _materialize(self, template_entry: Dict, target_title: Optional[str]) -> Dict:
        """Call the template's bash-generator fn and wrap as an element dict."""
        fn = template_entry["fn"]
        if template_entry.get("needs_target"):
            cmd_str = fn(target_title)
        else:
            cmd_str = fn()
        eid = f"rt_{template_entry['name']}_{self.rng.randint(0, 99999)}"
        return {
            "id": eid,
            "category": template_entry.get("category", "unknown"),
            "recovery_cost": template_entry.get("cost", 0),
            "once": template_entry.get("once_default", True),
            "command": ["bash", "-c", cmd_str],
            "template_name": template_entry["name"],
        }


__all__ = [
    "RuntimeNoiseSampler",
    "HELDOUT_TEMPLATE_NAMES",
    "TIER_COST_CAP",
    "tier_for_success_rate",
    "fires_for_task_eval",
]
