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
]
