from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


logger = logging.getLogger("desktopenv.noise")


@dataclass
class NoiseFireResult:
    element_id: str
    category: str
    recovery_cost: int


class NoiseScheduler:
    """Action-triggered noise scheduler.

    The scheduler is intentionally conservative:
    - does nothing unless initialized with elements
    - fires at most one element per step
    - preserves clean behavior when disabled

    Elements follow the repo's `noise_meta.elements[]` schema.
    """

    def __init__(
        self,
        setup_controller,
        elements: List[Dict[str, Any]],
        rng_seed: Optional[int] = None,
        post_fire_pause_sec: float = 0.2,
    ):
        self.setup_controller = setup_controller
        self.elements = [dict(e) for e in elements]
        self.rng = random.Random(rng_seed)
        self.post_fire_pause_sec = post_fire_pause_sec
        self._fired_once: set[str] = set()
        self.total_fires = 0

    def on_step(self, probability: float) -> List[NoiseFireResult]:
        if probability <= 0.0 or not self.elements:
            return []

        candidates = [e for e in self.elements if not (e.get("once", True) and e.get("id") in self._fired_once)]
        if not candidates:
            return []

        if self.rng.random() >= probability:
            return []

        chosen = self.rng.choice(candidates)
        self._fire_element(chosen)
        if chosen.get("once", True):
            self._fired_once.add(chosen.get("id", "unknown"))
        self.total_fires += 1

        return [
            NoiseFireResult(
                element_id=str(chosen.get("id", f"noise_{self.total_fires}")),
                category=str(chosen.get("category", "unknown")),
                recovery_cost=int(chosen.get("recovery_cost", 0)),
            )
        ]

    def _fire_element(self, element: Dict[str, Any]) -> None:
        command = element.get("command")
        if not command:
            logger.warning("noise element %s has no command; skipping", element.get("id"))
            return

        try:
            if isinstance(command, list):
                self.setup_controller._launch_setup(command, shell=False)
            elif isinstance(command, str):
                self.setup_controller._launch_setup(command, shell=True)
            else:
                logger.warning("noise element %s has unsupported command type %s", element.get("id"), type(command).__name__)
                return

            if self.post_fire_pause_sec > 0:
                time.sleep(self.post_fire_pause_sec)

            logger.info(
                "noise fired: id=%s category=%s recovery_cost=%s",
                element.get("id"),
                element.get("category"),
                element.get("recovery_cost", 0),
            )
        except Exception as exc:
            logger.warning("noise element %s failed: %s", element.get("id"), exc)
