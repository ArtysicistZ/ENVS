from __future__ import annotations

import logging
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
    """v4 deterministic-schedule noise scheduler.

    Each element carries a `fire_step` index (set by
    `RuntimeNoiseSampler.sample_fire_schedule`). On every agent step
    `on_step()` is called; if the current step index matches any element's
    `fire_step`, that element fires exactly once. No probability dice — the
    schedule is pre-determined per `(task_id, training_step)` so all `n`
    rollouts of a GRPO group see identical firings.
    """

    def __init__(
        self,
        setup_controller,
        elements: List[Dict[str, Any]],
        post_fire_pause_sec: float = 0.2,
    ):
        self.setup_controller = setup_controller
        self.elements = [dict(e) for e in elements]
        self.post_fire_pause_sec = post_fire_pause_sec
        # Index by fire_step for O(1) lookup. Multiple elements per step are
        # allowed (rare; sampler usually de-dups bucket spacing).
        self._by_step: Dict[int, List[Dict[str, Any]]] = {}
        for el in self.elements:
            fs = int(el.get("fire_step", -1))
            if fs >= 0:
                self._by_step.setdefault(fs, []).append(el)
        self._fired_once: set[str] = set()
        self._step_counter: int = 0
        self.total_fires = 0

    def on_step(self, probability: float = 1.0) -> List[NoiseFireResult]:  # noqa: ARG002 — probability kept for API back-compat
        """Fire any element whose `fire_step` matches this step's index.

        `probability` is accepted for back-compat with v3 callers but ignored
        — the schedule is deterministic.
        """
        idx = self._step_counter
        self._step_counter += 1
        due = self._by_step.get(idx, [])
        if not due:
            return []
        results: List[NoiseFireResult] = []
        for chosen in due:
            eid = chosen.get("id", f"noise_{self.total_fires}")
            if chosen.get("once", True) and eid in self._fired_once:
                continue
            self._fire_element(chosen)
            if chosen.get("once", True):
                self._fired_once.add(eid)
            self.total_fires += 1
            results.append(NoiseFireResult(
                element_id=str(eid),
                category=str(chosen.get("category", "unknown")),
                recovery_cost=int(chosen.get("recovery_cost", 0)),
            ))
        return results

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
