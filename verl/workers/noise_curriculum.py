from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class _TaskNoiseState:
    p_noise: float
    tier: int
    sr_ema: float = 0.0
    initialized: bool = False
    promote_streak: int = 0


@dataclass
class NoiseCurriculum:
    """Per-task capability-adaptive noise curriculum.

    Design goals:
    - seed each task from a static prior (`initial_tier`, MCTS SR, etc.)
    - adapt noise probability and tier from observed rollout success
    - keep the API aligned with `ray_trainer.py`

    The update rule is intentionally conservative for v1:
    - maintain an EMA of success per task
    - promote when the task stays comfortably above target for `k_up` rounds
    - demote / relax noise when the task falls below the lower threshold
    """

    target_success_rate: float = 0.5
    lr: float = 0.05
    p_min: float = 0.0
    p_max: float = 0.2
    tau_up: float = 0.6
    tau_down: float = 0.15
    k_up: int = 3
    ema_alpha: float = 0.2
    initial_tier: int = 1
    initial_p: float = 0.1
    tier_min: int = 0
    tier_max: int = 5
    states: Dict[str, _TaskNoiseState] = field(default_factory=dict)

    def get(self, task_id: str) -> float:
        return self.get_p(task_id)

    def get_p(self, task_id: str) -> float:
        return self._get_state(task_id).p_noise

    def get_tier(self, task_id: str) -> int:
        return self._get_state(task_id).tier

    def seed_task(self, task_id: str, initial_tier: int, initial_probability: float | None = None) -> None:
        self.seed_tier(task_id, initial_tier, initial_probability)

    def seed_tier(self, task_id: str, initial_tier: int, initial_probability: float | None = None) -> None:
        state = self._get_state(task_id)
        state.tier = self._clamp_tier(initial_tier)
        if initial_probability is not None:
            state.p_noise = self._clamp_p(initial_probability)

    def update(self, task_id: str, success: float) -> None:
        state = self._get_state(task_id)
        success = float(success)
        if not state.initialized:
            state.sr_ema = success
            state.initialized = True
        else:
            state.sr_ema = (1 - self.ema_alpha) * state.sr_ema + self.ema_alpha * success

        # If a task is consistently succeeding, increase noise pressure slowly.
        if state.sr_ema >= self.tau_up:
            state.promote_streak += 1
            if state.promote_streak >= self.k_up:
                state.tier = self._clamp_tier(state.tier + 1)
                state.p_noise = self._clamp_p(state.p_noise + self.lr)
                state.promote_streak = 0
            return

        # If it is struggling badly, back off quickly so we preserve signal.
        if state.sr_ema < self.tau_down:
            state.promote_streak = 0
            state.p_noise = self._clamp_p(state.p_noise - self.lr)
            if state.p_noise <= self.initial_p and state.tier > self.tier_min:
                state.tier = self._clamp_tier(state.tier - 1)
            return

        # Mid-band: nudge p toward target while holding tier fixed.
        state.promote_streak = 0
        delta = state.sr_ema - self.target_success_rate
        state.p_noise = self._clamp_p(state.p_noise + self.lr * delta)

    def snapshot(self) -> Dict[str, Dict]:
        return {
            task_id: {
                "p_noise": state.p_noise,
                "tier": state.tier,
                "sr_ema": state.sr_ema,
                "promote_streak": state.promote_streak,
            }
            for task_id, state in self.states.items()
        }

    def _clamp_p(self, value: float) -> float:
        return max(self.p_min, min(self.p_max, float(value)))

    def _clamp_tier(self, value: int) -> int:
        return max(self.tier_min, min(self.tier_max, int(value)))

    def _get_state(self, task_id: str) -> _TaskNoiseState:
        if task_id not in self.states:
            self.states[task_id] = _TaskNoiseState(
                p_noise=self._clamp_p(self.initial_p),
                tier=self._clamp_tier(self.initial_tier),
            )
        return self.states[task_id]
