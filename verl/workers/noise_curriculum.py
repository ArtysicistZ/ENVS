from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class _TaskNoiseState:
    p_noise: float
    tier: int
    sr_ema: float = 0.0
    recovery_ema: float = 0.0
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
    recovery_ema_alpha: float = 0.2
    initial_tier: int = 1
    initial_p: float = 0.1
    tier_min: int = 0
    tier_max: int = 5
    states: Dict[str, _TaskNoiseState] = field(default_factory=dict)

    def has_task(self, task_id: str) -> bool:
        return task_id in self.states

    def get(self, task_id: str) -> float:
        return self.get_p(task_id)

    def get_p(self, task_id: str) -> float:
        return self._get_state(task_id).p_noise

    def get_tier(self, task_id: str) -> int:
        return self._get_state(task_id).tier

    def get_sr(self, task_id: str) -> float:
        """Live EMA of observed success rate for this task. Used by the v4
        SR-bucket mapping (`fires_for_sr`) to size the per-rollout noise
        budget. If the task has never been updated, returns the seeded value
        (0.0 by default; trainer seeds from MCTS SR file at init)."""
        return self._get_state(task_id).sr_ema

    def seed_task(self, task_id: str, initial_tier: int, initial_probability: float | None = None,
                  initial_sr: float | None = None) -> None:
        self.seed_tier(task_id, initial_tier, initial_probability, initial_sr)

    def seed_tier(self, task_id: str, initial_tier: int, initial_probability: float | None = None,
                  initial_sr: float | None = None) -> None:
        state = self._get_state(task_id)
        state.tier = self._clamp_tier(initial_tier)
        if initial_probability is not None:
            state.p_noise = self._clamp_p(initial_probability)
        if initial_sr is not None:
            # Bootstrap sr_ema from a static prior (e.g. MCTS SR file). Marks
            # the state as initialized so subsequent `update()` blends instead
            # of overwriting on the first observation.
            state.sr_ema = max(0.0, min(1.0, float(initial_sr)))
            state.initialized = True

    def update(self, task_id: str, success: float, recovery_burden: float = 0.0) -> None:
        state = self._get_state(task_id)
        success = float(success)
        recovery_burden = float(recovery_burden)
        if not state.initialized:
            state.sr_ema = success
            state.recovery_ema = recovery_burden
            state.initialized = True
        else:
            state.sr_ema = (1 - self.ema_alpha) * state.sr_ema + self.ema_alpha * success
            state.recovery_ema = (
                (1 - self.recovery_ema_alpha) * state.recovery_ema
                + self.recovery_ema_alpha * recovery_burden
            )

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

        # Mid-band: nudge p toward target while holding tier fixed. Recovery
        # burden acts as a braking signal so tasks that are technically still
        # successful but already paying a large recovery tax do not get pushed
        # into noise escalation too quickly.
        state.promote_streak = 0
        delta = state.sr_ema - self.target_success_rate
        burden_brake = min(1.0, state.recovery_ema / 3.0)
        state.p_noise = self._clamp_p(state.p_noise + self.lr * delta * (1.0 - 0.5 * burden_brake))

    def snapshot(self) -> Dict[str, Dict]:
        return {
            task_id: {
                "p_noise": state.p_noise,
                "tier": state.tier,
                "sr_ema": state.sr_ema,
                "recovery_ema": state.recovery_ema,
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
