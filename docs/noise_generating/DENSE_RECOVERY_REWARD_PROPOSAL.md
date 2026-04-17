# Proposal: Dense Noise-Recovery Rewards for ARPO Training

**Author**: Auto-generated from deep codebase analysis  
**Date**: 2026-04-15  
**Status**: Proposal  

---

## 1. Problem Statement

Noisy ARPO training with GRPO on 86 OSWorld tasks (UI-TARS-1.5-7B) fails to learn. Eval success rate oscillates around 0.37 across 95 training steps with zero upward trend.

### Root Cause

GRPO uses **outcome-only reward**: a binary 0/1 placed on the last token of each rollout, with the same advantage broadcast to all 15 steps. When noise (modal dialogs, occlusion windows, focus steals) fires mid-rollout:

1. **Zero-gradient groups**: With CRN (Common Random Numbers), all n=8 rollouts see identical noise. If noise makes the task impossible, all 8 get reward=0, GRPO std=0, advantage=0 → **zero gradient** for that task.
2. **No credit assignment**: Even when some rollouts succeed, the broadcast advantage gives equal credit to correct pre-noise actions and futile post-noise actions.
3. **No recovery learning**: The model has no signal about whether dismissing noise is useful, because recovery is invisible to the outcome-only reward.

### What We Cannot Do

We **cannot label arbitrary steps as correct or incorrect** mid-trajectory. The task evaluator only runs at the end. There is no ground-truth intermediate progress measure.

### What We Can Verify

There is exactly **one verifiable dense signal**: when the agent successfully brings the task application window back to the foreground after noise displaced it. This is observable by checking which window has focus after each agent step.

---

## 2. Related Work

### PRIME — Process Reinforcement through Implicit Rewards
**[Cui et al., 2025. arxiv 2502.01456](https://arxiv.org/abs/2502.01456)**

Implicit process reward: `r_t = β · log(π_φ(a_t|s_t) / π_ref(a_t|s_t))` where β=0.05. Trains an implicit PRM online with binary outcome labels. Compatible with GRPO — modifies only advantage estimation. Achieves 2.5× training acceleration over outcome-only rewards.

**Relevance**: We can use PRIME's implicit step rewards as an additional (optional) layer on top of explicit recovery rewards. Both `old_log_probs` and `ref_log_probs` are already computed in our training pipeline.

### iStar — Agentic RL with Implicit Step Rewards
**[ICLR 2026. arxiv 2509.19199](https://arxiv.org/abs/2509.19199)**

Combined advantage: `A(a_t) = A^E(τ) + α · A^S(a_t)` where A^E is episode-level (GRPO) and A^S is step-level (implicit PRM via trajectory DPO). Compatible with GRPO, RLOO, DAPO.

**Relevance**: Validates the approach of combining outcome-level and step-level advantages. The mixing coefficient α controls the balance.

### ADMIRE — Adaptive Milestone Reward for GUI Agents
**[2026. arxiv 2602.11524](https://arxiv.org/abs/2602.11524)**

Extracts milestones from successful trajectories via LLM. Matches current steps to milestones via sentence-BERT (cosine similarity > 0.75). Achieves +11.2% SR on AndroidWorld over outcome-only reward.

**Relevance**: Validates dense rewards for GUI agents. However, ADMIRE requires LLM-based milestone extraction and sentence-BERT matching, which adds significant complexity. Our noise-recovery signal is simpler and directly verifiable.

---

## 3. Proposed Design

### 3.1 Core Insight: Recovery = Task Window Regains Focus

Noise elements push the task application out of focus. The correct recovery is **not** closing the noise window — it is **raising the task application back to the foreground**. The agent might:
- Click the task app's visible edge behind the modal
- Alt-Tab to cycle back to the task window
- Click the task app's icon in the taskbar
- Press Escape to dismiss a modal, revealing the task window

**Detection**: After each agent step, check which window is active/on-top via `xdotool getactivewindow getwindowname`. If the task application regained focus after being displaced by noise → **recovery event**.

This is robust because:
- We don't need to track noise window titles (which vary per element)
- We check the **positive state** (task app active) rather than the negative (noise gone)
- Any recovery method works — we reward the outcome, not the specific action
- The noise window may still exist but be behind the task window — that's fine

### 3.2 Recovery Detection Flow

```
Step t:   Noise fires → active window changes to noise window
          ↓ _noise_displaced_task_window = True
Step t+1: Agent acts → check active window
          → If task window active: recovery_detected = True, reward = 0.1 × cost
          → If noise window still active: no recovery, continue
Step t+2: Agent acts → check again
          → Same check
...
Step t+k: Agent restores task window → recovery detected
```

### 3.3 Environment-Side Implementation

In `OSWorld/desktop_env/desktop_env.py`, modify `step()`:

```python
def step(self, action, pause=2, noise_probability=0.0):
    # Record active window BEFORE agent action
    pre_action_window = self._get_active_window()
    
    # Execute agent action (existing code)
    self.controller.execute_python_command(action['command'])
    
    # Fire noise (existing code — may change active window)
    noise_fired = False
    if self.noise_enabled and self.noise_scheduler:
        fired = self.noise_scheduler.on_step(...)
        noise_fired = len(fired) > 0
    
    time.sleep(pause)
    observation = self._get_obs()
    
    # --- NEW: Check recovery ---
    post_action_window = self._get_active_window()
    recovery_detected = False
    recovery_cost_recovered = 0
    
    if self._noise_displaced_task_window:
        # Task window was displaced by noise in a previous step
        if self._is_task_window(post_action_window):
            recovery_detected = True
            recovery_cost_recovered = self._displaced_recovery_cost
            self._noise_displaced_task_window = False
    
    if noise_fired and not self._is_task_window(post_action_window):
        self._noise_displaced_task_window = True
        self._displaced_at_step = self._step_no
        self._displaced_recovery_cost = sum(
            evt.recovery_cost for evt in fired
        )
    
    info["noise_recovery"] = recovery_detected
    info["noise_recovery_cost"] = recovery_cost_recovered
    info["noise_recovery_steps"] = (
        self._step_no - self._displaced_at_step if recovery_detected else 0
    )
    
    return observation, reward, done, info

def _get_active_window(self):
    """Get the currently focused window's title."""
    try:
        return self.controller.execute_command(
            "xdotool getactivewindow getwindowname 2>/dev/null"
        ).strip()
    except Exception:
        return ""

def _is_task_window(self, window_name):
    """Check if window belongs to the task's target application."""
    if not window_name or not self._task_app_pattern:
        return False
    return self._task_app_pattern.lower() in window_name.lower()
```

The `_task_app_pattern` is derived from the task config's application domain during `reset()`:

| Task Domain | Pattern | Example Window Titles |
|-------------|---------|----------------------|
| browser_* | `"chrome\|chromium\|firefox"` | "Google Chrome", "New Tab - Chromium" |
| gimp | `"gimp"` | "GNU Image Manipulation Program" |
| libreoffice_* | `"libreoffice\|writer\|calc\|impress"` | "Untitled 1 - LibreOffice Writer" |
| thunderbird | `"thunderbird"` | "Mozilla Thunderbird" |
| terminal | `"terminal\|bash"` | "Terminal" |
| vscode | `"code\|visual studio"` | "Visual Studio Code" |
| file_manager | `"files\|nautilus"` | "Files" |

### 3.4 Env Server Propagation

In `scripts/servers/remote_env_server.py`, add recovery info to `noise_burden`:

```python
noise_burden["recovery_detected"] = info.get("noise_recovery", False)
noise_burden["recovery_reward"] = (
    0.1 * info.get("noise_recovery_cost", 0) if info.get("noise_recovery") else 0.0
)
```

### 3.5 Training-Side Integration

In `verl/trainer/ray_trainer.py`, after building the batch (~line 2065):

```python
# Extract per-step recovery rewards from metadata
recovery_rewards = self._build_recovery_reward_tensor(
    batch.non_tensor_batch.get("rollout_step_metadata", []),
    batch.batch["labels"],
    response_len=batch.batch["responses"].size(1),
)
# Add recovery rewards to token_level_scores (alongside outcome on last token)
batch.batch["token_level_scores"] = batch.batch["token_level_scores"] + recovery_rewards
```

The recovery reward is placed on the **last token of the recovery step** so that GRPO's `sum(dim=-1)` includes it in the total score. This creates variance within the GRPO group:

| Rollout | Recovery | Outcome | Total Reward |
|---------|----------|---------|--------------|
| 1 | 0.1 | 0 | 0.1 |
| 2 | 0 | 0 | 0 |
| 3 | 0.1 | 0 | 0.1 |
| 4 | 0 | 0 | 0 |
| 5 | 0 | 1 | 1.0 |
| 6 | 0.1 | 1 | 1.1 |
| 7 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 |

GRPO: mean=0.2875, std=0.41 → **non-zero advantages for all rollouts**, even when most outcomes are 0.

### 3.6 PRIME Layer (Optional, Additive)

For additional step-level credit beyond just recovery, enable PRIME implicit step rewards:

```
r_implicit_t = β · (old_log_probs_t - ref_log_probs_t)    # β = 0.05
```

Step advantage via leave-one-out baseline within GRPO group. Combined:

```
A_combined = A_outcome + α · A_step    # α = 0.5
```

Both `old_log_probs` and `ref_log_probs` are already computed in the existing pipeline (`ray_trainer.py` lines 2222-2231). This requires only modifying `compute_advantage()` in `core_algos.py`.

---

## 4. Warm-Start Phase

The model should learn clean task structure before facing noise. Add a warm-start gate in `_annotate_task_configs_with_noise()`:

```python
if is_val:
    return
warm_start = getattr(self.config.algorithm, 'noise_warm_start_episodes', 0)
if warm_start > 0 and getattr(self, '_current_episode', 0) < warm_start:
    return  # No noise during warm-start
```

**Training phases**:
1. **Episodes 0-4** (clean): Model learns task structure. Achieves ~35-45% SR.
2. **Episode 5+** (noise on, tier 1): Noise introduced gently. Recovery rewards begin.
3. **Curriculum adapts**: As model learns recovery, SR improves, curriculum increases noise.

---

## 5. Files to Modify

### Environment Side

| File | Change |
|------|--------|
| `OSWorld/desktop_env/desktop_env.py` | Add `_get_active_window()`, `_is_task_window()`, recovery tracking state, modify `step()` to detect recovery, set `_task_app_pattern` from task config in `reset()` |
| `scripts/servers/remote_env_server.py` | Add `recovery_detected`, `recovery_reward` to `noise_burden` dict |

### Training Side

| File | Change |
|------|--------|
| `verl/trainer/ray_trainer.py` | `_build_recovery_reward_tensor()` helper, add recovery rewards to `token_level_scores`, warm-start gate in `_annotate_task_configs_with_noise()` |
| `verl/trainer/core_algos.py` | (Optional) `compute_grpo_dense_advantage()` for PRIME integration |
| `verl/trainer/config.py` | Add `noise_recovery_reward`, `noise_warm_start_episodes`, `use_prime_step_rewards`, `prime_beta`, `prime_alpha` |
| `configs/arpo_8gpu_noise_full.yaml` | Enable recovery reward + warm-start + entropy |

---

## 6. Configuration

```yaml
algorithm:
  # --- Recovery reward ---
  noise_recovery_reward: 0.1       # Base reward per recovery (× recovery_cost)
  
  # --- Warm-start ---
  noise_warm_start_episodes: 5     # Clean training before noise begins
  noise_initial_tier: 1            # Gentler noise after warm-start (was 2)
  
  # --- Exploration ---
  entropy_coef: 0.01               # Prevent policy collapse on hard tasks
  
  # --- PRIME (optional, additive) ---
  use_prime_step_rewards: false     # Enable after validating recovery reward
  prime_beta: 0.05                  # Implicit reward scaling
  prime_alpha: 0.5                  # Step-vs-outcome advantage mixing
```

---

## 7. Expected Impact

| Metric | Current | Expected |
|--------|---------|----------|
| Eval SR | ~0.37 (flat) | >0.45 (upward trend) |
| GRPO group std | Often 0 (CRN all-fail) | >0 (recovery creates variance) |
| Recovery rate | Not tracked | Increasing over training |
| Policy entropy | Decreasing (collapse) | Stable ~3.0 (entropy regularization) |

---

## 8. Verification Plan

1. **Recovery detection accuracy**: Run smoke test with noise. Manually verify that logged recovery events match observed modal dismissals / window focus changes in screenshots.
2. **Reward variance**: With CRN, check GRPO groups have std>0 when some rollouts successfully recover.
3. **Learning curve**: SR should trend upward within 5-10 episodes after noise begins (vs. flat in current training).
4. **Recovery rate metric**: Plot `recovery_events / noise_events` ratio across training steps. Should increase as model learns to dismiss noise.
5. **Ablation**: Compare (a) recovery reward only, (b) recovery + PRIME, (c) recovery + warm-start, (d) all three.

---

## 9. References

- [PRIME: Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456) — Cui et al., 2025. Implicit process rewards for dense credit assignment, compatible with GRPO.
- [iStar: Agentic Reinforcement Learning with Implicit Step Rewards](https://arxiv.org/abs/2509.19199) — ICLR 2026. Combined episode + step advantages for agentic RL.
- [ADMIRE: Adaptive Milestone Reward for GUI Agents](https://arxiv.org/abs/2602.11524) — 2026. Milestone-based dense rewards for GUI agents, +11% SR on AndroidWorld.
- [PRIME Blog: Implementation Details](https://huggingface.co/blog/ganqu/prime) — β=0.05, leave-one-out baseline, separate return computation.
