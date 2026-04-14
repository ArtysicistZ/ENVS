# Noise-Curriculum ARPO Spec

This document specifies the intended robustness-training design for ARPO under realistic desktop noise. It turns the current ad hoc noise assets into an algorithmically complete system:

- domain-specific noise library
- four operational difficulty tiers plus clean baseline
- action-triggered runtime firing
- adaptive per-task probability
- recovery-cost validation
- replay/recovery-aware ARPO extensions

The design goal is not to make tasks randomly harder. The goal is to reduce overfitting to clean OSWorld layouts and train policies that are invariant to realistic interruption patterns.

## Objectives

We want a training/evaluation pipeline that:

1. Preserves the original clean OSWorld benchmark as the default regime.
2. Injects noise only when explicitly enabled.
3. Makes noise realistic, domain-aware, and diverse enough to avoid template memorization.
4. Controls difficulty by recovery burden, not by arbitrary popup count.
5. Adapts noise pressure based on task mastery.
6. Supports held-out noise families for OOD robustness measurement.

## Training Regimes

We evaluate the following sequence:

1. `ARPO-clean`
2. `ARPO-fixed-noise`
3. `ARPO-noise-curriculum`
4. `ARPO-noise-curriculum + entropy-regularization`
5. `ARPO-noise-curriculum + entropy + replay/recovery-aware training`

We always compare against:

1. `Base clean`
2. `Base noisy`
3. `MCTS-SFT clean`
4. `MCTS-SFT noisy`
5. `ARPO clean`
6. `ARPO noisy`

Metrics:

- clean success
- noisy success
- held-out noise OOD success
- held-out task generalization
- degradation gap from clean to noisy
- median extra actions under noise
- recovery success rate after interruption
- loop/repeat rate under interruption

## Core Concepts

### 1. Static Task Difficulty

Static task difficulty is a prior over how fragile the underlying task already is before noise is introduced. It should not depend on one particular noisy rollout.

We compute a static task difficulty score:

`D_static(task) = w_sr * D_success_rate + w_h * D_horizon + w_dom * D_domain + w_ui * D_ui_fragility`

Where:

- `D_success_rate`: inverse of clean success rate on the task
- `D_horizon`: normalized expected action horizon from successful trajectories
- `D_domain`: domain prior for inherent interface brittleness
- `D_ui_fragility`: heuristic for how easy it is to derail the UI state

Suggested defaults:

- `w_sr = 0.45`
- `w_h = 0.25`
- `w_dom = 0.15`
- `w_ui = 0.15`

Practical sources:

- clean base / MCTS-SFT / ARPO eval results
- MCTS successful trajectories
- task metadata: domain, related_apps, multi-app requirement

### 2. Realized Noise Difficulty

Realized difficulty is how much burden a sampled noisy rollout actually imposes.

We define:

`D_realized(rollout) = alpha * C_recovery + beta * C_interruptions + gamma * C_occlusion + delta * C_focus_loss`

Where:

- `C_recovery`: cumulative planned recovery cost from sampled noise elements
- `C_interruptions`: number of noise firings during rollout
- `C_occlusion`: weighted count of modal/overlay/focus-steal events
- `C_focus_loss`: count of events that require re-focusing or window recovery

Suggested defaults:

- `alpha = 0.50`
- `beta = 0.20`
- `gamma = 0.20`
- `delta = 0.10`

### 3. Mediated Difficulty

This answers the question: we have an initial task prior, but the final difficulty depends on extra actions required. How do we mediate them?

We use:

`D_target = lambda * D_static + (1 - lambda) * D_realized_budget`

Where:

- `D_static` is task prior
- `D_realized_budget` is the intended difficulty of sampled noise, before rollout starts
- `lambda` defaults to `0.6`

Then after rollout we log:

`D_observed = lambda_obs * D_target + (1 - lambda_obs) * D_realized(rollout)`

Where:

- `lambda_obs` defaults to `0.4`

Interpretation:

- static difficulty chooses how aggressive we are allowed to be
- realized sampled difficulty tells us what we actually injected
- observed realized burden tells us whether the sample was too weak or too strong

This avoids the failure mode of treating all noisy tasks as equally difficult just because they have the same number of perturbations.

## Noise Tiers

Tiering is defined by cumulative recovery cost and interruption semantics.

### Tier 0: Clean

- no noise
- used for baseline and warm start

### Tier 1: Ambient

- mostly cost-0 or cost-1 elements
- passive notifications
- background windows
- harmless transient banners

Target effect:

- visual clutter without serious derailment

### Tier 2: Mild Interruption

- 1-2 recovery-cost budget
- cookie banners
- one-off modal dialogs
- focus-steal by non-target app
- partial occlusion

Target effect:

- agent must dismiss/refocus, but task state remains stable

### Tier 3: Moderate Interruption

- 2-3 recovery-cost budget
- multiple interruptions
- target-window shove
- browser/site error flows
- app-specific prompts

Target effect:

- requires recovery decisions and state tracking

### Tier 4: Compositional/OOD

- 3-4+ recovery-cost budget
- chained or compositional events
- held-out rendering variants
- rare recovery-path variants

Target effect:

- stress-test robustness and generalization

## Domain-Specific Noise Families

Noise must be domain-aware. The same perturbation library should not be applied uniformly.

### Chrome

- ads
- cookie banners
- newsletter popups
- login walls
- consent dialogs
- VPN / captive portal / SSL warning
- 404 / network error pages
- download prompts

### VS Code

- trust workspace dialog
- extension recommendation/update prompts
- file modified externally prompt
- unsaved-change prompt
- sidebar/panel layout changes

### LibreOffice Writer / Impress / Calc

- recovery dialogs
- template pickers
- sidebar/layout occlusion
- export/save warnings
- autosave conflict prompts

### Thunderbird

- sync/account prompts
- notification bursts
- folder/tree focus loss
- connection retry dialogs

### GIMP / VLC

- plugin/update dialogs
- tool panel shifts
- file warning dialogs
- open-media prompts

### OS / Multi-App

- file manager overlaps
- settings dialogs
- background notifications
- network/system warnings

## Diversity Axes

To avoid overfitting, every category should vary across multiple axes:

1. visual rendering
2. semantic message type
3. trigger timing
4. spatial placement
5. composition pattern
6. recovery path

The runtime library should prefer sampling diverse combinations rather than repeated identical zenity dialogs.

## Recovery-Cost Validation

Every noise element must declare:

- `recovery_cost`
- `recovery_actions`
- `touches_target_app`
- `category`
- `once`

Hard rule:

- any single element requiring more than 3 standard agent actions to recover is invalid

The curriculum budget is enforced on cumulative recovery cost, not element count.

## Curriculum State

Per task we maintain:

- `p_noise(task)`: probability that a pending noise element fires after an action
- `tier(task)`: current difficulty tier
- `sr_ema(task)`: rolling success estimate
- `recovery_ema(task)`: rolling recovery burden estimate

Update rule:

- if `sr_ema(task)` rises above threshold, increase `tier` or `p_noise`
- if `sr_ema(task)` falls below threshold, reduce `p_noise` first, then reduce `tier`

Recommended initial thresholds:

- promote if `sr_ema >= 0.70` for 3 consecutive windows
- hold if `0.35 <= sr_ema < 0.70`
- demote if `sr_ema < 0.35`

Recommended probability caps:

- Tier 0: `p=0.00`
- Tier 1: `p=0.05`
- Tier 2: `p=0.10`
- Tier 3: `p=0.15`
- Tier 4: `p=0.20`

## Replay / Recovery-Aware ARPO

The final ARPO variant should distinguish:

- clean successful trajectories
- noisy successful trajectories
- recovery-success trajectories

Desired replay policy:

- prefer noisy successful trajectories when training in noisy mode
- preserve a clean/noisy mixture to avoid catastrophic regression on clean tasks
- log recovery-type tags so replay can cover different interruption families

## Entropy-Regularized ARPO

Entropy regularization is added only after the fixed-noise and curriculum baselines are working.

Reason:

- noise increases observation ambiguity
- without entropy regularization, policies collapse too early to brittle routines

Suggested effect:

- stronger exploration under noisy observation
- better recovery path discovery

## OOD Noise Split

We use a held-out subset of the runtime template library for evaluation only.

This produces:

- in-distribution noisy eval
- held-out noisy eval

The held-out split is what makes the result meaningful from a robustness perspective instead of memorization.

## Implementation Stages

### Stage 1

- config fields for noise
- remote env protocol carries `noise_probability`
- env-side `NoiseScheduler`
- use provided `noise_meta` if task already has it

### Stage 2

- runtime sampling from the universal template library
- task reset can synthesize per-rollout noise from the clean task

### Stage 3

- `NoiseCurriculum` with per-task EMA and tier updates
- logging of firing rate, tier, and realized burden

### Stage 4

- entropy-regularized ARPO
- replay/recovery-aware trajectory tagging and sampling

## Design Answers To The Two Core Questions

### Q1. How do we measure task difficulty across domains?

We do not rely on domain alone. We use a hybrid prior:

- clean success rate
- clean horizon from successful trajectories
- domain fragility prior
- UI-state brittleness prior

This yields a domain-aware but data-grounded static difficulty estimate.

### Q2. We have initial difficulty, but final difficulty depends on extra actions. How do we mediate this?

We separate:

- `static task difficulty`
- `sampled noise difficulty`
- `observed realized burden`

And combine them with weighted mediation.

This prevents:

- over-noising already fragile tasks
- under-noising easy tasks
- treating “same category, different realization” as identical

That mediation is the core of a principled curriculum, and it is what makes the system defensible rather than heuristic-only.
