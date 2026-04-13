# Noise-Augmented GUI Agent Training: A Research Framing

*Deep framing document for the noise-generation pipeline. Synthesizes
domain-randomization, unsupervised environment design, procedural-generation,
data-centric AI, reward-from-demonstrations, and GUI-robustness literatures
into a defensible research contribution.*

---

## 1. One-sentence claim

We train GUI agents to become robust to the messy, ambient, concurrent-use
conditions of real desktops by co-evolving the agent with an **adversarial
noise curriculum** over a **diverse, procedurally-randomized noise
distribution**, supervised by **dense reward derived from MCTS-waypoint
demonstrations** — producing agents that generalize to noise distributions
held out from training.

This differs from every prior line of work on GUI agents in one clear axis:
we treat the training environment itself as a first-class, adaptive design
surface, not a fixed dataset.

---

## 2. What problem are we actually solving?

### 2.1 The sim-to-real gap for GUI agents is structurally large

OSWorld, WebArena, and every widely-used GUI benchmark evaluates agents on
cleaned, minimal-clutter desktop environments: one app is open, no
notifications, no cookie banners, no concurrent users, no background
processes. The production target — an office worker's laptop, a lab machine,
a kiosk — is none of these things.

Agents trained purely on clean OSWorld show catastrophic degradation on real
desktops. This has been documented most recently by GUI-Robust [Yang et al.
2025; arxiv.org/abs/2506.14477], which introduced **seven categories of
real-world anomalies** (layout shifts, transient pop-ups, appearance
variations, etc.) and found that state-of-the-art GUI agents suffer
substantial performance degradation under them. OSWorld-MCP [2025] later
added 25 "distractor tools" to measure agent focus under interruption. The
field has converged on the diagnosis: **GUI agents are overfit to
sanitized training distributions and do not transfer to realistic
deployment conditions.**

### 2.2 Procedural generation is the well-established cure in sibling fields

In pixel-based RL, Cobbe et al. [Procgen, 2020; arxiv.org/abs/1912.01588]
showed empirically that **agents require access to as many as 10,000
procedurally-generated levels to close the generalization gap**. Fewer
levels and the agent memorizes, regardless of model size or training time.
This is the single most important empirical result in the generalization
literature for RL: **narrow training distributions produce memorization, not
skill.**

In robotics sim-to-real, Tobin et al. [Domain Randomization, 2017;
arxiv.org/abs/1703.06907] established the analogous principle: train a
policy on a randomized distribution of simulator configurations so broad
that the real world appears as just another sample from the training
distribution. This has produced production-grade results in OpenAI's
Dexterity work and many downstream systems.

The GUI-agent equivalent of Procgen levels and robotics simulator
randomization does not yet exist. **Our contribution is to build it.**

### 2.3 Fixed noise distributions fail; adaptive ones succeed

Domain randomization in its original form has a known weakness:
"domain randomization cannot generate structure or adapt the difficulty of
the environment to the agent's learning progress" [Dennis et al. PAIRED,
NeurIPS 2020]. On a fixed randomized distribution, early training is
dominated by too-hard samples (wasted gradient) and late training by
too-easy samples (saturated signal).

The cure is **unsupervised environment design (UED)**: a family of methods
that train a *second* player whose job is to propose environments at the
frontier of the protagonist's current capability. PAIRED uses a regret
signal to balance challenge and solvability. POET [Wang et al. 2019] uses
population-based evolution. ACCEL [Parker-Holder et al. 2022] edits prior
high-regret environments to generate new ones just beyond the agent's
current skill. **These methods empirically dominate fixed domain
randomization on zero-shot transfer to unseen environments.**

Our pipeline adopts the UED structure with a key simplification: our
"antagonist" is a rule-based curriculum controller, not a learned adversary.
The noise elements are human-designed primitives; the curriculum only
decides which elements fire, how often, and in which compositions. This is
enough to produce capability-adaptive difficulty without the instability of
training a second network adversarially.

### 2.4 Sparse reward breaks the curriculum

One mechanism we must address up front: **a capability-adaptive curriculum
needs a clean skill signal, and sparse 0/1 task-success reward is too noisy
for that.** With 8 samples per task per step, the standard error on EMA
success rate is ~0.15. The curriculum tries to read the agent's skill
through this noise and advances/regresses at random. Adaptive curricula
with sparse reward empirically degenerate toward random difficulty
assignment.

Dense reward — ideally derived from demonstrations, not hand-shaped —
resolves this. Recent work [Beyond Imitation, arxiv.org/abs/2510.02493]
established that SFT is implicitly dense per-token reward derived from
expert demonstrations via inverse Q-learning; we lift the same idea to
the trajectory level for GUI agents, deriving dense screenshot-level
rewards from our existing MCTS-successful trajectories.

### 2.5 The combined hypothesis

Training GUI agents that transfer to real-world noise requires three
design choices to hold simultaneously:

1. **Diverse, procedurally-randomized noise** that covers the mode space of
   real desktop interruption (not just tokens drawn from a narrow
   rendering).
2. **Capability-adaptive curriculum** that tracks per-task agent skill and
   adjusts noise difficulty in response — avoiding both gradient-starvation
   (too hard) and signal-saturation (too easy).
3. **Dense reward** derived from demonstrations, supplying per-step
   learning signal that remains legible under noise.

Missing any one of these breaks the other two. Our contribution is the
integrated system.

---

## 3. Related work, by contribution axis

### 3.1 Domain randomization (visual + dynamics)
* Tobin et al., *Domain Randomization for Transferring Deep Neural Networks
  from Simulation to the Real World*, 2017 [arxiv.org/abs/1703.06907]
* OpenAI Dexterity (solving the Rubik's cube on a real robot via
  randomized simulation)
* DROPO, *Sim-to-Real Transfer with Offline Domain Randomization*, 2023
* Surveys: Sim-to-Real in Deep RL [arxiv.org/abs/2009.13303]; Lilian Weng's
  *Domain Randomization for Sim2Real Transfer*, 2019

**What we inherit from DR:** the core premise that a wide, randomized
training distribution produces transferable policies. Our noise library
*is* the randomization. Visual diversity (Axis 1 in our library),
behavioral diversity (Axis 2), and compositional randomization (Axis 4) are
direct transplants of DR principles to the GUI domain.

**What DR alone is missing:** the randomization does not adapt to the
agent's skill (the problem that motivated UED).

### 3.2 Unsupervised environment design
* Dennis et al., *Emergent Complexity and Zero-shot Transfer via
  Unsupervised Environment Design* (PAIRED), NeurIPS 2020
  [arxiv.org/abs/2012.02096]
* Wang et al., *POET: Paired Open-Ended Trailblazer*, 2019
* Jiang et al., *Prioritized Level Replay* (PLR), 2021
* Parker-Holder et al., *ACCEL: Adversarially Compounding Complexity by
  Editing Levels*, 2022
* Azad et al., *CLUTR: Curriculum Learning via Unsupervised Task
  Representation Learning*, ICML 2023
* More recent: *Adversarial Environment Design via Regret-Guided
  Diffusion Models*, 2024 [arxiv.org/abs/2410.19715]

**What we inherit from UED:** the three-axis structure of our curriculum
(tier, probability, pool composition) is directly analogous to ACCEL's
level-editing operators; our tier-advancement rule (advance on 3 consecutive
mastery signals) is analogous to PLR's staleness-weighted prioritization.

**What we simplify:** no learned adversary. Our noise library is finite and
human-designed; the curriculum only orders and gates its elements. This
avoids PAIRED's antagonist-training overhead while retaining the adaptive
benefit.

### 3.3 Procedural generation & generalization
* Cobbe et al., *Leveraging Procedural Generation to Benchmark Reinforcement
  Learning* (Procgen), 2019–2020 [arxiv.org/abs/1912.01588]
* *ProcGen* benchmark suite, OpenAI

**What we inherit:** the generalization-first methodology. We adopt
Procgen's key experimental design: **hold out a subset of procedurally-
generated instances and measure agent performance on them.** We apply this
to noise elements — 80/20 train/test split of the noise library — so we can
measure generalization across noise kinds, not just transfer within a
single noise distribution.

**Critical quantitative finding Procgen established:** agents trained on
few levels overfit badly; 10,000 levels can be needed to close the gap.
This directly motivates our emphasis on **visual, behavioral, compositional,
and recovery-path diversity** within our noise library.

### 3.4 Data-centric AI
* Ng, *A Chat with Andrew Ng about AI Shifting to Data-Centric*, 2021
  [spectrum.ieee.org/andrew-ng-data-centric-ai]
* Landing AI data-centric AI program
* Zha et al., *Data-Centric Artificial Intelligence: A Survey*, 2023
  [arxiv.org/html/2212.11854v4]
* Chinchilla scaling laws [Hoffmann et al., 2022]: data quantity/quality
  trades off with parameter count at a well-defined rate

**What we inherit:** the framing that **what the agent sees in training
dominates what the agent becomes**. Our experiment holds the training
method (GRPO/ARPO) and model (UI-TARS-1.5-7B) fixed while varying only the
training distribution (clean / noisy / noisy+diverse / noisy+adaptive).
Large deltas on this axis support the data-centric claim.

**What we contribute:** a concrete data-centric intervention for GUI
agents — not just "more data" or "cleaner data" but *procedurally diverse
interruption data*, which is the specific kind of data missing from current
GUI-agent training pipelines.

### 3.5 Reward shaping from demonstrations
* Ng, Harada & Russell, *Policy Invariance Under Reward Transformations:
  Theory and Application to Reward Shaping*, ICML 1999 (potential-based
  shaping)
* *Conservative Reward Shaping from Demonstrations* (CRSfD)
* *Beyond Imitation: Recovering Dense Rewards from Demonstrations*, 2024
  [arxiv.org/html/2510.02493]
* *Attention-Based Reward Shaping* (ARES), 2024
  [arxiv.org/html/2505.10802v1]
* Vecerík et al., *Leveraging Demonstrations for Deep RL on Robotics
  Problems with Sparse Rewards* (DDPGfD), 2017

**What we inherit:** the observation that expert demonstrations can be
transformed into per-step dense reward that dominates hand-shaped reward
on sample efficiency. Our MCTS-successful trajectories (already in
`checkpoints/mcts_trajectories_v2/`) are exactly such demonstrations; each
contains the intermediate screen states along a successful trajectory.

**How we apply it:** pre-embed all MCTS-trajectory screenshots with CLIP
(or a small vision encoder) to form per-task "waypoint embedding indices."
At rollout time, embed the agent's current screenshot and compute the max
cosine similarity to any MCTS waypoint the agent *has not already passed*.
Reward = similarity gain from the previous step.

Potential-based (Ng-Harada-Russell-safe) because the shaping depends only on
state, not on trajectory history beyond the waypoint index. We will also
clip shaped reward to ≤ 0.3 × final-task reward, ensuring that the final
0/1 outcome remains the dominant learning signal and that shaped reward
cannot hijack the policy away from task completion.

### 3.6 GUI-agent robustness literature (contemporaneous)
* Xie et al., *OSWorld*, NeurIPS 2024
* Zhou et al., *WebArena*, 2023
* Yang et al., *GUI-Robust: A Comprehensive Dataset for Testing GUI Agent
  Robustness in Real-World Anomalies*, NeurIPS 2025
  [arxiv.org/abs/2506.14477]
* OSWorld-MCP, 2025 (distractor-tool robustness extension)
* Zhao et al., *On the Robustness of GUI Grounding Models Against Image
  Attacks*, CVPR-W 2025 [arxiv.org/abs/2504.04716]
* *On the Robustness of Multimodal Language Model towards Distractions*,
  2025 [arxiv.org/abs/2502.09818]
* WorldGUI, *An Interactive Benchmark for Desktop GUI Automation from Any
  Starting Point*, 2025 [arxiv.org/abs/2502.08047]

**Critical positioning:** GUI-Robust and OSWorld-MCP both *measure*
robustness. They are benchmarks. Our contribution is orthogonal: we
propose a **training pipeline** that makes agents more robust under the
same measurement. A successful noise-augmented agent should show improved
scores on GUI-Robust and OSWorld-MCP — that's a legitimate paper-result
pointer, not competing territory.

---

## 4. Theoretical framing: three interacting mechanisms

### 4.1 Diversity → generalization (the Procgen principle)

Let **D_train** be the training noise distribution and **D_test** the real-
world noise distribution. Agent performance on D_test decomposes as:

> performance(agent, D_test) ≈
>   performance(agent, D_train) − overfit_penalty(D_train → D_test)

where the overfit penalty shrinks with support(D_train) and with coverage
of D_test by D_train. Procgen established that this penalty is large and
slow-decaying without extensive procedural diversity; we transplant the
implication: **our noise library must aggressively randomize across every
axis that real desktops vary along, or the agent will memorize
rendering-specific artifacts of our library.**

### 4.2 Adaptation → sample efficiency (the UED principle)

For capability-adaptive curricula, the relevant theoretical object is the
*regret* at each training step — the gap between an idealized agent and
the current policy. PAIRED's contribution was showing that *optimizing
environment selection for regret produces a natural curriculum* that
dominates uniform sampling on both training speed and zero-shot transfer.

We inherit this in simplified form. Our `NoiseCurriculum` tracks per-task
EMA success rate; a rise past threshold τ_up unlocks the next noise tier
(adding harder elements), a fall below τ_down regresses. This is a
proxy for regret-based environment selection, tuned to our specific
library structure rather than learning an adversary.

### 4.3 Dense reward → adaptation fidelity (the reward-shaping principle)

The adaptive curriculum from 4.2 needs a legible skill signal. With sparse
binary reward, the signal-to-noise ratio is governed by rollout count per
task (N = 8 for us) and binary variance. The curriculum tries to read
signal across a 15% standard-error band and transitions randomly inside it.

Dense reward (from 3.5) reduces per-step variance by ~4× in practice
(empirical for similar waypoint-shaping schemes). This in turn makes the
curriculum's tier advancement rule actually track agent skill rather than
sampling noise. **Dense reward is not independent of the curriculum — it
is a prerequisite for the curriculum to function as designed.**

### 4.4 Combined: the three mechanisms are complementary, not substitutable

|  | Diverse noise | Adaptive curriculum | Dense reward |
|---|---|---|---|
| Diverse noise alone | — | Wasted on unsolvable / trivial | Wasted on noisy signal |
| Curriculum alone | Agent memorizes | — | Degenerates to random |
| Dense reward alone | No robustness gain | No sample-efficiency gain | — |
| **All three** | **Generalization + efficient learning + stable training** | | |

The paper's strongest claim follows from this decomposition: **each
mechanism is necessary, none is sufficient, and published prior work
establishes each one individually but has not combined all three for
GUI agents.**

---

## 5. System design

### 5.1 Four-axis noise diversification (Procgen-inspired)

Every noise primitive must randomize across all four axes. Narrow
diversity along any one axis invites overfitting along it.

**Axis 1 — Visual rendering.** Same behavior class, different visual
artifact. A "dismissible modal" can be rendered as: zenity-info,
zenity-warning, xmessage, yad, a custom Python/tkinter popup with
randomized background color + border style + button layout,
`notify-send --urgency critical`, or a browser-rendered modal launched in
a non-target Chrome window. Each visual variant maps to the same
behavioral class.

**Axis 2 — Behavioral class.** The taxonomy of 16 noise-behavior classes
we defined earlier: passive_notification, modal_dialog, focus_steal,
occlude_from_above, partial_overlap, target_shove, target_shrink,
target_wobble, persistent_overlay, app_internal_prompt, os_device_event,
state_drift, visual_flicker, resource_activity, accidental_cover,
cursor_behavior.

**Axis 3 — Recovery path.** The agent's corrective action must vary per
element. Same behavior class with different required recovery: modal
requiring click-OK vs. modal requiring click-X vs. modal requiring
Escape-key vs. modal that auto-dismisses (correct action: *wait*). The
final category — *no agent action correct* — is critical; without it,
agents overfit to "always try to dismiss everything."

**Axis 4 — Compositional.** Multi-element noise firings within a single
step: modal-then-overlay, focus-steal-then-shove, three-notification-burst.
Randomize order, delays, and which elements co-fire. Prevents the agent
from memorizing the firing schedule.

### 5.2 Held-out OOD evaluation (Procgen-inspired)

The noise library is split **80/20** into train and held-out sets along a
category-preserving stratification: for every behavior class, 80% of its
variants are in train, 20% in test. The agent never sees held-out variants
during training.

Primary generalization measurement: agent performance on held-out noise
variants. A well-trained robust agent should perform within 20% of its
in-distribution performance on held-out noise. An agent that memorizes the
training noise will drop sharply.

This single metric — **held-out noise generalization** — is the paper's
centerpiece.

### 5.3 Three-axis adaptive curriculum

Per task t, the curriculum tracks:
* `tier[t]`: integer in 0..5; gates which elements are in the active pool
* `p[t]`: probability in [0.02, 0.40]; fires each active element
  independently per agent step
* `pool_size[t]`: integer in 1..8; randomly samples that many elements from
  the active tier on each rollout reset

Update rule based on EMA of rolling-8-rollout success on task t, using the
**dense reward signal** (not raw 0/1 task success) for lower variance:

```
if EMA_dense_reward[t] rises past τ_up for k_up consecutive evals:
  advance tier if not at max
  else: raise p[t] toward p_max
if EMA_dense_reward[t] falls below τ_down:
  regress tier if not at min
  else: lower p[t] toward p_min
```

Parameters (to be tuned in initial runs): τ_up = 0.5, τ_down = 0.15,
k_up = 3, p_min = 0.02, p_max = 0.30.

### 5.4 MCTS-waypoint dense reward

Pre-computation (one-time per task):
* Load successful MCTS trajectories from `mcts_trajectories_v2/trees/`
* Extract sequence of (action, screenshot) pairs per trajectory
* CLIP-embed all screenshots with a small ViT (e.g., `ViT-B/32` via `open_clip`)
* Store per-task as `waypoint_embeddings[t] ∈ R^{k × d}` where k is the
  number of waypoints and d is the CLIP embedding dim

Per-step reward computation:
* Embed agent's current screenshot with the same CLIP encoder
* Compute cosine similarity to every waypoint for this task
* Let `max_sim` be the max; let `max_sim_so_far` be the running max seen
  this rollout
* Reward = `max(0, max_sim − max_sim_so_far)` (monotonic progress)
* Update `max_sim_so_far ← max(max_sim_so_far, max_sim)`

Properties:
* **Non-negative**: agents can only gain from waypoint matching, never
  lose shaped reward for noise-recovery actions
* **Monotonic**: credit is only given when progress is *new* (prevents
  oscillating between two waypoints)
* **Bounded**: total shaped reward across a rollout is ≤ 1.0 by
  construction
* **Task-survival-compatible**: bounded by a hyperparameter relative to
  the final-task reward, guaranteed not to dominate

A small additional **recovery bonus** of +0.05 per step where the
scheduler fired noise *and* the agent's next action restored focus /
dismissed the interrupter. Detected by comparing the focused window ID
before and after the agent's action.

### 5.5 Opt-in feature gating

All of the above activates only when `algorithm.enable_noise=true` in the
config. Default is false. Clean training and clean eval remain bit-for-bit
identical to the pre-feature pipeline. This preserves backward
compatibility for all existing ARPO, GRPO, SFT-eval configurations.

---

## 6. Experimental plan

### 6.1 Primary experiment: the 2×2 interaction

Two binary factors, four training runs. All with fixed model (UI-TARS-1.5-
7B), fixed method (GRPO, no replay), fixed 86 trainable OSWorld tasks, same
compute budget.

| Run | Noise | Reward |
|---|---|---|
| A | off (clean) | sparse (status quo) |
| B | fixed `p=0.1`, full library | sparse |
| C | off (clean) | dense (MCTS-waypoint) |
| D | **adaptive curriculum, full library** | **dense (MCTS-waypoint)** |

Eval every trained checkpoint on:
* **E1**: clean 86 trainable (in-distribution, clean)
* **E2**: noisy 86 trainable, fixed `p=0.1` on training noise distribution
* **E3**: noisy 86 trainable, fixed `p=0.1` on **held-out** noise
  distribution (OOD generalization)
* **E4**: clean 214 held-out tasks (task-level generalization)
* **E5**: noisy 214 held-out tasks, held-out noise (joint generalization)

E3 and E5 are the paper's headline metrics.

### 6.2 Hypothesized results pattern

| | E1 (clean train) | E2 (train noise) | E3 (OOD noise) | E4 (held-out task) | E5 (both OOD) |
|---|---|---|---|---|---|
| A clean+sparse | baseline | ↓↓ | ↓↓↓ | ↓ | ↓↓↓ |
| B noise+sparse | ≈ or ↓ | ↑↑ | ↑? (risk: memorized) | ≈ | ↑? |
| C clean+dense | ≈ or ↑ (faster convergence) | ≈ | ≈ | ↑ | ≈ |
| **D noise+dense+adaptive** | **≈** | **↑↑** | **↑↑↑** | **↑↑** | **↑↑↑** |

**Key test:** does D − B on E3 exceed D − B on E1?  (i.e., does the
adaptive curriculum + dense reward add generalization-to-novel-noise that
fixed-noise+sparse cannot?)

**Falsification conditions:**
* If D degrades E1 by > 3 pts, noise is too aggressive — claim weakens.
* If B ≈ D on E3, the curriculum and dense reward add no value — claim
  fails.
* If E3 ≪ E2 for run D (held-out ≪ train-distribution), the noise
  diversity is insufficient — we revisit Axis 1-4 above.

### 6.3 Ablations (budget-permitting, after the 2×2)

* D without adaptive curriculum (fixed `p=0.1`): isolates curriculum's
  contribution
* D without dense reward (sparse 0/1): isolates reward shaping's
  contribution
* D with only Tier-1 human-task noise: isolates the "concurrent human"
  story
* D with only Tier-2B application/browser interruption: isolates the
  "application-internal" story
* D on a differently-sized task pool (e.g. 43 tasks, or 172 tasks): data
  quantity vs. quality
* D with noise held-out reversed (train on held-out-set, test on
  train-set): symmetry check

### 6.4 Downstream: evaluate on GUI-Robust

If the pipeline works, run the GUI-Robust benchmark on the final D-trained
agent. Compare to the same agent trained without noise (run A). This is
independent, published evaluation territory — strong external validation.

---

## 7. Expected contributions (what we claim, what we don't)

### 7.1 Strong claims (defensible if experiments succeed)

1. **A training pipeline that produces GUI agents robust to realistic
   desktop noise.** The first such pipeline combining procedural noise
   diversity, capability-adaptive curriculum, and demonstration-derived
   dense reward.

2. **Empirical evidence that training-data distribution dominates
   training-method choice for GUI-agent robustness.** Measured via the
   magnitude of D − A on E3 compared to the magnitude of ARPO − GRPO on
   E1 (already have this comparison: ~7 pts).

3. **The first held-out OOD-noise benchmark for GUI agents**, applied
   rigorously via Procgen-style train/test split of noise variants.

4. **Demonstration that MCTS trajectories can be converted to per-step
   dense reward** via CLIP-waypoint matching, producing a ~4× variance
   reduction in the skill signal that drives a capability-adaptive
   curriculum.

### 7.2 Weaker / hypothesized claims

1. **That noise-augmented training does not degrade clean-eval performance
   by more than 2 pts.** We hypothesize this but cannot guarantee it
   without running the experiment.

2. **That the combined system generalizes to held-out tasks better than
   any component alone** (the hopeful result). May hold; may not. Either
   way we report honestly.

### 7.3 What we explicitly do not claim

* That noise-augmented training improves *clean* OSWorld benchmark
  performance. (Prior analysis suggests this is unlikely.)
* That our 15-axis noise library covers all real-world noise. (It is a
  procedural proxy; the actual distribution of real-desktop noise has long
  tails we do not model.)
* That the MCTS-waypoint dense reward is the only way, or the optimal way,
  to supply dense reward. (Alternative methods in 3.5 may work equally
  well; we choose this because the MCTS data already exists.)

---

## 8. Connection to the "data > model" framing

A data-centric paper-level framing makes the contribution sharper than a
method-level one:

> *Given fixed model (UI-TARS-1.5-7B) and fixed RL algorithm (GRPO), we
> show that switching the training-data distribution from clean to
> diverse-noise-augmented with adaptive curriculum produces a larger gain
> on held-out noise eval than switching the algorithm (SFT → GRPO → ARPO)
> produces on any eval.*

This is testable. The method-effect magnitude is already known from our
earlier results (SFT vs ARPO on 86 trainable: ~7 pts greedy). The
data-effect magnitude will be measured by D − A on E3. If data-effect
exceeds method-effect on held-out noise by a factor of ≥ 1.5×, the data-
centric claim is supported with reasonable quantitative margin.

This reframes the paper's contribution away from leaderboard-climbing and
toward a methodological argument:

> **For GUI agents, the training data distribution — specifically its
> coverage of realistic noise and interruption — dominates architectural
> and algorithmic choices in determining deployment-worthy performance.**

This is the claim the broader ML community will evaluate us on, and it is
well-scoped, measurable, and supported by the parallel body of work in
data-centric AI [Ng / Landing AI; Zha et al. 2023].

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Noise curriculum degenerates to random tier assignments | Dense reward for variance reduction; conservative thresholds τ_up/τ_down |
| MCTS-waypoint reward enables reward hacking (agent mimics waypoints without completing task) | Shaped reward cap at 0.3 × final reward; monotonic (non-negative) shaping; final task reward remains dominant |
| Agent overfits to our specific noise-library rendering | Four-axis diversity (visual, behavioral, recovery-path, compositional); held-out OOD noise measurement |
| Tier advancement traps agent at a plateau (curriculum stalls) | Periodic forced tier perturbation every N training steps; regression rule restores old tier on failure |
| Training under adaptive noise is slower than clean (extra computation, extra rollout variance) | Feature-flag off by default; measure compute cost explicitly and report in paper |
| Infrastructure cost: 4+ training runs × 15 hours × 8 GPUs | Reduce to 3 runs for initial ablation; prioritize A vs D comparison |
| VLM-critic reward alternative outperforms MCTS-waypoint | Implement MCTS-waypoint first; add VLM-critic as a secondary variant in ablations |

---

## 10. Open questions (worth pursuing in follow-up work)

* Can the adversary be made *learned* (an actual neural noise-selector)
  rather than rule-based? PAIRED suggests yes; cost is training overhead.
* Does noise generalize *across tasks* — does a model trained on noisy
  Chrome tasks handle noisy GIMP tasks?
* What is the relationship between **noise diversity** (number of
  variants) and **generalization** (OOD performance)? Procgen suggests a
  roughly logarithmic curve requiring ~10k levels for full generalization.
  How many distinct noise elements do we need?
* Can we measure agents' *per-category noise immunity* and use it to
  target-generate noise? (e.g., if the agent handles modals well but
  struggles with window overlaps, increase overlap firing rate.)
* Does dense reward from MCTS-waypoints still work when the agent's
  trajectory diverges substantially from MCTS (e.g., the agent discovers
  a novel but valid path)?

---

## 11. Concrete next-step roadmap

**Phase A (1-2 days) — Diversification.**
1. Implement the four-axis diversification in `templates.py`: add 15+
   visual-rendering variants across modals, notifications, persistent
   overlays; add varied recovery paths (5+ kinds); add 5+ compositional
   templates.
2. Split the library 80/20 into train and held-out sets. Tag each template
   `in_train=True/False`.

**Phase B (1-2 days) — Dense reward.**
3. Pre-compute CLIP embeddings for all MCTS-trajectory screenshots from
   `checkpoints/mcts_trajectories_v2/combined_all/trees/`. Store as
   per-task pickled NumPy arrays.
4. Implement `WaypointDenseReward` class: per-step embed → nearest-neighbor
   in per-task waypoint index → return monotonic similarity gain, clipped.
5. Wire into `ray_trainer.py` in parallel with the existing sparse reward
   (sum them, not replace).

**Phase C (2-3 days) — Adaptive curriculum.**
6. Extend `NoiseScheduler` with tier-gated element pools.
7. Implement 3-axis `NoiseCurriculum` (tier, p, pool_size) with update
   rule using `EMA_dense_reward`.
8. Wandb logging for per-task tier/p/pool-size trajectories.

**Phase D (image + launch, 1 day) — Infrastructure.**
9. Rebuild osworld:latest with `libnotify-bin`, `gnome-calculator`,
   `xdotool`, plus new tools required by Phase A's visual diversity (`yad`
   optional, `python3-tk` for custom popups, `xmessage`).
10. Redistribute image to all 3 cluster hosts.

**Phase E (core experiment, 3-5 days) — 2×2 grid.**
11. Run A (clean+sparse) — baseline reproduction of existing GRPO/ARPO.
12. Run B (noise+sparse) — fixed noise distribution.
13. Run C (clean+dense) — isolates dense reward.
14. Run D (noise+dense+adaptive) — the full pipeline.
15. Eval all four on E1-E5.
16. Analyze; write up.

Total: ~2 weeks of engineering + training time for the core experiment.

---

## References

1. Tobin et al. (2017). *Domain Randomization for Transferring Deep Neural
   Networks from Simulation to the Real World.* arxiv.org/abs/1703.06907
2. Dennis et al. (2020). *Emergent Complexity and Zero-shot Transfer via
   Unsupervised Environment Design.* arxiv.org/abs/2012.02096 (PAIRED)
3. Wang et al. (2019). *Paired Open-Ended Trailblazer (POET).*
   arxiv.org/abs/1901.01753
4. Parker-Holder et al. (2022). *Evolving Curricula with Regret-Based
   Environment Design.* accelagent.github.io (ACCEL)
5. Jiang et al. (2021). *Prioritized Level Replay.* arxiv.org/abs/2010.03934
6. Azad et al. (2023). *CLUTR: Curriculum Learning via Unsupervised Task
   Representation Learning.* ICML.
7. Cobbe et al. (2020). *Leveraging Procedural Generation to Benchmark
   Reinforcement Learning.* arxiv.org/abs/1912.01588 (Procgen)
8. Ng (2021). *A Chat with Andrew Ng about AI Shifting to Data-Centric.*
   IEEE Spectrum. spectrum.ieee.org/andrew-ng-data-centric-ai
9. Zha et al. (2023). *Data-Centric Artificial Intelligence: A Survey.*
   arxiv.org/html/2212.11854v4
10. Hoffmann et al. (2022). *Training Compute-Optimal Large Language
    Models.* arxiv.org/abs/2203.15556 (Chinchilla)
11. Ng, Harada, Russell (1999). *Policy Invariance Under Reward
    Transformations.* ICML. (Potential-based shaping)
12. Vecerík et al. (2017). *Leveraging Demonstrations for Deep RL on
    Robotics Problems with Sparse Rewards.* arxiv.org/abs/1707.08817
    (DDPGfD)
13. *Beyond Imitation: Recovering Dense Rewards from Demonstrations.*
    arxiv.org/html/2510.02493 (2024)
14. *Attention-Based Reward Shaping for Sparse and Delayed Rewards.*
    arxiv.org/html/2505.10802v1 (2024, ARES)
15. Xie et al. (2024). *OSWorld: Benchmarking Multimodal Agents for
    Open-Ended Tasks in Real Computer Environments.* NeurIPS.
    os-world.github.io
16. Zhou et al. (2023). *WebArena: A Realistic Web Environment for
    Building Autonomous Agents.* arxiv.org/abs/2307.13854
17. Yang et al. (2025). *GUI-Robust: A Comprehensive Dataset for Testing
    GUI Agent Robustness in Real-World Anomalies.* NeurIPS Datasets
    Track. arxiv.org/abs/2506.14477
18. OSWorld-MCP (2025). github.com/X-PLUG/OSWorld-MCP
19. Zhao et al. (2025). *On the Robustness of GUI Grounding Models
    Against Image Attacks.* CVPR Workshops. arxiv.org/abs/2504.04716
20. *On the Robustness of Multimodal Language Model towards
    Distractions.* arxiv.org/html/2502.09818 (2025)
21. WorldGUI (2025). *An Interactive Benchmark for Desktop GUI Automation
    from Any Starting Point.* arxiv.org/html/2502.08047
22. ARPO (2025). *End-to-End Policy Optimization for GUI Agents with
    Experience Replay.* arxiv.org/abs/2505.16282
