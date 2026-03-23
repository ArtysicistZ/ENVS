# MCTS SFT Training Strategy: Weighting and Step-Level Filtering

> Research document for training on MCTS-generated GUI agent trajectories.
> Date: 2026-03-22

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Per-Task Weighting Schemes](#2-per-task-weighting-schemes)
3. [Step-Level Filtering: Remove Real Errors, Keep Recovery](#3-step-level-filtering-remove-real-errors-keep-recovery)
4. [Empirical Analysis of Our MCTS Trajectories](#4-empirical-analysis-of-our-mcts-trajectories)
5. [Recommended Approach](#5-recommended-approach)
6. [References](#6-references)

---

## 1. Problem Statement

Our MCTS trajectory collection produces wildly imbalanced data across 86 tasks:

| Metric | Value |
|--------|-------|
| Total successful trajectories | 1,792 |
| Tasks with successes | 83 (3 tasks have 0) |
| Min successes per task | 1 |
| Max successes per task | 77 |
| Median | 16 |

After `expand_episode` (converting each trajectory into per-step SFT examples), the imbalance amplifies:

- **Top task**: 752 SFT steps (68 trajs x 11.1 avg steps)
- **Bottom task**: 6 SFT steps (1 traj x 6 steps)
- **Raw gradient imbalance**: 125x
- **Top 10 tasks**: 32.2% of all SFT steps
- **Bottom 10 tasks**: 1.3% of all SFT steps

Without intervention, the model learns 77 ways to add a GIMP layer (easy) and barely learns how to switch a Linux user (hard). Two questions arise:

1. **How should we weight samples across tasks?**
2. **Within each trajectory, which steps should we train on?**

---

## 2. Per-Task Weighting Schemes

### 2.1 The Temperature Sampling Framework

The standard approach for rebalancing imbalanced groups comes from multilingual pretraining. Given tasks with `n_t` examples each, sample task `t` with probability proportional to `n_t^α`:

```
p_t ∝ n_t^α
```

- **α = 1.0**: Natural distribution. Sample proportional to group size. Large tasks dominate.
- **α = 0.0**: Task-uniform. Every task equally likely regardless of size.
- **0 < α < 1**: Interpolation. Smaller α = more upsampling of rare tasks.

The total gradient mass per task is `G_t = n_t^α`, giving a gradient mass ratio between the largest (n=77) and smallest (n=1) task of **77^α**.

To implement this via per-sample loss weights rather than resampling, each sample from task t gets weight `w_t = n_t^(α−1)` (so that `n_t × w_t = n_t^α`).

> **Convention warning**: Some formulations use a weight exponent β where `w = 1/n^β`, which inverts the convention: **β = 1−α**. XLM-R's sampling α=0.3 corresponds to weight β=0.7. Confusing these conventions produces wrong ratio calculations — e.g., writing `w = 1/N^0.3` but expecting ratio 77^0.3 ≈ 3.7x, when that formula actually gives ratio 77^(1−0.3) = 77^0.7 ≈ 20.9x. **This document uses the sampling exponent α throughout.**

**Correct ratio table** (for N_max=77, N_min=1):

| Scheme | Sampling α | Per-sample weight | 77:1 gradient ratio | Used by |
|--------|-----------|-------------------|-------------------|---------|
| Natural | 1.0 | 1 (uniform per sample) | **77x** | No balancing |
| mBERT-style | 0.7 | n^(−0.3) | **20.9x** | mBERT (Devlin, 2018) |
| Square root | 0.5 | 1/√n | **8.8x** | Common default |
| XLM-R / mT5 | 0.3 | n^(−0.7) | **3.7x** | XLM-R, mT5 |
| Task-uniform | 0.0 | 1/n | **1x** | Full inverse frequency |

### 2.2 Literature Review

#### XLM-R (Conneau et al., ACL 2020)

Used **α = 0.3** for sampling across 100 languages, finding it optimal for overall cross-lingual transfer:

> "When considering overall performance, we found 0.3 to be an optimal value for α, and use this for XLM-R."

Higher α favors high-resource languages; lower α favors low-resource. α=0.3 was a significant shift from mBERT's α=0.7 — XLM-R upsamples low-resource languages much more aggressively.

**Context**: XLM-R trains on 2.5TB of CommonCrawl data across 100 languages. Even the lowest-resource languages have millions of tokens. This is a fundamentally different data scale from our ~1,800 trajectories (~18K steps).

#### mT5 (Xue et al., NAACL 2021)

Tested three values: α=0.2 (from MMNMT/Arivazhagan et al. 2019), α=0.3 (from XLM-R), α=0.7 (from mBERT). Results on XNLI zero-shot (mT5-Large):

| α | XNLI accuracy |
|---|--------------|
| **0.3** | **81.1** |
| 0.2 | 80.7 |
| 0.7 | 80.7 |

Differences are small (0.4 points) but α=0.3 consistently won. mT5 adopted α=0.3 following XLM-R.

#### UniMax (Chung et al., ICLR 2023)

Showed that temperature sampling causes pathological overfitting when minority groups are very small — those examples get repeated dozens of times per epoch. **UniMax** caps each source at a maximum of K repetitions per epoch, then distributes the remaining sampling budget uniformly.

UniMax consistently outperforms standard temperature sampling and benefits persist at scale. **Directly relevant** to our 1-3 trajectory tasks, which would otherwise be repeated excessively under aggressive upsampling.

#### DART-Math "Prop2Diff" (Tong et al., NeurIPS 2024)

Showed that vanilla rejection sampling creates severe bias toward easy queries: the hardest queries produce few or zero synthetic responses. **Prop2Diff** allocates synthetic responses proportional to query difficulty (fail rate): harder queries get more responses. Results: +4.5 points over vanilla rejection tuning on MATH/GSM8K.

This is a **collection-time** strategy — it cannot be applied retroactively, but it motivates our rerun_low23 re-collection for hard tasks.

#### Class-Balanced Loss (Cui et al., CVPR 2019)

Introduces "effective number of samples": `E_n = (1 − β^n) / (1 − β)`, with per-class weight proportional to 1/E_n.

- β = 0 → no reweighting (all classes equal weight per sample)
- β → 1 → full inverse frequency (1/n)
- β = 0.999 (their recommended default) → for our group sizes (1-77), 0.999^77 = 0.926, so E_77 ≈ 74.2 ≈ n. This approximates **full inverse frequency (1/n)**, not moderate smoothing.
- For moderate smoothing comparable to √n on group sizes 1-77, β ≈ 0.9 is more appropriate (E_77 ≈ 7.3).

#### Reward-Weighted Regression (Peters & Schaal, 2007; Peng et al., 2019)

The classical RL framework for converting demonstrations into training signal:

```
max_θ  E_{(s,a) ~ D} [ w_i * log π_θ(a_i | s_i) ]
```

where `w_i = exp(A_i / β)` and β is a temperature parameter. Weight clipping at w_max ~ 20-100 prevents gradient explosion.

#### Importance-Weighted SFT (Springenberg et al., 2025)

Formal proof that SFT on curated data optimizes an RL lower bound. The importance weight per trajectory is `w(τ) = q(τ) / π_ref(τ)`, computed via per-token log-probability ratios with clipping for variance control.

**Practical caveat**: Byrd & Lipton (ICML 2019) showed that in overparameterized models, the effect of importance weights diminishes over training epochs. The implicit bias of gradient descent toward max-margin solutions makes the final solution less sensitive to per-sample weights.

**Implication**: For large VLMs, **data-level resampling** (controlling what enters the training set) may be more robust than **loss-level reweighting** (multiplying the loss).

#### Resampling vs Reweighting (An et al., ICLR 2021)

An, Ying & Zhu prove that resampling outperforms reweighting under stochastic gradients, even though they are equivalent in expectation. The key difference: resampling introduces beneficial implicit regularization through gradient noise, while extreme loss weights increase harmful gradient variance without this regularizing effect.

### 2.3 Comparison for Our Data

Our setting differs from multilingual NLP in critical ways:

| Factor | Multilingual NLP | Our MCTS trajectories |
|--------|-----------------|----------------------|
| Groups | 100+ languages | 84 tasks |
| Size range | Millions–billions of tokens | 1–77 trajectories (6–752 steps) |
| Rare group quality | High (curated web text) | Variable (barely-succeeded MCTS explorations) |
| Overfitting risk | Low (massive data per group) | High (1 trajectory = 6 steps) |
| Total data | Terabytes | ~18K steps |

**The key difference is overfitting risk.** XLM-R and mT5 can afford aggressive upsampling (α=0.3) because even their rarest languages have millions of tokens — no risk of memorization. We have tasks with a single 6-step trajectory. Repeating those 6 steps dozens of times per epoch invites memorization of noisy, barely-succeeded behavior.

**Approximate gradient mass distribution under each scheme:**

| Scheme (α) | Top 10 tasks | Bottom 10 tasks | Rare tasks (≤3 trajs, ~12 tasks) |
|------------|-------------|-----------------|--------------------------------|
| Natural (1.0) | ~32% | ~1.3% | ~0.7% |
| mBERT (0.7) | ~28% | ~2.4% | ~1.5% |
| **√n (0.5)** | **~23%** | **~4%** | **~3%** |
| XLM-R (0.3) | ~17% | ~7% | ~6% |
| Uniform (0.0) | 11.9% | 11.9% | 14.3% |

### 2.4 The Noise-Quality Tradeoff

Our tasks with ≤3 trajectories have genuinely noisy data:

- **ac1b39ff** (impress: move table): 7 steps, coherent but found only via branching at steps 1 and 3
- **a462a795** (os: switch user): 6 steps, confused intermediate steps (`sudo -i` → "user doesn't exist"), yet eval=1.0
- **b3d4a89c** (os: turn on Bluetooth): 9 steps, wanders through Chrome, system tray, terminal, gets "address family not supported" error

The dilemma:
- **Too little upsampling** (α → 1.0): Rare tasks get <1% of gradient. The model barely learns from hard tasks.
- **Too much upsampling** (α → 0.0): Noisy rare-task samples get amplified. Under task-uniform (α=0), each sample from a 1-trajectory task gets **77x** the per-sample weight of samples from the 77-trajectory task. This heavily amplifies confused wandering behavior.

**Resolution**: Combine moderate upsampling with safeguards:
1. **Square root sampling (α=0.5)**: Reduces 77:1 to 8.8:1 — meaningful improvement without extreme per-sample weights
2. **Repetition capping** (UniMax-style): Prevent any trajectory from appearing >K times per epoch
3. **Step-level filtering** (Section 3): Clean noise before upsampling, so amplified samples are higher quality
4. **Quality-aware adjustment**: Optionally further downweight tasks whose trajectories remain noisy after filtering

### 2.5 Resampling vs Reweighting

Two ways to implement the same effective distribution:
- **Resampling**: Build a dataloader that samples tasks with probability ∝ n_t^α, then uniformly selects trajectories within each task. All samples have equal loss weight.
- **Reweighting**: Use the natural data distribution but multiply each sample's loss by w_t = n_t^(α−1).

An et al. (ICLR 2021) showed resampling outperforms reweighting under SGD. Byrd & Lipton (ICML 2019) showed loss weights diminish in effect for overparameterized models at convergence. **We recommend resampling** — implement the balancing at the dataloader level, not the loss level.

---

## 3. Step-Level Filtering: Remove Real Errors, Keep Recovery

### 3.1 The Core Idea

In a successful MCTS trajectory, the agent often takes a wrong action — clicks the wrong menu, opens the wrong tool, navigates to the wrong place — and then in a later step, **realizes it made a mistake** and corrects course. These two types of steps have completely different training value:

- **The wrong action itself** (e.g., clicking Colors menu instead of Image menu): This step teaches the model to do the wrong thing. If we train on it, the model learns "when trying to set palette mode, click Colors." This is harmful.

- **The step where the agent recognizes the error** (e.g., "I just clicked on the Colors menu, but that's not what I need. The palette settings should be under the Image menu."): This step is **extremely valuable**. It teaches the model how to diagnose a mistake from the current screen state and plan a recovery. This is the exact capability that makes a robust agent.

Therefore: **remove the real error steps, keep the error-recognition and recovery steps.** After filtering, the trajectory becomes a demonstration of "even when things go wrong, here's how to get back on track."

### 3.2 Why This Cannot Be Done with Scripts

A python regex or heuristic cannot distinguish between these step types. Consider:

**Step 5** (real error — should be REMOVED):
> Thought: I need to adjust the palette settings, so I'll click on the Colors menu to find the relevant options.
> Action: click(start_box='<|box_start|>(513,150)<|box_end|>')

**Step 6** (error recognition — should be KEPT):
> Thought: I just clicked on the Colors menu, but it seems I couldn't find the option I was looking for. The palette settings should be located under the Image menu. I need to reopen the Image menu.
> Action: click(start_box='<|box_start|>(413,153)<|box_end|>')

Both steps contain "Colors menu" and "palette." Step 5 is the agent confidently doing the wrong thing. Step 6 is the agent diagnosing the mistake and correcting course. No regex can tell these apart — it requires **reading the thought, understanding the intent, comparing with the actual UI outcome, and judging whether the action was productive or mistaken.**

This is fundamentally a semantic judgment that requires the same kind of reasoning a human data annotator would apply. The only way to do it correctly is to have agents (acting as human-like reviewers) read every trajectory step by step and make per-step judgments.

### 3.3 Literature Support

#### WebSTAR (2512.10962): Step-Level > Trajectory-Level Filtering

He et al. train a StepRM (Step Reward Model) to score each step 0-10, then apply binary masking — train only on steps scoring >5. Full trajectories are preserved as context (the model sees errors in the input) but loss is computed only on quality steps.

**Results**: Step-level filtering achieves **39.6%** vs trajectory-level **29.9%** on WebVoyager — nearly 10 points improvement using roughly half the data volume.

**Key design**: The model still sees error steps as context. It just doesn't optimize on them. This is exactly our approach — we remove error steps from the loss but keep them in the conversation history so the model understands the full trajectory context.

#### Agent-R (2501.11425): MCTS-Based Recovery Training

Yuan et al. construct **revision trajectories** directly from the MCTS tree. The agent identifies the "first error" in its failed trajectories, truncates at that point, and continues with a successful sibling branch. Training on these revision trajectories teaches self-correction.

**Direct relevance**: Our MCTS tree already has the structure for this. When a branch fails and its sibling succeeds at the same divergence point, the failed prefix contains the error steps, and the successful suffix contains the recovery.

#### STeP (2505.20023): Partial Masking of Error Steps

Chen et al. apply partial masking to self-reflected trajectories:

```
L_PM = -E[Sum delta_i * log pi(a_i | ...)]
```

where delta_i = 0 for error steps, delta_i = 1 for correct steps. Error steps contribute zero loss but remain as context. The result: the model learns to recover from errors it can see in context without being trained to reproduce those errors.

#### GUI-Reflection (2506.08012): Three-Phase Recovery Training

Wu et al. decompose recovery into: verification (recognizing errors), backtracking (undo actions), reattempt (corrected actions). Their three-phase training pipeline specifically separates "the wrong action" from "recognizing the wrong action" — training on the latter, not the former.

#### DART (2509.23866): Entropy-Based Step Selection

Li et al. compute per-step entropy and train only on steps whose entropy exceeds the 20th percentile. Low-entropy steps (where the model is highly confident) contribute zero gradient. The rationale: high-entropy steps are "critical forks" where training signal matters most.

#### GUI-Libra (2602.22190): Agreement Filtering

Yang et al. run the VLM 10 times stochastically and discard steps with re-prediction accuracy < 0.3. Also uses action-aware token reweighting (action tokens 2x, coordinate tokens 4x vs reasoning tokens).

### 3.4 The Filtering Process

The filtering must be done by **agents reading every trajectory** — functioning exactly as human data annotators would. No automated scripts or regex. The reason is that error classification is a semantic judgment that requires:

1. Reading the agent's thought to understand its intent
2. Looking at what action it took
3. Comparing with the next step's thought to see if the action achieved what was intended
4. Judging whether the step moved toward the goal or away from it

For each step in each trajectory, an agent must classify:

**REMOVE (loss weight = 0)**: The step is a real error — the agent takes an action that leads it to the wrong place. The thought may sound confident, but the action is objectively wrong given the task. Examples:
- Clicking the wrong menu when trying to find a setting
- Opening the wrong application
- Typing the wrong command
- Navigating to an irrelevant part of the UI
- Repeating the same failed action multiple times with no new reasoning

**KEEP (loss weight = 1)**: The step is either:
- **A correct productive step**: The agent does the right thing with clear reasoning.
- **An error-recognition step**: The agent explicitly recognizes a previous mistake and plans recovery. The thought contains diagnosis of what went wrong and the action takes a corrective path. These are among the most valuable steps in the entire dataset — they teach the model the critical skill of self-correction.

After filtering, the trajectory still contains the full conversation context (error steps appear in the input/history), but the model only optimizes on correct steps and error-recognition steps. It learns: "given that I just made a mistake (visible in context), here's how to diagnose it and recover."

### 3.5 Scale of the Audit

- ~1,800 trajectories, ~17,000 steps
- ~100 parallel agents, each reviewing ~18 trajectories (~170 steps)
- Each agent reads the raw thought+action text for every step, in sequence, and produces a per-step label (REMOVE or KEEP)
- Expected outcome: ~10-15% of steps marked REMOVE (the real error actions), ~85-90% KEEP (productive steps + error-recognition steps)

---

## 4. Empirical Analysis of Our MCTS Trajectories

### 4.1 Step-Level Statistics

Across all 17,334 steps in 1,792 successful trajectories:

| Category | Count | Fraction |
|----------|-------|----------|
| **Productive** (correct actions with clear reasoning) | 7,895 | 45.5% |
| **Neutral** (descriptive narration, no strong signal) | 7,799 | 45.0% |
| **Error-related** (contains error language) | 1,640 | 9.5% |

The error-related 9.5% is a mix of both **real error steps** (the wrong action) and **error-recognition steps** (diagnosing a previous wrong action). Only the agent audit can separate these two — the real errors should be removed, the error-recognition steps should be kept.

### 4.2 Error Distribution Patterns

**57.6%** of trajectories contain at least one error-related step. Only 28.5% are entirely clean.

**Errors cluster early** in trajectories:
- First third: 19.8% error rate
- Middle third: 17.7%
- Last third: 11.3%

**Errors are self-reinforcing**: 37.1% chance an error step is followed by another error. But productivity is even stickier: 49.5% chance a productive step is followed by another productive step. Once the agent finds the right track, it tends to stay there.

**Most error runs are short**: 73.4% are isolated single steps. Mean error run length: 1.54 steps.

### 4.3 Trajectory Structure

**58.7% of long trajectories** (10+ steps) follow a "wander then productive" pattern where early errors give way to a clean execution phase.

**Average wasted steps**: 1.77 out of 9.67 mean steps (18.3% waste ratio).

**Error rate scales with trajectory length**:
- 4-5 step trajectories: 7-8% error rate
- 10-12 step trajectories: 15-19%
- 15 step trajectories (max): 17%

### 4.4 MCTS Divergence and Step Quality

**Pre-divergence vs post-divergence error rates are nearly identical**:
- Pre-divergence steps (shared with parent): 14.1% error rate
- Post-divergence steps (new branches): 13.9%

This means the MCTS tree branches at points of genuine uncertainty, not necessarily after clean prefixes. The shared prefix is NOT significantly cleaner.

### 4.5 Quality Concerns with Max-Step Trajectories

38.3% of trajectories from hard tasks (<=5 successes) hit the max_steps=15 ceiling, vs only 5.8% for easy tasks. These represent "barely scraped by" behavior and likely contain a higher proportion of real error steps.

---

## 5. Recommended Approach

### 5.1 Level 1: Per-Task Resampling with Square Root Weighting

**Strategy**: Resample training data so each task's gradient mass is proportional to √n_t, combined with per-trajectory length normalization and repetition capping.

#### Sampling Distribution

For each training step, sample with probability:

```
P(step s from trajectory j of task t) ∝ n_t^(α−1) / L_tj
```

where:
- `n_t` = number of trajectories for task t
- `L_tj` = number of steps in trajectory j of task t
- `α = 0.5` (square root sampling)

This gives per-sample weight `w = n_t^(−0.5) / L_tj = 1 / (√n_t × L_tj)`.

This achieves:
1. **Task-level balance**: Total gradient per task ∝ √n_t. The 77-trajectory task gets **8.8x** the gradient of the 1-trajectory task (not 77x, not 1x).
2. **Trajectory-level equalization**: Within each task, every trajectory contributes equally regardless of length. A 3-step efficient solution gets the same total weight as a 15-step wandering solution.

#### Equivalent Loss Weights (for implementation)

If implementing via loss weighting rather than a resampled dataloader:

```
w(t, j, s) = 1 / (√n_t × L_tj)
```

Normalize so weights sum to the total number of training steps (preserving learning rate calibration):

```
w_normalized = w / mean(w)
```

#### Repetition Capping (UniMax-style)

To prevent overfitting on rare tasks, cap each trajectory at **K = 3** maximum appearances per epoch:

- A task with 1 trajectory × 6 steps → at most 18 step-appearances per epoch
- A task with 77 trajectories × 10 avg steps → ~770 step-appearances before √n down-weighting
- After capping, redistribute excess sampling budget to uncapped tasks

This addresses the core failure mode of aggressive upsampling: a single noisy trajectory getting memorized through excessive repetition.

#### Why √n (α=0.5) as Starting Point

Square root sampling is the de facto default for imbalanced fine-tuning. The key reason we start here rather than at XLM-R's α=0.3:

| Factor | XLM-R/mT5 setting (α=0.3) | Our setting |
|--------|---------------------------|-------------|
| Data scale | 2.5TB, millions of tokens per language | ~18K steps total |
| Rare group size | Millions of tokens | 6 steps (single trajectory) |
| Overfitting risk | Negligible | High |
| Rare group quality | Curated web text | Barely-succeeded MCTS explorations |

At α=0.3, rare tasks (~12 tasks with ≤3 trajectories) would get ~6% of gradient mass — and those trajectories are our noisiest. At α=0.5, they get ~3% — still 4x more than under no balancing (0.7%), but with less noise amplification.

**If hard-task eval performance lags**: Push α down to 0.3 (XLM-R/mT5 style) in a follow-up experiment. The step-level filtering (Level 2) reduces noise in rare-task data, making more aggressive upsampling safer. But start conservative.

#### Why Resampling, Not Loss Reweighting

Implement the balancing at the dataloader level (resampling), not the loss level:
- An et al. (ICLR 2021): Resampling outperforms reweighting under SGD due to implicit regularization
- Byrd & Lipton (ICML 2019): Importance weights diminish in effect for overparameterized models

#### Interaction with Level 2

Step-level filtering and per-task resampling work together:
- **Filtering alone** (no resampling): Data is cleaner, but easy tasks still get ~32% of gradient. The model barely learns from hard tasks.
- **Resampling alone** (no filtering): Rare tasks get more gradient, but that gradient amplifies noisy error steps from barely-succeeded trajectories.
- **Both together**: Filtering cleans the data first, then resampling gives cleaned rare-task trajectories appropriate representation. This is the correct order — clean first, then rebalance.

### 5.2 Level 2: Agent-Audited Step-Level Filtering

**Remove real error steps. Keep everything else — especially error-recognition steps.**

The filtering is performed by launching ~100 parallel agents that read every trajectory step by step, just as a human annotator would. No python scripts, no regex, no heuristics. Each agent reads the raw thought and action for each step, understands the task context, and makes a semantic judgment:

- **Is this step a real mistake?** The agent took an action that led it to the wrong place — wrong menu, wrong tool, wrong command, pointless repeated action. → **REMOVE** (set loss weight to 0, but keep in conversation context)

- **Is this step an error-recognition moment?** The agent explicitly recognizes a previous mistake, diagnoses what went wrong, and plans a corrective action. → **KEEP** (these are the most valuable steps in the dataset — they teach self-correction)

- **Is this step a normal productive step?** The agent does the right thing. → **KEEP**

The result is a per-step weight map stored alongside the trajectory data. The training script consumes this map: error steps contribute zero loss but remain as context, so the model sees the full trajectory including mistakes, but only optimizes on correct actions and recovery actions.

**Why agents, not scripts**: The difference between "clicking Colors menu (wrong)" and "I just clicked Colors menu but that was wrong, I need Image menu instead" is purely semantic. It requires reading the thought, understanding intent, and comparing with outcome. This is a human-level judgment that no regex can make.

### 5.3 Future: Agent-R Revision Trajectories from MCTS Tree

Beyond filtering naturally-occurring trajectories, we can construct additional error-recovery training data directly from the MCTS tree:

1. Find sibling nodes where one branch succeeded and the other failed
2. Take the failed prefix up to the divergence point
3. Insert a reflection signal ("The previous approach didn't work because...")
4. Continue with the successful branch's actions

This directly teaches error recovery using the tree structure we already have. The `diverged_at_step` and `branch_path` metadata in each trajectory provides exactly the information needed.

### 5.4 Future: MCTS-Aware Step Weighting

Our MCTS tree provides natural step-quality signal that can complement the agent audit:
- **Branch points** (where multiple clusters were detected) = high entropy = decision points worth emphasizing
- **Per-step success rate** = fraction of downstream leaves with eval_score > 0 — can be propagated back through the tree as continuous step weights
- Steps that appear in many successful trajectories but few failed ones = high-confidence productive steps
- Steps unique to a single path = idiosyncratic, lower confidence

---

## 6. References

### Per-Task Weighting

| Paper | ID | Key Contribution |
|-------|----|------------------|
| XLM-R | Conneau et al., ACL 2020 (1911.02116) | Temperature sampling with α=0.3 for 100 languages |
| mT5 | Xue et al., NAACL 2021 (2010.11934) | Ablation of α={0.2, 0.3, 0.7}; α=0.3 optimal |
| UniMax | Chung et al., ICLR 2023 | Capped temperature sampling prevents overfitting on rare groups |
| DART-Math | Tong et al., NeurIPS 2024 (2407.13690) | Prop2Diff: difficulty-proportional sampling beats uniform |
| Class-Balanced Loss | Cui et al., CVPR 2019 (1901.05555) | Effective number of samples, β interpolation |
| iw-SFT | Springenberg et al., 2507.12856 | Formal proof SFT on curated data = RL lower bound |
| RWR | Peters & Schaal, ICML 2007 | exp(R/T) weighting with temperature control |
| AWR | Peng et al., 2019 | Practical RWR with advantage estimation and clipping |
| Why Resampling > Reweighting | An et al., ICLR 2021 (2009.13447) | Resampling outperforms reweighting via implicit regularization |
| Importance Weighting in DL | Byrd & Lipton, ICML 2019 (1812.03372) | Importance weights diminish in effect for overparameterized models |
| DoReMi | Xie et al., NeurIPS 2023 (2305.10429) | Learned domain weights via minimax proxy optimization |
| WED for ExIT | 2006.00283 | Episode duration normalization for self-play |
| Expert Iteration | Anthony et al., NeurIPS 2017 | Original ExIT framework |
| STaR | Zelikman et al. | Self-taught reasoner with rationalization |
| V-STaR | COLM 2024 | DPO verifier from both positive and negative trajectories |
| ReST/ReST^EM | Google DeepMind, 2308.08998 | Grow-improve loop with binary filtering |

### Step-Level Filtering

| Paper | ID | Key Contribution |
|-------|----|------------------|
| WebSTAR | 2512.10962 | Binary step masking with StepRM, +10 pts over traj-level |
| DART (agent) | 2509.23866 | 4-level adaptive curation, entropy-based step selection |
| Agent-R | 2501.11425 | MCTS-based revision trajectory construction |
| STeP | 2505.20023 | Partial masking of error steps in reflected trajectories |
| GUI-Reflection | 2506.08012 | Three-phase reflection training for GUI agents |
| GUI-Libra | 2602.22190 | Action-aware token reweighting + agreement filtering |
| ReST-MCTS* | NeurIPS 2024, 2406.03816 | Tree search for automatic step reward inference |
| OmegaPRM | 2406.06592 | MCTS binary search for step error identification |
| CSO | 2602.03412 | Verified critical step optimization (16% of steps) |
| HGPO | 2602.22817 | Hierarchical step advantage estimation |
| GiGPO | 2505.10978 | Anchor state grouping for step-level credit |
| SELAUR | 2602.21158 | Uncertainty-aware rewards for step weighting |
| iStar | 2509.19199 | Implicit step rewards via trajectory DPO |
| ECHO | 2510.10304 | Hindsight trajectory rewriting for LLM agents |
| SAMULE | EMNLP 2025, 2509.20562 | Multi-level reflection synthesis (micro/meso/macro) |
| WebSynthesis | 2507.04370 | World-model MCTS for web trajectory synthesis |
| ExACT | 2410.02052 | Reflective MCTS for web agents (Exploratory Learning) |
| TreeRL | ACL 2025, 2506.11902 | On-policy RL with tree search, process supervision |
| SPAE | 2601.03823 | Step potential advantage from confidence probing |
| AgentProcessBench | 2603.14465 | Benchmark for step-level quality in agent trajectories |
