# Recovery / exploration hypothesis at n=8, t=1
Re-runs the n=1, t=0 analysis (`../recovery_hypothesis_n1t0/REPORT.md`) on the
8-rollout, temperature-1 evaluation. 86 paired tasks × 8 rollouts each side =
688 rollouts per model.

## Topline

| | SFT init | ARPO step 35 |
|---|---|---|
| pass@1 (mean over 8) | 0.387 | **0.406** |
| pass@8 (≥1 success) | 71 / 86 (82.6 %) | **74 / 86 (86.0 %)** |
| total successful rollouts | 266 / 688 | 279 / 688 |

Transitions on **pass@8** basis: PP = 63, regression PF = 8, gain FP = 11, FF = 4.
Net **+3 tasks** for ARPO (same direction as n=1, but the absolute numbers are
much closer to SFT-equivalence).

The big shift vs n=1: **only 12 / 86 task outcomes flip at pass@8 (14 %)**, vs
36 % at n=1. Most of what looked like "ARPO instability" at n=1 was sampling
variance — ARPO's *policy support* is wide enough to solve almost everything
SFT solves once you draw 8 samples.

---

## H1 (main): ARPO can't explore mid-trajectory — **WEAKENED**

The cleanest n=8 form of H1 is to compare **successful rollouts** of each model
(if H1 holds, ARPO success would look more "tunneled" — fewer fns, higher
max-run, lower recovery — than SFT success). They don't.

| metric (per rollout) | SFT success (n=266) | ARPO success (n=279) |
|---|---|---|
| unique action-fn types | 3.82 | 3.80 |
| unique click clusters | 6.32 | 6.21 |
| max consec same-fn run | 4.78 | 4.76 |
| self-correction phrases | 0.32 | 0.32 |
| back-and-forth A→B→A | 1.03 | 1.10 |
| recovery rate | **0.935** | **0.935** |
| trajectory length | 10.17 | 10.19 |

Within failures the two are equally indistinguishable (recovery 0.857 vs 0.851;
unique fns 3.45 vs 3.42). And on the 63 tasks where both models pass at least
once, paired success rollouts are identical to 2 decimals across all six
metrics. **At n=8 there is no global exploration deficit in ARPO trajectories.**

The H1 signal does still appear, but only on a narrow slice — the **8 regression
tasks** where SFT solves the task at least once and ARPO goes 0 / 8:

| metric | SFT successful rollouts (n=13) | ARPO failed rollouts (n=64) |
|---|---|---|
| unique action-fn types | **4.38** | 3.50 |
| recovery rate | **0.91** | 0.84 |
| self-correction phrases | 0.15 | **0.83** |
| max consec same-fn run | 4.38 | **5.61** |
| back-and-forth A→B→A | 1.08 | **1.97** |

Same fingerprint as the n=1 regressions (the "says I'm stuck but doesn't change
the action" gap reappears: ARPO emits **5×** more self-correction language while
its recovery rate is lower). It is a real effect on the tasks ARPO genuinely
can't solve, but it is no longer the *dominant* story it was at n=1.

**Updated framing:** at greedy n=1 the policy collapses to the argmax and the
exploration deficit shows up everywhere; at t=1 with 8 samples, sampling noise
re-injects enough diversity that successful rollouts look indistinguishable
between SFT and ARPO, and the deficit only shows up on tasks where ARPO's
*entire* mass of 8 samples lands inside a wrong basin.

See `fig_success_rollouts.png`, `fig_paired_success.png`, `fig_paired_features.png`.

---

## H2: ARPO wanders too long — **REFUTED (again)**

| | SFT | ARPO |
|---|---|---|
| mean steps on success | 10.17 | 10.19 |
| mean steps on failure | 13.32 | 13.32 |
| mean steps on the 4 both-fail tasks (paired) | 14.75 | 14.91 |

Identical at every aggregation level. See `fig_length.png`.

---

## H3: first few key steps wrong — **REFUTED (again)**

Pooled across both runs (1 376 rollouts):

| first-3-bad | success rate | N |
|---|---|---|
| 0 (clean opener) | 37.7 % | 220 |
| ≥1 (shaky opener) | 40.0 % | 1 156 |

Knowing the opener is clean *lowers* the success rate by 2 pp — opposite
direction of the hypothesis, same as in the n=1 analysis. See `fig_first3.png`.

---

## H4 (new): cross-rollout policy collapse — **NOT SUPPORTED**

For each task, compute per-step entropy of action-fn across the 8 rollouts and
average over the first 12 steps. If ARPO's policy is collapsed, this should be
lower for ARPO at fixed task.

| cohort | SFT entropy | ARPO entropy | N tasks |
|---|---|---|---|
| both-pass (PP)        | 0.690 | 0.658 | 63 |
| regression (PF)       | 0.696 | **0.740** | 8 |
| gain (FP)             | 0.761 | **0.808** | 11 |
| both-fail (FF)        | 0.550 | 0.500 | 4 |
| **mean (paired Δ)**   | 0.693 | 0.677 | (Δ = +0.016 nats) |

Aggregate gap is in the H4 direction (ARPO entropy 2 % lower) but *small* and
*reversed in two of the four cohorts*, including regression. ARPO sampling
noise at t=1 produces about as much cross-rollout diversity as SFT — the policy
is sharper but not collapsed. See `fig_entropy.png`.

---

## H5 (new): pass@8 − pass@1 exploration headroom — **REFUTED**

If the policy is too tight to luck into wins across resamples, ARPO's pass@8
should sit much closer to its pass@1 than SFT's does.

| | SFT | ARPO |
|---|---|---|
| mean (pass@8 − pass@1), all tasks | 0.439 | 0.455 |
| mean, restricted to tasks with pass@8 > 0 | 0.532 | 0.529 |

Effectively identical. See `fig_headroom.png`, `fig_pass1_scatter.png`.

---

## What the n=8, t=1 evidence actually says

1. **The "ARPO is unstable" framing was largely a greedy / single-sample
   artifact.** 36 % outcome churn at n=1 → 14 % at n=8. Net direction stays the
   same (ARPO +3 tasks) but the magnitude shrinks roughly 2.5×.
2. **Within-rollout exploration features are statistically the same** between
   SFT and ARPO at n=8/t=1, both on success and on failure cohorts. The H1
   "trajectory-level RL erodes mid-trajectory recovery" claim is *not visible*
   when the policy is sampled at t=1.
3. **The deficit is concentrated, not diffuse.** It exists on the 8 regression
   tasks where ARPO whiffs all 8 samples — there the n=1 fingerprint reappears
   (5× self-correction language, lower recovery, longer same-fn runs). On
   everything else, ARPO and SFT are interchangeable on these features.
4. **MCTS-SFT detour-imitation framing is consistent but no longer
   *necessary*** to explain the data: the gap between models is too small at
   n=8 to demand a structural explanation, and could plausibly come from
   training noise + 8 task-level whiffs.

## Re-verifying earlier hypotheses

- *"ARPO performs unstably because it wanders too much"* — refuted at n=1,
  refuted again at n=8. Step counts paired both-fail are 14.75 vs 14.91; success
  paths are identical (10.17 vs 10.19). Wandering is what *failure* looks like
  for both models.
- *"ARPO performs unstably because the first few steps are wrong"* — refuted
  at n=1, refuted again at n=8. Clean openers give 37.7 % success vs 40.0 % for
  shaky ones. Effect is in the wrong direction.
- *"ARPO is unstable per se"* — n=1 made this look like a 36 % churn problem;
  at n=8 it shrinks to 12 / 86 task flips with ARPO net positive. Most of the
  apparent instability was n=1 sampling variance.

---

## Other hypotheses worth ruling in / out next

These remain open after the n=8 evidence:

1. **Regression-task localisation.** The H1 fingerprint *survives* on 8 / 86
   tasks where ARPO collapses to 0 / 8 success. Inspecting these task
   instructions and the trajectories themselves would tell us whether they
   share a structural property (e.g. tasks where MCTS-SFT data contained the
   only successful detour pattern, ARPO over-shifted away from). The pairs
   table (`tasks_table.csv`) marks them as cohort `PF`.
2. **t=0 vs t=1 gap of the same checkpoint.** The mismatch between n=1/t=0
   (ARPO −) and n=8/t=1 (ARPO ≈ +) suggests the deficit lives in the
   greedy-mode argmax. Sampling at t=1 from ARPO with n=1 (no aggregation)
   would isolate whether it is *temperature* or *aggregation* doing the work.
3. **MCTS-SFT detour signature, direct.** A single-side check on SFT
   trajectories: are *successful* rollouts longer and more diverse than
   *minimal* rollouts of the same task? If yes, the SFT data itself contains
   the pattern H1 ascribes to it — independent of any ARPO comparison.
4. **Per-step credit shaping.** Even if H1 doesn't hold globally, it still
   describes the regression cohort cleanly. Per-segment shaping ("did the
   screen change?", "did the next thought stop saying *I'm stuck*?") would
   target exactly the regression failure mode without affecting the 78 tasks
   where the two models are interchangeable.
5. **ARPO does generalise mildly better.** ARPO **gains** 11 tasks (FP) and
   **loses** 8 (PF). The non-trivial structural question is whether the +11
   share a property different from the −8: e.g. ARPO may be picking up tasks
   with unambiguous single-path solutions while losing the tasks whose only
   solution path is a detour — exactly the credit-assignment story, but as a
   net-positive trade rather than a net-negative one.

---

## Files

- `summary_table.csv` — per-rollout feature means by success/fail × SFT/ARPO
- `tasks_table.csv` — per-task: pass@1, pass@8, cohort, cross-rollout entropy
- `raw_metrics.json` — machine-readable hypothesis numbers
- `fig_success_rollouts.png` — **headline H1 chart**: features of successful rollouts, SFT vs ARPO
- `fig_paired_success.png` — same, restricted to tasks where both solve at least once
- `fig_paired_features.png` — 4-panel feature comparison by transition cohort (PP / PF / FP / FF)
- `fig_entropy.png` — H4: cross-rollout per-step action-fn entropy by cohort
- `fig_headroom.png` — H5: pass@8 − pass@1 distribution
- `fig_pass1_scatter.png` — per-task pass@1 ARPO vs SFT scatter (red = regression, blue = gain, green = both-pass, grey = both-fail)
- `fig_length.png` — H2: trajectory length by rollout outcome
- `fig_first3.png` — H3: opener-quality vs success rate
- `fig_transitions.png` — outcome transitions by paired task
