# Why is ARPO unstable on top of MCTS-SFT?
Tests three hypotheses on paired n=1, t=0 trajectories from
`sft_init/trajectories_at_0.jsonl` vs `arpo_step35/trajectories_at_35.jsonl`.

86 paired tasks. SFT pass = 44, ARPO pass = 47.

| transition | count | meaning |
|---|---|---|
| SFT pass → ARPO pass | 30 | retained |
| SFT pass → ARPO **fail** | 14 | **regressions** |
| SFT fail → ARPO pass | 17 | gains |
| SFT fail → ARPO fail | 25 | both stuck |

Net +3 tasks, but **31/86 (36 %) of outcomes flip**. That churn is what "unstable" means here.

---

## H1 (main): ARPO can't explore mid-trajectory — **SUPPORTED**

The strongest test is the 14 *regression* tasks (same task, SFT solves it, ARPO doesn't). Paired means:

| metric | SFT (passed) | ARPO (failed) | direction |
|---|---|---|---|
| unique action-fn types | **3.79** | 2.86 | ARPO uses 25 % fewer distinct fns |
| unique click clusters (~30 px) | **6.36** | 4.71 | ARPO clicks 26 % fewer distinct regions |
| self-correction phrases ("let me try X") | 0.86 | **1.71** | ARPO *says* it's stuck 2× more |
| recovery rate after duplicate / self-correction | **0.87** | 0.69 | ARPO actually changes action only 69 % of the time, vs SFT 87 % |
| back-and-forth A→B→A oscillations | 1.43 | **2.79** | ARPO bounces between 2 actions more |

The smoking gun is the gap between *self-correction language* and *recovery rate*. ARPO trajectories articulate "I'm stuck, let me try something else" **twice as often** as SFT, but then **fail to actually change** the action 31 % of the time (vs SFT's 13 %). The planner-prose stayed; the action-distribution diversity needed to *act* on it shrank. That is precisely what trajectory-level RL would do: it sharpens the conditional argmax without rewarding exploration that pays off mid-rollout.

Cohort-level numbers tell the same story but more diluted:

| metric | SFT pass | ARPO pass | SFT fail | ARPO fail |
|---|---|---|---|---|
| unique action-fn types | 3.73 | 3.68 | 3.10 | **2.67** |
| recovery rate | 0.89 | 0.90 | 0.65 | 0.63 |

When ARPO passes, it looks just like SFT. The exploration deficit shows up in **failures** (action-fn vocabulary collapses to 2.67) and especially in **regressions** above.

**MCTS-SFT side mechanism (your framing):** SFT trajectories were drawn from MCTS rollouts that succeeded *despite* taking detours — the data literally contains "wrong move → re-evaluate → different move → success" patterns, and SFT imitated that whole pattern. ARPO with sparse trajectory-level reward only credit-assigns the final outcome; the model never learns that *the corrective move is what made the success*, so the corrective behaviour decays first. The recover-rate column above is the direct measurement of that decay.

See `fig_paired_features.png` (Regression panel), `fig_recovery_rate.png`, `fig_unique_fns.png`.

---

## H2: ARPO wanders too long — **REFUTED**

| | SFT | ARPO |
|---|---|---|
| mean steps on success | 11.11 | **9.81** (faster) |
| mean steps on failure | 13.79 | 14.38 |
| mean steps on **same** failed tasks (paired both-fail, N=25) | 13.76 | 14.72 |

When ARPO solves a task it does so in fewer steps than SFT. When it fails, it uses the same ~14 steps SFT does on the same tasks. RL did not introduce wandering — wandering is what *failure* looks like for either model, and ARPO actually wins faster when it wins. See `fig_length.png`.

---

## H3: first few key steps are wrong — **REFUTED**

Marginal success rate as a function of first-3-step quality, pooling both runs:

| first-3-bad signals | success rate | N |
|---|---|---|
| 0 (clean opener) | 55.6 % | 27 |
| ≥1 (shaky opener) | 52.4 % | 145 |

Knowing the opener is clean barely moves the needle (~3 pp). On regressions, SFT and ARPO openers are equally shaky (1.43 vs 1.50 bad signals) yet outcomes are opposite. On gains, openers are equally shaky (1.24 vs 1.18) yet outcomes flip the other way. The *opener* is not the driver — mid-trajectory recovery is. See `fig_first3.png`.

---

## Other hypotheses worth ruling in / out next

These are not tested here but follow naturally from the H1 finding:

1. **Distributional sharpening / entropy collapse.** Action-fn vocabulary on failures dropped from 3.10 → 2.67. A direct entropy / unique-token check on the action-token logits at step 35 would confirm this is policy-side and not just sampling noise (n=1).
2. **Loss of credit on corrective sub-trajectories.** A natural fix is re-introducing per-step or per-segment shaping ("did the screen change?", "did the next thought stop saying *I'm stuck*?") — these would directly target the recover-rate gap.
3. **KL-anchor erosion.** If KL to SFT is too low or decays during training, the regression set is exactly where it would show up. A KL-coefficient sweep at fixed step 35 would quantify it.
4. **Sampling variance from n=1.** 36 % outcome churn at n=1 is partly stochastic. The pass@8 numbers being collected for `sft_init_n8_t1` / `arpo_step35_n8_t1` will tell us how much of "instability" is policy drift vs single-rollout noise. If the pass@8 gap shrinks markedly, the n=1 churn was variance — but the *recovery-rate gap* is independent of that and is the real behavioural signal.

---

## Files

- `summary_table.csv` — cohort-level metric means / medians
- `pairs_table.csv` — per-task paired SFT vs ARPO features
- `raw_metrics.json` — machine-readable hypothesis numbers
- `fig_paired_features.png` — **headline chart**: 6-panel paired SFT vs ARPO across regression / gain / both-pass / both-fail
- `fig_recovery_rate.png` — recovery rate by cohort (the H1 smoking gun)
- `fig_unique_fns.png`, `fig_unique_clusters.png` — action diversity
- `fig_self_correct.png`, `fig_bandf.png` — self-correction language and oscillation
- `fig_max_run.png` — loop tightness
- `fig_length.png` — H2 step-count distribution
- `fig_first3.png` — H3 opener-quality distribution
- `fig_transitions.png` — outcome transition counts
