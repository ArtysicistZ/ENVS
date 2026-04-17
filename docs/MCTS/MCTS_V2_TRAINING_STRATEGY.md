# MCTS v2 Training Strategy: Step-Level Tree Training

> Date: 2026-03-26
> Based on: v2 collection across 6 rounds (base_86, rerun_low35, rerun2_low23, rerun3a/b/c_low18)

## 1. Problem Statement

MCTS SFT v1 achieved 88-92% doable on trained tasks (n=8) but lost to naive SFT on untrained tasks (12-13 vs 18 doable, n=8) and on overall doable rate (87-89 vs 93). Two root causes:

**Root cause 1: Data imbalance.** Naive SFT uses difficulty-aware caps (hard: 15 trajectories, easy: 6) and diversity selection (greedy maximin). MCTS v1 had the opposite bias — easy tasks produced many trajectories (77), hard tasks few (1-5). Among the 26 trained tasks where naive SFT beat MCTS SFT, the average MCTS trajectory count was 16.2. Among the 40 where MCTS won, it was 27.6.

**Root cause 2: Prefix duplication.** v1 trained on full root-to-leaf trajectories. In a tree, many trajectories share prefixes — the root's steps appear in EVERY trajectory. A task with 50 successful leaves trains on the root steps 50×. This wastes gradient on already-learned early steps and under-represents the unique late-step decisions where branches actually diverge. The model over-learns opening moves and under-learns the hard decisions.

## 2. v2 Data Inventory

### 2.1 Collection Rounds

86 trainable tasks collected across 6 rounds. Hard tasks (<=15 successes) were repeatedly rerun with 112 VMs each round to build up data coverage.

| Round | Tasks | VMs/task | New Successes | Cumulative |
|-------|-------|----------|---------------|------------|
| base_86 | 86 | 112 | 2,385 | 2,385 |
| rerun_low35 | 35 (<=15 succ) | 112 | 254 | 2,639 |
| rerun2_low23 | 23 (<15 succ) | 112 | 94 | 2,733 |
| rerun3a_low18 | 18 (<15 succ) | 112 | 62 | 2,795 |
| rerun3b_low18 | 18 (<15 succ) | 112 | 65 | 2,860 |
| rerun3c_low18 | 18 (<15 succ) | 112 | 67 | 2,927 |

### 2.2 Combined Results (all rounds)

| Tier | Tasks | Successes | Runs/task | SR range |
|------|-------|-----------|-----------|----------|
| Zero (0) | 0 | 0 | — | — |
| Very Hard (1-5) | 4 | 6 | 672 | 0.1-0.6% |
| Hard (6-15) | 6 | 67 | 336-672 | 1.8-4.2% |
| Medium (16-40) | 51 | 1,247 | 112-672 | 3.9-39.3% |
| Easy (41-70) | 17 | 893 | 112-672 | 37.5-67.9% |
| Very Easy (>70) | 8 | 714 | 112 | 69.6-95.5% |
| **Total** | **86** | **2,927** | — | — |

All 86 tasks now have at least 1 success. The 4 Very Hard tasks (1-5 successes after 672 runs each) represent the true difficulty floor for this model.

SR_t is computed per task as n_success / n_total_runs (varies: 112 for base-only tasks, up to 672 for tasks in all 6 rounds).

### 2.3 Trainable Steps (node-level)

With step-level tree training (Section 4.1), each node's own_steps are trained exactly once:

| | Nodes | own_steps |
|---|---|---|
| Successful leaves | 2,927 | — |
| Internal nodes on success paths | — | — |
| **Total on success paths** | **3,965** | **31,165** |
| **After ~12% REMOVE mask** | — | **~27,400** |

Each tree node averages ~7.9 steps. The internal steps are shared prefixes that would have been duplicated N times under trajectory-level training — now each is trained exactly once.

## 3. Literature Insights

### 3.1 DART-Math Prop2Diff (NeurIPS 2024)

**Key finding:** Allocating training examples proportional to task difficulty (+6.9 pts over vanilla). Vanilla rejection sampling severely under-represents hard tasks (51.1% of hardest queries get zero examples).

**Implication:** Motivates difficulty-aware weighting. However, linear Prop2Diff `w = (1 - SR)` can be too aggressive — see Section 4.1 for our power-scaled variant.

### 3.2 WebSTAR Step-Level Masking (2512.10962)

**Key finding:** Step-level filtering (mask steps scoring <=5) achieves 39.6% vs trajectory-level 29.9% — nearly 10 points improvement with half the data. Error steps remain as context but contribute zero loss.

**Implication:** Our agent-audited step masks (KEEP/REMOVE) should be applied to v2 data. Step masks are applied per node's own_steps, so each mask decision is made once.

### 3.3 STeP Partial Masking (2505.20023)

**Key finding:** Partial masking (d_i = 0 for error steps) prevents learning incorrect reasoning while maintaining context. Removing partial masking hurts seen-task performance.

**Implication:** Confirms our KEEP/REMOVE masking approach. V2 SFT should use the same binary masks.

### 3.4 GUI-Reflection Three-Phase Training (2506.08012)

**Key finding:** Separating error recognition from error correction in training. Models trained on synthetic error-recovery data improve 14.6% -> 34.7% on AndroidWorld through online self-correction.

**Implication:** Our revision trajectories (Agent-R style) teach error recovery — the failed prefix is context, the successful correction is the loss target. This is GUI-Reflection's "reattempt" phase applied to MCTS tree data.

### 3.5 Agent-R Revision Trajectories (2501.11425)

**Key finding:** Splicing failed prefix with successful sibling continuation at tree branch points teaches self-correction. 4 candidates per expansion, iterative self-training with progressive difficulty. Uses SFT loss with eta=0.2 mixing ratio for revision vs general data.

**Implication:** Our 983 revision trajectories from v2 trees can be mixed into SFT training at low ratio to teach error recovery.

### 3.6 Key Principle: Never SFT on Wrong Actions

All papers agree: SFT maximizes log pi(action | state). Training on wrong actions teaches the model to reproduce them. The only safe ways to use negative data:
- **As context** (zero loss, visible in prompt history) — WebSTAR, STeP, our KEEP/REMOVE masks
- **As DPO rejected** (negative gradient signal) — Tree-GRPO, AlphaLLM-CPL
- **As revision prefix** (context for recovery, loss only on correction) — Agent-R, GUI-Reflection

## 4. Training Design: Step-Level Tree SFT

### 4.1 Core Idea: Train on Nodes, Not Trajectories

**Principle: Each step in the tree is learned exactly once.**

v1 trained on full root-to-leaf trajectories. This causes prefix duplication — shared parent steps get trained N times (once per descendant leaf). v2 trains on **tree nodes** directly:

For each node on a successful path (Q > 0 or ancestor of a successful leaf):
- **Context (loss=0):** full prefix reconstructed from root -> parent chain
- **Loss target:** this node's `own_steps` only, with KEEP/REMOVE masks applied
- **Weight:** power-scaled difficulty weighting (see 4.2)

```
Tree structure:

    root (steps 1-3)         <- trained ONCE, not 50x
       /    |    \
     A       B       C       <- each trained once
    / \     / \     / \
  A1  A2  B1  B2  C1  C2    <- each trained once (leaf own_steps only)
```

This is directly supported by the tree_io.py serialization — each node stores `own_steps` (unique steps after branching from parent) and the prefix is reconstructed via `reconstruct_steps()`.

**Training example for node N:**
```python
# Context: ancestor chain (loss=0)
prefix_steps = reconstruct_steps(tree_data, parent_id)
# Loss target: this node's unique steps
loss_steps = node["own_steps"]  # with KEEP/REMOVE masks applied
# Full conversation = prefix_steps + loss_steps
# Loss mask: [0]*len(prefix_steps) + [mask]*len(loss_steps)
```

**What this achieves:**
1. **No duplication:** root steps trained 1x, not Nx
2. **Gradient on decisions, not context:** the model learns branch-point actions (where uncertainty exists) rather than re-learning the common opening
3. **Natural deduplication of tree structure:** the tree already stores each step once; now training matches storage

### 4.2 Power-Scaled Difficulty Weighting

The weight for each training example (one node):

```
w(task t, node n) = (1 - SR_t)^beta / L_n
```

Where:
- `SR_t` = task success rate = n_success_t / n_total_runs_t (varies per task: 112 for base-only, 224 for rerun tasks)
- `L_n` = number of KEEP steps in node n's own_steps
- `beta` = power parameter (tunable, start with 0.5)
- Normalize so all weights average to 1.0 (preserves LR calibration)

**Why power scaling instead of linear Prop2Diff?**

| Weighting | VeryHard (SR~2%) | Hard (SR~10%) | Medium (SR~27%) | Easy (SR~55%) | VeryEasy (SR~80%) |
|-----------|------------------|---------------|------------------|---------------|-------------------|
| Linear (beta=1.0) | w=0.98 | w=0.90 | w=0.73 | w=0.45 | w=0.20 |
| sqrt (beta=0.5) | w=0.99 | w=0.95 | w=0.85 | w=0.67 | w=0.45 |
| Task-uniform (beta=0) | w=1.00 | w=1.00 | w=1.00 | w=1.00 | w=1.00 |

Linear Prop2Diff (beta=1.0) suppresses VeryEasy to 1.1% of total gradient — almost zero signal from tasks that teach basic GUI interaction. It also gives VeryHard 23.2% of gradient, amplifying noise from the few (1-5) lucky successes on those tasks.

Power scaling with beta=0.5 is gentler: VeryEasy still gets meaningful gradient (foundational skills are worth learning), while Hard/VeryHard get moderate upweighting. The naive SFT distribution (which achieved 93 doable) was quite balanced:

| Tier | Naive SFT | Prop2Diff (beta=1) | sqrt (beta=0.5) | Task-uniform (beta=0) |
|------|-----------|-------------------|-----------------|----------------------|
| VeryHard | 6.7% | 23.2% | ~19% | 17.5% |
| Hard | 27.6% | 27.9% | ~25% | 22.5% |
| Medium | 42.7% | 35.9% | ~37% | 35.0% |
| Easy | 18.3% | 11.8% | ~14% | 18.8% |
| VeryEasy | 4.6% | 1.1% | ~5% | 6.2% |

beta=0.5 produces a distribution closer to naive SFT's proven distribution than beta=1.0 does. The main improvement over naive SFT should come from **3x more data** and **step masking**, not from aggressive weighting.

**Recommendation:** Start with beta=0.5. If trained-task performance is too low, increase toward 1.0. If untrained-task generalization is poor, decrease toward 0.

### 4.3 Agent-Audited Step Masks

**Scale:** 4,401 nodes on successful paths, 35,548 own_steps to classify. 200 parallel subagents, ~22 nodes / ~178 steps each.

**Audit format per node:**
- Agent sees: task instruction + ancestor steps (read-only context) + this node's own_steps (to label)
- Agent outputs: KEEP or REMOVE for each own_step
- REMOVE steps: loss=0 but remain as context in the conversation

**Classification rules for each step:**

KEEP (loss=1) — train on this step:
- **Correct productive step:** The agent does the right thing. The action moves toward completing the task.
- **Error-recognition step:** The agent's Thought explicitly recognizes a previous mistake and plans recovery. Even though it references an error, this teaches self-correction. Example: "I clicked Colors menu but that's wrong, I need Image menu instead."
- **Neutral observation step:** The agent describes what it sees and plans next. Shows correct situational awareness.

REMOVE (loss=0) — do NOT train on this step:
- **Real error step:** The agent takes an action that is objectively wrong for the task. The Thought may sound confident, but the action leads to the wrong place. Examples:
  - Clicking the wrong menu when trying to find a setting
  - Opening the wrong application or dialog
  - Typing the wrong command or entering wrong values
  - Navigating to an irrelevant part of the UI
  - Repeating the same failed action with no new reasoning

**The critical distinction:** Does the Thought show AWARENESS that something went wrong?
- Yes -> KEEP (error-recognition, teaches recovery)
- No, agent confidently does wrong thing -> REMOVE (real error, teaches bad behavior)

**When in doubt:** Default to KEEP. Better to train on a slightly ambiguous step than miss a valuable error-recovery step. Only REMOVE when the step is a clear mistake with no self-awareness.

**Concrete example (from v1 doc):**
- Step 5 (REMOVE): "I need to adjust palette settings, so I'll click on the Colors menu." -> Confidently wrong.
- Step 6 (KEEP): "I clicked Colors menu, but that's not what I need. Palette settings should be under Image menu." -> Diagnoses mistake, corrects course.

**Expected outcome:** ~10-12% REMOVE rate based on v1 audit statistics. ~31,300 KEEP steps for training.

**Batch files:** `combined_all/audit_batches/v2_audit_000.json` through `v2_audit_199.json`
**Output:** `combined_all/step_masks_v2.json` mapping `{task_id}:{round}:{node_id}` -> `[1, 0, 1, 1, ...]`

### 4.4 Revision Trajectory Mixing (optional)

Mix ~150 revision trajectories from hard tasks at ratio eta=0.15:
- Shared parent prefix as context (zero loss)
- Failed child's action as context (zero loss)
- Revision signal + successful child's continuation as loss target
- Teaches error recovery — a skill naive SFT cannot learn

Already implemented in `tree_io.py:get_revision_trajectories()`.

### 4.5 Training Config

- Model: ByteDance-Seed/UI-TARS-1.5-7B (base)
- LR: 5e-6 (naive SFT sweet spot)
- Epochs: 1
- Effective batch: 32
- Cosine schedule, 3% warmup
- FSDP full-shard, bf16, gradient checkpointing

### 4.6 DPO: Skip for Now

2,550 sibling pairs are available, but DPO is unlikely to help at this stage:

1. **Rejected actions were minority clusters** — the model already assigns them lower probability. DPO reinforces a preference the model already has.
2. **Main gap is untrained tasks** (12-13 vs 18 doable). DPO on trained-task sibling pairs provides zero signal for untrained tasks.
3. **Risk of mode collapse** — DPO can make the model overly conservative, avoiding exploration that sometimes leads to success.
4. **Agent-R used SFT (not DPO)** on revision trajectories and got strong results.

**If v2 SFT still loses on trained tasks**, DPO is the next lever. The sibling pairs are saved in the trees.

## 5. Why This Should Beat Naive SFT

| Factor | Naive SFT | v1 MCTS SFT | v2 MCTS SFT |
|--------|-----------|-------------|-------------|
| Training unit | Trajectory | Trajectory | **Node (step-level)** |
| Training data | 715 trajectories (capped) | 571 resampled | **All nodes on successful paths** |
| Prefix duplication | N/A (independent trajs) | Root trained Nx | **Each step trained 1x** |
| Hard task handling | Caps (15 vs 6) | alpha=0.3 resample | **(1-SR)^0.5 weighting** |
| Step quality | No filtering | KEEP/REMOVE masks | **KEEP/REMOVE masks** |
| Error recovery | Not taught | Not taught | **Revision trajectories** |
| Diversity source | Greedy maximin | Random (few branches) | **MCTS branching (112 VMs)** |

The key improvements over v1:
1. **No prefix duplication** — gradient goes to unique decisions, not repeated openings
2. **Proper difficulty weighting** — power-scaled (1-SR)^beta matches proven distributions
3. **3x+ more data** — v2 collection with 112 VMs per task (vs 80) plus rerun of hard tasks
4. **Revision trajectories** — teaches error recovery from tree branch failures

The key improvements over naive SFT:
1. **Step masking** — agent-audited KEEP/REMOVE removes error steps (v1's biggest win: +38pts on trained tasks)
2. **More raw data** — all successful nodes, not capped to 6-15 per task
3. **Tree diversity** — MCTS branching naturally explores diverse strategies (vs greedy maximin on independent rollouts)
4. **Error recovery** — revision trajectories teach self-correction, impossible without tree structure

## 6. What NOT to Do

1. **Do NOT SFT on failed trajectories** — teaches wrong actions
2. **Do NOT train on full root-to-leaf trajectories** — causes prefix duplication; train on nodes
3. **Do NOT use Q-weighted SFT** — Q values are mostly binary (0 or 1) at leaf level. Only 31% of internal nodes have mixed Q. Not useful for step-level weighting.
4. **Do NOT cap easy-task trajectories** — train on ALL successful nodes, use weighting instead
5. **Do NOT use linear Prop2Diff (beta=1)** — too aggressive, suppresses easy tasks to near-zero. Use power-scaled beta=0.5 as starting point.
6. **Do NOT include zero-success tasks** — no positive signal available

## 7. Implementation Plan

1. **Identify nodes on successful paths** from v2 trees (Q > 0 or ancestor of successful leaf)
2. **Audit step masks** for each node's own_steps (reuse v1 masks where possible, fresh audit for new nodes)
3. **Build training dataset** — one example per node: prefix (loss=0) + own_steps (with masks)
4. **Compute per-node weights** using `(1 - SR_t)^0.5 / L_n`, normalize to mean 1.0
5. **Extract revision trajectories** from trees for hard tasks
6. **Train** with step masks + difficulty weighting + revision mixing
7. **Evaluate** n=1 and n=8 on 300 tasks
8. **Compare against:** naive SFT (93/300 doable n=8) and MCTS SFT v1 (87-89/300 doable n=8)

If beta=0.5 underperforms on trained tasks, try beta=0.7 or 1.0.
If beta=0.5 underperforms on untrained tasks, try beta=0.3 or 0.

## 8. References

| Paper | Key Contribution for Our Approach |
|-------|----------------------------------|
| DART-Math (NeurIPS 2024) | Difficulty-proportional allocation beats uniform |
| WebSTAR (2512.10962) | Step-level masking >> trajectory-level, +10 pts |
| STeP (2505.20023) | Partial masking: d_i=0 for errors, keep as context |
| GUI-Reflection (2506.08012) | Three-phase error recovery training |
| Agent-R (2501.11425) | Revision trajectories from MCTS tree siblings |
| Tree-GRPO (2509.21240) | Intra-tree advantages = step-level DPO |
| AlphaLLM-CPL (2410.06508) | Curriculum DPO from MCTS sibling pairs |
