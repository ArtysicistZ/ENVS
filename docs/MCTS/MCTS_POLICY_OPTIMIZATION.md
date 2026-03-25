# MCTS Policy Optimization: Full-Tree Training for GUI Agents

> Design document for the next iteration of MCTS-based training.
> Date: 2026-03-25

## Table of Contents

1. [Motivation: Why MCTS SFT v1 Falls Short](#1-motivation)
2. [Root Cause Analysis: The Hard-Task Data Gap](#2-root-cause-analysis)
3. [Literature: Using Tree Structure for Training](#3-literature)
4. [Proposed Approach: Full-Tree MCTS Collection + Training](#4-proposed-approach)
5. [Implementation Plan](#5-implementation-plan)
6. [Expected Impact](#6-expected-impact)
7. [References](#7-references)

---

## 1. Motivation: Why MCTS SFT v1 Falls Short

### 1.1 Current Results

MCTS SFT v1 (lr=3e-6, step-masked, task-balanced resampling) achieves strong per-attempt success rate but loses to naive SFT (lr=5e-6) on overall doable rate at n=8:

| Metric (n=8, 300 tasks) | Naive SFT | MCTS SFT |
|--------------------------|-----------|----------|
| Overall doable | **93 (31.0%)** | 87 (29.0%) |
| Trained doable (84 tasks) | 75 (89.3%) | 74 (88.1%) |
| Untrained doable (216 tasks) | **18 (8.3%)** | 13 (6.0%) |
| Trained success rate | 42.1% | **48.8%** |

MCTS SFT is more reliable per attempt (+6.7pp success rate on trained tasks) but solves fewer total tasks. The gap comes from **untrained tasks** (18 vs 13) and **hard trained tasks with sparse MCTS data**.

### 1.2 The Paradox

MCTS SFT has 2.16× more training data (15,920 vs 7,361 examples), uses step-level quality filtering (12% error steps removed), and employs task-balanced resampling. Despite all this, it cannot beat a simpler method. Why?

---

## 2. Root Cause Analysis: The Hard-Task Data Gap

### 2.1 Naive SFT's Difficulty-Aware Selection

The naive SFT training pipeline uses `scripts/select_sft_trajectories.py` which deliberately over-represents hard tasks:

| Difficulty Tier | Base Model Success Rate | Trajectory Cap per Task |
|----------------|------------------------|------------------------|
| Easy | > 50% | 6 |
| Medium | 25–50% | 8 |
| Hard | 12.5–25% | 10 |
| Very Hard | < 12.5% | **15** |

Hard tasks get **2.5× more trajectories** than easy tasks. This is Prop2Diff (DART-Math, NeurIPS 2024) applied at the collection level — allocate more data where the model struggles most.

### 2.2 MCTS Collection's Inverse Bias

MCTS tree search naturally produces fewer successful trajectories for hard tasks:

| MCTS Trajectory Count | Number of Tasks | Avg Base Model SR | Tier |
|----------------------|-----------------|-------------------|------|
| 1–5 | 12 tasks | 8.3% | V.Hard |
| 6–10 | 14 tasks | 6.4% | V.Hard |
| 11–20 | 26 tasks | 14.2% | Hard–V.Hard |
| 21–50 | 21 tasks | 31.4% | Medium–Hard |
| 51+ | 11 tasks | 71.6% | Easy |

The hardest tasks (base SR < 12.5%) produce only 1–10 MCTS trajectories, while easy tasks produce 50–77. This is the exact opposite of what we want.

### 2.3 Per-Task Correlation

Among 84 trained tasks evaluated at n=8:

| Winner | Count | Avg MCTS Trajectories |
|--------|-------|-----------------------|
| MCTS SFT wins | 40 | **27.6** |
| Naive SFT wins | 26 | **16.2** |
| Tie | 18 | 18.2 |

MCTS SFT wins wherever it has sufficient data. It loses on hard tasks purely because the data is scarce.

**The 12 critical tasks where naive beats MCTS and MCTS has ≤10 trajectories:**

| Task ID | MCTS Trajs | Naive Cap | Naive SR | MCTS SR | Diff |
|---------|-----------|-----------|----------|---------|------|
| 2c9fc0de | 3 | 15 | 50.0% | 12.5% | -37.5 |
| f201fbc3 | 6 | 15 | 50.0% | 12.5% | -37.5 |
| 276cc624 | 9 | 15 | 75.0% | 50.0% | -25.0 |
| 9cf05d24 | 4 | 15 | 37.5% | 12.5% | -25.0 |
| dbbf4b99 | 5 | 15 | 37.5% | 12.5% | -25.0 |
| 37887e8c | 6 | 10 | 25.0% | 0.0% | -25.0 |
| cb130f0d | 8 | 15 | 37.5% | 12.5% | -25.0 |
| 9bc3cc16 | 8 | 15 | 25.0% | 0.0% | -25.0 |
| 2b9493d7 | 3 | 15 | 12.5% | 0.0% | -12.5 |
| a462a795 | 3 | 15 | 62.5% | 50.0% | -12.5 |
| 58d3eeeb | 4 | 15 | 25.0% | 12.5% | -12.5 |
| ac1b39ff | 5 | 10 | 12.5% | 0.0% | -12.5 |

Every single one is a hard task where MCTS had 3–9 trajectories but naive had 10–15. The method isn't failing — the data is.

### 2.4 The Untrained Task Gap

Naive SFT solves 18 untrained tasks vs MCTS SFT's 13. Two factors:
1. Naive SFT's hard-task over-representation teaches more generalizable GUI skills (hard tasks require novel interaction patterns that transfer to new tasks)
2. MCTS SFT's resampling upsamples rare tasks but can't create data that doesn't exist — the gradient signal for hard tasks is fundamentally weaker

### 2.5 What We're Throwing Away

Currently, the MCTS orchestrator runs tree search on 80 VMs per task, producing a tree with dozens of nodes — but saves **only the successful leaf trajectories**. For a hard task:
- Tree has ~30 nodes (using 30 of 80 VMs)
- 3 nodes succeed (eval_score=1.0)
- 27 nodes fail (eval_score=0.0)
- We save the 3 successes, discard the 27 failures
- Training signal: 3 SFT examples

With the full tree, we would have:
- 3 successful trajectories (SFT signal)
- 27 failed trajectories (negative signal via DPO)
- ~10 branch points with sibling comparisons (step-level credit assignment)
- ~10 revision trajectories (error recovery training)
- Training signal: 3 SFT + 10 DPO pairs + 10 revision examples = **23× more data for the same compute**

**Hard tasks produce MORE tree structure, not less.** The tree explores extensively when the task is hard — exactly where we need the most training signal. Currently we throw this away.

---

## 3. Literature: Using Tree Structure for Training

### 3.1 Tree-GRPO (Ji et al., 2509.21240, ICLR 2026)

**Key insight:** Tree-structured rollouts enable step-level process supervision from outcome-only rewards.

**Mechanism:** At each branch point in the tree, sibling branches that share a common prefix diverge at one step. By comparing their outcomes:
- The successful branch's divergence action gets positive advantage
- The failed branch's divergence action gets negative advantage
- This is proven equivalent to **step-level DPO** — no process reward model needed

**Intra-tree advantage:** Within one tree, comparing siblings at each depth provides step-level credit. Siblings share the same state (same prefix) so the comparison is grounded — the ONLY difference is the action taken at the branch point.

**Inter-tree advantage:** Across different tree rollouts for the same task, comparing full trajectories provides trajectory-level signal (standard GRPO).

**Efficiency:** Tree search with shared prefixes achieves **4× more rollouts per fixed token budget** compared to independent chain sampling. The prefix is generated once, then branches explore different continuations.

**Results:** Outperforms chain-based GRPO on multi-step agent tasks with the same compute budget. The advantage grows with task horizon — longer tasks benefit more from tree structure.

### 3.2 ReST-MCTS* (Zhang et al., 2406.03816, NeurIPS 2024)

**Key insight:** MCTS tree search can automatically infer step-level process rewards without human annotation.

**Quality value formula:**
```
v_k = max(v_{k-1} + w_{s_k}, 0)

w_{s_k} = (1 - v_{k-1}) / (m_k + 1) × (1 - 2 × r_{s_k})
```

Where:
- `r_{s_k}` = process reward model score for step k
- `m_k` = minimum remaining steps to reach correct answer
- `v_k` = cumulative quality value (bounded in [0, 1])

**Self-training loop:**
1. Run MCTS* search with current PRM guidance → collect reasoning traces
2. Verify final answers → label traces as correct/incorrect
3. Train policy model on correct traces (SFT)
4. Extract (partial_solution, v_k) pairs from tree nodes → train PRM
5. Repeat with improved policy + PRM

**Cold-start:** Initial PRM trained on synthetic data (correct solutions from training set + incorrect continuations from weak LLM). No human step-level annotation needed.

**Relevance:** Shows that the tree search itself provides sufficient signal to train a process reward model. We can use this to bootstrap step-level rewards for GUI agent tasks from binary task completion rewards.

### 3.3 Agent-R (Yuan et al., 2501.11425)

**Key insight:** Construct revision trajectories by splicing failed prefixes with successful sibling continuations from the MCTS tree.

**Revision trajectory construction:**
1. From MCTS tree, take a failed trajectory τ_bad
2. Use the actor model to identify the **first error step** t' in τ_bad
3. Find a successful sibling trajectory τ_good that shares the same parent node (diverges at the same point)
4. Construct: `τ_revision = (τ_bad[0:t'], revision_signal, τ_good[t'+1:])`
5. The revision signal is a randomly sampled correction thought (e.g., "The previous approach didn't work because...")

**Training objective (SFT):**
```
L(θ) = η · { E[log π_θ(τ_good | u)] + E[log π_θ(revision_signal, τ_good[t'>t'] | u, τ_bad[t≤t'])] }
      + (1-η) · E[log π_θ(y | x)]
```
With η=0.2, mixing good trajectories, revision trajectories, and general capability data.

**MCTS details:** 4 candidate actions per expansion, k=8 rollouts for Monte Carlo estimation, tree depth d=20, UCT exploration constant c=0.25.

**Iterative loop:** Three iterations. Each iteration re-collects revision trajectories with the updated model, progressively increasing the reward threshold α (0.5 → 0.7 → 1.0) to converge toward optimal trajectories.

**Results:** 63.9% on WebShop, 70.2% on ScienceWorld, 78.0% on TextCraft — outperforming GPT-4o and expert-trajectory-trained agents.

**Direct relevance:** Our MCTS tree already has the exact structure needed for Agent-R's revision trajectories. At each branch point, successful and failed siblings share the same parent state. We just need to save the tree instead of discarding it.

### 3.4 AlphaLLM-CPL (Zhang et al., 2410.06508, AAAI 2025)

**Key insight:** Extract preference pairs from MCTS tree siblings and train with curriculum DPO.

**Sibling pair extraction:** For any parent node with children, extract (winner, loser) pairs where the Q-value gap exceeds threshold τ. Both stepwise pairs (single-step divergence) and trajectory pairs (full path comparison) are used.

**Curriculum learning:** Training pairs are ordered by difficulty:
- **Preference reward gap** (static): Large gap = easy example (train first)
- **Policy prediction gap** (dynamic): Re-ranked each epoch based on current model's confidence
- Combined with balance rate α for progressive training

**Results:** LLaMA2-7B: 14.6% → 36.5% on GSM8K. Mistral-7B: 38.5% → 57.3%. Self-improvement without additional annotations.

---

## 4. Proposed Approach: Full-Tree MCTS Collection + Training

### 4.1 Overview

Re-collect MCTS trajectories for all 86 tasks, but this time **save the full tree** — every node (successful AND failed), parent-child relationships, eval scores, and back-propagated Q-values. Then extract training signals in a carefully staged pipeline:

**Stage 1 — Cold-Start SFT (safe, no training on failures):**
1. **Q-weighted SFT** on successful trajectories only (improved step weighting from tree confidence)
2. **Revision trajectory SFT** from failed→successful splices (Agent-R style — failed prefix as CONTEXT only, loss on recovery action only)

**Stage 2 — Preference Refinement (after SFT warm-start):**
3. **Step-level DPO** from sibling comparisons at branch points (Tree-GRPO style — requires a competent base model to learn from negative signal)

> **Critical design principle:** SFT optimizes `log π(action | state)` — it trains the model to REPRODUCE whatever action is in the loss. Failed actions must NEVER enter the SFT loss. They can appear as conversation context (visible but not optimized) or as rejected examples in DPO (negative signal). This is the same principle behind our step-masking audit in v1: error steps as context, not as training targets.

### 4.2 Full-Tree MCTS Collection

**What changes from current collection:**

| Aspect | Current (v1) | Proposed (v2) |
|--------|-------------|---------------|
| Nodes saved | Successful leaves only | ALL nodes (success + failure) |
| Eval | Only terminal nodes | ALL leaves (terminal + stuck + timeout) |
| Tree structure | Discarded | Saved (parent-child, branch metadata) |
| Q-values | Not computed | Back-propagated from leaf eval scores |
| Output format | `mcts_success.jsonl` (flat) | `mcts_trees/{task_id}.json` (tree) |
| Output size | ~3GB (screenshots in successful trajs) | ~10-15GB (all nodes with screenshots) |

**The collection process itself is unchanged** — same 80 VMs per task, same K=32 probing, same fingerprint-based branching logic. The only difference is what we save afterward.

**New: Evaluate all leaf nodes.** Currently only nodes that reach `finished()` or `fail()` get evaluated. Nodes that are `stuck` or hit `max_steps` are marked done but not evaluated. In v2, we evaluate ALL leaf nodes — a stuck node might have partially completed the task (eval_score > 0 but < 1.0 for partial credit tasks, or simply 0.0 for binary tasks). This gives more signal for Q-value computation.

### 4.3 Q-Value Back-Propagation

After all leaves are evaluated, back-propagate Q-values through the tree:

```
Q(leaf) = eval_score              # 1.0 for success, 0.0 for failure
Q(internal_node) = mean(Q(child) for child in node.children)
```

This gives every node in the tree a Q-value in [0, 1] representing the fraction of its descendants that succeeded. Properties:
- Root Q = overall success rate for this task
- Nodes on high-success branches have high Q
- Nodes on failed-only branches have Q = 0
- Branch points where SOME children succeed and others fail have intermediate Q — these are the most informative nodes

### 4.4 Training Signal Extraction

#### Signal 1: Q-Weighted SFT (replaces binary step masks)

For each successful trajectory, weight each step by the Q-value of the tree node at that step:

```
loss_weight(step_k) = Q(node_at_step_k) × mask(step_k)
```

Where `mask` is the existing KEEP/REMOVE binary mask from agent auditing.

**Effect:** Steps on high-Q paths (many siblings also succeeded) get full weight. Steps on marginal paths (barely succeeded, most siblings failed) get reduced weight. This is softer than binary masking and directly uses tree confidence.

**Addressing the hard-task gap:** Easy tasks with Q ≈ 0.9 everywhere get near-uniform weights (no change). Hard tasks with Q ≈ 0.1 get lower weights overall — but the revision trajectories (below) compensate with additional SFT-safe examples that are abundant for hard tasks. The DPO signal in Stage 2 further sharpens decision-making.

**No failed trajectories in the loss:** Q-weighted SFT only applies to successful trajectories. Failed branches are NOT trained on — they only contribute to Q-value computation (affecting weights on successful trajectories) and to DPO/revision data extraction.

#### Signal 2: Revision Trajectories (Agent-R Style) — Stage 1 SFT

For each failed leaf node, find the nearest successful sibling (shares the same parent) and construct a revision trajectory:

```
τ_revision = failed_prefix[0:branch_step] + "The previous approach was wrong because..." + successful_suffix[branch_step:]
```

**Construction:**
1. For each failed leaf L_fail:
   - Walk up to its parent P
   - Find a sibling L_success with eval_score > 0 under P (or under P's parent, etc.)
   - The revision point = the step where L_fail and L_success diverge
   - Failed prefix = L_fail's actions up to the revision point
   - Revision signal = a templated correction thought
   - Successful suffix = L_success's actions from the revision point onward
2. In SFT loss: the failed prefix is **CONTEXT only** (labels masked to -100). The model sees the errors but only optimizes on the revision signal + successful suffix. No failed action enters the loss.

**Why this is SFT-safe:** The failed prefix appears in the conversation history (like our current REMOVE steps that stay visible as context). The loss is computed ONLY on the revision thought + successful continuation. The model learns "given that I just made this mistake (visible in context), here's how to diagnose and recover" — the exact same pattern our step-masking audit identified as the most valuable training signal.

**Why this fixes the hard-task gap:** A hard task with 3 successes and 20 failures, with ~10 branch points where siblings diverge, produces ~10 revision trajectories. Combined with 3 Q-weighted SFT examples: **13 SFT-safe training examples instead of 3** — a 4.3× increase in hard-task data with zero additional VM cost.

**Why this helps generalization:** Revision trajectories teach a universal skill — error recognition and recovery — not just task-specific correct actions. The recovery pattern ("I clicked the wrong menu, let me try the correct one") transfers to untrained tasks. Our step-masking audit already showed that error-recognition steps are among the most valuable training signals.

#### Signal 3: Step-Level DPO from Sibling Branches (Tree-GRPO) — Stage 2 Only

> **This signal is ONLY used in Stage 2 (after SFT warm-start).** DPO requires a competent base model to learn meaningful preferences. Applying DPO directly to the base model without SFT warm-start risks catastrophic forgetting or degenerate behavior.

At each branch point in the tree where siblings have different Q-values:

```
chosen = trajectory through high-Q child
rejected = trajectory through low-Q child
```

The chosen and rejected trajectories share an identical prefix (same state up to the branch point) and diverge at exactly one step. This is the purest form of step-level preference — the ONLY difference is the action taken at the branch point.

**Construction:**
1. For each internal node with ≥2 children:
   - Sort children by Q-value
   - Take (highest Q child, lowest Q child) as a DPO pair
   - The "prompt" = conversation history up to the branch point
   - chosen = high-Q child's action + continuation
   - rejected = low-Q child's action + continuation
2. Filter: only use pairs where Q difference > threshold (e.g., 0.3)

**Mathematical justification (Tree-GRPO):** The intra-tree sibling comparison is equivalent to step-level DPO:

```
L_DPO = -log σ(β · (log π(a_chosen | s) - log π(a_rejected | s)))
```

Where `s` is the state at the branch point, `a_chosen` and `a_rejected` are the actions taken by the successful and failed children. This gives direct policy gradient at the exact step where the decision mattered.

**Why Stage 2, not Stage 1:** DPO trains the model to PREFER chosen over rejected. The rejected trajectory enters as a negative signal — pushing probability mass AWAY from the wrong action. This is fundamentally different from SFT (which only has positive signal). A model needs to first know WHAT to do (from SFT) before it can learn WHAT NOT to do (from DPO). Applying DPO to an untrained model teaches it to avoid random actions without providing a good alternative.

### 4.5 Training Pipeline

**Stage 1: Q-Weighted SFT + Revision Trajectories** (1 epoch, cold-start safe)
- Q-weighted SFT on **successful trajectories only** — same as current MCTS SFT but with continuous Q-value weights instead of binary masks
- Revision trajectories mixed in at η=0.2 ratio — failed prefix as context (not in loss), recovery action as training target
- Step-level auditing masks still applied on top of Q-weights (REMOVE steps = 0 regardless of Q)
- Task-balanced resampling (α=0.3) with UniMax capping (K=3) — same as current
- **No failed actions in the loss. No DPO. Pure positive-signal SFT.**
- This produces a warm-start model that knows WHAT to do + HOW to recover from errors

**Stage 2: Step-Level DPO** (1 epoch, preference refinement)
- Fine-tune the Stage 1 model with DPO on sibling pairs from tree branch points
- Use β=0.1 (standard DPO temperature)
- 1 epoch over all DPO pairs (filtered by Q-value gap > 0.3)
- This sharpens the model's decision-making at critical branch points — teaching WHAT NOT to do
- **This is the only stage where failed actions influence training**, and they enter as negative signal (push probability mass away), not positive signal

**Stage 3: Evaluation**
- n=1 on 300 tasks (quick sanity check)
- n=8 on 300 tasks (definitive comparison)
- Compare trained/untrained split against all baselines
- Ablation: Stage 1 only vs Stage 1 + Stage 2 to measure DPO's marginal contribution

### 4.6 Why This Should Outperform Naive SFT

| Advantage | Mechanism | Stage |
|-----------|-----------|-------|
| **Hard-task data ×4** | Revision trajectories from failed→successful splices give 4× more SFT-safe examples for hard tasks | Stage 1 (SFT) |
| **Error recovery skill** | Revision trajectories teach generalizable recovery patterns that transfer to untrained tasks | Stage 1 (SFT) |
| **Data quality** | Q-weighted SFT uses tree confidence as continuous quality signal — better than binary masks | Stage 1 (SFT) |
| **Step-level credit** | Sibling DPO pinpoints EXACTLY which action mattered — no agent auditing or PRM needed | Stage 2 (DPO) |
| **Same compute** | Full tree is already generated during collection. We just save it instead of discarding it. Zero additional VM cost. | Collection |

**How the hard-task gap gets fixed (SFT-safe, Stage 1 only):**

| Task Difficulty | Current v1 SFT Signal | v2 Stage 1 SFT Signal |
|----------------|----------------------|----------------------|
| V.Hard (3 successes, 20 failures) | 3 SFT examples | 3 Q-weighted SFT + ~10 revision trajectories = **13 examples** |
| Hard (8 successes, 15 failures) | 8 SFT examples | 8 Q-weighted SFT + ~8 revision trajectories = **16 examples** |
| Easy (50 successes, 5 failures) | 50 SFT examples | 50 Q-weighted SFT + ~3 revision trajectories = **53 examples** |

The data imbalance inverts: hard tasks go from 3 examples to 13 (4.3× increase), while easy tasks barely change (50 → 53, 1.06×). **Revision trajectories are naturally difficulty-proportional** — harder tasks produce more failures, which produce more revision opportunities. This is Prop2Diff (DART-Math) emerging automatically from the tree structure.

The naive SFT's advantage was hard-task data quantity (cap 15 for V.Hard tasks). Our full-tree approach matches this through revision trajectories (13 examples for V.Hard) and adds qualitatively richer signal: error recovery patterns that generalize to untrained tasks. Stage 2 DPO further sharpens decision-making at critical branch points.

---

## 5. Implementation Plan

### 5.1 Phase 1: Save Full Trees

**Files to modify:**

| File | Change |
|------|--------|
| `verl/mcts/tree.py` | Add `q_value` field to `TreeNode`, `propagate_q_values()` and `get_sibling_pairs()` methods to `MCTSTree` |
| `verl/mcts/orchestrator.py` | Evaluate ALL leaf nodes (not just terminal), call `propagate_q_values()`, save full tree |
| `verl/mcts/trajectory_io.py` | New `save_mcts_tree()` / `load_mcts_tree()` for full tree serialization (all nodes, parent-child, Q-values, screenshots) |
| `configs/mcts_collection_v2.yaml` | New config with `save_full_tree: true` |

**Output format:** `mcts_trees/{task_id}.json` containing:
```json
{
  "task_id": "...",
  "instruction": "...",
  "tree_summary": {"n_nodes": 35, "n_successful": 3, "n_failed": 27, "root_q": 0.086},
  "nodes": [
    {
      "node_id": "node_001",
      "parent_id": null,
      "children_ids": ["node_002", "node_003"],
      "depth": 0,
      "q_value": 0.086,
      "eval_score": null,
      "done": false,
      "steps": [
        {"screenshot_b64": "...", "action": "Thought: ... Action: click(...)"}
      ],
      "diverged_at_step": 0,
      "branch_path": "root"
    },
    ...
  ]
}
```

Screenshots are stored per-node (not per-step-of-shared-prefix) to avoid duplication. Child nodes reference parent's steps via `parent_id`.

### 5.2 Phase 2: Extract Training Data

**New file: `scripts/extract_tree_training_data.py`**

Input: `mcts_trees/{task_id}.json` for all tasks
Output:
- `mcts_sft_q_weighted.jsonl` — successful trajectories with per-step Q-weights
- `mcts_dpo_pairs.jsonl` — sibling DPO pairs from branch points
- `mcts_revision_trajs.jsonl` — revision trajectories (failed→successful splices)

### 5.3 Phase 3: Re-collect All 86 Tasks

```bash
# Same infrastructure, just save full tree
python -m verl.mcts.run_collection --config configs/mcts_collection_v2.yaml
```

Expected: ~80 nodes per task × 86 tasks = ~6,880 total nodes. With binary eval (GUI tasks), expect ~20% success rate overall → ~1,370 successful + ~5,500 failed nodes.

### 5.4 Phase 4: Training

**Stage 1: Q-weighted SFT + Revision trajectories**
```bash
torchrun --nproc_per_node=8 scripts/train_mcts_sft_v2.py \
    --sft_data mcts_sft_q_weighted.jsonl \
    --revision_data mcts_revision_trajs.jsonl \
    --revision_mix_ratio 0.2 \
    --lr 3e-6 --num_epochs 1
```

**Stage 2: Step-level DPO (optional)**
```bash
torchrun --nproc_per_node=8 scripts/train_mcts_dpo.py \
    --model_path checkpoints/mcts_sft_v2/epoch_1 \
    --dpo_data mcts_dpo_pairs.jsonl \
    --lr 1e-6 --beta 0.1 --num_epochs 1
```

### 5.5 Phase 5: Evaluation

```bash
# n=1 quick check
sudo -E python -m verl.trainer.main config=configs/sft_eval_300tasks_clean_n1.yaml \
    worker.actor.model.model_path=checkpoints/mcts_sft_v2/epoch_1

# n=8 definitive
sudo -E python -m verl.trainer.main config=configs/sft_eval_300tasks_clean_n8.yaml \
    worker.actor.model.model_path=checkpoints/mcts_sft_v2/epoch_1
```

**Success criteria:**
- Trained tasks (n=8): doable rate ≥ 90% (currently 88-92%)
- Trained tasks (n=8): success rate ≥ 48% (currently 42-49%)
- Untrained tasks (n=8): doable rate ≥ 8% (currently 6%, naive SFT achieves 8.3%)
- Overall (n=8): doable rate ≥ 31% (beat naive SFT's 31%)

---

## 6. Expected Impact

### 6.1 Why Full-Tree Fixes the Hard-Task Gap

**Stage 1 (SFT-safe signal only):**

| Task Difficulty | v1 SFT Signal | v2 Stage 1 Signal | Increase |
|----------------|---------------|-------------------|----------|
| Easy (50+ trajs) | 50+ SFT | 50+ Q-SFT + ~3 revision | 1.06× |
| Medium (20-50 trajs) | 20-50 SFT | 20-50 Q-SFT + ~10 revision | 1.3–1.5× |
| Hard (5-10 trajs) | 5-10 SFT | 5-10 Q-SFT + ~12 revision | **2.5–3.4×** |
| V.Hard (1-5 trajs) | 1-5 SFT | 1-5 Q-SFT + ~15 revision | **4–16×** |

**Stage 2 adds DPO on top (not included in above counts):**

| Task Difficulty | DPO Pairs from Tree |
|----------------|-------------------|
| Easy | Few (~3 branch points with mixed outcomes) |
| Hard | Many (~15 branch points) |
| V.Hard | Most (~20 branch points) |

**The data imbalance inverts at both stages.** Hard tasks go from data-poorest to signal-richest because the tree explores extensively before finding success. Every failed branch that has a successful sibling produces a revision trajectory (Stage 1) and a DPO pair (Stage 2).

### 6.2 Why This Should Improve Untrained Task Generalization

1. **Revision trajectories teach error recovery** — a skill that transfers across all tasks, not just the 84 trained ones. The model learns "when I click the wrong menu, I should recognize the error and try the correct one" — this pattern applies universally.

2. **DPO teaches discrimination** — the model learns to prefer correct actions over incorrect ones at decision points. This sharpens its action selection generally, not just on trained tasks.

3. **Hard-task patterns transfer most** — hard tasks require novel GUI interaction patterns (unusual menu paths, complex multi-step workflows). Training on these transfers to untrained tasks that require similar novel patterns. Naive SFT's advantage on untrained tasks likely comes from its hard-task over-representation — our full-tree approach produces even richer hard-task signal.

### 6.3 Compute Cost

**Collection:** Same as current — 80 VMs per task, ~30 minutes per task, ~86 tasks = ~43 hours. The only additional cost is evaluating stuck/timeout nodes (~5% overhead) and saving the full tree (~2× storage).

**Training:** Stage 1 (SFT) is same cost as current (~3 hours). Stage 2 (DPO) adds ~2 hours. Total: ~5 hours vs current ~3 hours.

**Net:** ~45 hours total (same collection + marginally more training) for expected significant improvement in both trained and untrained task performance.

---

## 7. References

### Tree Structure for LLM Training

| Paper | ID | Key Contribution |
|-------|----|------------------|
| Tree-GRPO | 2509.21240 (ICLR 2026) | Intra-tree sibling advantage ≡ step-level DPO; 4× token efficiency |
| ReST-MCTS* | 2406.03816 (NeurIPS 2024) | Automatic step reward inference from tree; iterative self-training |
| Agent-R | 2501.11425 | Revision trajectories from failed→successful sibling splicing |
| AlphaLLM-CPL | 2410.06508 (AAAI 2025) | Sibling preference pairs with curriculum DPO; self-improvement without annotation |
| Tree-OPO | 2509.09284 | Off-policy MCTS with step-level advantage estimation |

### Step-Level Supervision for Agents

| Paper | ID | Key Contribution |
|-------|----|------------------|
| AgentPRM | 2511.08325 | Process reward evaluating progress + promise for agents |
| iStar | 2509.19199 | Implicit step rewards via trajectory DPO |
| GuidNav | 2504.16073 | Process rewards for VLM agent navigation |
| CM2 | 2602.12268 | Checklist rewards for multi-turn tool use |
| MCTS-EP | 2509.17116 | MCTS + preference optimization for embodied agents |

### GUI/Web Agent Training

| Paper | ID | Key Contribution |
|-------|----|------------------|
| ARPO | 2505.16282 | GRPO + replay buffer for GUI agents (our baseline) |
| DigiRL | 2406.11896 | Online RL for Android agents; offline-to-online training |
| ExACT | 2410.02052 | Reflective MCTS for web agents; exploratory learning |
| WebSynthesis | 2507.04370 | World-model MCTS for web trajectory synthesis |
| WebSTAR | 2512.10962 | Step-level binary masking with StepRM |

### Data Balancing

| Paper | ID | Key Contribution |
|-------|----|------------------|
| DART-Math (Prop2Diff) | NeurIPS 2024 | Difficulty-proportional data allocation |
| UniMax | ICLR 2023 | Capped temperature sampling prevents rare-group overfitting |
| XLM-R | ACL 2020 | α=0.3 temperature sampling for imbalanced groups |
