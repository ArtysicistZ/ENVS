# Hybrid Cold-Start for GUI Agent Training: Plan-MCTS for Diversity + EvoCUA-32B Distillation for Coverage

**Date:** 2026-03-14
**Context:** ARPO replication on OSWorld, UI-TARS-1.5 (Qwen2.5-VL 7B), 8x A100 80GB

---

## 1. Problem: ARPO's Concrete Bottlenecks

ARPO achieves 83.9% on 128 training tasks and **~24% overall** on OSWorld (368 tasks). Three specific bottlenecks limit further improvement:

### Bottleneck 1: 65% of Tasks Are Wasted — No Training Signal

ARPO filters tasks by running 16 rollouts with the base 7B model. Tasks with 0/16 success are discarded — **~240 of 368 tasks produce zero training signal.** These include complex multi-app workflows, unfamiliar UI patterns, and tasks requiring domain knowledge the 7B model simply does not have (e.g., specific LibreOffice Calc formulas, GIMP filter chains, multi-app orchestration).

**The fundamental issue:** MCTS or any search method using the same 7B model **cannot create capability that doesn't exist.** If the model doesn't know how to use GIMP's "Curves" tool, no amount of plan diversity or reflective refinement will produce a successful trajectory. For these tasks, we need a **stronger model** to provide the missing knowledge.

### Bottleneck 2: Low Strategy Diversity on Solvable Tasks

For the 128 tasks the 7B model *can* solve, ARPO uses 8 temperature-sampled rollouts per task. These tend to produce **variations of the same strategy** — the model has a dominant mode (e.g., always using the menu bar) and temperature sampling produces minor coordinate variations, not fundamentally different approaches. Havrilla et al. (2024) show RL pass@96 saturates early — **RL does not explore beyond strategies present in SFT initialization.** The pretrained model's behavioral diversity is the ceiling for RL.

### Bottleneck 3: Cold-Start Vanishing Gradients

In early ARPO epochs, the replay buffer is empty and most rollouts fail. GRPO advantages are all zero → no learning. The model must "get lucky" before training can begin.

---

## 2. Core Insight: Diversity and Coverage Are Different Problems

These are fundamentally different problems requiring different tools:

| Problem | Root Cause | Right Tool |
|---------|-----------|------------|
| **Low diversity** on solvable tasks | Model always follows its dominant mode | **MCTS with 7B model** — structured exploration forces different strategies |
| **Zero coverage** on hard tasks | Model lacks domain knowledge | **EvoCUA-32B** — provides knowledge the 7B doesn't have |
| **Empty replay buffer** | No prior successes cached | Both tracks contribute successful trajectories |

**Key constraint:** ARPO trains with **max 15 steps** per rollout. OSWorld tasks are generally designed for ≤15 steps. The ~210 hard tasks are **"knowledge-hard"** (model doesn't know the right GIMP filter or Calc formula) not **"length-hard"** (needing 50 steps). So all cold-start data must be collected with a 15-step budget to match ARPO's training.

---

## 3. The Hybrid Cold-Start Pipeline

```
Current ARPO:
  Pretrained 7B (~24% overall) → ARPO RL (128 tasks) → ~30% overall

Proposed:
  ┌───────────────────────────────────────────────────────────┐
  │              Phase 0: Hybrid Data Generation               │
  │                                                            │
  │  Track A: MCTS with 7B model on 128 solvable tasks        │
  │    → Goal: 3-5 diverse strategies per task                 │
  │    → Output: ~400-600 diverse successful trajs             │
  │                                                            │
  │  Track B: EvoCUA-32B rollouts on ~210 hard tasks           │
  │    → 15-step budget (matching ARPO training)               │
  │    → Goal: find ≥1 solution for as many as possible        │
  │    → Output: ~80-120 new successful trajs                  │
  │                                                            │
  │  Combined: ~500-720 high-quality trajectories              │
  └───────────────────────────────────────────────────────────┘
                           ↓
  ┌───────────────────────────────────────────────────────────┐
  │  Phase 1: SFT (2 epochs) on combined MCTS + EvoCUA data   │
  │    → 7B model learns diverse strategies + new tasks        │
  └───────────────────────────────────────────────────────────┘
                           ↓
  ┌───────────────────────────────────────────────────────────┐
  │  Phase 2: ARPO RL on expanded task set (200+ tasks)        │
  │    → Replay buffer pre-populated from Phase 0              │
  │    → Better-initialized model from Phase 1                 │
  └───────────────────────────────────────────────────────────┘
```

---

## 4. Track A: Plan-MCTS with 7B Model (Diversity Engine)

### 4.1 Why MCTS, Not Just More Rollouts?

For the 128 tasks the 7B model can already solve, the goal is **strategy diversity**, not mere success. Consider a simple task: "Save the current document."

**Temperature sampling (8 rollouts):**
All 8 rollouts: Click File menu → Click Save. Minor variations in click coordinates.
Result: 1 strategy, 8 near-identical trajectories.

**Plan-Level MCTS (same compute budget):**
- Plan A: File → Save (2 rollouts)
- Plan B: Ctrl+S keyboard shortcut (2 rollouts)
- Plan C: Right-click tab → Save (2 rollouts)
- Plan D: Ctrl+Shift+S → Save As dialog (2 rollouts)
Result: 4 fundamentally different strategies.

Why this matters for downstream RL: when ARPO runs GRPO on the SFT'd model, each rollout group now contains **different approaches**. The advantage signal can distinguish which strategy works better in which context, rather than comparing near-identical attempts.

### 4.2 Why Plan-Level, Not Action-Level?

OSWorld's Docker environments only support reset to baseline (OverlayFS wipe). **No intermediate snapshots** — you cannot save VM state at step 3 and branch from there. This makes action-level MCTS (which requires state rollback at every node) infeasible.

**Plan-Level MCTS** works within this constraint:
- Each MCTS "simulation" = one full rollout from VM baseline reset (5-8 seconds)
- Branching happens at the plan level: the LLM generates K different high-level plans
- Each plan is executed as a complete trajectory from the initial state
- No intermediate snapshots needed

### 4.3 Algorithm: Track A

```
TRACK_A_PLAN_MCTS(solvable_tasks, model_7b, K=5, N=20):
  """
  Generate diverse strategies for tasks the 7B model can already solve.
  K = initial plan families per task
  N = total MCTS iterations (rollouts) per task
  """
  D_diverse = []

  for task in solvable_tasks:  # 128 tasks
    tree = PlanTree(root=task.instruction)

    # ── Step 1: Generate K fundamentally different plans ──
    plans = model_7b.generate(
      prompt = f"""
        Task: {task.instruction}
        Screenshot: {task.initial_screenshot}

        Generate {K} fundamentally different strategies to accomplish this
        task. Each strategy MUST use a different mechanism:
        1. Via menu navigation (clicking through menus)
        2. Via keyboard shortcut (if applicable)
        3. Via right-click context menu (if applicable)
        4. Via terminal/command line (if applicable)
        5. Via alternative UI path (toolbar, sidebar, etc.)

        For each strategy, list the concrete steps.
        If a mechanism doesn't apply, create a variation of another.
      """,
      temperature = 0.9
    )
    for plan in plans:
      tree.root.add_child(PlanNode(plan))

    # ── Step 2: MCTS Iterations ──
    for iteration in range(N):

      # Selection: UCB1 picks which plan family to try next
      node = UCB1_SELECT(tree.root)
      # UCB1 = V(n)/N(n) + c * sqrt(ln(N(parent)) / N(n)),  c = 1.41

      # Expansion: if node visited before and failed, generate refinements
      if node.is_leaf() and node.visit_count > 0 and node.value < 0.8:
        refinements = model_7b.generate(
          prompt = f"""
            Task: {task.instruction}
            Original plan: {node.plan}
            Previous failures: {node.failure_analyses}

            This plan has been tried but often fails. Generate 2 refined
            variants that fix the identified failure points.
          """,
          temperature = 0.7
        )
        for ref in refinements:
          node.add_child(PlanNode(ref))
        node = node.children[0]

      # Simulation: full rollout from VM reset
      env.reset(task)                          # 5-8s
      traj = EXECUTE_WITH_PLAN(                # 20-40s
        plan=node.plan, env=env,
        model=model_7b, max_steps=15
      )
      reward = env.evaluate()                  # 0 or 1

      # Progress score for UCB (LLM-as-judge on failures)
      if reward == 0:
        progress = model_7b.judge_progress(task, traj)  # 0.0-1.0
        node.failure_analyses.append(
          model_7b.analyze_failure(task, traj)
        )
      else:
        progress = 1.0

      # Backpropagation
      backprop(node, progress)

      # Collect successful trajectory
      if reward > 0:
        D_diverse.append({
          'task': task,
          'trajectory': traj,
          'plan_family': node.plan_family_id,
          'source': 'mcts_7b'
        })

  return D_diverse
```

### 4.4 What MCTS Adds Over Temperature Sampling: Concrete Mechanisms

1. **Explicit plan diversity**: The prompt forces K *fundamentally different* approaches (menu vs shortcut vs terminal). Temperature sampling produces variations *within* the model's dominant mode.

2. **UCB1 compute allocation**: If plan family "keyboard shortcuts" succeeds quickly, UCB1 shifts budget to under-explored families like "terminal commands." Equal-budget sampling would waste compute re-confirming what already works.

3. **Reflective refinement**: When a plan fails, the LLM sees *why* and generates a targeted fix. Example: "Right-click save failed because the file was read-only → add chmod step first." Independent sampling would just re-try the same failing approach.

4. **Cross-plan preference pairs**: The tree structure naturally produces (good_plan, bad_plan) pairs per task — useful for DPO if needed.

---

## 5. Track B: EvoCUA-32B Distillation (Coverage Engine)

### 5.1 The Capability Gap Problem

For ~210 tasks where the 7B model achieves 0/16 success, the model **lacks the domain knowledge** to solve them regardless of search strategy. Examples from OSWorld:

- **GIMP tasks**: "Apply Gaussian blur with radius 5" — the 7B model doesn't know where GIMP's Gaussian blur filter is
- **LibreOffice Calc**: "Create a VLOOKUP formula" — the model doesn't understand VLOOKUP syntax
- **Multi-app**: "Copy data from Chrome to a LibreOffice spreadsheet" — requires coordinating two applications with clipboard
- **OS**: "Configure custom keyboard shortcut in GNOME settings" — requires navigating system settings the model has never seen

No amount of MCTS with the 7B model will solve these. We need a model with broader GUI knowledge.

### 5.2 EvoCUA-32B as Knowledge Source

**EvoCUA-32B** (`meituan/EvoCUA-32B-20260105`) is the strongest open-weight GUI agent model:

| Property | Value |
|----------|-------|
| OSWorld score | **56.7%** (OSWorld-Verified, 50-step budget) |
| Parameters | 32B (based on Qwen3-VL-32B-Thinking) |
| Open weight | Yes (HuggingFace) |
| GPU requirement | **2x A100 80GB** (vLLM, tensor-parallel-size=2) |
| License | Open |

EvoCUA-32B is specifically trained for GUI tasks — it knows application layouts, menu structures, keyboard shortcuts, and multi-step workflows across desktop environments.

### 5.3 The 15-Step Constraint

EvoCUA-32B's published 56.7% uses a **50-step budget**. We must run it with **15 steps** to match ARPO's training budget. Why this still works:

- **OSWorld tasks are designed for ≤15 steps.** The standard `max_steps` across all ARPO configs is 15. Tasks that need 50 steps are rare edge cases.
- **The hard tasks are "knowledge-hard", not "length-hard."** The 7B model fails because it doesn't know *what to do* (which menu, which formula), not because it needs more steps. A model that knows the right approach can execute it in 5-10 steps.
- **Estimated EvoCUA-32B performance at 15 steps: ~35-45%.** Lower than 56.7%, but on the ~210 hard tasks specifically, many are solvable in ≤15 steps with the right knowledge.

**Rollout math with 15-step budget:**

With per-attempt success rate ~40% on hard tasks (at 15 steps):
- P(≥1 success in 5 attempts) = 1 - 0.6^5 = **92.2%**
- Expected tasks unlocked: ~210 × 0.92 × (fraction solvable in 15 steps) ≈ **80-130 tasks**

### 5.4 Algorithm: Track B

```
TRACK_B_EVOCUA(hard_tasks, evocua_32b, attempts=5, max_steps=15):
  """
  Use EvoCUA-32B to find solutions for tasks the 7B model cannot solve.
  Simple rollouts — no MCTS needed (EvoCUA is already capable enough).
  15-step budget to match ARPO training.
  """
  D_coverage = []

  for task in hard_tasks:  # ~210 tasks
    for attempt in range(attempts):
      env.reset(task)
      traj = EXECUTE_WITH_MODEL(
        env=env, model=evocua_32b, max_steps=15
      )
      reward = env.evaluate()

      if reward > 0:
        D_coverage.append({
          'task': task,
          'trajectory': traj,
          'source': 'evocua_distill'
        })
        break  # One success is enough for coverage

  return D_coverage
```

### 5.5 Why Not Just Use EvoCUA-32B for Everything?

If EvoCUA-32B is stronger, why not use it for the solvable tasks too and skip MCTS entirely?

Because **EvoCUA-32B gives you coverage but not diversity.**

EvoCUA-32B on the 128 solvable tasks would produce 128 trajectories — one solution per task, using EvoCUA-32B's own dominant mode. You'd get the same diversity problem as temperature sampling.

**MCTS with the 7B model** generates 3-5 different strategies per task because it *forces* the search to explore different plan families. And these strategies come from the 7B model's own distribution, which matters for SFT — the data is in-distribution for the student model. EvoCUA-32B's action phrasing, reasoning style, and coordinate prediction are based on Qwen3-VL (a different architecture family from the Qwen2.5-VL that UI-TARS uses), so pure EvoCUA data has a distribution gap.

**The hybrid is optimal:**
- Track A (MCTS/7B): **in-distribution** diversity for tasks the 7B can solve
- Track B (EvoCUA-32B): **cross-distribution** coverage for tasks the 7B cannot solve

After SFT, the 7B model has both diverse strategies for familiar tasks AND new capabilities for unfamiliar tasks. Then ARPO RL brings everything into the model's own distribution through online optimization.

---

## 6. POMDP Motivation

GUI interaction is a POMDP — the screenshot (observation) does not reveal the full system state (files on disk, clipboard, background processes, other workspaces). No existing GUI agent paper uses an explicit POMDP formulation; all use implicit MDP approximations.

This matters for two concrete reasons:

### 6.1 Theoretical Justification for Exploration Over Imitation

He et al. (2024, ICML) prove that naive imitation in POMDPs produces **linear regret**, while exploration-based methods achieve sublinear regret. This justifies Track A: even though the 7B model *can* solve the 128 tasks via imitation-like sampling, MCTS exploration produces better training data for the POMDP setting.

### 6.2 Belief-Aware Plan Generation

In our MCTS plan generation prompts, we explicitly ask the LLM to reason about uncertainty:

> "What hidden state might affect success? What information-gathering steps are needed first?"

This produces plans with **diagnostic actions** — checking file existence, scrolling to verify content, checking system settings. These information-gathering behaviors are critical for robust GUI interaction but systematically absent from imitation-based data.

### 6.3 Implementation

We do **not** use particle filters or explicit belief representations. The LLM's context window (full action-observation history) serves as implicit belief state. The POMDP framing is a **theoretical motivation and prompt design principle**, not an algorithmic component.

---

## 7. Detailed Pipeline and Data Flow

### 7.1 Phase 0: Hybrid Data Generation

**GPU Allocation (8x A100 80GB):**
```
GPU 0-1:  EvoCUA-32B via vLLM (tensor-parallel-size=2)  → Track B inference
GPU 2:    UI-TARS-1.5-7B via vLLM                        → Track A inference
GPU 3-7:  Free for SFT training (Phase 1) / ARPO RL (Phase 2)
```

Both tracks run in parallel — Track A uses the 7B model on GPU 2, Track B uses EvoCUA-32B on GPUs 0-1. VMs are shared across both tracks.

```
┌──────────────────────────────────────────────────────────────────┐
│                  Track A: MCTS Diversity (Parallel)               │
│                                                                   │
│  Input: 128 solvable tasks, UI-TARS-7B on GPU 2                 │
│  Method: Plan-MCTS with K=5 plans, N=20 iterations per task     │
│  Compute: 128 × 20 = 2,560 rollouts                             │
│  Per rollout: ~45s (8s reset + 25s execution + 12s LLM@7B)      │
│  Serial time: ~32 hours                                           │
│  With 16 parallel VMs: ~2 hours                                  │
│                                                                   │
│  Expected output:                                                 │
│    ~400-600 successful trajectories (3-5 per task × 128)         │
│    ~200 near-miss trajectories (high progress, failed)           │
│    ~300 preference pairs (cross-plan comparisons)                │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│               Track B: EvoCUA-32B Coverage (Parallel)             │
│                                                                   │
│  Input: ~210 hard tasks, EvoCUA-32B on GPUs 0-1                  │
│  Method: Simple rollouts, 5 attempts per task, max_steps=15      │
│  Compute: 210 × 5 = 1,050 rollouts                              │
│  Per rollout: ~75s (8s reset + 50s execution + 17s LLM@32B)     │
│  Serial time: ~22 hours                                           │
│  With 16 parallel VMs: ~1.4 hours                                │
│  Cost: $0 (local inference)                                       │
│                                                                   │
│  Expected output:                                                 │
│    ~80-120 successful trajectories (15-step budget)              │
│    (EvoCUA-32B at 15 steps solves ~40% of hard tasks)            │
│    30 infeasible tasks skipped from the start                    │
└──────────────────────────────────────────────────────────────────┘

Combined output: ~500-720 successful trajectories
  - 400-600 diverse trajectories from Track A (in-distribution)
  - 80-120 coverage trajectories from Track B (cross-distribution)
  - 200 near-miss trajectories
  - 300 preference pairs
```

### 7.2 Phase 1: SFT on Combined Data

```
Input:  Combined D_sft (500-720 successful + 200 near-miss trajectories)
Model:  Qwen2.5-VL 7B (UI-TARS-1.5 base)
GPUs:   3-7 (5 A100s)
Method: Standard SFT, 2 epochs only

Why 2 epochs:
  - Havrilla et al. (2024): 2-epoch SFT → 3.7 unique solutions
                            4-epoch SFT → 2.9 unique solutions
  - More epochs = model memorizes strategies = lower RL ceiling

Optional: 1 epoch of DPO on the ~300 preference pairs

Output: SFT'd 7B model with:
  - Multiple strategies for familiar tasks (from Track A)
  - New capabilities for previously unsolvable tasks (from Track B)
```

### 7.3 Phase 2: ARPO RL (Existing, Better Initialized)

```
Model:  SFT'd 7B model from Phase 1
Tasks:  Expanded set — re-filter with SFT'd model
        Expected: 180-220 trainable tasks (up from 128)
Replay: Pre-populated with successful trajectories from Phase 0
Method: Standard ARPO (GRPO + experience replay), unchanged
GPUs:   Full 8x A100

Why this improves ARPO:
  1. More trainable tasks (180-220 vs 128) → more training signal
  2. Better initialization → higher starting success rate
  3. Pre-populated replay → no vanishing gradient from epoch 0
  4. Diverse strategies → higher RL ceiling (Havrilla constraint)
```

---

## 8. Why Each Track Is Necessary (Neither Alone Suffices)

### Track A alone (MCTS/7B only):
- Generates diverse strategies for 128 solvable tasks
- **Cannot unlock hard tasks** — the 7B model simply doesn't know how
- Trainable tasks stay at ~128-140
- Improvement limited to better strategy diversity on existing tasks

### Track B alone (EvoCUA-32B only):
- Unlocks 80-120 new tasks via stronger model
- **Doesn't improve diversity** — one solution per task
- Cross-distribution data (Qwen3-VL style ≠ Qwen2.5-VL style) → SFT mismatch risk
- RL ceiling stays low due to strategy homogeneity

### Hybrid (Track A + Track B):
- **Diversity** from MCTS for solvable tasks (raises RL ceiling)
- **Coverage** from EvoCUA-32B for hard tasks (expands training set)
- Track A data is in-distribution → clean SFT signal
- Track B data provides new knowledge → SFT teaches 7B new capabilities
- ARPO RL brings cross-distribution Track B data into the model's own distribution

---

## 9. Concrete Execution Details

### 9.1 EXECUTE_WITH_PLAN (Track A Helper)

```python
def execute_with_plan(plan, env, model, max_steps=15):
    """Execute a high-level plan by grounding each step to concrete actions."""
    trajectory = Trajectory()
    obs = env.get_screenshot()

    for step in range(max_steps):
        action_text = model.generate(
            messages=[
                system_prompt,
                user_msg(
                    instruction=task.instruction,
                    plan=plan,
                    history=trajectory.history,
                    screenshot=obs
                )
            ],
            temperature=0.3
        )

        parsed = parse_action(action_text)
        if parsed.type in ('FINISH', 'FAIL'):
            break

        obs, done = env.step(parsed)
        trajectory.add_step(obs, action_text, parsed)
        if done:
            break

    return trajectory
```

### 9.2 UCB1 Selection

```python
def ucb1_select(root, c=1.41):
    """Select child with highest UCB1 score."""
    best_score, best_child = -inf, None
    for child in root.children:
        if child.visit_count == 0:
            return child
        exploitation = child.total_value / child.visit_count
        exploration = c * sqrt(log(root.visit_count) / child.visit_count)
        score = exploitation + exploration
        if score > best_score:
            best_score, best_child = score, child
    return best_child
```

### 9.3 Data Formatting for SFT

Each trajectory is formatted as a multi-turn conversation matching UI-TARS's training format:

```
System: You are a GUI agent. Given a screenshot and instruction, output an action.

User: [screenshot_1] Task: {instruction}
Assistant: Thought: I need to open the File menu to save the document.
           Action: click(start_box='(45,12)')

User: [screenshot_2]
Assistant: Thought: The File menu is open. I see "Save" option.
           Action: click(start_box='(45,87)')

User: [screenshot_3]
Assistant: Thought: The document has been saved. Task complete.
           Action: FINISH
```

Both Track A and Track B data are formatted identically — no plan hints in the final SFT data. The model learns diverse strategies purely from exposure to different action sequences.

**Track B format conversion:** EvoCUA-32B's output format (Qwen3-VL based) must be converted to UI-TARS's action format before SFT. This is a straightforward text transformation (both use similar click/type/scroll primitives with coordinate output).

### 9.4 Near-Miss Trajectory Handling

Gandhi et al. (2025) show that **structure matters more than correctness** for cold-start. For near-miss trajectories (high progress, task failed), we include truncated versions:

```python
def format_near_miss(task, trajectory, progress_score):
    if progress_score < 0.6:
        return None
    good_prefix = model.identify_good_prefix(task, trajectory)
    return format_sft_example(task, good_prefix)
```

---

## 10. Differentiation from Existing Work

### 10.1 vs. AgentQ (MCTS during RL)

| Aspect | AgentQ | Ours |
|--------|--------|------|
| When MCTS runs | Every DPO iteration (recurring) | Once at cold-start (one-time) |
| Strong model usage | None | EvoCUA-32B for coverage (local, no API) |
| Cost structure | O(RL_iterations × MCTS_cost) | O(MCTS_cost + distill_cost) |
| RL algorithm | DPO only | Any (GRPO, PPO, DPO) |
| Task expansion | No | Yes (EvoCUA-32B unlocks hard tasks) |

### 10.2 vs. ProAct (Iterative MCTS → SFT)

| Aspect | ProAct | Ours |
|--------|--------|------|
| Pipeline | Iterative (search → SFT → search → ...) | One-shot cold-start → RL |
| Downstream | SFT only (no RL phase) | SFT → ARPO RL |
| Capability gap | Single model | Hybrid: 7B for diversity + 32B for coverage |
| Formulation | Implicit MDP | POMDP-motivated |

### 10.3 vs. Pure Distillation (EvoCUA-32B → SFT)

| Aspect | Pure EvoCUA Distillation | Ours |
|--------|--------------------------|------|
| Strategy diversity | Low (EvoCUA's dominant mode) | High (MCTS forces K plans) |
| Distribution match | Cross-distribution only | Hybrid: in-dist (Track A) + cross-dist (Track B) |
| Cost | Higher (32B on all 368 tasks) | Lower (32B only on ~210 hard tasks) |
| RL ceiling | Limited by homogeneous data | Raised by diverse MCTS data |

### 10.4 vs. EvoCUA (Current SOTA, 56.7% OSWorld)

| Aspect | EvoCUA | Ours |
|--------|--------|------|
| Data generation | Evolutionary random rollouts | Structured MCTS + targeted distillation |
| Exploration | Unstructured (random mutations) | Structured (UCB1, reflective refinement) |
| Model size | 32B (final model) | 7B (final model, much cheaper to deploy) |
| Training method | Evolutionary self-play | Cold-start SFT → ARPO RL |

**Key advantage over EvoCUA:** Our final model is **7B** (cheap to deploy and run at scale), while EvoCUA's is 32B. We use the 32B model only during the one-time cold-start phase.

---

## 11. Expected Results

### 11.1 Baseline Performance

| Model | Size | 128 Training Tasks | Overall OSWorld (368) |
|-------|------|-------------------|----------------------|
| UI-TARS-1.5 (Base) | 7B | 68.7% | **~24%** |
| UI-TARS-1.5 + GRPO | 7B | 72.9% | 26.0% |
| UI-TARS-1.5 + ARPO | 7B | 83.9% | **~30%** |
| EvoCUA-32B (reference) | 32B | — | **56.7%** (50 steps) |

### 11.2 Projected Results

| Metric | ARPO Baseline | + Track A Only | + Track B Only | + Hybrid (A+B) |
|--------|---------------|----------------|----------------|----------------|
| Trainable tasks | 128 | ~135 | ~210-250 | ~220-260 |
| Training success rate | 83.9% | 86% | 82% | 87% |
| **Overall OSWorld** | **~30%** | **~32%** | **~35%** | **~38-42%** |
| Cold-start waste | 2-3 epochs | 0 | 0 | 0 |
| Final model size | 7B | 7B | 7B | **7B** |

### 11.3 Success Criteria

1. **Minimum viable**: Track A produces ≥3 distinct strategies per task on average. Track B solves ≥60 additional tasks. Overall OSWorld improves by ≥5%.
2. **Good**: Overall OSWorld ≥38%. Clear ablation showing both tracks contribute.
3. **Excellent**: Overall OSWorld ≥42%, closing the gap with EvoCUA (56.7%) despite using a 7B model vs 32B.

---

## 12. Compute Budget

**All local — zero API cost.**

| Component | GPUs Used | VM Hours | Wall Clock (16 VMs) |
|-----------|-----------|----------|---------------------|
| Track A: MCTS/7B (128 × 20 rollouts) | GPU 2 | 32h serial | ~2 hours |
| Track B: EvoCUA-32B (210 × 5 rollouts) | GPUs 0-1 | 22h serial | ~1.4 hours |
| Phase 1: SFT (2 epochs) | GPUs 3-7 | — | ~2-4 hours |
| Phase 2: ARPO RL (expanded) | All 8 GPUs | Same + ~40% more | Same + ~40% |
| **Total cold-start overhead** | — | — | **~5-8 hours** |

Tracks A and B run in parallel (different GPUs), so Phase 0 wall clock ≈ max(2h, 1.4h) ≈ **2 hours**.

Total additional cost vs vanilla ARPO: **~6-10 hours of wall clock time, zero dollars.** ARPO training itself takes 40+ hours. The cold-start overhead is **~15-25%** of total training time.

---

## 13. Implementation Plan

### 13.1 New Components

| Component | Lines (est.) | Description |
|-----------|-------------|-------------|
| `plan_mcts.py` | ~300 | Tree data structure, UCB1, backpropagation |
| `plan_generator.py` | ~200 | LLM prompts for plan generation, refinement, failure analysis |
| `mcts_cold_start_runner.py` | ~400 | Orchestrates Track A across tasks and parallel VMs |
| `evocua_coverage_runner.py` | ~200 | Orchestrates Track B with EvoCUA-32B |
| `sft_data_formatter.py` | ~150 | Converts trajectories to SFT format; handles EvoCUA→UI-TARS format conversion |

### 13.2 Modified Components

| Component | Change |
|-----------|--------|
| `replay_buffer.py` | Add `load_from_cold_start()` to pre-populate |
| `ray_trainer.py` | Config option for pre-loaded replay buffer |
| Training configs | New YAML for SFT phase; modified ARPO config |

### 13.3 Reusable Infrastructure

All existing infrastructure works without modification:
- `RemoteEnvWorker` / `LocalEnvWorker` for VM interaction
- `parse_action_to_structure_output()` for action parsing
- `OSWorldTaskConfigDataset` for task loading
- vLLM server (already used for UI-TARS inference)
- `ImageProcessMixin` for screenshot processing

### 13.4 EvoCUA-32B Deployment

```bash
# Deploy EvoCUA-32B on GPUs 0-1
vllm serve meituan/EvoCUA-32B-20260105 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --port 8001

# Deploy UI-TARS-7B on GPU 2 (already exists in codebase)
vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
  --tensor-parallel-size 1 \
  --port 8002
```

---

## 14. Risk Analysis

### Risk 1: EvoCUA-32B at 15 Steps Solves Fewer Tasks Than Expected
**Probability**: Medium. EvoCUA-32B's 56.7% uses 50 steps. At 15 steps, success rate drops. Some hard tasks may genuinely need >15 steps even with the right knowledge.
**Mitigation**: (a) Increase to max_steps=20 as a compromise if 15 is too limiting. (b) Focus on the tasks EvoCUA-32B solves quickly (≤10 steps) — these are most likely knowledge-hard, not length-hard. (c) Even 60-80 additional tasks is a meaningful expansion from 128.

### Risk 2: Distribution Mismatch from EvoCUA Data Hurts SFT
**Probability**: Medium. EvoCUA-32B uses Qwen3-VL-32B-Thinking (different architecture family from UI-TARS's Qwen2.5-VL). Action format, reasoning style, and coordinate prediction may differ.
**Mitigation**: (a) Convert all Track B data to UI-TARS's exact output format. (b) Use Track B data with lower weight in SFT loss. (c) Ablation: Track A only vs A+B to isolate effect.

### Risk 3: MCTS Doesn't Produce Meaningfully Different Strategies
**Probability**: Low. The prompt explicitly requests different mechanisms. K=5 with forced diversity should produce ≥3 distinct approaches for most tasks.
**Mitigation**: Measure diversity quantitatively (edit distance between action sequences) and set a minimum threshold.

### Risk 4: SFT Overfitting Despite 2-Epoch Limit
**Probability**: Low. With 500-720 trajectories and 2 epochs, overfitting is unlikely for a 7B model.
**Mitigation**: Monitor validation loss. Reduce to 1 epoch if needed.

### Risk 5: EvoCUA-32B's Output Format Incompatible
**Probability**: Low-Medium. EvoCUA may use different action format than UI-TARS.
**Mitigation**: Build a format converter in `sft_data_formatter.py`. Both models use similar GUI primitives (click, type, scroll with coordinates) — the conversion is text manipulation, not semantic.

---

## 15. Ablation Studies

| # | Experiment | Tests | Expected Finding |
|---|------------|-------|------------------|
| 1 | Track A only vs Hybrid | Value of EvoCUA coverage | Hybrid > A due to task expansion |
| 2 | Track B only vs Hybrid | Value of MCTS diversity | Hybrid > B due to RL ceiling |
| 3 | MCTS vs Best-of-N (same budget) | Value of structured search | MCTS > Best-of-N on diversity |
| 4 | With vs without near-miss data | Value of partial trajectories | With ≥ without (Gandhi et al.) |
| 5 | Pre-populated replay only (no SFT) | Replay vs SFT contribution | SFT adds more |
| 6 | 1 vs 2 vs 4 SFT epochs | Optimal SFT duration | 2 epochs optimal |
| 7 | K=3 vs K=5 vs K=8 plan families | Optimal diversity | K=5 sweet spot |
| 8 | max_steps=15 vs 20 vs 25 for Track B | Step budget sensitivity | 15-20 likely sufficient |

---

## 16. Novelty Claims

**To our knowledge, no prior work has:**

1. **Combined structured MCTS exploration with strong-model distillation** in a hybrid cold-start pipeline, separating diversity (MCTS/weak model) from capability (strong model), with both running locally at zero API cost.

2. **Used Plan-Level MCTS specifically for SFT cold-start diversity** before RL in GUI agent training. AgentQ uses MCTS during RL (recurring cost); ProAct uses tree search iteratively (no RL). We use MCTS once.

3. **Expanded the trainable task set** via cold-start distillation from a stronger open-weight model. All existing GUI agent RL systems (ARPO, DigiRL, WebRL) filter to tasks where the base model already succeeds.

4. **Provided POMDP-grounded justification** for exploration-based cold-start over imitation-based cold-start in GUI environments, backed by He et al.'s (2024) linear regret theorem.

---

## Appendix A: Key Papers

| Paper | Contribution to This Proposal |
|-------|-------------------------------|
| **ARPO** (Lu et al., 2024) | Baseline: GRPO + experience replay, ~24% → ~30% on OSWorld |
| **EvoCUA** (Xue et al., 2026) | Parent model for Track B (56.7% OSWorld, open-weight 32B) |
| **Plan-MCTS** (Zhang et al., 2026) | Plan-space search for GUI environments |
| **ProAct** (Yu et al., 2026) | MCTS → SFT distillation precedent |
| **AgentQ** (Putta et al., 2024) | MCTS in agent RL (competitor) |
| **ExACT** (Yu et al., 2024) | Reflective MCTS + contrastive learning |
| **Havrilla et al.** (2024) | RL doesn't explore beyond SFT — ceiling argument |
| **He et al.** (2024, ICML) | Linear regret of imitation in POMDPs |
| **Chu et al.** (2025) | SFT memorizes, RL generalizes |
| **Gandhi et al.** (2025) | Structure > correctness for cold-start |
| **ASTER** (Zhang et al., 2026) | 4K diverse > large homogeneous |
| **DeepSeek-R1** (2025) | Cold-start → RL pipeline reference |
| **Compute-Optimal Sampling** (Bansal, 2024) | Weaker models produce more diverse data |

## Appendix B: Why the Hybrid Follows Compute-Optimal Sampling

Bansal et al. (2024) show that **weaker models generate more diverse training data** than stronger models. The stronger model's outputs cluster around its strong mode, while the weaker model's outputs spread across more strategies.

This directly supports our hybrid:
- **Track A uses the 7B model** (weaker) for diversity → more diverse strategies per task
- **Track B uses EvoCUA-32B** (stronger) for coverage → solves hard tasks but with less diversity

The hybrid naturally follows the compute-optimal principle: weak model where diversity matters, strong model where capability matters. And both run locally on 8x A100 80GB — no API costs, no external dependencies.

---

*Document compiled: 2026-03-14*
*Revised: EvoCUA-32B (local, zero API cost) + 15-step budget*
*For the ARPO GUI agent training research project*
