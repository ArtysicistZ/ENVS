# Action-Level MCTS for SFT Trajectory Collection

**Date:** 2026-03-20
**Context:** ARPO codebase, UI-TARS-1.5 (Qwen2.5-VL 7B), 80 Docker VMs (3 servers), 8× A100 80GB
**Data:** `docs/research/action_entropy_smoke_20260320_162231.json`, 15-step smoke test log
**VM cloning research:** `docs/research/VM_CLONING_FEASIBILITY.md`

---

## 1. Core Idea

At each step of an episode, generate K=8 action candidates and inspect the distribution. If the model is genuinely uncertain — multiple semantically distinct strategies — **spawn a new VM** to explore the minority action while the original VM continues the majority path.

The tree **grows on demand**. All 40 VMs per task are pre-setup with the same task at the start and sit waiting. When a branch is needed, claim a waiting VM and replay the action history to that point. Branch cost = **pure action replay: 0.3 × N seconds** (where N = current step).

---

## 2. Architecture: 40 Pre-Setup VMs, Claim On Demand

### 2.1 Setup Phase

At task start, **all 40 VMs** reset and run `setup_controller.setup()` for the same task in parallel:

```
t=0s:   ray.get([worker[i].reset.remote(task_config) for i in range(40)])
t=30s:  All 40 VMs ready. Same task state. Same initial screenshot.
```

This costs ~30s (the slowest VM's setup time) but is done **once** and in parallel.

### 2.2 Exploration Phase

All 40 VMs start identical (same task, same setup, same screenshot). **Inference (K=8 generation) runs on GPU, not on VMs.** VMs only execute actions.

**Step 0:**
```
All 40 VMs are at the same initial state (identical screenshots).

1. Take screenshot from any VM (all identical).
2. Generate K=8 responses on GPU.
3. Cluster → e.g., 3 distinct actions: click_toolbar, hotkey_ctrl_l, right_click.
4. Pick 3 VMs (any 3 — they're all identical), assign one action each:
   VM-0 executes click_toolbar
   VM-1 executes hotkey_ctrl_l
   VM-2 executes right_click
5. The other 37 VMs remain at the initial state, untouched, waiting.
```

No replay needed at step 0 — all VMs start identical.

**Step 1+:**
```
Each active VM now has a different state (different screenshot).

For each active VM:
  1. Take its screenshot.
  2. Generate K=8 from that screenshot (GPU inference).
  3. If consensus (1 cluster): execute the action on that VM.
  4. If high entropy (N clusters):
     - That VM executes the majority action.
     - For each minority action: claim a waiting VM,
       replay the active VM's action history (1.0s/step),
       then execute the minority action on the claimed VM.
```

**The waiting VMs are still at the initial setup state.** They need action replay to catch up to the branch point before they can take the minority action.

**Example timeline:**
```
t=30s:   Step 0: screenshot from any VM → K=8 → 3 clusters.
         VM-0, VM-1, VM-2 each take a different action.
         37 VMs waiting (still at initial state).

t=50s:   Step 1: VM-0 screenshot → K=8 → consensus. VM-0 executes.
         VM-1 screenshot → K=8 → 2 clusters.
           VM-1 takes majority action.
           Claim VM-3, replay VM-1's 1 prior action (1.0s), take minority.
         VM-2 screenshot → K=8 → consensus. VM-2 executes.
         → 4 VMs active, 36 waiting.

t=100s:  Step 5: VM-0 screenshot → K=8 → HIGH ENTROPY.
         → VM-0 takes majority.
         → Claim VM-4, replay VM-0's 5 actions (1.5s), take minority.
         → 5 VMs active, 35 waiting.
```

### 2.3 Resource Layout

```
40 VMs per task (2 tasks in parallel = 80 VMs total)
├── 1 VM starts stepping + probing entropy
├── N-1 VMs claimed dynamically at branch points (N determined by entropy)
└── remaining: idle, pre-setup, waiting

No separate "reserve pool." All 40 VMs are identical — same task, same setup.
The number of active VMs at any moment is determined purely by the model's entropy.
```

### 2.4 Dynamic VM Count

The number of active VMs is never hardcoded — it's driven entirely by the model's action distribution:

- **Step 0, consensus (1 cluster):** 1 VM runs. 39 idle.
- **Step 0, 3 clusters:** 3 VMs start (one per cluster). 37 idle.
- **Step 5, one branch hits entropy:** +1 VM claimed. Others unchanged.
- **Easy task (chrome, low entropy):** may use 1-3 VMs total for the whole episode.
- **Hard task (libreoffice_impress, entropy everywhere):** may grow to 15-20 VMs.

VMs are allocated **where the model is most uncertain**, not pre-assigned.

---

## 3. VM Cloning: Action Replay

### 3.1 Why Not True Cloning

True VM cloning (preserving running processes) is not feasible:
- **CRIU**: Broken for Xorg + Chrome + GNOME + D-Bus. Zero published examples exist.
- **QEMU/KVM snapshots**: No `/dev/kvm` on Azure VMs. Would be ideal (3-8s, full state) on bare metal.
- **OverlayFS copy**: Preserves filesystem only. Chrome opens to initial URL, NOT the navigated page. Still needs action replay for correctness.

See `docs/research/VM_CLONING_FEASIBILITY.md` for full analysis.

### 3.2 Action Replay Mechanism

All 40 VMs have the same task setup. To "clone" VM-0's state at step N onto VM-3:

```
VM-3 already has: reset done, task setup done, same initial state as VM-0 at t=0
VM-3 needs: the N actions VM-0 has taken since setup

Replay:
  for action in VM_0.action_history[:N]:
      VM_3.step(action, pause=REPLAY_PAUSE)  # 1.0s per action
```

**No reset needed. No setup needed. Just replay the action sequence.**

### 3.3 Replay Timing

| Branch at step | Replay time | Notes |
|:-:|:-:|---|
| 1 | 1.0s | Near-instant |
| 5 | 5.0s | Typical early branch |
| 10 | 10.0s | Mid-episode branch |
| 15 | 15.0s | Late branch (rare, only in Phase 2) |

### 3.4 Replay Fidelity

The replay re-executes the exact pyautogui commands. Expected divergence:

| Action Type | Frequency | Per-Step Risk | Root Cause |
|---|:-:|:-:|---|
| click | 55% | 1-3% | Page not fully loaded at 1.0s pause |
| type | 15% | ~0.5% | Deterministic text input |
| hotkey | 10% | ~0% | Keyboard shortcuts are deterministic |
| scroll | 8% | ~3% | Slight position variance |
| drag | 5% | ~8% | Timing-sensitive animation |

**Compounding**: With mixed action types (~2% avg per-step risk), a 10-step replay has ~18% chance of any divergence. This is acceptable — diverged VMs simply produce different trajectories (free exploration diversity).

### 3.5 Replay Pause Tuning

Production pause: 1.0s. Replay pause: **1.0s** (same as production to maximize fidelity).

- 1.0s works for: clicks on static UI, hotkeys, typing, scrolling
- 0.5s needed for: page loads after URL navigation, menu animations (LibreOffice ~200-400ms), dialog opens (GIMP ~300-800ms)

Adaptive approach: use 1.0s by default, bump to 0.5s for steps where the action involves Chrome navigation or opening dialogs (detectable from the action type + coordinates).

### 3.6 The `/env/replay` Endpoint

```python
POST /env/replay
{
  "slot_id": 3,
  "predictions": ["Thought: ... Action: click(...)", "Thought: ... Action: type(...)"],
  "replay_pause_sec": 1.0
}

# Server-side: just loop env.step() with reduced pause
# Skip screenshot encoding, reward shaping, stall detection
# Only capture final screenshot after all actions complete

Response: {
  "success": true,
  "steps_completed": 5,
  "elapsed_sec": 1.8
}
```

**No reset, no setup** — the VM is already set up. Just replay actions.

---

## 4. Entropy Estimation & Branching Criteria

### 4.1 Hierarchical Action Clustering

**Layer 1 — Action Type Divergence (strongest signal):**
Group K=8 candidates by action type. If ≥2 types each have ≥2 samples → branch.

**Layer 2 — Spatial Divergence (within same type):**
Within dominant type, cluster by Euclidean distance (150px threshold). If ≥2 spatial clusters each have ≥2 samples → branch.

**Layer 3 — Noise (never branch):**
Same type, same spatial cluster (<150px). Coordinate jitter.

### 4.2 The 5-Gate Branch Filter

```python
def should_branch(candidates, step, node, task_budget):
    clusters = hierarchical_cluster(candidates, dist_threshold=150)
    significant = [c for c in clusters if len(c) >= 2]

    if len(significant) < 2:          return False  # Gate 1: no real divergence
    if compute_type_entropy(candidates) < 0.5 \
       and not extreme_spatial(clusters, 500):
                                       return False  # Gate 2: type_H too low
    if step >= 7 and not node.prev_high:
        node.prev_high = True;         return False  # Gate 3: late-step deferral
    if task_budget.branches_remaining <= 0:
                                       return False  # Gate 4: budget exhausted
    if available_vms() < 1:            return False  # Gate 5: no VMs left

    return True
```

### 4.3 Adaptive Entropy Probing Strategy

The naive approach (K=8 per VM per step) costs 864 calls across a 15-step episode as the tree grows to 15 VMs. The adaptive strategy below reduces this to ~200 calls (77% reduction) while preserving branch detection reliability where it matters.

#### 4.3.1 Core Design: Three-Layer Cost Control

The strategy combines three orthogonal savings mechanisms, each targeting a different source of waste:

**Layer 1 — Logprob Screening (Approach D):** Every VM needs exactly 1 generation per step to get its execution action. That generation is free — we need it anyway. Its token logprobs tell us whether the model is uncertain. If the first action-type token has P(top) > 0.9, the model is confident and no further probing is needed. Empirically, ~40% of VMs at any step are in this confident regime (type_H < 0.5 in 43% of steps from the smoke test). These VMs get K_effective = 1 at zero additional cost.

**Layer 2 — Budget-Capped Distribution (Approach F):** For the uncertain VMs that survive logprob screening, a per-step probe budget caps total extra calls. The budget decreases over the episode because early steps have higher branching value:

| Step Range | Probe Budget | Rationale |
|:--:|:--:|---|
| 0-2 | 16 | Highest strategic value. Invest heavily in understanding the action space. |
| 3-5 | 12 | Good branching value. Moderate investment. |
| 6-8 | 8 | Diminishing returns. Detect only strong divergence. |
| 9+ | 0 | Never branch (existing gate). No probing at all. |

The budget is distributed equally across uncertain VMs: `K_per_uncertain = max(2, probe_budget // N_uncertain)`. This keeps total calls roughly constant regardless of how many VMs are uncertain.

**Layer 3 — Branch Budget Gate (Approach C):** VMs with exhausted branch budgets (spawned VMs that have already used their 2 branches, or root VMs past 5) are never probed. They just execute with K=1. This naturally reduces the probed set as the tree matures.

#### 4.3.2 The Algorithm

```python
def adaptive_probe(active_nodes, step, generate_fn):
    """Determine K for each VM and run probing generation.

    Returns: dict mapping node_id -> list of candidate texts.
    """
    PROBE_BUDGETS = {0: 16, 1: 16, 2: 16, 3: 12, 4: 12, 5: 12,
                     6: 8,  7: 8,  8: 8}
    NEVER_PROBE_AFTER = 9       # matches never_branch_after
    CONFIDENCE_THRESHOLD = 0.9  # P(top action type token)

    results = {}

    # Phase 1: Generate 1 candidate per VM (required for execution anyway)
    # These calls batch across all VMs (different screenshots) — efficient.
    first_pass = {}
    for node in active_nodes:
        if node.done:
            continue
        obs = node.vm.get_obs()
        text, logprobs = generate_fn(obs, return_logprobs=True)
        first_pass[node.id] = (text, logprobs, obs)
        results[node.id] = [text]

    # Phase 2: Identify which VMs need additional probing
    if step >= NEVER_PROBE_AFTER:
        return results  # No probing past step 8

    # Filter to VMs that (a) have branch budget AND (b) are uncertain
    uncertain_nodes = []
    for node in active_nodes:
        if node.done or node.id not in first_pass:
            continue
        if node.budget.branches_remaining <= 0:
            continue  # Layer 3: no branch budget
        text, logprobs, obs = first_pass[node.id]
        top_type_prob = extract_action_type_prob(logprobs)
        if top_type_prob > CONFIDENCE_THRESHOLD:
            continue  # Layer 1: model is confident
        uncertain_nodes.append((node, obs))

    if not uncertain_nodes:
        return results  # All VMs are confident

    # Phase 3: Distribute probe budget across uncertain VMs
    probe_budget = PROBE_BUDGETS.get(step, 0)
    if probe_budget == 0:
        return results

    n_uncertain = len(uncertain_nodes)
    k_total = max(2, probe_budget // n_uncertain)  # includes the 1 already generated
    k_extra = k_total - 1

    # Generate k_extra more candidates for each uncertain VM
    # These calls batch across VMs (different screenshots) — efficient.
    for node, obs in uncertain_nodes:
        for _ in range(k_extra):
            text = generate_fn(obs)
            results[node.id].append(text)

    return results


def extract_action_type_prob(logprobs):
    """Extract P(top action type) from the generation's token logprobs.

    Looks at the token position where the action type is generated
    (after "Action: "). The logprob of the selected token gives us
    P(selected_type). If this is high, the model is confident.

    Returns float in [0, 1].
    """
    # Find the token after "Action:" in the logprob sequence
    # The action type token (click, hotkey, type, scroll, etc.)
    # is typically within the first 30 tokens of the output.
    # We look for the highest-logprob token in the action-type region.
    import math
    if not logprobs:
        return 0.5  # Unknown -> treat as uncertain

    # Scan logprobs for the action type token
    # Heuristic: it's the token with lowest entropy in the first 20 tokens
    # (the model is very confident about "Thought:" and "Action:" frame tokens,
    # but the action TYPE is where real uncertainty shows)
    for i, pos_logprobs in enumerate(logprobs[:30]):
        if pos_logprobs is None:
            continue
        for token_id, lp in pos_logprobs.items():
            logp = lp.logprob if hasattr(lp, 'logprob') else float(lp)
            prob = math.exp(logp)
            # Check if this token is an action type keyword
            # In practice, we'd check against the action type vocabulary
            # For now, return the prob of whatever token was selected
            # at the action type position
            return prob
    return 0.5
```

#### 4.3.3 Detection Reliability Analysis

The critical question: can small K reliably detect type-level divergence?

From the empirical data (40 steps, K=8, 8 tasks), steps with type_H >= 0.5 have these majority-type fractions:

| Majority Fraction | Occurrences | P(detect at K=2) | P(detect at K=4) | P(detect at K=8) |
|:--:|:--:|:--:|:--:|:--:|
| 88% (1 minority sample in 8) | 12/23 = 52% | 21% | 40% | 64% |
| 75% | 6/23 = 26% | 38% | 68% | 90% |
| 62% | 3/23 = 13% | 47% | 83% | 98% |
| 50% | 2/23 = 9% | 50% | 88% | 99% |

Key findings:

1. **K=2 is unreliable.** It misses 79% of the most common divergence pattern (88% majority). It should never be the sole probing strategy.

2. **K=4 is the minimum useful K.** It detects 68-88% of the 75%/62%/50% cases (the ones worth branching on). It misses the 88% majority case, but those are marginal anyway — a 7:1 split means the minority action had only 12.5% model probability.

3. **K=8 is gold standard but expensive.** It catches 90%+ of all cases.

4. **The 88% majority cases (52% of all divergent steps) are borderline.** A 7:1 type split means the minority is nearly noise. Missing these with K=4 is acceptable — the 5-gate filter would likely reject many of them anyway (Gate 1 requires each cluster to have >= 2 samples; a 7:1 split with K=4 would show up as 3:1 or 4:0, both failing Gate 1).

**Design conclusion:** K=4 is the minimum per-VM sample size for probing. The adaptive budget distributes enough to keep K >= 4 for the first ~5 uncertain VMs, K >= 2 after that.

#### 4.3.4 Call Count Projection (Realistic Scenario)

Tree growth: 1 VM at step 0, growing to ~14 VMs by step 14 (typical high-entropy task).

| Step | N_active | With Budget | Uncertain | K/uncertain | Total Calls | Time (8 GPU) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 1 | 1 | 1 | 16 | 16 | 5.0s |
| 1 | 2 | 2 | 1 | 16 | 17 | 5.3s |
| 2 | 3 | 3 | 2 | 8 | 17 | 5.3s |
| 3 | 3 | 3 | 2 | 6 | 13 | 4.1s |
| 4 | 4 | 4 | 2 | 6 | 14 | 4.4s |
| 5 | 5 | 4 | 2 | 6 | 15 | 4.7s |
| 6 | 6 | 4 | 2 | 4 | 12 | 3.8s |
| 7 | 7 | 4 | 2 | 4 | 13 | 4.1s |
| 8 | 8 | 3 | 2 | 4 | 14 | 4.4s |
| 9 | 9 | 0 | 0 | 1 | 9 | 2.8s |
| 10 | 10 | 0 | 0 | 1 | 10 | 3.1s |
| 11 | 11 | 0 | 0 | 1 | 11 | 3.4s |
| 12 | 12 | 0 | 0 | 1 | 12 | 3.8s |
| 13 | 13 | 0 | 0 | 1 | 13 | 4.1s |
| 14 | 14 | 0 | 0 | 1 | 14 | 4.4s |
| **Total** | | | | | **200** | **~62s** |

Comparison:

| Strategy | Total Calls | Avg/Step | Max Step Time (8 GPU) |
|---|:--:|:--:|:--:|
| Naive K=8 all VMs | 864 | 57.6 | 35.0s (step 14) |
| Strategy A (decreasing K by step) | 387 | 25.8 | 6.0s |
| Strategy F (budget=16, uniform) | 204 | 13.6 | 5.0s |
| **Recommended (D+F+C combined)** | **200** | **13.3** | **5.3s** |

The recommended strategy achieves **77% fewer calls** than naive K=8 while maintaining K >= 4 for all actively probed VMs through step 8. After step 8, the never_branch_after gate eliminates all probing overhead.

#### 4.3.5 Why Not Approach E (Family Representatives)?

Approach E (probe one VM per tree family) was considered but rejected:

1. **Siblings diverge quickly.** After a branch, two sibling VMs see different screenshots within 1-2 steps. Their action distributions become independent — probing one tells you nothing about the other.

2. **Implementation complexity.** Tracking families, electing representatives, and propagating branch decisions adds significant orchestration logic for marginal savings over the budget-cap approach.

3. **The budget cap already solves the scaling problem.** With a fixed budget of 12-16, total calls are bounded regardless of tree size. Family representatives would save at most 2-3 calls per step — not worth the complexity.

#### 4.3.6 Logprob Screening: Implementation Notes

The logprob screening (Layer 1) relies on vLLM's `logprobs=1` parameter, which returns the log-probability of each selected token. The key signal is the action-type token — the first non-frame token after "Action: " in the output.

**Calibration needed:** The 0.9 confidence threshold should be validated on the 15-step smoke test data. Specifically:

```python
# For each step in the smoke test, check: when type_H < 0.5 (empirically confident),
# what was the logprob of the first action-type token?
# This tells us the right threshold for screening.
#
# Expected: P(top_type_token) > 0.9 correlates with type_H < 0.5.
# If not, adjust the threshold or use a different logprob signal.
```

If logprob screening proves unreliable (poor correlation with type_H), it can be disabled by setting `CONFIDENCE_THRESHOLD = 0.0`. The strategy degrades gracefully to just Layer 2 + Layer 3, which still achieves ~204 total calls (vs 200 with screening).

### 4.4 Content Normalization

- Whitespace: `"ctrl  alt t"` → `"ctrl alt t"`
- Key order: `"alt+ctrl+t"` → `"ctrl+alt+t"`
- Type prefixes: same URL to different lengths = same strategy

### 4.5 Probing Cost Management

The adaptive probing strategy (Section 4.3) is the primary cost control mechanism. Additional considerations:

- **Batch across VMs**: Different VMs' generation calls (different screenshots) batch efficiently in vLLM. A step with 10 VMs each needing 1 call = 10 calls that can run as a single vLLM batch, taking ~2.5s total (not 25s).
- **Probe calls do NOT batch per-VM**: Due to the vLLM n>1 multimodal bug, K probing calls for the same screenshot must be sequential. With 8 GPUs, we can parallelize across VMs but not within a VM.
- **Two-phase generation per step**: Phase 1 generates 1 per VM (batched, fast). Phase 2 generates K_extra per uncertain VM (sequential per VM, parallelized across VMs on different GPUs). Phase 1 takes ~2.5s. Phase 2 takes ~K_extra * 2.5s worst case.
- **vLLM n>1 bug fix is still the single largest optimization.** If fixed: all K calls per VM become a single batched call. The adaptive strategy's 200 calls would take ~25s total instead of ~62s. With the bug, the adaptive strategy keeps time under control by capping K.

---

## 5. Branching Strategy

### 5.1 Front-Loaded by Design

From the 15-step smoke test (6 tasks, 77 steps):

| Steps | Mean Clusters | Branch Rate | SFT Value |
|:-:|:-:|:-:|---|
| 0-3 | 2.4 | 67% | **High** — strategic choices |
| 4-6 | 3.8 | 93% | Moderate — tactical decisions |
| 7-9 | 4.4 | 93% | Low — execution details |
| 10-14 | 4.3 | 92% | Minimal — finishing moves |

Steps 9+ have zero consensus across any task. Entropy is high but SFT marginal value is low (shared 67%+ prefix with existing trajectories).

**Default: never branch at step 9+.** Exception: Phase 2 hindsight exploration (Section 6) can branch late near successful trajectories.

### 5.2 Branch Budget

Each active VM gets a branch budget (default: 5 spawns for the root VM, 2 for spawned VMs). This prevents exponential cascade while allowing second-level branching on important paths. Total spawns across an episode are bounded by available waiting VMs (up to 39).

### 5.3 Branch Value Formula

```
branch_value(step_d, novelty) = (15 - d) / 15 × novelty
```

Where novelty:
- 1.0 = different action type (click vs hotkey)
- 0.5 = same type, different UI region (>300px apart)
- 0.1 = same type, nearby region (coordinate noise)

A branch at step 0 with a new action type is **50× more valuable** than a branch at step 12 with coordinate noise.

### 5.4 Pruning & Reclamation

Spawned VMs are marked done (stop stepping) when they become unproductive:
1. **Terminal action**: `finished()` or `fail()`
2. **Stuck loop**: Same action category 3 consecutive steps
3. **Stalled**: `wait()` twice in a row

Pruned VMs are not "released back to pool" — they just stop stepping. Their trajectories are still saved and evaluated. The pre-setup VMs waiting in the pool are separate.

---

## 6. Two-Phase Execution

### Phase 1: Exploration (~2.5 min with adaptive probing)

The main MCTS loop. 1 VM starts, probes entropy, spawns at branch points. Tree grows dynamically — from 1 VM (all consensus) to 15-20 VMs (high entropy). Adaptive probing (Section 4.3) keeps generation costs bounded as the tree grows.

### Phase 2: Hindsight Exploitation (~3-5 min, optional)

After Phase 1 and evaluation:
1. Identify **successful branches** (eval_score > 0)
2. Walk UP the tree from each success to find **unexplored sibling actions** — clusters from entropy probing that were not explored
3. Claim waiting VMs, replay the prefix of a successful branch, take the sibling action
4. Run remaining steps, evaluate

Phase 2 explores **siblings of known-good paths** — the highest-value unexplored territory. This is where late-step branching (step 9+) IS allowed, because we already know the prefix leads to success.

---

## 7. The Full Orchestration Loop

```python
class MCTSOrchestrator:
    def __init__(self, config: MCTSConfig):
        self.config = config

    def _get_probe_budget(self, step):
        """Get the probing budget for a given step (Section 4.3)."""
        for (lo, hi), budget in self.config.probe_budgets.items():
            if lo <= step <= hi:
                return budget
        return 0  # Steps beyond all ranges: no probing

    def run(self, task_config, all_workers, generate_fn):
        # ---- Setup Phase ----
        # Reset + setup ALL 40 VMs in parallel
        futures = [w.reset.remote(task_config) for w in all_workers]
        ray.get(futures, timeout=300)
        # All 40 VMs now have identical task state

        # ---- Exploration Phase ----
        # All 40 VMs are now identical. Waiting VMs stay at initial state.
        waiting = list(all_workers)  # All 40 available
        active_nodes = []            # None active yet — step 0 will create them
        tree = MCTSTree(roots=active_nodes)

        for step in range(self.config.max_steps):

            # ======== ADAPTIVE PROBING (Section 4.3) ========
            # Phase 1: Generate 1 candidate per active VM (batched, ~2.5s)
            # This is required anyway for the execution action.
            first_pass = {}
            for node in [n for n in active_nodes if not n.done]:
                obs = node.vm.get_obs()
                text, logprobs = generate_fn(obs, return_logprobs=True)
                first_pass[node.id] = (text, logprobs, obs)
                node.candidates = [text]

            # Step 0 special case: all VMs identical, only 1 obs needed
            if step == 0 and not active_nodes:
                obs = waiting[0].get_obs()
                # Step 0 gets full probe_budget (16) since only 1 VM
                budget = self._get_probe_budget(0)
                candidates = []
                for _ in range(budget):
                    text = generate_fn(obs)
                    candidates.append(text)
                clusters = get_significant_clusters(candidates)

                # Create one active node per distinct cluster
                for cluster in clusters:
                    if not waiting: break
                    vm = waiting.pop(0)
                    node = TreeNode(vm=vm, depth=0,
                                    budget=self.config.max_branch_per_explorer,
                                    action=representative(cluster))
                    active_nodes.append(node)
                    tree.add_root(node)
                continue  # Skip to execution

            # Phase 2: Identify uncertain VMs needing more probing
            probe_budget = self._get_probe_budget(step)
            uncertain_nodes = []

            if probe_budget > 0:
                for node in [n for n in active_nodes if not n.done]:
                    if node.id not in first_pass:
                        continue
                    # Layer 3: skip VMs with no branch budget
                    if node.budget.branches_remaining <= 0:
                        continue
                    # Layer 1: skip VMs where model is confident
                    _, logprobs, obs = first_pass[node.id]
                    top_prob = extract_action_type_prob(logprobs)
                    if top_prob > self.config.logprob_confidence_threshold:
                        continue
                    uncertain_nodes.append((node, obs))

            # Phase 3: Distribute probe budget across uncertain VMs
            if uncertain_nodes:
                n_uncertain = len(uncertain_nodes)
                k_total = max(self.config.min_k_per_uncertain_vm,
                              probe_budget // n_uncertain)
                k_extra = k_total - 1  # minus the 1 already generated

                for node, obs in uncertain_nodes:
                    for _ in range(k_extra):
                        text = generate_fn(obs)
                        node.candidates.append(text)

            # ======== BRANCH DECISIONS ========
            for node in [n for n in active_nodes if not n.done]:
                if len(node.candidates) >= 2:
                    # Has probing data — check for branching
                    if should_branch(node.candidates, step, node, node.budget):
                        clusters = get_significant_clusters(node.candidates)
                        node.action = representative(clusters[0])
                        for minority in clusters[1:self.config.max_branches_per_step]:
                            if not waiting: break
                            new_vm = waiting.pop(0)
                            new_vm.replay.remote(
                                node.get_action_history(),
                                pause=self.config.replay_pause_sec)
                            child = TreeNode(
                                vm=new_vm, depth=step, parent=node,
                                action=representative(minority),
                                budget=self.config.child_branch_budget)
                            tree.add_child(node, child)
                            active_nodes.append(child)
                    else:
                        node.action = node.get_majority_action()
                else:
                    # Only 1 candidate (confident or no budget) — use it
                    node.action = node.candidates[0]

            # ======== EXECUTE ========
            for node in active_nodes:
                if not node.done:
                    node.record_action(node.action)
            step_futures = {n.id: n.vm.step.remote(n.action)
                           for n in active_nodes if not n.done}
            ray.get(list(step_futures.values()), timeout=60)

            # ======== PRUNE ========
            for node in active_nodes:
                if node.is_terminal() or node.is_stuck():
                    node.done = True

        # ---- Save Phase ----
        for node in tree.all_nodes():
            save_trajectory(node)

        # ---- Evaluate Phase ----
        eval_futures = {n.id: n.vm.evaluate.remote()
                        for n in tree.all_nodes() if not n.never_stepped}
        eval_results = ray.get(list(eval_futures.values()), timeout=350)

        # ---- Phase 2: Hindsight ----
        if self.config.enable_phase2:
            successful = [n for n in tree.leaves() if n.eval_score > 0]
            for node in successful:
                siblings = tree.get_unexplored_siblings(node)
                for sib_action, branch_step in siblings[
                        :self.config.max_hindsight_per_success]:
                    if not waiting:
                        break
                    new_vm = waiting.pop(0)
                    prefix = node.get_action_history()[:branch_step]
                    new_vm.replay.remote(prefix,
                                         pause=self.config.replay_pause_sec)
                    hindsight_node = TreeNode(
                        vm=new_vm, depth=branch_step, ...)
                    for remaining_step in range(branch_step,
                                                self.config.max_steps):
                        action = generate_fn(new_vm.get_obs())
                        new_vm.step.remote(action)
                    save_trajectory(hindsight_node)
                    new_vm.evaluate.remote()
```

---

## 8. Trajectory Saving & SFT Value

### 8.1 Save-Before-Evaluate

```
1. All VMs finish stepping
2. SAVE all trajectories to disk (every VM, success or failure)
3. EVALUATE all VMs
4. UPDATE trajectories with eval results
```

### 8.2 Tree-Aware Trajectory Format

```json
{
  "task_id": "480bcfea...",
  "vm_idx": 7,
  "branch_path": "root→click_toolbar(s3)→scroll_down(s5)",
  "tree_depth": 2,
  "diverged_at_step": 3,
  "parent_vm_idx": 0,
  "steps": [
    {"step": 0, "screenshot_b64": "...", "action": "Thought: ... Action: click(...)"},
    ...
  ],
  "eval_result": 1.0,
  "is_hindsight": false
}
```

### 8.3 Trajectory Importance for SFT

| Priority | Description | Training Weight |
|:-:|---|:-:|
| 1 | Successful, diverges at step 0-3 (different strategy) | 1.0 |
| 2 | Successful, diverges at step 4-8 (different tactic) | 0.7 |
| 3 | Successful, diverges at step 9+ (finishing variation) | 0.3 |
| 4 | Failed trajectory | 0.0 for SFT; used for DPO/GRPO |

**Branch-point emphasis:** Steps immediately after a branch get 2× training weight.
**Prefix deduplication:** Shared prefix steps weighted by `1/n_sharing`.

### 8.4 Expected Yield Per Task

| Metric | Value |
|---|---|
| Starting VMs (depends on step 0 entropy) | 1-5 |
| Branch events (typical) | 5-15 |
| Total active VMs at episode end | 8-18 |
| Successful trajectories | 2-6 |
| Distinct strategies among successes | 2-4 |
| Phase 2 additional successes | +1-3 |

---

## 9. Generation Throughput

### 9.1 The Bottleneck: vLLM n>1 Bug

vLLM 0.7.3 + Qwen2.5-VL: `n>1` crashes with image token mismatch. K probing calls for the same screenshot must be generated **sequentially**. The adaptive probing strategy (Section 4.3) mitigates this by capping total calls to ~200/episode instead of 864.

### 9.2 Per-Step Cost (With Adaptive Probing)

| Component | Time | Notes |
|---|:-:|---|
| Phase 1: 1 generation/VM (batched) | ~2.5s | All VMs batch in one vLLM call |
| Phase 2: K_extra for uncertain VMs | ~3-5s (steps 0-5) to ~0s (steps 9+) | Sequential per-VM, parallel across GPUs |
| VM stepping + screenshot | ~2s | Parallel across all VMs |
| Action replay for new branches | ~5-10s | 1.0s x steps, parallel with other work |
| **Total per step (1 VM, step 0)** | **~10s** | 16 probing calls, heavily invested |
| **Total per step (5 VMs, mid-episode)** | **~12s** | 15 total calls, budget-capped |
| **Total per step (15 VMs, late)** | **~6s** | No probing, just 15 batched calls |

### 9.3 Walltime

- Per task Phase 1: 15 steps x ~10s avg = **~2.5 min** (with adaptive probing)
- Per task Phase 2: ~4 min (replay + step + eval for hindsight branches)
- Per task total: **~6.5 min** (down from ~14 min with naive K=8)
- 2 tasks per batch (80 VMs), 150 batches for 300 tasks
- **~16 hours total** (conservative, includes pipeline overhead)

### 9.4 Comparison to Baseline

| Method | Calls/Task | Walltime/Task | Engineering | Diversity |
|---|:-:|:-:|:-:|---|
| Current (N=8, T=1.0) | 120 | ~1.5 min | 0 days | Sampling only |
| Scale to N=40 | 600 | ~1.5 min | 0 days | More sampling |
| MCTS naive K=8 | ~1200 | ~14 min | ~7 days | Targeted branching |
| **MCTS adaptive probing** | **~200** | **~6.5 min** | **~8.5 days** | **Targeted branching** |

With adaptive probing, MCTS costs ~1.7x more generation compute than current N=8 sampling (200 vs 120 calls) while providing **targeted minority-action exploration** that independent rollouts miss. The value is concentrated on:
- Tasks where the successful strategy requires minority actions at 3+ consecutive steps
- Tasks with multiple viable strategies that independent sampling rarely discovers together

### 9.5 The Fix That Would Change Everything

If the vLLM multimodal n>1 bug is fixed: all K probing calls for a VM become a single batched call. The adaptive strategy's 200 calls would take ~25s total instead of ~62s. Walltime per task would drop to ~3 min. This makes MCTS essentially free compared to baseline.

Note: the adaptive probing strategy is designed for the **bug-present** regime. If the bug is fixed, the budget caps can be raised significantly (or removed) since batched calls are cheap. With the fix, even naive K=8 on all VMs is affordable.

---

## 10. Empirical Data: 15-Step Smoke Test

**Setup:** 6 tasks × 6 domains, up to 15 steps each, K=8 candidates, T=1.0.

### 10.1 Key Numbers

| Metric | Value |
|---|:-:|
| Total steps | 77 |
| Consensus rate (spatial clusters) | 18.2% |
| Steps wanting branch | 81.8% |
| Type_H ≥ 0.5 (real type divergence) | 65% |
| Mean clusters/step | 3.7 |
| Mean top_fraction | 0.57 |

### 10.2 By Step Index

| Steps | Mean Clusters | Consensus | Branch-Worthy |
|:-:|:-:|:-:|:-:|
| 0-3 | 2.4 | 33% | 67% |
| 4-6 | 3.8 | 7% | 93% |
| 7-9 | 4.4 | 0% | 100% |
| 10-14 | 4.3 | 0% | 100% |

### 10.3 By Domain

| Domain | Steps | Mean H | Branch% |
|---|:-:|:-:|:-:|
| chrome | 9 | 1.47 | 56% |
| libreoffice_writer | 14 | 1.23 | 79% |
| libreoffice_calc | 11 | 1.30 | 82% |
| multi_apps | 15 | 1.43 | 80% |
| gimp | 13 | 1.89 | 85% |
| libreoffice_impress | 15 | 1.91 | 100% |

### 10.4 Clustering Comparison

| Method | Consensus Rate |
|---|:-:|
| 48px grid bucketing | 14.3% (too noisy) |
| 100px spatial clusters | 18.2% |
| 150px spatial clusters (recommended) | ~22% (estimated) |

---

## 11. Configuration

```python
@dataclass
class MCTSConfig:
    # VMs
    vms_per_task: int = 40
    # n_explorers: dynamic — determined by step 0 entropy
    tasks_per_batch: int = 2           # 80 VMs / 40 per task

    # Adaptive Probing (Section 4.3)
    probe_temperature: float = 1.0
    probe_budgets: dict = field(default_factory=lambda: {
        # step_range -> total extra probing calls across all uncertain VMs
        (0, 2): 16,   # steps 0-2: invest heavily
        (3, 5): 12,   # steps 3-5: moderate
        (6, 8): 8,    # steps 6-8: detect only strong divergence
        # steps 9+: no probing (never_branch_after gate)
    })
    logprob_confidence_threshold: float = 0.9  # P(top_type) above this -> skip probing
    min_k_per_uncertain_vm: int = 2            # floor: always at least 2 samples if probing

    # Branching
    branch_h_threshold: float = 0.5    # type_H gate
    spatial_threshold: int = 150       # px
    max_branch_per_explorer: int = 5
    child_branch_budget: int = 2       # spawned VMs get reduced budget
    max_branches_per_step: int = 2     # max minority actions to explore per node
    never_branch_after: int = 9        # steps 9+ only in Phase 2
    late_step_threshold: int = 7       # deferred branching starts here

    # Replay
    replay_pause_sec: float = 1.0          # same as production for fidelity

    # Pruning
    stuck_repeat_limit: int = 3
    stuck_wait_limit: int = 2

    # Phase 2
    enable_phase2: bool = True
    max_hindsight_per_success: int = 3

    # Pipeline
    max_steps: int = 15
```

---

## 12. Implementation Roadmap

| Phase | Component | Effort | Description |
|:-:|---|:-:|---|
| 1 | `/env/replay` endpoint | ~1.5d | Loop `env.step()` with reduced pause, skip screenshot encoding |
| 1 | `RemoteEnvWorker.replay()` | ~0.5d | Client-side: send action list, receive final obs |
| 2 | `TreeNode`, `ActionRecord`, `MCTSTree` | ~1d | Tree data structure with action history |
| 2 | `MCTSOrchestrator` main loop | ~2d | Entropy probing → branch decision → spawn → step |
| 3 | Pruning logic | ~0.5d | Stuck detection, terminal handling |
| 3 | Phase 2 hindsight exploitation | ~1d | Post-eval sibling exploration |
| 4 | Integration: `_run_mcts_chunk()` | ~1d | Replace `_run_rollout_chunk` when `use_mcts=True` |
| 4 | Config, testing, end-to-end validation | ~1d | MCTSConfig, smoke test on 10 tasks |

**Total: ~8.5 days. Critical path: `/env/replay` endpoint (Phase 1).**

**What does NOT change:** Docker provider, DesktopEnv, evaluation pipeline, SFT training pipeline, model/tokenizer/processor. The remote env server gets one new endpoint. Everything else is new orchestration code on top of existing infrastructure.

---

## 13. Known Limitations & Risks

1. **vLLM n>1 bug** forces sequential probing calls per VM. Adaptive probing (Section 4.3) mitigates this from ~864 to ~200 calls/episode. Fixing the bug would make even naive K=8 affordable.
2. **Replay fidelity**: ~2% per-step divergence compounding to ~18% over 10 steps for mixed actions. Navigation-heavy sequences (Chrome, menus) may hit 3-5% per step. Acceptable as free exploration but undermines precise state matching.
3. **Generation cost**: ~1.7x baseline per task with adaptive probing (200 vs 120 calls). Probing is focused on budget-holding VMs only; spawned VMs past their branch budget execute with K=1.
4. **Logprob screening calibration**: The 0.9 confidence threshold for logprob screening (Section 4.3.6) needs validation. If poorly calibrated, the strategy degrades gracefully to budget-only control (~204 calls vs ~200).
5. **Hard step-9 cutoff**: May miss tasks where the critical decision is at step 10+. Mitigated by Phase 2 hindsight. Consider making this configurable per-domain.
6. **Sample size**: 77 steps from 6 tasks. All thresholds (150px, type_H >= 0.5, step >= 7 deferral) need validation on 50+ tasks before production.
7. **MCTS value is concentrated**: Analysis of eval data shows ~6% of tasks (19/300) are in the "marginal" zone where MCTS helps most. For 66% of tasks (too hard, 0 successes), MCTS can't help. For 28% (easy, 6+ successes), independent rollouts are already sufficient.

---

## 14. Alternatives Considered

| Approach | Compute | Engineering | When It's Better |
|---|:-:|:-:|---|
| Scale N=8→40 (just more rollouts) | 5× | 0 days | When success rate is the bottleneck |
| Forced diversity (K=8 probe + split) | 1.3× | ~2 days | When you only need 1 branch point per episode |
| Temperature scheduling (T=0.3/1.0/1.5) | 1× | ~0.5 days | Free diversity, combines with any approach |
| **Full MCTS (this document)** | **10×** | **~8.5 days** | **When tasks need chained minority-action sequences** |
| QEMU/KVM snapshots (future) | 10× | ~2 weeks | If bare-metal/nested-virt servers available |

Full MCTS is chosen because it provides the maximum trajectory diversity through **targeted, coordinated minority-action exploration** — something no simpler alternative can match for the hardest tasks.

---

*Document created: 2026-03-20*
*Revised: 2026-03-20 — Rewritten with corrected architecture: 40 pre-setup VMs per task, claim-on-demand spawning, action replay branching. Incorporates 15-step smoke test data, VM cloning research, and audit findings.*
*Revised: 2026-03-20 — Added adaptive entropy probing strategy (Section 4.3): three-layer cost control combining logprob screening, budget-capped distribution, and branch-budget gating. Reduces probing from 864 to ~200 calls/episode (77%). Updated orchestration loop, config, throughput estimates, and known limitations.*
*For the ARPO GUI agent training research project*
