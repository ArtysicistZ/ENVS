# Action-Level MCTS for SFT Trajectory Collection

**Date:** 2026-03-20
**Status:** Implemented and validated (GPU smoke tests passing)
**Context:** ARPO codebase, UI-TARS-1.5 (Qwen2.5-VL 7B), 80 Docker VMs (3 servers), 8x A100 80GB

---

## 1. Core Idea

At each step, generate K action candidates and inspect the distribution. If the model is genuinely uncertain (multiple semantically distinct strategies), **spawn a new VM** to explore the minority action while the original VM continues the majority path.

The tree **grows on demand**. All 40 VMs per task are pre-setup with the same task at the start and sit waiting. When a branch is needed, claim a waiting VM and replay the action history to that point. The hard VM cap is `config.max_active_vms` -- the single place to control maximum resource utilization.

---

## 2. Architecture: 40 Pre-Setup VMs, Claim On Demand

### 2.1 Setup Phase

At task start, all 40 VMs reset and run `setup_controller.setup()` for the same task in parallel (~30s, bounded by the slowest VM). Done once.

### 2.2 Exploration Phase

**Step 0 uses the same logic as all other steps.** There is no special-cased step 0. One root VM probes K=16 candidates, branches if clusters exist. The number of initial VMs is determined by the data, not hardcoded.

```
Unified step loop (all steps including step 0):

1. Get screenshots from all active nodes
2. Generate K candidates per node (K=16 at step 0, K=8 otherwise)
3. Cluster by fingerprint -> branch decisions
4. Spawn child VMs for minority clusters (replay parent's physical actions)
5. Execute actions on all active VMs
6. Prune stuck/terminal VMs
```

### 2.3 Resource Layout

```
40 VMs per task (2 tasks in parallel = 80 VMs total)
+-- 1 VM starts stepping at step 0
+-- N-1 VMs claimed dynamically at branch points (N determined by entropy)
+-- remaining: idle, pre-setup, waiting

Hard VM cap: config.max_active_vms (default 40)
```

### 2.4 Dynamic VM Count

The number of active VMs is never hardcoded. It is driven entirely by the model's action distribution and gated by `max_active_vms`.

---

## 3. VM Cloning: Action Replay

### 3.1 Mechanism

All 40 VMs have the same task setup. To "clone" a parent's state at step N onto a new VM, replay the parent's **physical** action sequence:

```python
# new_vm already has: reset done, task setup done, same initial state
# Replay the parent's physical action sequence:
env_client.replay(parent_node.get_physical_action_sequence(), replay_pause_sec=1.0)
```

No reset needed. No setup needed. Just replay the action sequence.

### 3.2 The `/env/replay` Endpoint

```
POST /env/replay
{
  "slot_id": 3,
  "predictions": ["Thought: ... Action: click(...)", "Thought: ... Action: type(...)"],
  "replay_pause_sec": 1.0
}
```

### 3.3 Replay Fallback

If `/env/replay` returns 404 (not implemented on the server), `MCTSEnvClient` automatically falls back to sequential `step()` calls. This is implemented in `env_client.py`:

```python
except requests.exceptions.HTTPError as e:
    if e.response is not None and e.response.status_code == 404:
        return self._replay_fallback(predictions, replay_pause_sec)
```

---

## 4. Clustering: 50px Fingerprint Grid

### 4.1 Action Fingerprinting

Replaces the earlier 150px single-linkage spatial clustering. Each candidate action gets a discrete fingerprint:

- **Spatial actions** (click, drag, scroll with coordinates): Quantize `(x, y)` to a 50px grid. Example: `click@(12,8)` means coordinates in the 50px cell at grid position (12, 8).
- **Non-spatial actions** (hotkey, type, etc.): Group by content/key string. Example: `hotkey:k:ctrl+s`, `type:c:hello world`.

Two actions with the same fingerprint are "the same strategy." Two with different fingerprints are candidates for branching.

### 4.2 Singletons Count

`min_cluster_size=1`. A single candidate with a unique fingerprint forms its own cluster. This means even rare actions are eligible for branching.

### 4.3 Content Normalization

- Whitespace: `"ctrl  alt t"` -> `"ctrl alt t"`
- Key order: `"alt+ctrl+t"` -> `"ctrl+alt+t"`
- Type content: truncated to 30 chars

---

## 5. Probing: Per-Node K

Every node probes **K=8 candidates at every step**. This is a per-node budget, not shared across nodes. At step 0: **K=16** (higher investment since only 1 node exists).

All K candidates are generated via the VLLMPool, distributed round-robin across GPUs.

---

## 6. Branching Strategy

### 6.1 Simplified 4-Gate Branch Filter

```
Gate 1: >= 2 distinct fingerprint clusters           (no real divergence -> skip)
Gate 2: Late-step deferral at step >= 10             (require 2 consecutive high-entropy steps)
Gate 3: Branch budget remaining                      (root: 5, child: 2)
Gate 4: Step cutoff = 15                             (effectively disabled)
```

**No type_H threshold.** The old 0.5 entropy gate was removed. If there are 2+ distinct clusters, the signal is sufficient.

### 6.2 max_branches_per_step = 3

Each node can spawn up to 3 child VMs per step (one per minority cluster). Consensus from literature: all papers use 3-5.

### 6.3 Branch Budget

Root VMs get 5 branch events. Spawned child VMs get 2. This prevents exponential cascade while allowing second-level branching.

### 6.4 Pruning

Nodes are marked done (stop stepping) when:
1. **Terminal action**: `finished()` or `fail()`
2. **Stuck loop**: Same action repeated 3 consecutive steps
3. **Stalled**: `wait()` twice in a row

Pruned nodes still have their trajectories saved and evaluated.

---

## 7. Physical vs Logical Action Sequences

The tree distinguishes two types of action sequences per node:

- **`get_action_history()`** (logical): Walks the parent chain. Used for building model input prompts -- the model sees the full decision history from root.
- **`get_physical_action_sequence()`** (physical): `replay_prefix + action_history`. This is what was actually executed on this node's VM. Used for replaying state on new VMs.

Each child node stores `replay_prefix` at spawn time -- the parent's physical action sequence at the moment of branching.

---

## 8. 8-GPU VLLMPool

8 independent vLLM engines (TP=1 each), one per GPU. Generation requests are distributed round-robin.

```
Per GPU: ~16GB model + ~52GB KV cache = ~68GB on 80GB A100
K=16 candidates on 8 GPUs: 2 per GPU -> ~4s total
```

**Validated**: 10x speedup (153s vs 1510s) on the 8-GPU smoke test.

---

## 9. Trajectory Format: SFT Compatible

MCTS trajectories are a **strict superset** of the standard trajectory format. The existing SFT pipeline (`trajectory_sft.py`, `expand_episode`, `select_sft_trajectories.py`) reads `task_id`, `instruction`, `eval_result`, `limit_images`, `steps` and ignores unknown fields. No changes needed to the SFT pipeline.

MCTS-specific fields (`branch_path`, `tree_depth`, `diverged_at_step`, `parent_vm_idx`, `node_id`) are metadata only.

---

## 10. Validated Smoke Test Results

### 10.1 1-GPU, 10 VMs, 15 steps, GIMP brightness task

| Metric | Value |
|---|---|
| VMs used | 10/10 (all claimed by step 3) |
| Successful trajectories | 4/10 (eval=1.0) |
| Successes | node_001 (8 steps), node_003 (14 steps), node_006 (9 steps), node_010 (7 steps, grandchild) |
| SFT compatibility | OK (10 training examples per trajectory) |
| Total time | 1510s on 1 GPU |

### 10.2 8-GPU VLLMPool, 10 VMs, 15 steps, GIMP brightness

| Metric | Value |
|---|---|
| GPUs active | 8/8 (75GB each, 55-57% utilization) |
| Speedup | 10x (153s vs 1510s) |
| Successful trajectories | 2/2 |
| SFT compatibility | OK |

---

## 11. Configuration

```python
@dataclass
class MCTSConfig:
    # ---- VMs ----
    vms_per_task: int = 40
    max_active_vms: int = 40           # HARD CAP
    tasks_per_batch: int = 2

    # ---- Probing ----
    probe_temperature: float = 1.0
    k_per_node: int = 8                # K candidates per node per step
    k_step0: int = 16                  # K at step 0

    # ---- Branching ----
    spatial_grid_size: int = 50        # px grid for fingerprinting
    min_cluster_size: int = 1          # singletons count
    max_branch_per_explorer: int = 5   # root VM branch budget
    child_branch_budget: int = 2       # spawned VMs
    max_branches_per_step: int = 3     # max minority clusters to spawn per node
    never_branch_after: int = 15       # effectively disabled
    late_step_threshold: int = 10      # deferred branching

    # ---- Replay ----
    replay_pause_sec: float = 1.0

    # ---- Pruning ----
    stuck_repeat_limit: int = 3
    stuck_wait_limit: int = 2

    # ---- Phase 2 (Hindsight) ----
    enable_phase2: bool = True
    max_hindsight_per_success: int = 3

    # ---- Pipeline ----
    max_steps: int = 15

    # ---- GPU / Inference ----
    model_path: str = "ByteDance-Seed/UI-TARS-1.5-7B"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 32768
    limit_images: int = 3
    generation_max_tokens: int = 512
```

---

## 12. Implementation Files

```
verl/mcts/
+-- __init__.py
+-- config.py            # MCTSConfig with max_active_vms hard cap
+-- tree.py              # TreeNode with replay_prefix, BranchBudget, MCTSTree
+-- clustering.py        # 50px fingerprint grid, min_cluster_size=1, 4-gate should_branch()
+-- trajectory_io.py     # SFT-compatible JSONL format (strict superset)
+-- orchestrator.py      # Unified step loop, per-node K, hard VM cap
+-- env_client.py        # Lightweight HTTP client with /env/replay fallback
+-- vllm_pool.py         # 8-GPU parallel generation pool (TP=1 each, round-robin)

scripts/mcts/
+-- smoke_mcts_gpu.py    # GPU smoke test (1-GPU or 8-GPU)
```

---

*Document created: 2026-03-20*
*Revised: 2026-03-20 -- Rewritten to reflect validated implementation. Removed speculative content (adaptive logprob screening, per-step probe budgets, type_H thresholds, vLLM n>1 bug workarounds). Updated all parameters and results with validated GPU smoke test data.*
*For the ARPO GUI agent training research project*
