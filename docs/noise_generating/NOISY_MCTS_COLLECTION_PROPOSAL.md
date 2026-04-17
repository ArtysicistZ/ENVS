# Proposal: MCTS Trajectory Collection on Noisy Environments

**Date**: 2026-04-15  
**Status**: Proposal  

---

## 1. Objective

Collect MCTS trajectories on **noisy** OSWorld environments using 192 VMs across 4 clusters (48 VMs each). Maximum tree branches per task: 192. Unlike ARPO training which uses CRN (identical noise across rollouts), MCTS uses **truly random noise per branch** — each branch explores a different noise instance, maximizing diversity in collected recovery trajectories.

---

## 2. Current Clean MCTS System

### Architecture (from `verl/mcts/orchestrator.py`)

- **One task at a time**, all VMs dedicated to that task
- **Tree growth**: K action candidates per node → action fingerprint clustering → branch on divergence
- **Branching**: Child node gets a fresh VM, replays parent's full action sequence, then takes the branch action
- **No Docker snapshots**: State is reconstructed via action replay (deterministic execution assumed)
- **vLLM inference**: Multi-GPU batched generation for K candidates per node

### Previous Collection (v2, 86 tasks)

| Parameter | Value |
|-----------|-------|
| VMs | 112 (4 servers: 32+16+32+32) |
| k_max | 64 candidates per node |
| probe_budget | 96 prompts per step |
| max_branches/step | 2 |
| max_steps | 15 |
| Model | UI-TARS-1.5-7B (TP=8) |
| Results | 22,140 nodes, 2,927 successes (13.2% SR) |

### Key Files

| File | Purpose |
|------|---------|
| `verl/mcts/orchestrator.py` | Main MCTS loop (tree growth, branching, evaluation) |
| `verl/mcts/tree.py` | TreeNode, MCTSTree, BranchBudget |
| `verl/mcts/clustering.py` | Action fingerprinting, clustering, branch decisions |
| `verl/mcts/env_client.py` | HTTP client to remote env servers |
| `verl/mcts/vllm_pool.py` | Multi-GPU vLLM orchestration |
| `verl/mcts/tree_io.py` | Tree serialization, trajectory reconstruction |
| `verl/mcts/trajectory_io.py` | Per-node trajectory format for SFT/KTO |
| `scripts/mcts/run_mcts_collection.py` | Production entry point |

---

## 3. Key Design Decision: Truly Random Noise Per Branch

### Why Different From ARPO

| Aspect | ARPO Training | MCTS Collection |
|--------|---------------|-----------------|
| Goal | Fair advantage comparison within GRPO group | Diverse trajectory exploration |
| Noise seed | CRN: `hash(task_id, training_step)` — identical across n=8 | Random: `hash(task_id, node_id)` — unique per branch |
| Noise schedule | Same fires at same steps for all rollouts | Different fires at different steps per branch |
| Recovery diversity | All rollouts face same obstacle | Different branches face different obstacles |
| Downstream use | Policy gradient (needs fair comparison) | SFT/KTO training data (needs diversity) |

### What Random Noise Per Branch Gives Us

1. **Diverse recovery trajectories**: Branch A faces a modal at step 3, Branch B faces focus steal at step 7. Successful branches that recover from different noise types → rich training data.
2. **Noise-conditioned exploration**: Some branches may discover that certain action sequences are resilient to noise by accident → these become high-value SFT data.
3. **Better curriculum signal**: Per-task noise resilience measured across many different noise instances, not just one CRN instance.

### Implementation

In `orchestrator.py`, when creating a child node, assign a unique noise seed:

```python
# Current CRN (ARPO):
# noise_step_seed = hash(task_id, global_step)  — same for all branches

# Noisy MCTS:
# Each node gets its own noise seed
child_node.noise_seed = hash((task_id, node_id, branch_idx))
# Passed to env via task_config during reset or step
task_config["noise_step_seed"] = child_node.noise_seed
```

---

## 4. Noise Triggering Logic: Adaptive to Task Difficulty

### Core Principle

The number of noise fires should match what the task can absorb. Hard tasks already struggle without noise — adding noise destroys the learning signal. Easy tasks can handle more disruption and benefit from harder noise pressure.

### Fire Count Based on Clean MCTS SR

Use the per-task success rate from the v2 clean MCTS collection (`collection_results.json`) to determine noise intensity:

| Clean SR | Fire Count | Rationale |
|----------|-----------|-----------|
| SR < 0.15 | **0 fires** | Very hard task. Agent rarely succeeds clean. Noise would produce only failures — zero useful data. |
| SR 0.15–0.60 | **1 fire** | Hard/medium task. One noise event is recoverable. Any element type (no tier cap). |
| SR > 0.60 | **2 fires** | Easy task. Agent succeeds often. Two fires test whether the agent can handle sequential disruptions — a harder but learnable challenge. |

### No Tier Cap — Feasibility Constraint Instead

Any element from the full noise library (cost 0 to cost 3) can be selected. The constraint is **placement feasibility**: each fire must leave enough steps for recovery AND task completion.

```
For each fire:
  min_fire_step = 3                    # Let agent start the task
  max_fire_step = max_steps - recovery_cost - min_task_buffer
  
  Where min_task_buffer = 4            # Minimum steps to complete task after recovery
  
  Examples (max_steps = 15, 1 fire):
  - Cost-0 notification:  fire window = [3, 11]
  - Cost-1 modal:         fire window = [3, 10]
  - Cost-2 occlusion:     fire window = [3, 9]
  - Cost-3 composite:     fire window = [3, 8]

For 2 fires (easy tasks):
  Fire 1 placed in first half: [3, 7]
  Fire 2 placed in second half: [fire_1_step + recovery_cost_1 + 2, max_steps - recovery_cost_2 - 4]
  This ensures non-overlapping recovery windows.
```

### 20% Clean Control Branches

To preserve DPO contrastive pairs and baseline measurement:
- **80% of branches**: noise fires (count based on SR above)
- **20% of branches**: clean (no noise) — control group for comparison

### Why This Works

- **Very hard tasks (SR<0.15)**: No noise → collect more clean successes to strengthen the base policy
- **Hard/medium tasks (SR 0.15–0.60)**: 1 fire → the model learns to handle one interruption at a time, which is the most learnable unit of recovery
- **Easy tasks (SR>0.60)**: 2 fires → the model learns sequential recovery (dismiss first noise → continue → dismiss second noise → complete task), building compositional robustness
- **No tier cap**: Full diversity of noise types at every difficulty level. The feasibility constraint (not tier cap) ensures the agent has enough time to recover

---

## 5. Replay Under Random Noise (was §4)

### The Problem

When a child node replays the parent's action sequence to reach the branch point, the child sees **different noise** (different seed). The parent's actions were taken under the parent's noise state. If the child encounters noise at a step where the parent didn't, the replayed action may interact with a noise artifact instead of the task UI.

### Solution: Noise-Free Replay, Noise Starts at Branch Point

Disable noise during replay prefix, enable only from the branch step onwards:

```python
# In env_client.replay():
def replay(self, predictions, task_config, branch_step):
    """Replay parent's actions WITHOUT noise, then enable noise."""
    # Phase 1: Replay prefix (noise disabled)
    replay_config = dict(task_config)
    replay_config["enable_noise"] = False
    for i, action in enumerate(predictions[:branch_step]):
        self.step(action, noise_probability=0.0)
    
    # Phase 2: Enable noise from branch point onwards
    # Re-initialize noise scheduler with child's unique seed
    replay_config["enable_noise"] = True
    replay_config["noise_step_seed"] = child_noise_seed
    self._reinit_noise_scheduler(replay_config, start_step=branch_step)
```

This ensures:
- Replay fidelity: child reaches the same state as parent at the branch point
- Noise diversity: child encounters its own unique noise from the branch onwards
- No replay drift: actions are deterministic without noise interference

### Alternative: Noise During Replay (Aggressive)

If we want maximum diversity, allow noise during replay too. Accept that child's state may diverge from parent. The child explores from a slightly different starting state, which is fine for data collection (not training). This could be a config flag:

```yaml
mcts:
  noise_during_replay: false  # Conservative (default)
  # noise_during_replay: true  # Aggressive: more diversity, less replay fidelity
```

---

## 5. Infrastructure: 192 VMs Across 4 Clusters

### Cluster Layout

| Cluster | Host | Slots | IP |
|---------|------|-------|----|
| 1 | deepx-a100-40g-1 | 48 | 10.100.4.4:15001 |
| 2 | deepx-a100-40g-2 | 48 | 10.100.4.6:15001 |
| 3 | deepx-a100-40g-3 | 48 | 10.100.4.8:15001 |
| 4 | deepx-a100-40g-4 | 48 | 10.100.4.7:15001 |

**Total**: 192 VMs = 192 max branches per task

### Config

```yaml
# configs/mcts_collection_v3_noisy_192vm.yaml

mcts:
  remote_server_urls:
    - "http://10.100.4.4:15001"
    - "http://10.100.4.6:15001"
    - "http://10.100.4.8:15001"
    - "http://10.100.4.7:15001"
  remote_server_env_counts: [48, 48, 48, 48]
  max_active_vms: 192

  # Probing
  total_probe_budget: 128
  k_max: 64
  probe_temperature: 1.0

  # Branching
  spatial_grid_size: 50
  max_branch_per_explorer: 6    # Root can branch up to 6 ways
  child_branch_budget: 3        # Children can branch up to 3 ways
  max_branches_per_step: 3      # Up to 3 branches per decision point
  late_step_threshold: 10
  never_branch_after: 13        # Don't branch in last 2 steps

  # Noise — Adaptive to task difficulty, no tier cap
  enable_noise: true
  noise_mode: runtime_library
  noise_per_branch: true         # Unique noise seed per branch
  noise_during_replay: false     # Noise only from branch point onwards
  noise_branch_probability: 0.8  # 80% noisy, 20% clean control
  noise_min_fire_step: 3         # No noise in first 3 steps
  noise_min_task_buffer: 4       # Minimum 4 steps after recovery for task
  # Fire count per task SR (from v2 clean collection_results.json):
  #   SR < 0.15  → 0 fires (protect very hard tasks)
  #   SR 0.15–0.60 → 1 fire (any element, any cost, feasibility-constrained)
  #   SR > 0.60  → 2 fires (sequential, non-overlapping recovery windows)
  # No tier cap — element selection is unrestricted, placement ensures recovery is feasible

  # Full tree saving (same format as v2 clean collection)
  save_full_tree: true           # Save complete tree structure via tree_io.py
  save_trajectories: true        # Also save flat per-node trajectories

  # Model
  model: checkpoints/mcts_sft_v2.1/beta05_2e-6/epoch_1
  tensor_parallel_size: 8
  gpu_memory_utilization: 0.85
  max_model_len: 12000

  # Pipeline
  max_steps: 15
  replay_pause_sec: 1.0
  step_timeout_sec: 90          # Increased for noise overhead
  reset_timeout_sec: 450
  eval_timeout_sec: 450

data:
  task_file: OSWorld/evaluation_examples/test_86tasks_trainable.json
```

### VM Budget Per Task

With 192 VMs and up to 192 branches:
- Root: 1 VM
- Step 1-5: ~10-20 branches (action clustering triggers on divergent actions)
- Step 6-10: ~40-80 branches (tree widens)
- Step 11-15: ~80-192 branches (maximum exploration)
- Pruning: stuck/failed nodes release their VMs back to the pool

Expected utilization: ~60-80% of 192 VMs active at any step (some pruned, some waiting for replay).

---

## 6. Orchestrator Changes for Noisy MCTS

### 6.1 Per-Node Noise Seed

In `verl/mcts/tree.py`, add noise state to TreeNode:

```python
@dataclass
class TreeNode:
    # ... existing fields ...
    noise_seed: int = 0           # Unique noise seed for this node
    noise_tier: int = 2           # Noise tier for this node
    noise_recovery_events: int = 0  # Count of successful noise recoveries
```

### 6.2 Modified Branching in Orchestrator

In `verl/mcts/orchestrator.py`, when spawning children:

```python
def _spawn_child(self, parent, branch_action, branch_idx, vm_slot_id):
    child = TreeNode(
        node_id=f"node_{self._next_node_id()}",
        parent=parent,
        vm_slot_id=vm_slot_id,
        depth=parent.depth,
        replay_prefix=parent.get_physical_action_sequence(),
        # Unique noise seed per child
        noise_seed=hash((self.task_id, child.node_id, branch_idx)) & 0xFFFFFFFF,
        noise_tier=self._noise_tier_for_branch(parent),
    )
    # Reset VM and replay with noise-free prefix
    self._async_replay_child(child, branch_action)
    return child
```

### 6.3 Noise-Aware Evaluation

After tree exploration, evaluate terminal nodes:
- **Noisy eval**: Evaluate in the noisy state as-is → measures noise-resilient success
- **Clean eval** (optional): Re-run the successful trajectory without noise to verify the task was actually completed vs. noise accidentally helping

```python
def _evaluate_node(self, node):
    score = self.env_client.evaluate(node.vm_slot_id)
    node.eval_score = score
    # Also record noise metadata for this node's trajectory
    node.noise_metadata = {
        "seed": node.noise_seed,
        "tier": node.noise_tier,
        "recovery_events": node.noise_recovery_events,
        "total_noise_fires": node.total_noise_fires,
    }
```

### 6.4 Recovery Detection During Collection

Track noise recovery at each step (using the window-focus detection from the dense reward proposal):

```python
# In env_client.step(), parse response for recovery signal:
def step(self, slot_id, action, noise_probability=0.0):
    response = self._post(f"/env/step", ...)
    noise_burden = response.get("noise_burden", {})
    recovery = noise_burden.get("recovery_detected", False)
    return response, recovery
```

This enables labeling trajectories with:
- **clean_success**: Task completed, no noise encountered
- **recovery_success**: Task completed after recovering from noise
- **noisy_success**: Task completed with noise active but no recovery needed
- **noisy_failure**: Task failed due to noise
- **clean_failure**: Task failed without noise interference

---

## 7. Tree Output & Downstream Use

### 7.0 Full Tree Saving (Same as Clean v2)

Use the existing `tree_io.py` full tree serialization — same format as the v2 clean collection. Each tree is saved as a single JSON file containing:

- **Complete tree structure**: all nodes with parent/children pointers
- **Per-node own_steps**: only the steps executed by that node (not duplicated parent data)
- **Back-propagated Q-values**: leaf Q = eval_score, internal Q = mean(children Q)
- **Noise metadata per node**: seed, tier, fires, recoveries (new addition)

This preserves:
- `get_sibling_pairs()` for DPO (winner/loser at branch points)
- `get_revision_trajectories()` for failed→successful splices
- `reconstruct_trajectory()` for full root-to-leaf paths
- All existing SFT/KTO pipelines work unchanged on the tree structure

File: `verl/mcts/tree_io.py` — `save_mcts_tree()` / `load_mcts_tree()`
Output: `checkpoints/mcts_trajectories_v3_noisy/{task_id}/tree.json`

### 7.1 Enhanced Trajectory Format

```json
{
  "task_id": "uuid",
  "instruction": "user instruction",
  "eval_result": 1.0,
  "noise_metadata": {
    "seed": 123456,
    "tier": 2,
    "fires": [{"step": 3, "category": "modal"}, {"step": 8, "category": "focus_steal"}],
    "recovery_events": [{"step": 4, "recovered_from": "modal", "cost": 1}],
    "total_recovery_cost": 1
  },
  "trajectory_tag": "recovery_success",
  "steps": [
    {"screenshot_b64": "...", "action": "...", "noise_active": false},
    {"screenshot_b64": "...", "action": "...", "noise_active": false},
    {"screenshot_b64": "...", "action": "...", "noise_active": false},
    {"screenshot_b64": "...", "action": "click(450, 320)", "noise_active": true, "noise_event": "modal"},
    {"screenshot_b64": "...", "action": "click(960, 540)", "noise_active": false, "recovery": true},
    ...
  ]
}
```

### 7.2 Downstream Training Use

| Use | How |
|-----|-----|
| **SFT on recovery** | Train on `recovery_success` trajectories. The model learns: see modal → click dismiss → continue task. Step-level loss masking: full loss on recovery steps, normal loss on task steps. |
| **KTO / DPO pairs** | At branch points with noise: (winner = recovered + completed, loser = didn't recover). The tree structure naturally provides these pairs when siblings have different noise outcomes. |
| **Noise curriculum seeding** | Per-task noisy SR from MCTS → seeds ARPO curriculum with realistic noise-resilience priors (not just clean SR). |
| **Replay buffer init** | Pre-populate ARPO replay buffer with `recovery_success` trajectories, solving the cold-start problem. |
| **Recovery reward training data** | Steps where recovery was detected → used to validate and tune the dense recovery reward system. |

### 7.3 Sibling Pairs Under Noise

The tree produces powerful contrastive pairs:

```
Root (clean state)
├── Branch A (noise seed 111): modal at step 3
│   ├── Sub-branch A1: clicks dismiss → recovers → completes task (score=1.0)
│   └── Sub-branch A2: clicks randomly → stuck → fails (score=0.0)
└── Branch B (noise seed 222): focus steal at step 7  
    ├── Sub-branch B1: Alt-Tabs → recovers → completes task (score=1.0)
    └── Sub-branch B2: keeps clicking → wrong window → fails (score=0.0)
```

**Pairs for DPO**: (A1, A2) and (B1, B2) — each pair shows the SAME noise, different recovery behavior.

---

## 8. Scaling Considerations

### 8.1 Replay Cost

With 192 VMs and deep trees, replay cost grows:
- Branch at step 5: replay 5 actions (~5-10s)
- Branch at step 10: replay 10 actions (~10-20s)
- Branch at step 13: replay 13 actions (~13-26s)

Mitigation: **cluster-local branching**. Prefer branching to VMs on the same cluster (lower network latency for replay). If cluster A's VMs are exhausted, overflow to cluster B.

### 8.2 vLLM Throughput

With 192 active nodes at peak, each needing K candidates:
- K=4 per node × 192 nodes = 768 prompts per step
- 8 GPUs at TP=1 processing 8 prompts simultaneously
- ~96 batches × ~2s = ~3 minutes per step
- Probe budget cap (128) limits this to 128 prompts max per step

### 8.3 Task Throughput

Estimated per-task collection time:
- 15 steps × (probing + branching + execution) ≈ 30-45 minutes per task
- 86 tasks × 40 min ≈ 57 hours (~2.5 days)
- With noise overhead (replay, recovery detection): ~70 hours (~3 days)

### 8.4 Storage

Per task: ~500MB (screenshots + tree structure)
86 tasks: ~43GB total
Stored at: `checkpoints/mcts_trajectories_v3_noisy/`

---

## 9. Implementation Phases

### Phase 1: Infrastructure Setup
- Deploy 48-slot env servers on all 4 hosts
- Verify all 192 VMs healthy
- Add noise support to `/env/replay` endpoint (noise-free replay mode)
- Add recovery detection to env step (window focus check)

### Phase 2: Orchestrator Modifications
- Per-node noise seed in TreeNode
- Noise-free replay for children
- Noise initialization at branch point
- Recovery event tracking during collection
- Enhanced trajectory output with noise metadata

### Phase 3: Collection Run
- Start with 5-task smoke test (verify noise + branching works)
- Full 86-task collection (~3 days)
- Incremental result saving with resume support

### Phase 4: Downstream Integration
- SFT training on noisy trajectories (recovery-weighted loss)
- KTO/DPO on noise-conditioned sibling pairs
- Update ARPO curriculum seeding with noisy SR priors
- Pre-populate replay buffer with recovery trajectories

---

## 10. Files to Modify

| File | Change |
|------|--------|
| `verl/mcts/tree.py` | Add `noise_seed`, `noise_tier`, `noise_recovery_events` to TreeNode |
| `verl/mcts/orchestrator.py` | Per-node noise seeding, noise-free replay, recovery tracking |
| `verl/mcts/env_client.py` | Support noise config in reset/step/replay, recovery detection |
| `verl/mcts/trajectory_io.py` | Enhanced format with noise metadata and trajectory tags |
| `verl/mcts/tree_io.py` | Serialize noise metadata in tree output |
| `scripts/mcts/run_mcts_collection.py` | Accept noise config, 192-VM setup |
| `scripts/servers/remote_env_server.py` | Noise-free replay mode, recovery detection in step |
| `OSWorld/desktop_env/desktop_env.py` | Recovery detection (window focus check) |
| `configs/mcts_collection_v3_noisy_192vm.yaml` | New config for noisy 192-VM collection |

---

## 11. References

- Current clean MCTS: `configs/mcts_collection_v2_86tasks.yaml` (112 VMs, 86 tasks, 22K nodes)
- Dense recovery reward proposal: `docs/noise_generating/DENSE_RECOVERY_REWARD_PROPOSAL.md`
- Noise curriculum spec: `docs/noise_generating/CURRICULUM_SPEC.md`
- Research framing: `docs/noise_generating/RESEARCH_FRAMING.md`
