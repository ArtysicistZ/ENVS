# Per-GPU-Group Async Step Loop

## Problem

In the step loop (`_run_rollout_chunk`), all 32 VMs must finish their `step()` before
ANY GPU can generate the next token. With DP=8 / TP=1, each GPU independently processes
only 4 VMs — it doesn't need the other 28 VMs' data at all. The slowest VM across all 32
blocks every GPU.

## Architecture Facts

### DP=8, TP=1 — each GPU is independent

All configs use `tensor_parallel_size: 1`. With 8 GPUs:

```
dp_size = world_size // tp_size = 8 // 1 = 8
```

`generate_sequences` uses `Dispatch.DP_COMPUTE_PROTO` → `DataProto.chunk(8)` splits the
batch by position:

```
Batch of 32 prompts → chunk(8) →
  GPU 0: prompts[0:4]    (VMs 0-3)     — one independent vLLM engine
  GPU 1: prompts[4:8]    (VMs 4-7)     — one independent vLLM engine
  ...
  GPU 7: prompts[28:32]  (VMs 28-31)   — one independent vLLM engine
```

Each GPU runs its OWN vLLM engine on its 4 prompts. No cross-GPU data dependency during
generation.

### `generate_sequences` is a collective call

Despite each GPU being independent, `generate_sequences` dispatches to ALL 8 GPUs in one
call and waits for ALL 8 to return. There is no per-GPU dispatch mode in verl.

This means we can't literally "start GPU 0 while GPU 7 waits." But we CAN call
`generate_sequences` with a **partial batch** (only ready VMs) and pad to 8-divisible.

### How padding actually works (IMPORTANT)

`pad_dataproto_to_divisor` does NOT create dummy 1-token prompts. It **duplicates real
data** from the batch: `padding_protos.append(data[:take_size])`. So with 4 real prompts
padded to 8, all 8 GPUs get full 36K-token prompts (4 unique + 4 copies).

**Why this is fine:** Each GPU processes 1 prompt (after `chunk(8)` splits 8 → 1 each).
With batch=32 (full lockstep), each GPU processes 4 prompts. So `generate(8 padded)`
takes approximately **1/4 the time** of `generate(32)`. The duplicate results are
discarded by `unpad_dataproto`.

Total GPU work is constant: 32 prompts × 36K tokens = 1.15M tokens, regardless of
how many generate calls we split it across. The only extra cost is per-call overhead
(Python dispatch, ray.remote/get) — approximately 0.5s per call.

### What's NOT a constraint

- `prepare_generate_sequences()` / `finish_generate_sequences()` are called ONCE per
  chunk. Multiple `generate_sequences` calls within the step loop cost nothing extra.
- `generate_sequences` accepts variable batch sizes (must be divisible by world_size).
- vLLM doesn't care about step_idx — prompts at different steps can be batched together.

## Current Bottleneck

```python
# ray_trainer.py:1131-1137 — WAITS FOR ALL 32 VMs
_step_futures = [worker.step.remote(action_text) for ...]
env_outputs = _ray_get_robust(_step_futures, timeout=330, ...)
```

Concrete waste with 32 VMs:

```
time 0:     step.remote() dispatched to all 32 VMs
time 5:     Group 0-6 (28 VMs) done — 7 GPUs idle
time 20:    Group 7 (4 VMs) finally done
             → 15s × 15 steps × 8 chunks = 30 min wasted per training step
```

## Solution: Per-GPU-Group Async Stepping

### Core idea

Divide 32 VMs into 8 **groups** of 4, matching the DP split. Each group advances through
steps independently. When a group's 4 VMs all finish their current step, that group is
ready for the next generation call — it doesn't wait for other groups.

```
group_size = num_envs // world_size   (32 // 8 = 4)

Group 0: VMs [0, 1, 2, 3]     → all 4 finish step → group 0 ready
Group 1: VMs [4, 5, 6, 7]     → still running...
Group 2: VMs [8, 9, 10, 11]   → all 4 finish step → group 2 ready
...
```

When enough groups are ready, call `generate_sequences` for those groups. Other groups'
VMs keep running in the background.

### Why per-group, not per-VM

- **Statistical gain**: E[max(4)] ≈ p85, E[max(32)] ≈ p99. Per-group eliminates most
  tail-latency waste. Going per-VM gives diminishing returns (E[max(1)] = p50 vs p85).
- **Simpler state tracking**: 8 group counters vs 32 VM counters.
- **Matches the GPU split**: The DP dispatch naturally groups by 4.
- **Within-group wait is cheap**: Waiting for the slowest of 4 is seconds, not minutes.

### Detailed design

```
Phase 1: Initial generate (all 32 VMs from reset)
  ├─ prepare_vllm_inputs_full(all 32 reset outputs)
  ├─ generate_sequences(batch=32)  → DP split → 4 prompts per GPU
  ├─ Decode actions
  └─ Dispatch step.remote() to all 32 VMs
      Each VM now has an in-flight step future

Phase 2: Async group loop (replaces current lockstep for step_idx in range(max_steps))

  while any groups still active:
    │
    ├─ ray.wait() on ALL in-flight step futures
    │   Wait for at least 1 full group to complete (all 4 VMs in a group done)
    │
    ├─ For each newly-completed group:
    │   ├─ Collect 4 step results
    │   ├─ Accumulate format_rewards
    │   ├─ Handle is_done: launch evaluate.remote(), add to done_slots
    │   ├─ If group still has active VMs and step < max_steps:
    │   │     Mark group as "ready for next generate"
    │   └─ If all VMs in group are done: mark group as finished
    │
    ├─ Batch all ready groups together
    │   e.g., groups 0, 2, 5 ready → 12 active VMs → batch of 12, padded to 16
    │
    ├─ generate_sequences(padded_batch)
    │   8 GPUs participate: 4 get ~2 real prompts each, 4 get ~2 padding each
    │   Padding GPUs finish near-instantly (1-token dummy prompts)
    │
    ├─ Decode actions for ready groups only
    │
    └─ Dispatch step.remote() to ready groups' active VMs
        These become new in-flight futures → loop continues

Phase 3: Post-loop (same as current)
  ├─ Collect eval_results (ray.get on eval futures)
  ├─ Launch get_train_dict.remote() on all workers
  └─ Return (process_results, eval_results, format_rewards, task_configs)
```

### Implementation sketch

```python
def _run_rollout_chunk(self, task_configs, timing_raw, chunk_idx, total_chunks,
                       prefetch_reset_futures=None):
    active_workers = self.env_workers[:len(task_configs)]
    world_size = self.actor_rollout_wg.world_size
    num_workers = len(active_workers)
    group_size = num_workers // world_size  # 4
    local_timing = {}

    format_rewards = [0.0] * num_workers
    eval_results_objects = [None] * num_workers
    rollout_step_metadata_by_job = [[] for _ in task_configs]
    done_slots = set()

    # ── Reset (unchanged) ──
    with _timer("env_reset", local_timing):
        if prefetch_reset_futures is not None:
            reset_outputs = _ray_get_robust(prefetch_reset_futures, ...)
        else:
            reset_outputs = _ray_get_robust(
                [w.reset.remote(tc) for w, tc in zip(active_workers, task_configs)], ...)

    # Build slot/env_idx mappings (unchanged)
    active_worker_by_env_idx = {o["env_idx"]: w for o, w in zip(reset_outputs, active_workers)}
    slot_by_env_idx = {o["env_idx"]: s for s, o in enumerate(reset_outputs)}

    # ── Group setup ──
    groups = {}  # group_id → list of slots
    for g in range(world_size):
        slots = list(range(g * group_size, min((g + 1) * group_size, num_workers)))
        groups[g] = slots

    group_step = {g: 0 for g in groups}        # current step per group
    group_active = {g: True for g in groups}    # is group still stepping?

    # Track in-flight step futures
    ref_to_slot = {}          # ObjectRef → slot index
    slot_to_group = {}        # slot → group_id
    for g, slots in groups.items():
        for s in slots:
            slot_to_group[s] = g

    in_flight_refs = []       # flat list of all pending step ObjectRefs
    group_completed = {g: {} for g in groups}  # g → {slot: env_output}

    # Observations ready for next generate (group_id → [env_outputs])
    ready_groups = {}

    # ── Initial is_done check + first generate ──
    env_outputs = [o for o in reset_outputs if o.get("obs_messages") is not None]
    if not env_outputs:
        # All resets failed
        return [fallback_process_result] * num_workers, [0.0] * num_workers, format_rewards, task_configs

    # Check is_done from reset
    is_done_futures = [w.is_done.remote() for w in active_workers]
    is_done_results = _ray_get_robust(is_done_futures, timeout=30, fallback_fn=lambda _: True, ...)
    for slot, is_done_val in enumerate(is_done_results):
        if is_done_val:
            done_slots.add(slot)
            eval_results_objects[slot] = active_workers[slot].evaluate.remote()

    # Remove done VMs from env_outputs
    env_outputs = [o for o in env_outputs if slot_by_env_idx[o["env_idx"]] not in done_slots]
    if not env_outputs:
        break  # all done from reset

    # First generate: all active VMs
    vllm_batch, valid_env_idx = self.prepare_vllm_inputs_full(env_outputs)
    if vllm_batch is None:
        return ...  # all failed

    vllm_batch_pad, pad_size = pad_dataproto_to_divisor(vllm_batch, world_size)
    gen_batch = vllm_batch_pad.pop(batch_keys=[...], non_tensor_batch_keys=[...])
    action_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
    action_batch_output = unpad_dataproto(action_batch_output, pad_size=pad_size)

    response_texts = self.tokenizer.batch_decode(action_batch_output.batch["responses"], ...)
    response_texts, step_metadata, _, _ = self._retry_invalid_actions_once(...)

    # Record metadata
    for env_idx, meta in zip(valid_env_idx, step_metadata):
        rollout_step_metadata_by_job[slot_by_env_idx[env_idx]].append(meta)

    # Dispatch step.remote() to all active VMs
    cur_valid_envs = [active_worker_by_env_idx[eidx] for eidx in valid_env_idx]
    for worker, action, eidx in zip(cur_valid_envs, response_texts, valid_env_idx):
        ref = worker.step.remote(action)
        slot = slot_by_env_idx[eidx]
        ref_to_slot[ref] = slot
        in_flight_refs.append(ref)

    # Mark group steps
    for g in groups:
        if any(s not in done_slots for s in groups[g]):
            group_step[g] = 1

    # ── Async group loop ──
    _step_timeout = float(os.environ.get("ROLLOUT_STEP_TIMEOUT", "330"))
    max_steps = self.config.env.max_steps

    while in_flight_refs:
        # Wait for at least 1 future to complete
        newly_ready, still_pending = ray.wait(in_flight_refs, num_returns=1, timeout=_step_timeout)
        # Greedily collect ALL other ready futures (non-blocking) to batch groups together
        if still_pending:
            more_ready, still_pending = ray.wait(still_pending, num_returns=len(still_pending), timeout=0)
            newly_ready.extend(more_ready)
        in_flight_refs = list(still_pending)

        if not newly_ready:
            # Hard timeout — mark all remaining as failed
            for ref in in_flight_refs:
                slot = ref_to_slot[ref]
                done_slots.add(slot)
            in_flight_refs = []
            break

        # Process each completed future
        for ref in newly_ready:
            slot = ref_to_slot.pop(ref)
            group = slot_to_group[slot]
            try:
                result = ray.get(ref, timeout=0)
            except Exception:
                result = {"env_idx": slot, "obs_messages": None, "is_done": True, "format_reward": 0.0}

            format_rewards[slot] += float(result.get("format_reward", 0.0))

            if result.get("is_done") or result.get("obs_messages") is None:
                done_slots.add(slot)
                if eval_results_objects[slot] is None:
                    eval_results_objects[slot] = active_workers[slot].evaluate.remote()
                # Don't add to group_completed — this VM is finished
            else:
                group_completed[group][slot] = result

            # Check if this group is fully done with its current step
            active_in_group = [s for s in groups[group] if s not in done_slots]
            pending_in_group = [s for s in active_in_group if s not in group_completed[group]]

            # A group is "ready" when ALL its active VMs have completed their current step
            if not pending_in_group and active_in_group:
                # All active VMs in this group finished → group ready for next generate
                group_obs = [group_completed[group][s] for s in active_in_group]
                ready_groups[group] = group_obs
                group_completed[group] = {}

            # If group has no active VMs left, it's done
            if not active_in_group:
                group_active[group] = False

        # ── Batch generate for all ready groups ──
        if ready_groups:
            all_ready_obs = []
            ready_group_ids = sorted(ready_groups.keys())
            for g in ready_group_ids:
                all_ready_obs.extend(ready_groups[g])

            print(f"[chunk={chunk_idx+1}] generating for {len(ready_group_ids)} groups "
                  f"({len(all_ready_obs)} VMs), groups at steps: "
                  f"{{{g}: group_step[g] for g in ready_group_ids}}}")

            vllm_batch, valid_env_idx = self.prepare_vllm_inputs_full(all_ready_obs)
            if vllm_batch is not None:
                vllm_batch_pad, pad_size = pad_dataproto_to_divisor(vllm_batch, world_size)
                gen_batch = vllm_batch_pad.pop(batch_keys=[...], non_tensor_batch_keys=[...])
                action_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                action_batch_output = unpad_dataproto(action_batch_output, pad_size=pad_size)

                response_texts = self.tokenizer.batch_decode(
                    action_batch_output.batch["responses"], ...)
                response_texts, step_metadata, _, _ = self._retry_invalid_actions_once(...)

                for env_idx, meta in zip(valid_env_idx, step_metadata):
                    rollout_step_metadata_by_job[slot_by_env_idx[env_idx]].append(meta)

                # Dispatch step.remote() for ready VMs
                for env_idx, action in zip(valid_env_idx, response_texts):
                    slot = slot_by_env_idx[env_idx]
                    worker = active_worker_by_env_idx[env_idx]
                    ref = worker.step.remote(action)
                    ref_to_slot[ref] = slot
                    in_flight_refs.append(ref)

            # Update group steps
            for g in ready_group_ids:
                group_step[g] += 1
                if group_step[g] >= max_steps:
                    # Max steps reached — mark remaining VMs in group as done
                    for s in groups[g]:
                        if s not in done_slots:
                            done_slots.add(s)
                            if eval_results_objects[s] is None:
                                eval_results_objects[s] = active_workers[s].evaluate.remote()

            ready_groups.clear()

        # Check termination
        if all(not active for active in group_active.values()):
            break

    # ── Post-loop: eval + train_dict (unchanged) ──
    ...
```

### How partial batches work with generate_sequences

Example: groups 0, 2, 5 are ready (12 active VMs). Groups 1, 3, 4, 6, 7 still stepping.

```
Batch: 12 prompts from groups 0, 2, 5
Padded to 16 (next multiple of 8) — padding = copies of prompts [0:4]
chunk(8) → 2 prompts per GPU

GPU 0: prompts[0:2]   (2 real)
GPU 1: prompts[2:4]   (2 real)
GPU 2: prompts[4:6]   (2 real)
GPU 3: prompts[6:8]   (2 real)
GPU 4: prompts[8:10]  (2 real)
GPU 5: prompts[10:12] (2 real)
GPU 6: prompts[12:14] (2 copies — duplicated padding)
GPU 7: prompts[14:16] (2 copies — duplicated padding)

All GPUs do equal work (padding = copies of real prompts, same token count).
Each GPU processes 2 prompts → latency ≈ generate(32)/4 (each GPU does 2 vs 4).
Duplicate results from GPU 6-7 are discarded by unpad_dataproto.
```

The GPU↔group mapping shifts with partial batches, but **this doesn't matter** — all
GPUs have identical model weights and produce identical outputs for the same input.
The mapping between outputs and VMs is maintained through `valid_env_idx`, not GPU
assignment.

### Concrete timing improvement

**Generate latency scales with per-GPU batch size:**
- generate(32): chunk(8) → 4 prompts/GPU → ~3s
- generate(16 padded): chunk(8) → 2 prompts/GPU → ~1.5s
- generate(8 padded): chunk(8) → 1 prompt/GPU → ~0.75s
- Per-call overhead (dispatch, ray, decode): ~0.5s

Current lockstep (32 VMs, 8 groups):
```
step.remote(32 VMs) → wait max(32 latencies) → generate(32) → ...
Typical: 20s wait + 3.5s generate = 23.5s per step
Over 15 steps: 352s
```

Per-group async with greedy batching:
```
time 0:   step.remote(all 32 VMs)
time 5:   groups 0-5 done (24 VMs). Generate(24 padded to 24) → ~2s
time 7:   dispatch next step for 24 VMs. Group 6 also done.
time 7:   generate(4+4=8 padded) → ~1.25s for groups 6+(just finished group 0)
time 20:  group 7 done. Generate for stragglers.
...

With greedy batch-collection, fast groups batch together → ~2 generate calls per
"logical step" instead of 8 separate calls.

Total generate time: ~2 calls × 2s = 4s (vs 3.5s lockstep — negligible increase)
Total wait time: ~max(4 latencies) per group ≈ 10s (vs max(32) ≈ 20s)

Net per step: 10s + 4s = 14s (vs 23.5s lockstep)
Over 15 steps: 210s vs 352s = ~140s savings (40%)
```

**Compounding gains**: Fast groups reaching max_steps or is_done early means fewer VMs
in later generate calls → even faster generation. With high latency variance (p50=5s,
p99=25s), savings can reach 25-40%.

### Interaction with existing done_slots

The current `done_slots` mechanism skips done VMs in `is_done` and `step` calls. The
async group loop naturally extends this:
- When a VM reports `is_done`, it's added to `done_slots` and `evaluate.remote()` fires
- When ALL VMs in a group are in `done_slots`, the group is marked inactive
- Groups with some done VMs continue stepping with only their active VMs
- A group's "readiness" is checked against active VMs only (not done ones)

### Interaction with evaluate/get_train_dict pipelining

After the async group loop ends:
- `eval_results_objects` has ObjectRefs for all done VMs (launched immediately when done)
- `train_dict_futures` are launched for all workers (as before)
- The existing pipelining (launch train_dict before waiting on eval) is unchanged

### Key edge cases

1. **All VMs in a group fail reset** (obs_messages=None): Group has 0 active VMs →
   immediately marked inactive. No generate calls for this group.

2. **One VM in a group is very slow**: The other 3 VMs in the group wait for it. But
   other groups proceed independently. Worst case: one group takes 25s per step while
   others take 5s. The slow group completes 15 steps in 375s. Other groups finish in
   ~75s and are idle for 300s. This is the same as lockstep's behavior for that group
   but other groups aren't penalized.

3. **Group size not evenly divisible** (e.g., 30 VMs / 8 GPUs): Last group has fewer
   VMs (30 - 7×3 = 9 in one group, or uneven split). Handle by computing:
   `group_size = num_workers // world_size` with remainder VMs in the last group.
   Generate pads to world_size-divisible anyway.

4. **Empty generate batch**: If all ready groups' VMs are done (no obs to generate for),
   skip the generate call. Just update done_slots.

5. **ray.wait() returns futures from multiple groups partially**: A group is only "ready"
   when ALL its active VMs complete. Partially-completed groups stay in the pending state.

## Robustness Audit

### CONFIRMED SAFE: Training still waits for ALL 32 VMs

The `while in_flight_refs:` loop exits ONLY when ALL step futures are resolved (completed
or hard-timed-out). After the loop:

```
1. missing_eval_idx → launch evaluate for VMs that hit max_steps without is_done
2. train_dict_futures → launch get_train_dict on ALL workers
3. eval_results → wait for ALL evaluations
4. process_results → wait for ALL train_dicts
5. return (process_results, eval_results, format_rewards, task_configs)
```

This is IDENTICAL to the current post-loop code (lines 1156-1217). The async design ONLY
changes what happens INSIDE the step loop. Training does NOT start until all 32 workers
have completed their rollouts. **No change to the training logic.**

### CONFIRMED SAFE: prepare_vllm_inputs_full works with any subset

`prepare_vllm_inputs_full(env_outputs)` extracts `env_idx` from each output (line 882),
filters out None obs_messages, and returns `(batch, valid_env_idx)`. No positional
assumption — works with any subset in any order.

### CONFIRMED SAFE: _retry_invalid_actions_once works with partial batches

Takes `gen_batch` (already padded) and calls `generate_sequences(gen_batch)` again for
retry. No assumption about batch size or worker count. Safe for partial batches.

### CONFIRMED SAFE: eval/get_train_dict pipelining unchanged

The pipelining pattern (launch train_dict before waiting on eval, lines 1166-1214) runs
AFTER the step loop exits. The async design fills `eval_results_objects` during the loop
and `done_slots` tracks completion — same semantics as current code.

### CONFIRMED SAFE: Prefetch reset pipeline unaffected

Prefetch reset (`_launch_prefetch_resets`) runs AFTER `_run_rollout_chunk` returns (in
`fit()` at line 1397). The async design only changes internals of `_run_rollout_chunk`.

### CONFIRMED SAFE: format_rewards accumulation

Accumulated per-slot in the async loop: `format_rewards[slot] += result["format_reward"]`.
Same as current code (line 1142). No dependency on step_idx or ordering.

### DESIGN FIX REQUIRED: Greedy batch-collection in ray.wait()

The sketch uses `ray.wait(in_flight_refs, num_returns=1)` which wakes up for EVERY single
future. With 32 VMs × 15 steps = 480 wakeups, each potentially triggering a small-batch
generate call. This creates excessive per-call overhead.

**Fix**: After the initial wakeup, greedily collect all other ready futures:

```python
# Wait for at least 1 future
newly_ready, still_pending = ray.wait(in_flight_refs, num_returns=1, timeout=_step_timeout)
# Immediately grab everything else that's also ready (timeout=0 = non-blocking)
if still_pending:
    more_ready, still_pending = ray.wait(still_pending, num_returns=len(still_pending), timeout=0)
    newly_ready.extend(more_ready)
in_flight_refs = list(still_pending)
```

This batches all simultaneously-completed futures into one processing round, maximizing
the chance of multiple groups being ready for a single generate call.

### DESIGN FIX REQUIRED: batch_skipped semantics

Current code sets `batch_skipped = True` when `prepare_vllm_inputs_full` returns None for
ALL workers (line 1086). This triggers special handling in the post-loop eval collection
(line 1176).

In the async design, `prepare_vllm_inputs_full` is called with subsets (ready groups). If
one group's VMs all fail (obs=None), the batch for that call is None. But other groups may
succeed.

**Fix**: Track `had_successful_generate = False`. Set to `True` after any successful
generate call. After the loop, set `batch_skipped = not had_successful_generate`. This
preserves the original semantics: batch_skipped=True only if ZERO successful generates
occurred across the entire chunk.

### DESIGN FIX REQUIRED: Group size when num_workers % world_size != 0

If 30 VMs / 8 GPUs: `group_size = 30 // 8 = 3`, remainder = 6. Either:
- Last group has 3 + 6 = 9 VMs (unbalanced) — BAD
- Use ceiling division and let some groups have 4, others 3 — BETTER

**Fix**: Use a proper partition:
```python
group_size = num_workers // world_size
remainder = num_workers % world_size
groups = {}
offset = 0
for g in range(world_size):
    size = group_size + (1 if g < remainder else 0)
    groups[g] = list(range(offset, offset + size))
    offset += size
```

In practice with 32 VMs / 8 GPUs, `32 % 8 = 0` so this is a non-issue for the current
config. But the implementation should handle it for future configs.

### VERIFIED: No blocking of downstream dependencies

The async design ONLY modifies the inner step loop within `_run_rollout_chunk`. The method
signature, return type, and semantics are unchanged:

```
Input:  task_configs, timing_raw, chunk_idx, total_chunks, prefetch_reset_futures
Output: (process_results, eval_results, format_rewards, task_configs)
```

Callers (`fit()` chunk loop at line 1369) see no difference. The `collate_fn_dataproto`,
`compute_log_probs`, `update_actor`, and all downstream GPU training code are unaffected.

### VERIFIED: `_apply_task_family_decoding_if_single`

Only activates when `len(valid_env_idx) == 1`. With group_size=4, the minimum batch is 4
(unless 3 VMs in the group are done, leaving 1 active). Handles partial groups correctly.

## Comparison to lockstep

| Aspect | Current lockstep | Per-group async |
|--------|-----------------|-----------------|
| Wait per step | max(32 latencies) | max(4 latencies) per group |
| Generate calls per step | 1 (all 32) | ~2 (partial batches) |
| GPU waste | 0% (full batch) | Padding GPUs duplicate real work (discarded) |
| Padding cost | None | ~0 extra wall time (smaller batches → faster per call) |
| State complexity | 1 step counter | 8 group counters |
| VM dropping | None | None |
| Code change | — | ~150 lines (step loop rewrite) |
| Estimated savings | — | 15-35% of rollout time |
