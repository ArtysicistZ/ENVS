# Pipeline Blocking Analysis: VM-GPU Synchronization Bottlenecks

This document identifies systematic blocking patterns in the training pipeline where
independent operations are forced into sequential execution, causing VMs and GPUs to
idle while waiting on each other.

**Setup context:** 48 remote-env VMs, 16 GPUs (2 nodes x 8), 6 tasks per step,
rollout.n=8, max_steps=15. Observed wall-clock: ~18min per training step.

---

## Blocking Point 1: In-Loop Evaluate Blocks All Other VMs' Next Step (CRITICAL)

### Location

`verl/trainer/ray_trainer.py:1006-1105` — the step loop inside `_run_rollout_chunk()`,
specifically the interaction between line 1102 (`evaluate.remote()`) and line 1007-1011
(`is_done.remote()` on the next iteration).

### What happens

The step loop runs up to 15 iterations. Within each iteration, after `step.remote()`
returns, VMs that report `is_done=True` immediately have `evaluate.remote()` launched:

```python
# Line 1098-1105: after step.remote() returns
for single_output in env_outputs:
    slot = slot_by_env_idx[single_output["env_idx"]]
    format_rewards[slot] += float(single_output.get("format_reward", 0.0))
    if single_output["is_done"] and eval_results_objects[slot] is None:
        eval_results_objects[slot] = active_workers[slot].evaluate.remote()  # <-- FIRE AND FORGET

if env_outputs and all(single_output["is_done"] for single_output in env_outputs):
    break
```

Then the **next step iteration** begins:

```python
# Line 1007-1011: NEXT ITERATION — called on ALL active_workers
is_done_stats = _ray_get_robust(
    [worker.is_done.remote() for worker in active_workers],  # includes VM_A!
    timeout=30.0, fallback_fn=lambda _: True,
    label=f"chunk{chunk_idx+1}/is_done",
)
```

### The blocking mechanism (Ray actor serialization)

`RemoteEnvWorker` is decorated with `@ray.remote(num_cpus=1)` (gui_agent.py:1026).
Ray actors are **single-threaded** — they process one method call at a time, queuing
the rest.

**Concrete scenario:**

1. **Step 6:** VM_A's `step.remote()` returns `is_done=True`. Line 1102 launches
   `active_workers[slot_A].evaluate.remote()`. The evaluate method makes an HTTP POST
   to the remote env server with `timeout=300s` and up to 4 retries with
   5s+10s+15s+20s backoff (gui_agent.py:1336-1349). This starts executing on VM_A's
   Ray actor.

2. **Step 7 begins:** Line 1007 calls `is_done.remote()` on **ALL** `active_workers`,
   including VM_A. Since VM_A's actor is busy running `evaluate()`, the `is_done()`
   call **queues behind it**.

3. **`_ray_get_robust` blocks:** `ray.get(futures, timeout=30)` waits for ALL 48
   `is_done` results. The 47 healthy VMs return instantly, but VM_A's `is_done` is
   queued behind `evaluate()`. The driver blocks for up to **30 seconds** until the
   timeout fires and fallback returns `True` for VM_A.

4. **This repeats every step:** For each remaining step (7, 8, 9, ..., 14), the
   `is_done` check on VM_A blocks for up to 30s until evaluate finishes.

5. **Worst case:** If VM_A finishes at step 6, there are 9 remaining steps. Each
   blocks 30s on the `is_done` timeout = **270 seconds (4.5 minutes) of pure waiting**
   just from the `is_done` check alone.

6. **It compounds:** If VM_B finishes at step 8 and VM_C at step 10, each adds its
   own 30s-per-remaining-step blocking. With 48 VMs finishing at different steps, the
   total wasted time accumulates significantly.

### Why this is fundamentally wrong

The evaluate call on VM_A is **completely independent** of the remaining VMs' step
loop. VM_A has finished its task — its evaluation score is not needed by the other VMs
to continue stepping. Yet because `evaluate.remote()` runs on the same single-threaded
Ray actor that handles `is_done()`, the evaluate occupies the actor and blocks all
subsequent method calls to it.

The `active_workers` list is **never filtered** — line 1008 always calls `is_done` on
all original workers, even those that have already finished and launched evaluate.

### Proposed fix direction

**Option A: Skip done workers in the step loop.**
After launching `evaluate.remote()`, remove the worker from subsequent `is_done` and
`step` calls. Track which workers are done separately:

```python
done_workers = set()
for step_idx in range(self.config.env.max_steps):
    still_active = [w for i, w in enumerate(active_workers) if i not in done_workers]
    is_done_stats = _ray_get_robust(
        [worker.is_done.remote() for worker in still_active],
        ...
    )
    # ... generate, step only for still_active ...
    for single_output in env_outputs:
        if single_output["is_done"]:
            done_workers.add(slot_by_env_idx[single_output["env_idx"]])
            eval_results_objects[slot] = active_workers[slot].evaluate.remote()
```

This completely eliminates the blocking because done workers are never called again
during the step loop. Their evaluate runs in the background on their own actor without
interfering with the step loop's progress.

**Option B: Use a separate actor or thread for evaluation.**
Instead of calling `evaluate.remote()` on the same `RemoteEnvWorker` actor, delegate
evaluation to a separate Ray actor or thread pool. This way the `RemoteEnvWorker`
actor remains available for `is_done()` calls while evaluation runs in parallel.

**Option A is strongly preferred** — it's simpler, requires no new actors, and the
semantics are clean: a done worker needs no further interaction in the step loop.

---

## Blocking Point 2: End-of-Chunk Evaluate Wait Blocks Everything After Rollout

### Location

`verl/trainer/ray_trainer.py:1117-1127` — the `evaluate_env` timer after the step loop.

### What happens

After the step loop completes (all VMs done or max_steps reached), evaluation futures
are collected synchronously:

```python
# Line 1117-1127: BLOCKING WAIT for all 48 evaluations
eval_results = _ray_get_robust(
    eval_results_objects,
    timeout=_eval_timeout,  # 350 seconds
    fallback_fn=lambda _: 0.0,
    label=f"chunk{chunk_idx+1}/evaluate",
)
```

### Why this is a blocking problem

Evaluation scores feed into the current step's training update. They are read at:
1. Line 1334: `batch.batch["eval_results"]` is set
2. Line 1342: `apply_replay()` reads `eval_results` (via line 1179) for replay sampling
3. Line 1386: reward computation `rewards = batch.batch["eval_results"] + 0.1 * format_rewards_clipped`

The scores do NOT influence the **next step's** rollout generation. But the entire
`_run_rollout_chunk` must return before `prepare_grpo_inputs`, GPU training phases, or
the next chunk can start.

If Blocking Point 1 is fixed (done workers are skipped), most evaluations will have
started earlier and may already be done by the time the step loop ends. But the
blocking `_ray_get_robust` at line 1122 still forces the driver to wait for the
slowest evaluation (up to 350s timeout).

### Dead VM cascade

If one VM's evaluation hangs, `_ray_get_robust` waits for the full 350s timeout
before falling back. All healthy VMs' results are already available but the driver
can't proceed.

### Proposed fix direction

Decouple evaluation collection from `_run_rollout_chunk`. Return eval futures as
pending and collect results later — right before line 1334 where they are first
needed. This allows `get_train_dict` (line 1133) and `prepare_grpo_inputs` (line 1316)
to proceed in parallel with evaluation.

Concretely:
- `_run_rollout_chunk` returns `(process_results, eval_futures, format_rewards, task_configs)`
  instead of resolved `eval_results`
- Eval futures are resolved with `ray.wait()` using a per-future timeout right before
  line 1334
- `get_train_dict` futures are launched immediately after the step loop (not after
  eval), and since done workers' evaluate calls have already been queued, the
  `get_train_dict` calls queue right behind them on each actor

---

## Blocking Point 3: `get_train_dict` Sequenced After Evaluation

### Location

`verl/trainer/ray_trainer.py:1133-1143` — inside `_run_rollout_chunk()`, after the
evaluation wait.

### What happens

```python
# Line 1117-1127: FIRST, wait for all evaluations (blocking)
eval_results = _ray_get_robust(eval_results_objects, timeout=350, ...)

# Line 1133-1143: THEN, fetch training data from all workers (blocking)
process_results = _ray_get_robust(
    [worker.get_train_dict.remote() for worker in active_workers],
    timeout=60.0,
    ...
)
```

`get_train_dict()` fetches trajectory data (input_ids, labels, attention_mask) that is
already finalized when the rollout steps finish — it has zero dependency on eval scores.

### Why this is a blocking problem

Both `evaluate()` and `get_train_dict()` are methods on the same `RemoteEnvWorker` Ray
actor. Since Ray actors serialize method calls, `get_train_dict.remote()` queues behind
`evaluate()` on each actor.

Currently, `get_train_dict` futures are launched AFTER the driver blocks on evaluation.
This means the driver waits for evaluate, THEN launches get_train_dict, THEN waits for
get_train_dict — fully sequential.

If instead `get_train_dict` futures are launched BEFORE waiting on evaluation, each
actor's queue becomes `[evaluate, get_train_dict]` with no driver round-trip in between.
The driver can then wait on both sets of futures simultaneously.

### Proposed fix direction

```python
# Launch get_train_dict immediately (queues behind evaluate on each actor)
train_dict_futures = [worker.get_train_dict.remote() for worker in active_workers]

# Now wait for evaluation
eval_results = _ray_get_robust(eval_results_objects, timeout=350, ...)

# get_train_dict is already done or nearly done (ran back-to-back with evaluate)
process_results = _ray_get_robust(train_dict_futures, timeout=60, ...)
```

---

## Blocking Point 4: No Pipelining Between Rollout and GPU Training

### Location

`verl/trainer/ray_trainer.py:1256-1313` — the outer training loop in `fit()`.

### What happens

The training loop is strictly sequential:

```python
# Line 1262: iterate over batches
for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=1):
    # Phase 1: Rollout on VMs (dominates wall time)
    rollout_chunks = self._build_rollout_chunks(batch_dict)
    self.actor_rollout_wg.prepare_generate_sequences()
    for chunk_idx, task_configs_chunk in enumerate(rollout_chunks):
        process_results, eval_results, ... = self._run_rollout_chunk(...)

    # Phase 2: Data preparation (CPU on driver)
    batch = collate_fn_dataproto(all_process_results, ...)

    # Phase 3: GPU compute (old logprobs, ref logprobs, values, advantages)
    old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
    ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
    values = self.critic_wg.compute_values(batch)
    batch = compute_advantage(batch, ...)

    # Phase 4: GPU training (critic update, actor update)
    critic_output = self.critic_wg.update_critic(batch)
    actor_output = self.actor_rollout_wg.update_actor(batch)

    # Phase 5: Validation + checkpoint (when scheduled)
    # ... then back to Phase 1 for next batch_dict
```

There is **commented-out prefetch code** at lines 1259-1260 and 1268-1269:

```python
# batch_dict_next_batch = next(iterator)
# task_configs_next_batch, reset_envs_object_next_batch = self.start_reset_envs(batch_dict_next_batch)
```

This was an attempt to start the next batch's environment resets while training runs,
but it was never implemented (`start_reset_envs()` does not exist in the codebase).

### Why this is a blocking problem

During the ~3min GPU training phase (Phases 2-4), all 48 VMs are idle. During the
~15min VM rollout phase, GPUs are mostly idle (only used for vLLM generation).

### Constraint

VM resets (environment setup — no model inference, no GPU usage) can safely overlap
with GPU training. VM rollout steps require `generate_sequences` which uses vLLM
weights synced in `prepare_generate_sequences()`, so they must wait until
`finish_generate_sequences()` + `update_actor()` complete.

The overlap window: **start resetting VMs for step N+1 while GPUs train on step N's
data**.

### Proposed fix direction

Implement a `start_reset_envs(batch_dict)` method that builds task configs and
launches `worker.reset.remote(task_config)` for all env workers, returning the futures
without blocking. Call it right after `finish_generate_sequences()` completes (line
1313), before GPU training begins. Collect the reset futures at the start of the next
step, before calling `prepare_generate_sequences()`.

---

## Blocking Point 5: Sequential Chunk Processing

### Location

`verl/trainer/ray_trainer.py:1289-1311` — the chunk for-loop inside `fit()`.

### Current config impact

With the 48-env config (`smoke_remote_env_16gpu_a100_n8_env48.yaml`):
- `rollout_batch_size=6`, `rollout.n=8` → `required_rollouts = 6 * 8 = 48`
- `num_envs=48` → `chunks = ceil(48/48) = 1`
- **Not a bottleneck for the current 48-env config.**

For smaller configs (e.g., `num_envs=16`, `required_rollouts=32`), chunks are processed
sequentially and wall time multiplies.

### Proposed fix direction

Ensure `num_envs >= required_rollouts` in production configs. For multi-chunk configs,
consider launching chunks concurrently on disjoint VM subsets (requires coordinating
shared GPU worker group access).

---

## Blocking Point 6: Validation Blocks Training

### Location

`verl/trainer/ray_trainer.py:1584-1594` — validation inside the training step loop.

### What happens

`_validate()` (lines 507-636) reuses the same `env_workers` and `actor_rollout_wg` as
training. It runs a full validation episode synchronously: reset, step loop, evaluate,
fetch history. Can take 5-15 minutes.

### Proposed fix direction

Run validation on dedicated resources (separate VMs, separate vLLM instance) as a
background Ray task. Or validate less frequently / on fewer tasks.

---

## Blocking Point 7: Checkpoint Saving Blocks Training

### Location

`verl/trainer/ray_trainer.py:1596-1598` — `_save_checkpoint()` inside the training loop.

### What happens

`_save_checkpoint()` (lines 708-727) synchronously saves actor/critic FSDP checkpoints
(all-gather across 16 GPUs) and dataloader state. Takes 1-3 minutes.

### Proposed fix direction

Use `torch.distributed.checkpoint.async_save()` to write to disk in the background
after snapshotting FSDP state dicts to CPU memory.

---

## Blocking Point 8: `prepare_vllm_inputs` Runs on Driver While VMs Wait

### Location

`verl/trainer/ray_trainer.py:1045-1046` and lines 879-915.

### What happens

Vision processing (image tokenization) for all 48 observations runs on the driver CPU
each step iteration via `ThreadPoolExecutor(max_workers=64)` (line 907). While the
driver processes observations, no generation or env stepping happens.

Pre-created `data_processor_workers` Ray actors (lines 420-427) exist but are **never
called** in the current code.

### Proposed fix direction

Use the pre-created `data_processor_workers` for true parallelism (separate processes,
no GIL). Or move processing into the env workers — `RemoteEnvWorker.step()` already
calls `process_message()` at line 1322 which does the same tokenization.

---

## Summary: Current vs. Proposed Timeline

### Current (with in-loop blocking)

```
Step loop (15 iterations):
  step 1:  [is_done] [prepare] [generate] [env_step]
  step 2:  [is_done] [prepare] [generate] [env_step]
  ...
  step 6:  [is_done] [prepare] [generate] [env_step] → VM_A done, evaluate.remote() launched
  step 7:  [is_done: BLOCKS 30s on VM_A] [prepare] [generate] [env_step]
  step 8:  [is_done: BLOCKS 30s on VM_A] [prepare] [generate] [env_step] → VM_B done, evaluate launched
  step 9:  [is_done: BLOCKS 30s on VM_A+B] [prepare] [generate] [env_step]
  ...                                    ↑ cumulative blocking from done VMs
  step 14: [is_done: BLOCKS 30s] [prepare] [generate] [env_step]

End-of-chunk: [wait for all evaluations: up to 350s] [get_train_dict: 60s]
GPU training: [old_logprobs] [ref_logprobs] [values] [advantages] [critic] [actor]
--- Next step starts from scratch ---
```

### Proposed (with fixes applied)

```
Step loop (15 iterations):
  step 1:  [is_done] [prepare] [generate] [env_step]
  ...
  step 6:  [is_done(active only)] [prepare] [generate] [env_step(active only)]
           → VM_A done, evaluate launched, VM_A removed from active set
  step 7:  [is_done(47 VMs)] [prepare] [generate] [env_step(47 VMs)]  ← no blocking!
  step 8:  [is_done(47 VMs)] [prepare] [generate] [env_step(47 VMs)]
           → VM_B done, removed from active set
  step 9:  [is_done(46 VMs)] [prepare] [generate] [env_step(46 VMs)]  ← no blocking!
  ...

End-of-chunk: [get_train_dict + eval collection in parallel]
              VM resets for next step launched here (prefetch) ──┐
GPU training: [old_logprobs] [ref_logprobs] [values] [adv]      │
              [critic] [actor]                                   │
              ← resets completing in background ─────────────────┘
--- Next step's rollout starts immediately with pre-reset VMs ---
```

---

## Blocking Point 9: Lockstep Step Loop — All 32 VMs Must Finish Before Any GPU Proceeds

### Location

`verl/trainer/ray_trainer.py:1130-1137` — `_ray_get_robust(_step_futures, ...)` inside the
per-step loop of `_run_rollout_chunk()`.

### Architecture context (CRITICAL)

The config uses **tensor_parallel_size=1** (all configs). With 8 GPUs and TP=1:

```
dp_size = world_size // tp_size = 8 // 1 = 8   (8 data-parallel groups)
```

`generate_sequences` uses `Dispatch.DP_COMPUTE_PROTO`, which calls `DataProto.chunk(8)`
to split the batch into 8 equal parts. Each GPU runs its own independent vLLM engine on
its share:

```
GPU 0: prompts[0:4]   → generates actions for VMs 0-3
GPU 1: prompts[4:8]   → generates actions for VMs 4-7
GPU 2: prompts[8:12]  → generates actions for VMs 8-11
...
GPU 7: prompts[28:32] → generates actions for VMs 28-31
```

**Each GPU only processes 4 VMs' data.** The generation for GPU K is fully independent of
the other 7 GPUs' data.

### What happens

The step loop runs in lockstep — ALL non-done VMs must complete their `step()` before
the next iteration:

```python
# Line 1131-1137: WAIT FOR ALL 32 VMs
_step_futures = [worker.step.remote(action_text) for worker, action_text in zip(cur_valid_envs, response_texts)]
env_outputs = _ray_get_robust(
    _step_futures,
    timeout=_step_timeout,   # 330s
    fallback_fn=lambda idx: {"env_idx": valid_env_idx[idx], "obs_messages": None, "is_done": True, "format_reward": 0.0},
    label=f"chunk{chunk_idx+1}/step{step_idx}",
)
```

After this returns, ALL observations are aggregated in `prepare_vllm_inputs_full`, then
`generate_sequences` dispatches to all 8 GPUs.

### Why this wastes time

**Concrete scenario (32 VMs, 8 GPUs, step latency distribution):**

VM step latency depends on screenshot capture, action execution, and HTTP round-trip.
Typical distribution:
- p50: ~5s (fast clicks, simple actions)
- p80: ~10s (typing, scrolling)
- p95: ~18s (complex rendering, network delays)
- p99/max: ~25-30s (timeouts, retries)

With 32 VMs, each step waits for the slowest VM (max latency). If 28 VMs finish in 5s
but 4 VMs (all on GPU 7's shard) take 20s:

```
time 0:     step.remote() dispatched to all 32 VMs
time 5:     28 VMs done — GPUs 0-6 HAVE the data needed to generate next tokens
time 5-20:  GPUs 0-6 IDLE, waiting for GPU 7's 4 VMs (15s wasted per step)
time 20:    all 32 VMs done — driver proceeds to generate
```

Over 15 steps: **15s × 15 steps = 225s (3.75 min) wasted per chunk** just from straggler
waiting. With 8 chunks per training step: up to 30 min wasted.

### The constraint: `generate_sequences` is a collective operation

Even though each GPU only needs 4 prompts, **`generate_sequences` dispatches to ALL 8
GPUs simultaneously.** The `DP_COMPUTE_PROTO` dispatch splits the batch across all GPUs,
calls `.remote()` on all 8 workers, and collects all 8 results. You cannot call it for
just one GPU — the framework has no per-GPU dispatch mode.

This means: even if GPU 0's 4 VMs are ready, you can't generate for just GPU 0 while
GPU 7 waits.

### Proposed approaches

#### Approach A: Threshold-based partial generation (RECOMMENDED)

Instead of waiting for ALL 32 VMs, wait for a **threshold** (e.g., 75%) and generate
for the ready subset. Slow VMs are deferred to the next generation call within the same
step.

```python
# Conceptual flow:
step_futures = {slot: worker.step.remote(action) for slot, worker, action in active_set}
per_vm_step_count = {slot: 0 for slot in range(num_workers)}

while active_slots:
    # Use ray.wait() to collect whatever is ready within a time budget
    min_ready = max(8, int(len(step_futures) * 0.75))  # 75% threshold, min 8 for padding
    ready_refs, pending_refs = ray.wait(
        list(step_futures.values()),
        num_returns=min(min_ready, len(step_futures)),
        timeout=max_step_timeout,
    )

    # Process ready VMs — update done_slots, format_rewards
    ready_outputs = [ray.get(ref) for ref in ready_refs]
    # ... handle is_done, accumulate rewards ...

    # Generate for ready VMs (pad to world_size divisor)
    if ready_for_next_step:
        vllm_batch = prepare_vllm_inputs_full(ready_outputs)
        vllm_batch_pad, pad_size = pad_dataproto_to_divisor(vllm_batch, world_size)
        action_output = generate_sequences(vllm_batch_pad)
        # Dispatch new step.remote() to ready VMs
        # Pending VMs stay in step_futures — picked up in next iteration
```

**How padding works:** With 24 ready VMs (24 prompts), padded to 32 (divisor of 8).
Each GPU gets 4 prompts. GPUs whose 4 VMs are all in the pending set get 4 padding
prompts — they compute garbage but quickly (short dummy prompts). The padding overhead
is small (vLLM decoding is dominated by the longest real prompt).

**Per-VM step tracking:** Each VM has its own step counter. VMs proceed at different
rates. A fast VM might be on step 8 while a slow VM is on step 3. This is fine — vLLM
generates independently per prompt. The step loop ends when all VMs either reach
max_steps or report is_done.

**Expected savings:**
- With 75% threshold (24/32): proceed when 24 VMs are ready
- If p75 step latency ≈ 8s vs max ≈ 20s: save ~12s per generation call
- ~2 generation calls per logical "step" (24 ready + 8 stragglers)
- Net: ~10s saved per step × 15 steps = **150s (2.5 min) per chunk**
- With 8 chunks: **~20 min saved per training step**

#### Approach B: Fully async event-driven step loop

Each VM runs independently. A central scheduler collects observations as they arrive and
batches them for generation. This is the maximum-throughput design.

```python
# Each VM has its own future in flight
in_flight = {}  # slot → ObjectRef
step_counts = {}  # slot → current step number

while any_active:
    # Wait for ANY VM to complete
    done_refs, _ = ray.wait(list(in_flight.values()), num_returns=1, timeout=1)

    for ref in done_refs:
        slot = ref_to_slot[ref]
        result = ray.get(ref)
        # ... handle done / accumulate reward ...
        if not result["is_done"] and step_counts[slot] < max_steps:
            pending_obs.append(result)

    # Batch generation when enough observations accumulated
    if len(pending_obs) >= min_batch_size:
        generate_and_dispatch(pending_obs)
        pending_obs.clear()
```

**Pros:** Maximum throughput, no straggler waste.
**Cons:** Much higher complexity, harder to debug, VMs at different steps make
logging/monitoring harder. Each generate call may have very small batches (low GPU
utilization).

#### Approach C: Adaptive straggler timeout (SIMPLEST)

Keep the lockstep loop but add a shorter "patience" timeout before the hard 330s:

```python
# Wait for 80% with short timeout, then proceed
fast_timeout = 10  # wait at most 10s for fast majority
ready, pending = ray.wait(_step_futures, num_returns=int(0.8 * len(_step_futures)), timeout=fast_timeout)

# If enough ready, proceed. Mark stragglers as timed-out (obs_messages=None, is_done=True)
if len(ready) >= 0.8 * len(_step_futures):
    ready_outputs = [ray.get(r) for r in ready]
    straggler_outputs = [fallback for _ in pending]
    env_outputs = ready_outputs + straggler_outputs
else:
    # Not enough ready — fall back to full wait
    env_outputs = _ray_get_robust(_step_futures, timeout=330, ...)
```

**Pros:** Minimal code change (~20 lines). Captures 80% of the savings.
**Cons:** Stragglers are DROPPED from subsequent steps (marked as done with 0 reward).
This reduces per-sample training signal quality. Acceptable if stragglers are rare.

### Feasibility assessment

| Approach | Savings | Code change | Risk | GPU efficiency |
|----------|---------|-------------|------|----------------|
| A. Threshold-based | ~150s/chunk (25%) | ~200 lines (major rewrite of step loop) | Medium — per-VM step tracking is complex | Good — pad waste ≤25% |
| B. Fully async | ~200s/chunk (35%) | ~300 lines (full rewrite) | High — many edge cases | Poor for small batches |
| C. Adaptive timeout | ~120s/chunk (20%) | ~20 lines | Low — simple logic | No waste (full batches) |

**Recommendation:** Start with **Approach C** (adaptive straggler timeout) for immediate
gains with minimal risk. Then evaluate whether the straggler drop rate justifies
upgrading to **Approach A**.

### Key constraint to remember

`generate_sequences` is a collective dispatch to ALL 8 GPUs via `DP_COMPUTE_PROTO`. The
batch is split by `DataProto.chunk(world_size)` — prompts 0-3 → GPU 0, prompts 4-7 →
GPU 1, etc. **There is no per-GPU dispatch mode.** Any partial-batch approach must pad to
`world_size` and accept some GPU waste on padding slots.

`prepare_generate_sequences()` / `finish_generate_sequences()` are called ONCE per chunk
(not per step iteration), so they don't add overhead to multiple generate calls within
the step loop. Each `generate_sequences` call within the loop is self-contained.

---

## Implementation Priority

| Priority | Fix | Savings | Complexity | Risk |
|----------|-----|---------|------------|------|
| **P0** | 1. Skip done workers in step loop | 30s × remaining_steps per done VM | Low | None — done VMs need no further interaction |
| **P1** | 2. Defer eval collection to before reward computation | Up to 350s per step | Medium | Low — eval results still collected before use |
| **P1** | 4. Pipeline VM resets with GPU training | Full reset duration per step | Medium | Medium — weight sync ordering |
| **P2** | 3. Launch get_train_dict before waiting on eval | Eliminates sequential driver wait | Low | None |
| **P2** | 8. Distribute prepare_vllm_inputs | Seconds per step × 15 steps | Low | Low — workers already created |
| **P3** | 5. Single-chunk config guarantee | N/A (current config OK) | Config change | None |
| **P3** | 6. Background validation | 5-15min when triggered | High | Medium |
| **P3** | 7. Async checkpoint | 1-3min when triggered | Medium | Low |
| **P2** | 9C. Adaptive straggler timeout | ~120s/chunk (20% rollout) | Low (~20 lines) | Low — simple cutoff |
| **P1** | 9A. Threshold-based partial gen | ~150s/chunk (25% rollout) | High (~200 lines) | Medium — step loop rewrite |
