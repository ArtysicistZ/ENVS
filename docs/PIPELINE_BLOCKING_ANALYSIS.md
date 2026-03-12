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
