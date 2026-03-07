# Scalability And Efficiency Notes

This document records the current understanding of ARPO training scalability in this codebase, based on:

- code audit of the trainer / worker / sequence-parallel path
- targeted multimodal SP tests
- a real smoke run on `8 x A100 80G`

It is meant to answer two questions:

1. What limits speed today?
2. What will or will not scale when moving toward many remote environments, such as `64 VMs`?

## 1. Current 8-GPU Findings

The current `8 x A100 80G` smoke run is functionally working after the multimodal SP fixes, but it is not throughput-efficient yet.

Observed first-update metrics from the live run:

- `time_per_step`: about `1105 s`
- `update_actor`: about `783 s`
- `old` (`compute_log_probs`): about `247 s`
- `gen`: about `49 s`
- `env_reset`: about `3.9 s`
- `env_step`: about `0.6 s`
- `max_memory_allocated_gb`: about `32.3`
- `max_memory_reserved_gb`: about `45.7`
- `mfu_actor`: about `0.002`

Conclusion:

- The main bottleneck is not the environment.
- The main bottleneck is actor-side compute:
  - recomputing `old_log_probs`
  - PPO update forward/backward
- GPU power being well below board TDP is expected here. This workload is dominated by long-context sequence compute, FSDP/SP communication, and small micro-batches. It is not a dense GEMM-only workload that naturally drives max power.

## 2. What Actually Scales

There are three different things that can scale, and they should not be mixed together:

### 2.1 Environment concurrency

More env workers / VMs allow more tasks to progress in parallel and reduce idle time waiting on remote desktops.

This helps only until the model side is saturated.

### 2.2 Rollout inference

Rollout uses vLLM. In the current `8 GPU` setup with:

- `worker.rollout.tensor_parallel_size: 4`

you effectively have `2` rollout groups, not `8` independent rollout GPUs.

That means `64 VMs` do not imply `64` simultaneous model generations. They mainly provide a larger pool of ready env states so the two rollout groups stay busy.

### 2.3 Training update

Actor training uses:

- `ulysses_sequence_parallel_size: 4`
- `8 GPUs total`

So you effectively have `2` data-parallel groups of `4` GPUs each.

This lowers per-GPU activation memory for long sequences, which is the main reason the current training path fits.

## 3. VRAM Scaling Rules

Peak per-GPU training VRAM does **not** automatically scale linearly with the number of VMs or env workers.

For actor update / old-logprob recompute, per-GPU VRAM is driven mainly by:

- model size
- FSDP sharding mode
- `ulysses_sequence_parallel_size`
- sequence length
- number of images / vision tokens
- `micro_batch_size_per_device_for_update`
- `micro_batch_size_per_device_for_experience`

If those stay fixed, increasing env count mostly increases:

- total samples produced
- update wall time
- CPU RAM / Ray object-store pressure
- rollout KV-cache pressure

It does **not** by itself cause a proportional increase in actor-update VRAM.

### 3.1 What can still OOM when scaling

The most likely OOM surfaces at larger scale are:

1. vLLM KV cache
2. CPU RAM / Ray object store
3. actor update if micro-batch size is increased too far

The current live run already reported roughly `112 GB` CPU memory usage. This means large-scale rollout will likely hit host memory pressure before actor VRAM becomes the first blocker.

## 4. Current Code Blockers For 64-VM Scale

These are not theoretical issues. They are concrete current-code limitations.

### 4.1 Remote env mode only creates one worker per URL

In [`ray_trainer.py`](/home/kevinzyz/yincheng/arpo/verl/trainer/ray_trainer.py#L302), `_create_envs()` does this:

- reads `env.remote_server_url`
- hardcodes `num_remote = 1`

So the current remote path is effectively single-server / single-worker unless the code is extended.

### 4.2 Sequential rollout scheduling is only correct for the smoke case

In [`ray_trainer.py`](/home/kevinzyz/yincheng/arpo/verl/trainer/ray_trainer.py#L842) and [`ray_trainer.py`](/home/kevinzyz/yincheng/arpo/verl/trainer/ray_trainer.py#L954):

- `start_reset_envs()` truncates `task_configs` to `num_envs`
- `_needs_sequential_rollouts` checks only `len(env_workers) < rollout.n`
- `seq_rollouts` is computed as `rollout_n // len(env_workers)`

That logic only works for `rollout_batch_size = 1`.

For larger batches, the real amount of required rollout work is:

- `rollout_batch_size * rollout.n`

So before scaling to many VMs with larger rollout batches, this scheduler needs to be rewritten to chunk the full rollout workload across env workers correctly.

## 5. Why The Current Run Is Slow

The current smoke config is:

- `rollout_batch_size: 1`
- `rollout.n: 2`
- `actor.global_batch_size: 4`
- `micro_batch_size_per_device_for_update: 1`
- `micro_batch_size_per_device_for_experience: 1`
- `max_prompt_length: 64000`
- `tensor_parallel_size: 4`
- `ulysses_sequence_parallel_size: 4`
- `offload_optimizer: true`

See [`smoke_remote_env_8gpu_a100.yaml`](/home/kevinzyz/yincheng/arpo/configs/smoke_remote_env_8gpu_a100.yaml#L7).

This means:

- the context is very long
- training runs with tiny micro-batches
- the actor does many small forwards/backwards
- optimizer state is moved through CPU even though `80 GB` cards still had memory headroom
- the rollout side is not the dominant cost

The two largest costs in the real run were:

1. `update_actor`
2. `compute_log_probs` for old policy probabilities

Important design choice for the implementation plan below:

- do **not** replace the current-policy logprob kernel with a denser “standard” implementation
- the current actor path is already using the memory-safe approach needed for long multimodal sequences
- the real waste is the redundant old-policy pass, not the current-policy pass

## 6. Decision-Complete Implementation Plan

This section is the implementation spec for improving immediate efficiency on the current `1 VM + 8 GPU` path while preserving an architecture that scales to `32/64 VMs` later.

Default assumptions for the plan:

- keep the current `8 GPU` topology as the foundation
- keep `ulysses_sequence_parallel_size = 4`
- keep `tensor_parallel_size = 4`
- do not introduce a separate near-term 4-GPU default
- keep `1 VM` support fully working throughout

### 6.1 Make rollout memory controls real

Required changes in [`vllm_rollout_spmd.py`](/home/kevinzyz/yincheng/arpo/verl/workers/rollout/vllm_rollout_spmd.py):

- pass `max_num_batched_tokens=config.max_num_batched_tokens` into `LLM(...)`
- pass `max_num_seqs=config.max_num_seqs` into `LLM(...)`
- add optional `worker.rollout.kv_cache_memory_bytes: int | null`
- precedence rule:
  - if `kv_cache_memory_bytes` is set, pass it to `LLM(...)`
  - otherwise rely on `gpu_memory_utilization`

Current code issue:

- `max_num_batched_tokens` is validated but not enforced
- `max_num_seqs` exists in config but is not used

Why this matters:

- rollout memory is the first likely GPU-side scale blocker
- without these caps, increasing concurrency can over-allocate KV cache even when the config suggests otherwise

`enable_chunked_prefill` should remain the rollout-side knob for higher concurrency. It is part of the later `32/64-VM` plan, not mandatory for the immediate `1 VM` path.

### 6.2 Remove the redundant old-policy logprob recompute

Required changes across [`vllm_rollout_spmd.py`](/home/kevinzyz/yincheng/arpo/verl/workers/rollout/vllm_rollout_spmd.py) and [`ray_trainer.py`](/home/kevinzyz/yincheng/arpo/verl/trainer/ray_trainer.py):

- set `SamplingParams.logprobs = 1`
- extend rollout output `DataProto` to include:
  - generated `responses` as today
  - per-token sampled logprobs aligned to those generated tokens

Use these rollout-time logprobs as the source of `old_log_probs` whenever possible instead of calling `actor_rollout_wg.compute_log_probs(batch)`.

#### Public knob

Add:

- `worker.rollout.old_logprob_source: "auto" | "rollout" | "recompute"`

Default:

- `"auto"`

Semantics:

- `"auto"`: use rollout logprobs when alignment succeeds, else fall back to recompute
- `"rollout"`: require rollout logprobs; fail if alignment cannot be proven
- `"recompute"`: preserve the current behavior for debugging

#### Alignment strategy

For the first iteration:

- keep `RemoteEnvWorker` trajectory assembly unchanged
- do not change how `history_messages`, `input_ids`, or `labels` are constructed

Instead:

- accumulate per-env, per-step rollout metadata in the trainer:
  - `response_ids`
  - sampled-token logprobs
- after `get_train_dict()` returns the full tokenized trajectory, build dense `old_log_probs` by aligning rollout `response_ids` to assistant-labeled spans in the final trajectory

Alignment rules:

- operate only on positions where `labels != -100`
- match left-to-right, step by step
- support left padding and right truncation
- fill masked-out positions with `0.0`; they are ignored by `response_mask`
- if a step cannot be aligned exactly:
  - `"auto"` falls back to recompute for the whole batch
  - `"rollout"` raises an error
  - record a counter / metric for fallback frequency

Why this is the standard PPO path here:

- rollout-time sampled token logprobs are exactly the old-policy probabilities PPO needs
- they remove one of the two dominant stage costs in the live run

### 6.3 Remove aggressive per-microbatch allocator churn

Required changes in [`dp_actor.py`](/home/kevinzyz/yincheng/arpo/verl/workers/actor/dp_actor.py):

- remove unconditional `torch.cuda.empty_cache()` before every micro-batch forward
- remove unconditional `torch.cuda.empty_cache()` before every micro-batch backward
- keep cache clearing only at vLLM wake/sleep boundaries, consistent with the existing sharding-manager comment

#### Public knob

Add:

- `worker.actor.empty_cache_policy: "boundary_only" | "aggressive" | "off"`

Default:

- `"boundary_only"`

Semantics:

- `"boundary_only"`: only clear cache at rollout wake/sleep boundaries
- `"aggressive"`: preserve the current per-microbatch behavior as a fragmentation fallback
- `"off"`: disable explicit cache clearing entirely

### 6.4 Throughput profile for A100 80G

The document should treat this as the first tuning profile to validate on the existing `1 VM + 8 GPU` path:

- `worker.actor.micro_batch_size_per_device_for_update: 2`
- `worker.actor.micro_batch_size_per_device_for_experience: 2`
- `worker.actor.offload.offload_optimizer: false`

Next gated experiment after that profile:

- if memory remains below budget, set `enable_gradient_checkpointing: false`

Why:

- `offload_optimizer` is a safety setting, not a throughput setting
- on `80 GB` cards it should not remain enabled in the high-throughput path unless memory proves it necessary

### 6.5 Fix the remote-env scheduler before any 32/64-VM scale work

Required changes in [`ray_trainer.py`](/home/kevinzyz/yincheng/arpo/verl/trainer/ray_trainer.py):

- add `env.remote_server_urls: list[str] | null`
- keep `env.remote_server_url` as a backward-compatible single-URL shorthand
- normalize both into one runtime list of remote endpoints
- create one `RemoteEnvWorker` per remote URL
- make validation batch sizing use the actual remote worker count, not a hardcoded `1`

The rollout scheduler rewrite is mandatory before scale:

- compute `required_rollouts = rollout_batch_size * rollout.n`
- chunk that full workload across available env workers
- support all cases:
  - `1 VM`
  - `N VMs` where `N < required_rollouts`
  - `N VMs` where `N >= required_rollouts`

Important policy choice:

- do not assume `rollout_batch_size == num_envs`
- for future scale, env count is a concurrency pool
- update batch should remain moderate even when env count grows

### 6.6 Immediate 1-VM / 8-GPU operating mode

This is the first-class operating mode to preserve while implementing the changes above.

Defaults:

- keep `8 GPUs`
- keep `SP = 4`
- keep `TP = 4`
- keep remote env count at `1`
- apply the old-logprob removal
- enforce rollout KV/cache caps
- apply allocator cleanup
- apply the A100 throughput profile before touching VM count

Explicit non-goal:

- the immediate objective is not to make `1 VM` saturate all 8 GPUs perfectly
- the objective is to remove avoidable inefficiency while keeping the same architecture that later supports `32/64 VMs`

## 7. Validation Matrix And Acceptance Criteria

### 7.1 Correctness tests

Required tests:

- unit test: rollout initialization passes through `max_num_batched_tokens`, `max_num_seqs`, and optional `kv_cache_memory_bytes`
- unit test: rollout logprob extraction returns a tensor shaped like `responses`
- unit test: dense `old_log_probs` assembly from rollout metadata aligns correctly for:
  - text-only sequences
  - multimodal sequences
  - left padding
  - right truncation
  - multiple assistant turns
- regression test: if alignment fails, `"auto"` falls back to recompute and logs the fallback
- regression test: `"rollout"` mode hard-fails on alignment mismatch
- scheduler test: with `rollout_batch_size > 1`, total collected rollouts always equal `rollout_batch_size * rollout.n`

### 7.2 Performance acceptance

Use the current first-step live run as baseline.

Targets:

- `old` stage time reduced by at least `80%` in `"rollout"` mode or successful `"auto"` mode
- actor update wall time reduced materially in the throughput profile
  - first target: at least `25%` improvement over the current baseline
- rollout memory caps become observable and bounded
  - increasing concurrency must not ignore `max_num_batched_tokens`
  - `max_num_seqs` must affect admission behavior
- `1 VM` smoke path remains functional end-to-end

### 7.3 Memory acceptance

Memory rules to preserve:

- actor/update VRAM budget is controlled by:
  - micro-batch size
  - sequence length
  - SP size
  - checkpointing
- rollout/KV memory budget is controlled by:
  - `gpu_memory_utilization` or `kv_cache_memory_bytes`
  - `max_num_batched_tokens`
  - `max_num_seqs`
- scaling env count alone must not be treated as a reason to increase micro-batch size

## 8. Recommended Execution Order

Priority order:

1. Enforce rollout memory caps in vLLM init.
2. Add rollout-time sampled token logprobs and dense `old_log_probs` assembly.
3. Add fallback logic for alignment mismatch and keep `"recompute"` mode for debugging.
4. Add `empty_cache_policy` and switch default behavior to boundary-only cache clearing.
5. Validate the A100 throughput profile on the existing `1 VM + 8 GPU` path.
6. Only after that, implement multi-URL remote env support and the full rollout scheduler rewrite.
7. Benchmark:
   - `num_envs = 16`
   - `rollout_batch_size = 4`
   - `rollout.n = 4`
8. Only then test `num_envs = 32/64`.

## 9. Practical Expectations For 32/64 VMs

If the remote scheduling code is fixed, then `32/64 VMs` should help:

- environment throughput
- data freshness
- utilization of the two rollout groups

But `32/64 VMs` alone will **not** make the cluster proportionally faster.

The hard limits remain:

- `2` rollout TP groups on `8 GPUs` with `tensor_parallel_size = 4`
- `2` actor DP groups on `8 GPUs` with `ulysses_sequence_parallel_size = 4`
- long-context actor forward/backward cost

So the realistic goal is:

- better utilization
- fewer idle gaps
- higher steady-state throughput

not linear speedup with the number of VMs.

## 10. Bottom Line

The current codebase is functionally correct for the 8-GPU multimodal SP smoke path, and it already achieves the core memory goal of splitting long-sequence training across `4` SP ranks.

The next work should therefore focus on:

- making rollout memory limits real
- removing the redundant old-policy logprob pass
- eliminating unnecessary allocator churn
- validating the `1 VM + 8 GPU` high-throughput profile
- only then extending the remote scheduler for `32/64-VM` scale

The main large-scale risk is still not that actor VRAM will automatically scale with VM count. The bigger risks are:

- rollout KV-cache growth
- CPU / Ray memory pressure
- update wall time exploding from too much accumulation
