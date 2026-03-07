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

See [`smoke_remote_env_8gpu_a100.yaml`](/home/kevinzyz/yincheng/arpo/configs/smoke_remote_env_8gpu_a100.yaml#L7).

This means:

- the context is very long
- training runs with tiny micro-batches
- the actor does many small forwards/backwards
- the rollout side is not the dominant cost

The two largest costs in the real run were:

1. `update_actor`
2. `compute_log_probs` for old policy probabilities

## 6. What To Change For Real 64-VM Efficiency

### 6.1 First fix correctness of remote scheduling

Before talking about throughput tuning, the trainer should support:

- multiple remote server URLs
- or another explicit remote-env pool abstraction

And the rollout scheduler must be based on:

- `required_rollouts = rollout_batch_size * rollout.n`

not just `rollout.n`.

Without this, large-scale remote rollout is not correctly represented in the trainer.

### 6.2 Do not equate `num_envs` with `rollout_batch_size`

At large scale, it is better to treat many VMs as a concurrency pool, not as a command to update on all of them every step.

Recommended approach:

- keep many remote envs available, for example `64`
- keep `rollout_batch_size` moderate, for example `4` or `8`
- choose `rollout.n` based on GRPO quality, ideally `8`

This keeps the model busy without forcing one giant PPO update every step.

### 6.3 Increase micro-batch size before increasing total batch aggressively

The current run has large headroom on `80 GB` cards:

- `max_memory_reserved_gb` was about `45.7`

So the first speed knob to test on A100 80G is:

- `micro_batch_size_per_device_for_update: 2`
- `micro_batch_size_per_device_for_experience: 2`

Why this matters:

- it reduces gradient-accumulation steps
- it reduces the number of tiny forward/backward launches
- it should improve MFU significantly

This is a better first move than simply increasing total rollout volume.

### 6.4 Consider disabling gradient checkpointing once memory is confirmed

Current config:

- `enable_gradient_checkpointing: true`

Checkpointing saves memory but costs time. On `80 GB` GPUs, once micro-batch `2` is validated, the next speed experiment should be:

- disable gradient checkpointing

That should materially speed actor update if memory remains within budget.

### 6.5 Treat old-logprob recompute as a major optimization target

`compute_log_probs` is too expensive at scale.

Medium-term optimization:

- return token log-probs directly from rollout when possible
- avoid a separate full actor forward just to recover old policy log-probs

This is a larger engineering change, but it targets one of the two biggest step costs directly.

### 6.6 Revisit `torch.cuda.empty_cache()` in the actor update loop

In [`dp_actor.py`](/home/kevinzyz/yincheng/arpo/verl/workers/actor/dp_actor.py#L413), `torch.cuda.empty_cache()` is called before forward and before backward for every micro-batch.

That is helpful as an OOM escape hatch, but it is expensive and can hurt allocator locality and throughput.

Recommendation:

- keep it only as a guarded fallback
- do not run it unconditionally on every micro-batch in the large-scale training path

### 6.7 Use env scale to hide latency, not to force huge per-step updates

With `64 VMs`, the goal should be:

- keep rollout groups fed continuously
- keep DP groups fed continuously
- avoid creating one gigantic PPO step with excessive accumulation

Good large-scale training is a balance between:

- enough on-policy data to keep GPUs busy
- small enough update chunks to keep wall time reasonable

## 7. Recommended Scaling Order

The order below is safer than jumping directly from the current smoke run to a large `64-VM` setting.

### Stage 1: single-node, correctness-first

- keep `8 GPUs`
- keep `ulysses_sequence_parallel_size: 4`
- keep `tensor_parallel_size: 4`
- validate remote multi-env scheduler after patching

### Stage 2: increase model efficiency on the same node

- raise `micro_batch_size_per_device_for_update` from `1` to `2`
- raise `micro_batch_size_per_device_for_experience` from `1` to `2`
- test with current `64K` prompt budget
- if stable, test disabling gradient checkpointing

### Stage 3: moderate rollout scale

Recommended first large-scale target:

- `num_envs`: `16` or `32`
- `rollout_batch_size`: `4`
- `rollout.n`: `4` or `8`

Only after that is stable should the system move toward:

- `num_envs: 64`
- `rollout_batch_size: 8`

### Stage 4: rollout memory tuning

For larger rollout concurrency, tune:

- `worker.rollout.gpu_memory_utilization`
- `worker.rollout.max_num_batched_tokens`
- `worker.rollout.enable_chunked_prefill`

These affect vLLM memory much more directly than actor update VRAM.

## 8. Practical Expectations For 64 VMs

If the remote scheduling code is fixed, then `64 VMs` should help:

- environment throughput
- data freshness
- utilization of the two rollout groups

But `64 VMs` alone will **not** make the cluster proportionally faster.

The hard limits remain:

- `2` rollout TP groups on `8 GPUs` with `tensor_parallel_size = 4`
- `2` actor DP groups on `8 GPUs` with `ulysses_sequence_parallel_size = 4`
- long-context actor forward/backward cost

So the realistic goal is:

- better utilization
- fewer idle gaps
- higher steady-state throughput

not linear speedup with the number of VMs.

## 9. Recommended Next Engineering Tasks

Priority order:

1. Add support for multiple remote env servers in config and trainer.
2. Rewrite remote rollout scheduling to use `rollout_batch_size * rollout.n`.
3. Add throughput-focused profiling logs per stage:
   - rollout generation
   - old log-prob recompute
   - actor update
   - CPU memory
4. Make per-microbatch `empty_cache()` optional.
5. Validate `micro_batch_size = 2` on `A100 80G`.
6. Benchmark with:
   - `num_envs = 16`
   - `rollout_batch_size = 4`
   - `rollout.n = 4`
7. Only then test `num_envs = 64`.

## 10. Bottom Line

The current codebase is now functional for the 8-GPU multimodal SP smoke path, and it does achieve the main memory goal of splitting long-sequence training across `4` SP ranks.

However, the path to efficient `64-VM` training still requires:

- fixing remote-env scheduling
- feeding the two DP / TP groups with better-sized batches
- increasing micro-batch sizes on `80 GB` cards
- reducing actor-side overheads, especially `old_log_probs` recompute

The main large-scale risk is not that actor VRAM will automatically scale with the number of VMs. The bigger risks are:

- rollout KV-cache growth
- CPU / Ray memory pressure
- update wall time exploding from too much accumulation
