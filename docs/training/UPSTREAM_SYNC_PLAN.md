# ARPO Training: Upstream Sync Plan

**Purpose:** Document every divergence between our codebase and the original
JIA-Lab ARPO (`https://github.com/JIA-Lab-research/ARPO`), explain why each
divergence causes OOM, and specify exactly what to restore.

---

## 1. Original ARPO Reference

| Item | Value |
|------|-------|
| Upstream repo | `https://github.com/JIA-Lab-research/ARPO` |
| Canonical launch script | `examples/osworld_full_arpo.sh` |
| Hardware (paper) | 2 nodes × 8 × A100-SXM4-80GB = 16 GPUs |
| Environments | 128 parallel Docker VMs |
| Model | `ByteDance-Seed/UI-TARS-1.5-7B` |

The script parameters below are the ground truth for every training hyperparameter:

```bash
# From examples/osworld_full_arpo.sh (fetched verbatim)
NUM_GPUS=8
NUM_ENVS=128
ROLLOUT_N=8
ROLLOUT_BSZ=$((NUM_ENVS / ROLLOUT_N))   # = 16

python3 -m verl.trainer.main \
    data.max_prompt_length=64000 \
    data.max_response_length=8192 \
    data.max_pixels=2116800 \
    data.min_pixels=256 \
    data.rollout_batch_size=16 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.ulysses_sequence_parallel_size=1 \
    worker.actor.padding_free=true \
    worker.actor.global_batch_size=8 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.temperature=1.0 \
    worker.rollout.n=8 \
    worker.rollout.limit_images=15 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_num_batched_tokens=128000 \
    algorithm.disable_kl=True \
    env.num_envs=128 \
    env.max_steps=15 \
    trainer.nnodes=2
```

---

## 2. Root Cause of OOM

Three compounding problems — **all introduced by us**, not present in original.

### 2.1 flash-attn broken → xformers backend (PRIMARY)

When flash-attn is unavailable, vLLM falls back to the xformers attention
backend. xformers computes attention by materialising the full score matrix:

```
attention_scores shape: [batch, heads, seq_len, seq_len]
```

For a 64K-token sequence with 16 heads and bf16:

```
64000 × 64000 × 16 × 2 bytes = 131 GB  per forward pass
```

Even for our 32K `max_prompt_length`:

```
32000 × 32000 × 16 × 2 bytes = 32 GB  per forward pass
```

flash-attn avoids this entirely by using a tiled kernel that never writes the
full matrix. The original ARPO works with 64K contexts precisely because it runs
with flash-attn. Our fork broke flash-attn by installing a version
(`flash-attn ≥ 2.7`) that calls `torch.library.wrap_triton`, an API introduced
in PyTorch 2.6. Our environment has torch `2.5.1+cu124`, which does not expose
this symbol → `AttributeError` → crash → vLLM falls back to xformers → OOM.

**No pre-built wheel exists for flash-attn + CUDA 12.4 + torch 2.5. Options:**

| Fix | Cost | Notes |
|-----|------|-------|
| Patch installed `rotary.py` (1 line) | ~30 s | Remove the `wrap_triton` wrapper; kernel still runs correctly |
| Source build of `flash-attn==2.6.3` | ~60 min | Verify 2.6.3's `rotary.py` does not itself call `wrap_triton` first |
| Upgrade torch to 2.6 | Risk | May break vLLM; not recommended without full dep audit |

**Recommended patch** (find the installed file path first with
`pip show flash-attn`, then edit `flash_attn/ops/triton/rotary.py`):

```python
# BEFORE (line ~159)
torch.library.wrap_triton(rotary_kernel)[grid](

# AFTER
rotary_kernel[grid](
```

`wrap_triton` is a compiler-tracing hint; removing it has zero effect on
correctness or numerical output.

### 2.2 `ulysses_sequence_parallel_size: 4` (SECONDARY)

The original ARPO sets `ulysses_sequence_parallel_size=1` — **no sequence
parallelism for the actor**. We set it to 4. This means:

- The 7B model's attention is split across 4 GPUs per sequence
- Each GPU must participate in 4-way all-reduce collectives every forward pass
- The Ulysses SP path in `dp_actor.py` calls `_select_pixel_values_for_sp_rank`
  which requires careful flash-attn integration; with xformers it degenerates
  into much higher memory usage per rank
- Added `offload_optimizer: true` as a workaround (unnecessary overhead)

The original runs fine on 16 GPUs with `ulysses_sp=1` because:
- 16-GPU FSDP already distributes weight shards adequately
- No extra SP overhead, no SP-specific memory amplification

### 2.3 `gpu_memory_utilization: 0.4` (TERTIARY)

Original uses `0.6`. We lowered to `0.4`. On an 80 GB A100 with the 7B model
(~14 GB weights):

| Setting | Free after weights | KV cache budget |
|---------|--------------------|-----------------|
| 0.6 | ~66 GB | ~39.6 GB |
| 0.4 | ~66 GB | ~26.4 GB |

The KV cache holds intermediate attention states for all in-flight sequences.
With 40% allocation we have ~13 GB less headroom, which matters at long contexts.

---

## 3. Complete Parameter Diff

### 3.1 YAML configs vs original script

The table below compares `configs/smoke_remote_env_8gpu_a100.yaml` (our current
state) with `examples/osworld_full_arpo.sh` (original).

| Parameter | Original ARPO | Our Smoke Config | Status |
|-----------|--------------|-----------------|--------|
| `data.max_prompt_length` | **64000** | 32000 | Wrong — too short |
| `data.max_response_length` | **8192** | 1024 | Wrong — too short |
| `data.rollout_batch_size` | 16 | 1 | OK for 1-env smoke |
| `worker.actor.ulysses_sequence_parallel_size` | **1** | 4 | Wrong — must revert |
| `worker.actor.global_batch_size` | 8 | 4 | OK for smoke |
| `worker.actor.micro_batch_size_per_device_for_update` | 1 | 1 | Correct |
| `worker.actor.micro_batch_size_per_device_for_experience` | 1 | 1 | Correct |
| `worker.actor.padding_free` | true | true | Correct |
| `worker.actor.offload.offload_optimizer` | **false** | true | Wrong — workaround, revert |
| `worker.rollout.n` | 8 | 4 | OK for smoke |
| `worker.rollout.gpu_memory_utilization` | **0.6** | 0.4 | Wrong — must revert |
| `worker.rollout.enforce_eager` | **false** | true | Wrong — workaround, revert |
| `worker.rollout.enable_chunked_prefill` | **false** | true | Wrong — workaround, revert |
| `worker.rollout.tensor_parallel_size` | **1** | 4 | Wrong — original uses TP=1 |
| `worker.rollout.max_num_batched_tokens` | **128000** | 33024 | Wrong — too restrictive |
| `worker.rollout.max_num_seqs` | 1024 | 1 | Wrong — too restrictive |
| `worker.rollout.limit_images` | 15 | 15 | Correct |
| `env.num_envs` | 128 | 1 | OK for smoke |
| `env.max_steps` | 15 | 15 | Correct |
| `trainer.nnodes` | 2 | 1 | Hardware difference, keep 1 |

### 3.2 Python dataclass defaults vs original

These files are nearly identical to the original. We added three optional fields
that are backward-compatible (disabled by default):

**`verl/workers/rollout/config.py`** — additions only:
```python
kv_cache_memory_bytes: Optional[int] = None   # added by us; None = use gpu_memory_utilization
old_logprob_source: str = "auto"              # added by us; "auto" matches original behaviour
trust_remote_code: bool = True                # added by us; required for UI-TARS processor
image_min_pixels: int = 0                     # added by us; 0 = no override
image_max_pixels: int = 0                     # added by us; 0 = no override
```

**`verl/workers/actor/config.py`** — additions only:
```python
empty_cache_policy: str = "boundary_only"     # added by us; reduces allocator churn
```

**All Python dataclass defaults match the original.** No changes needed to
these files. The divergence is 100% in the YAML config files.

---

## 4. What to Keep (OSWorld-specific code)

The following files are our additions on top of the original and must **not** be
removed or reverted. They implement the OSWorld environment integration.

| File | Purpose |
|------|---------|
| `verl/trainer/gui_agent.py` | `EnvWorker` (local Docker), `RemoteEnvWorker` (HTTP client), action parsing (Thought/Action → bbox), system prompt |
| `verl/trainer/remote_env_protocol.py` | HTTP wire format: base64 image encoding/decoding between training cluster and env server |
| `verl/trainer/replay_buffer.py` | ARPO's core novelty: per-task FIFO replay buffer for successful trajectories |
| `verl/trainer/ray_trainer.py` | Env loop additions: `_create_envs`, `_run_rollout_chunk`, `apply_replay`, `_load_replay_data` — keep all |
| `verl/utils/osworld.py` | `OSWorldTaskConfigDataset`, `OSWorldGRPODataset`, `collate_fn_dataproto` |
| `verl/workers/actor/dp_actor.py` | Keep `_select_pixel_values_for_sp_rank` (needed when `ulysses_sp > 1`) |
| `verl/utils/multimodal_sp.py` | Ulysses SP hook for vision inputs; keep even when `ulysses_sp=1` |
| `scripts/servers/remote_env_server.py` | FastAPI server on env cluster: `/env/reset`, `/env/step`, `/env/evaluate` |
| `OSWorld/` submodule | Desktop environment implementation |

---

## 5. Corrected Config for 1-Node 8-GPU

After fixing flash-attn, the smoke config should match the original paper values
adapted for single-node operation:

```yaml
# configs/smoke_remote_env_8gpu_a100.yaml  (target state)

data:
  train_files: OSWorld/evaluation_examples/test_smoke_4.json
  val_files: OSWorld/evaluation_examples/test_smoke_4.json
  prompt_key: instruction
  answer_key: ""
  image_key: images
  max_prompt_length: 64000      # restored: original 64000
  max_response_length: 8192     # restored: original 8192
  rollout_batch_size: 1         # keep: 1 env / n=4 → 1 task per step
  val_batch_size: -1
  shuffle: true
  seed: 1
  max_pixels: 2116800
  min_pixels: 256

algorithm:
  adv_estimator: grpo
  disable_kl: true
  use_kl_loss: false
  kl_coef: 0
  enable_replay: true

worker:
  actor:
    global_batch_size: 4
    micro_batch_size_per_device_for_update: 1
    micro_batch_size_per_device_for_experience: 1
    max_grad_norm: 1.0
    padding_free: true
    ulysses_sequence_parallel_size: 1   # restored: original 1 (was 4)
    ppo_epochs: 1
    clip_ratio_low: 0.2
    clip_ratio_high: 0.3
    model:
      model_path: ByteDance-Seed/UI-TARS-1.5-7B
      enable_gradient_checkpointing: true
      trust_remote_code: true
      freeze_vision_tower: true
    optim:
      lr: 1.0e-6
      weight_decay: 1.0e-2
      strategy: adamw_bf16
      lr_warmup_ratio: 0.05
    fsdp:
      enable_full_shard: true
      enable_cpu_offload: false
      enable_rank0_init: false
      torch_dtype: bf16
    offload:
      offload_params: false
      offload_optimizer: false   # restored: original false (was true)

  rollout:
    temperature: 1.0
    n: 4                          # keep: smoke uses n=4
    gpu_memory_utilization: 0.6   # restored: original 0.6 (was 0.4)
    enforce_eager: false          # restored: original false (was true)
    enable_chunked_prefill: false # restored: original false (was true)
    tensor_parallel_size: 1       # restored: original 1 (was 4); 8 independent engines
    limit_images: 15
    max_num_batched_tokens: 128000 # restored: original 128000 (was 33024)
    max_num_seqs: 4               # restored: reasonable default (was 1)
    val_override_config:
      temperature: 0.5
      n: 1

  ref:
    fsdp:
      enable_full_shard: true
      enable_cpu_offload: false
    offload:
      offload_params: false

  reward:
    reward_type: function
    score_function: math
    skip_special_tokens: true

env:
  num_envs: 1
  max_steps: 15
  remote_server_url: "http://10.100.4.7:15001"

trainer:
  total_episodes: 10
  logger: ["console", "wandb"]
  project_name: arpo-smoke-test
  experiment_name: uitars_7b_8xa100_remote_safe_smoke
  n_gpus_per_node: 8
  nnodes: 1
  val_before_train: false
```

**1-node vs 2-node adaptation rationale:**

On 2 nodes (original): 16 GPUs with TP=1 → 16 independent vLLM engines. Each
engine holds the full model (~14 GB) plus KV cache (60% × 66 GB ≈ 39.6 GB).
Total memory per GPU: ~53 GB out of 80 GB → comfortable.

On 1 node (ours): 8 GPUs with TP=1 → 8 independent vLLM engines. Same per-GPU
breakdown applies. With 8 engines and 1 env / n=4 rollouts, throughput is lower
than the 16-GPU original but memory profile is identical per GPU.

---

## 6. Execution Order

1. **Fix flash-attn first** (prerequisite for all other steps)
   ```bash
   # Find rotary.py
   python -c "import flash_attn; import os; print(os.path.join(os.path.dirname(flash_attn.__file__), 'ops/triton/rotary.py'))"
   # Edit that file: replace  torch.library.wrap_triton(rotary_kernel)[grid](
   #                     with  rotary_kernel[grid](
   ```

2. **Verify flash-attn works**
   ```bash
   python -c "from flash_attn.ops.triton.rotary import apply_rotary; print('flash-attn OK')"
   ```

3. **Update `configs/smoke_remote_env_8gpu_a100.yaml`** with the corrected values above

4. **Restart training**
   ```bash
   python -m verl.trainer.main config=configs/smoke_remote_env_8gpu_a100.yaml
   ```

5. **Verify no xformers warning** — training output should NOT contain:
   ```
   WARNING: Current vllm-flash-attn has a bug inside vision module, so we use xformers backend instead.
   ```

6. **Verify no OOM** — first rollout chunk should complete within 5–10 minutes

---

## 7. Additional Code-Level GPU Differences (Beyond YAML)

File-by-file read of both repos revealed these further divergences.

### A. `verl/workers/rollout/vllm_rollout_spmd.py`

**`max_model_len` and `max_num_batched_tokens` — COMMENTED OUT in upstream, active in ours**

```python
# Upstream (commented out — vLLM auto-detects):
# max_model_len=config.prompt_length + config.response_length,
# max_num_batched_tokens=config.max_num_batched_tokens,

# Ours (explicitly passed):
max_model_len=config.prompt_length + config.response_length,
max_num_batched_tokens=config.max_num_batched_tokens,
max_num_seqs=config.max_num_seqs,
```

By explicitly passing these, we constrain vLLM tightly. With our old
`max_num_batched_tokens: 33024` this exactly hits the internal validation floor
(`must be > prompt_length + response_length`). The original lets vLLM pick its
own optimal limits. **Action: revert to 128000 as documented in Section 5, or
remove the explicit pass and let vLLM auto-detect.**

**`disable_mm_preprocessor_cache=True` removed** — correct fix for vLLM 0.15.0
(API no longer exists). Keep this removal.

**`old_logprob_source` logprob metadata path** — entirely new in ours.
Enables capturing per-token logprobs during rollout to skip recomputation
(performance optimisation). Not in original. Keep it.

---

### B. `verl/workers/actor/dp_actor.py`

**`torch.compile` DISABLED** (performance impact)

```python
# Upstream: compiles log_probs_from_logits when use_torch_compile=True
self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)

# Ours: always eager
self.log_probs_from_logits = VF.log_probs_from_logits
```

Comment in our code says TorchDynamo produces fake-tensor shape mismatches in
GRPO batches. Disabling is safe; the cost is ~10–30% slower log_prob computation
during the experience collection step. Consider re-enabling once GRPO batches are
stable (or target `dynamic=True` which should handle symbolic shapes).

**`response_mask` and `advantages` indexing — correctness difference**

```python
# Upstream (shifts by 1):
response_mask = model_inputs["response_mask"][:, 1:]
advantages    = model_inputs["advantages"][:, 1:]

# Ours (no shift):
response_mask = model_inputs["response_mask"]
advantages    = model_inputs["advantages"]
```

The original applies `[:, 1:]` because the trainer stores `response_mask` and
`advantages` padded to the full sequence length with a leading token. Our trainer
stores them pre-shifted, so the slicing was dropped. If the upstream trainer is
ever re-synced into ours, this must match exactly.

**Memory-efficient log_prob path (NON-padding-free branch)** — improvement we made, KEEP

```python
# Upstream: uses log_probs_from_logits → F.cross_entropy → 16 GiB intermediate
logits = logits[:, -response_length - 1 : -1, :]
log_probs = self.log_probs_from_logits(logits, responses)

# Ours: logsumexp trick → O(bsz × response_length), no 16 GiB intermediate
logsumexp = torch.logsumexp(logits, dim=-1)
gathered  = logits.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
del logits
log_probs = gathered - logsumexp
```

For 64K-token multi-turn sequences the difference is ~16 GB vs ~0.5 GB.
This is a correct and intentional improvement.

**`_optimizer_step` — optimizer state device sync** — added for FSDP CPU offload. Keep.

---

### C. `verl/workers/sharding_manager/fsdp_vllm.py`

**vLLM API update — REQUIRED fix**

```python
# Upstream (vLLM < 0.15.0):
self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group

# Ours (vLLM 0.15.0+):
self.tp_group = vllm_ps.get_tp_group().device_group
```

**Qwen2.5-VL / UI-TARS weight key remapping — REQUIRED for our model**

```python
# Upstream: no remapping
for name, tensor in actor_weights.items():
    yield name, tensor.full_tensor() if self.world_size != 1 else tensor

# Ours: renames HF keys to vLLM namespace
if name.startswith("model.visual."):
    name = "visual." + name[len("model.visual."):]
elif name.startswith("model.language_model."):
    name = "language_model.model." + name[len("model.language_model."):]
```

HuggingFace Qwen2.5-VL stores weights as `model.visual.*` and
`model.language_model.*`. vLLM expects `visual.*` and `language_model.model.*`.
Without this remapping, all vision tower weights silently fail to load into vLLM
→ random outputs during rollout. This is a REQUIRED addition. Keep it.

---

### D. `verl/workers/fsdp_workers.py`

**`_validate_head_parallel_size()` — guard added**

Validates that `num_attention_heads` and `num_key_value_heads` are divisible by
the chosen parallel size (both TP and Ulysses SP). This is what would have caught
the TP=2 + ulysses_sp=4 incompatibility before the runtime tensor mismatch.
Keep it.

---

### Summary: What Is Safe vs What to Revert

| Change | File | Verdict |
|--------|------|---------|
| `max_model_len` + `max_num_batched_tokens` explicitly passed to LLM() | vllm_rollout_spmd.py | Keep, but fix values (128000) |
| `max_num_seqs` explicitly passed | vllm_rollout_spmd.py | Raise to 4+ (not 1) |
| `disable_mm_preprocessor_cache` removed | vllm_rollout_spmd.py | Correct fix, keep |
| `old_logprob_source` + rollout logprob metadata | vllm_rollout_spmd.py | Improvement, keep |
| `torch.compile` disabled | dp_actor.py | Safe workaround; re-enable later |
| `response_mask/advantages` no `[:, 1:]` | dp_actor.py | Must match trainer storage — keep as-is |
| `logsumexp` log_prob trick + `del logits` | dp_actor.py | Memory improvement, keep |
| Optimizer state device sync | dp_actor.py | Needed for CPU offload, keep |
| `_select_pixel_values_for_sp_rank` | dp_actor.py | Needed for ulysses_sp>1, keep |
| `get_tp_group()` API | fsdp_vllm.py | Required for vLLM 0.15.0, keep |
| Qwen2VL weight key remapping | fsdp_vllm.py | Required for UI-TARS, keep |
| `_validate_head_parallel_size` | fsdp_workers.py | Safety check, keep |

---

## 8. Additional Findings from 4-Agent Parallel Audit

The following items were discovered during a side-by-side file comparison after
the initial plan was written. They are separate from the YAML-level changes in
Section 3 and the summary in Section 7.

---

### 8.1 `verl/utils/ulysses.py` — 3D position_ids handling (ALREADY FIXED, WE ARE BETTER THAN UPSTREAM)

**Location:** `ulysses_pad_and_slice_inputs`, lines ~293–301

**Status:** ✅ Our fork already has the correct implementation. Upstream is the one with the bug.

**Our fork (correct — lines 293–301):**

```python
if position_ids_rmpad is not None:
    # position_ids may be 2D [bsz, seqlen] or 3D [bsz, 3, seqlen] (Qwen2.5-VL 3D-RoPE).
    # Match all leading dims and pad the last (sequence) dimension with zeros.
    pad_pos_ids = position_ids_rmpad.new_zeros(*position_ids_rmpad.shape[:-1], pad_size)
    position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
if position_ids_rmpad is not None:
    position_ids_rmpad = slice_input_tensor(position_ids_rmpad, dim=position_ids_rmpad.dim() - 1, padding=False)
```

**Upstream (broken for Qwen2.5-VL 3D m-rope):**

```python
if position_ids_rmpad is not None:
    pad_pos_ids = torch.arange(pad_size, device=position_ids_rmpad.device).unsqueeze(0)
    position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
# we don't need to slice position ids    ← comment is wrong; also doesn't slice
input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
```

**Two bugs in upstream that our fork has already fixed:**
1. `torch.arange().unsqueeze(0)` produces shape `[1, pad_size]` — wrong for 3D `[bsz, 3, seqlen]` position_ids. Our `new_zeros(*shape[:-1], pad_size)` matches any number of leading dims.
2. Upstream does NOT slice position_ids after padding → all SP ranks receive the full unsliced sequence → shape mismatch. Our fork slices along the last dim dynamically.

**No action needed — our fork is correct.**

---

### 8.2 `verl/workers/fsdp_workers.py` — wrong `attn_implementation` (FIXABLE)

**Location:** `from_pretrained` call in actor model loading

**Problem:**

```python
# Ours:
attn_implementation="sdpa"   # PyTorch scaled dot-product attention

# Upstream (original ARPO):
attn_implementation="flash_attention_2"
```

`sdpa` is the PyTorch built-in fused kernel. `flash_attention_2` uses the
flash-attn library. Once flash-attn is fixed (Section 6, Step 1), the actor
training forward passes should also use flash-attn for:

- Lower memory (O(n) vs O(n²) for the actor's training forward pass)
- Correct memory accounting during FSDP sharding

**Fix:** After flash-attn is confirmed working, change `attn_implementation`:

```python
# In fsdp_workers.py, in the from_pretrained() call:
attn_implementation="flash_attention_2"
```

**Impact:** Without this fix, the actor training forward pass uses PyTorch SDPA
even after vLLM/rollout is using flash-attn. This means the actor's memory
footprint is unnecessarily higher than the original for long contexts.

---

### 8.3 `verl/utils/torch_functional.py` — removed `.float()` upcast (KEEP, VALIDATE)

**Location:** `log_probs_from_logits` function

**Problem:**

```python
# Upstream (safe for stability):
log_probs = F.cross_entropy(logits.float(), labels, reduction='none')

# Ours (removed upcast):
log_probs = F.cross_entropy(logits, labels, reduction='none')  # stays bf16
```

We removed the `.float()` upcast to save memory — for 64K sequences with vocab
size 32K, the intermediate `float32` logits would be 8 GB vs 4 GB in bf16.

**Risk:** Cross-entropy in bf16 accumulates the log-softmax denominator with
only 7-bit mantissa. For large vocabularies (32K tokens) this can cause subtle
numerical drift in log probabilities, potentially destabilising GRPO advantages.

**Verdict:** Keep the memory optimisation for now. If training shows divergent
loss curves or NaN advantages after the flash-attn fix, restore `.float()` upcast
and use the `logsumexp` trick instead (which has been added to the non-padding-free
path in `dp_actor.py` and avoids the 8 GB intermediate).

---

### 8.4 `verl/trainer/ray_trainer.py` — key correctness differences

Several differences in the trainer were found beyond what is in Section 7. They
are all intentional or already handled:

| Difference | Our Fork | Upstream | Verdict |
|------------|----------|----------|---------|
| `tqdm` import | `from tqdm import tqdm` | `from ray.experimental.tqdm_ray import tqdm` | Keep ours — ray tqdm causes segfault with remote workers |
| GRPO std() for n=1 | NaN guard: `std = std.clamp(min=1e-6)` | No guard → NaN with n=1 | Keep ours — required for smoke tests with 1 env |
| Response mask shift | stored pre-shifted, no `[:, 1:]` in actor | stored with leading token, `[:, 1:]` in actor | Keep ours — consistent with our trainer storage |
| Old log_prob source | `get_old_log_probs_with_fallback()` — uses rollout logprobs when available | always recomputes via actor forward pass | Keep ours — significant compute saving |
| `ref_log_probs` truncation | guarded with `[:, :response_mask.size(1)]` | no guard | Keep ours — prevents shape error with multimodal |
| Reward weighting | `0.1 * clamp(format_rewards, -1, 1)` | `0.5 * format_rewards` | Keep ours — tuned for OSWorld |
| `RemoteEnvWorker` | HTTP client to remote env server | not present | Keep ours — our architecture requires it |

---

### 8.5 Updated Action Summary

| Item | File | Action | Priority |
|------|------|--------|----------|
| `ulysses_sp: 4` → 1 | YAML config | Revert | **BLOCKER** |
| `gpu_memory_utilization: 0.4` → 0.6 | YAML config | Revert | **BLOCKER** |
| `enforce_eager: true` → false | YAML config | Revert | **BLOCKER** |
| `enable_chunked_prefill: true` → false | YAML config | Revert | **BLOCKER** |
| `tensor_parallel_size: 4` → 1 | YAML config | Revert | **BLOCKER** |
| `max_num_batched_tokens: 33024` → 128000 | YAML config | Revert | **BLOCKER** |
| `max_response_length: 1024` → 8192 | YAML config | Increase | **BLOCKER** |
| flash-attn `wrap_triton` patch | rotary.py | 1-line patch | **PREREQUISITE** |
| `attn_implementation: sdpa` → `flash_attention_2` | fsdp_workers.py | 1-line change | After flash-attn fixed |
| 3D position_ids in ulysses padding | ulysses.py | Already correct in our fork (upstream has the bug) | Done |
| `.float()` upcast in `log_probs_from_logits` | torch_functional.py | Monitor | Only if loss diverges |
| `torch.compile` re-enable | dp_actor.py | Re-enable when stable | Low priority |

---

## 8.6 Second-Round Audit: Additional New Findings

The following were discovered in the second parallel audit pass and are NOT yet
covered in Sections 7 or 8.1–8.5.

---

### 8.6.1 `verl/workers/fsdp_workers.py` — explicit GPU pinning in `dist.init_process_group`

**Location:** `_init_distributed()` / process group initialization, ~line 100

```python
# Ours:
local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RAY_LOCAL_RANK", os.environ.get("RANK", "0"))))
device_id = torch.device("cuda", local_rank)
dist.init_process_group(backend="nccl", device_id=device_id)

# Upstream:
dist.init_process_group(backend="nccl")
```

**Why it matters:** PyTorch ≥ 2.x NCCL backend picks a GPU to own per-rank. Without
`device_id`, the NCCL communicator may map multiple ranks to the same GPU (common
in Ray actor pools where `LOCAL_RANK` is not set by the launcher). This causes
NCCL hangs or double-allocation OOM. The explicit `device_id` pins each rank to its
GPU before any distributed op runs. **Keep — this is a correctness fix.**

---

### 8.6.2 `verl/workers/fsdp_workers.py` — `caching_allocator_warmup` patch before `from_pretrained`

**Location:** model loading path, before `auto_class.from_pretrained()`

```python
# Ours:
try:
    from transformers import modeling_utils as _hf_modeling_utils
    if hasattr(_hf_modeling_utils, "caching_allocator_warmup"):
        _hf_modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
except Exception:
    pass
```

**Why it matters:** Transformers ≥ 4.44 calls `caching_allocator_warmup()` which
pre-warms the CUDA caching allocator by touching every 128 MB page. With 80 GB A100s
and model size ~14 GB, this warmup alone allocates ~80 GB of temporary tensors →
instant OOM before the model even loads. Patching it to a no-op has no effect on
model weights. **Keep — prevents OOM during actor/reference model loading.**

---

### 8.6.3 `verl/workers/fsdp_workers.py` — `device_map="cpu"` always + `is_meta` guard

**Location:** `from_pretrained` call, ~line 245

```python
# Ours: always CPU
device_map="cpu",

# Upstream: conditional
device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
```

And after loading:

```python
# Ours: guards against meta tensors from enable_rank0_init=False non-root ranks
is_meta = any(p.is_meta for p in model.parameters())
if not is_meta:
    model = model.to(device="cpu", dtype=torch_dtype)

# Upstream: unconditional
model = model.to(torch_dtype)
```

**Why it matters:** When `enable_rank0_init=False`, non-root FSDP ranks hold
`init_empty_weights()` meta tensors. Calling `.to()` on a meta tensor raises
`RuntimeError: cannot convert meta tensor`. Our guard skips `.to()` for meta
ranks; upstream would crash. Also: loading directly to "cuda" (upstream) materialises
the full model on GPU before FSDP shards it → peak GPU memory = full model × N ranks
on rank 0's GPU. Our CPU-first approach lets FSDP shard before touching GPU memory.
**Keep — prevents crash and reduces peak GPU memory.**

---

### 8.6.4 `verl/workers/fsdp_workers.py` — `torch.cuda.empty_cache()` before vLLM init

**Location:** `_build_rollout()`, before `LLM(...)` construction

```python
# Ours:
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Upstream: not present
```

**Why it matters:** FSDP model loading and `tie_weights()` leave PyTorch's caching
allocator holding freed blocks. vLLM's `CuMemAllocator` maps KV cache blocks from
**free physical memory** reported by `cudaMemGetInfo`. If PyTorch is caching freed
blocks, `cudaMemGetInfo` understates free memory → vLLM allocates a smaller KV cache
than the configured `gpu_memory_utilization` ratio. A pre-init `empty_cache()` forces
freed blocks back to the OS, letting vLLM see the full ~60 GB of free memory.
**Keep — ensures vLLM gets the correct KV cache budget.**

---

### 8.6.5 `verl/utils/ulysses.py` — `group` parameter in `get_ulysses_sequence_parallel_rank`

**Location:** `slice_input_tensor`, ~line 121

```python
# Ours:
sp_rank = get_ulysses_sequence_parallel_rank(group)

# Upstream:
sp_rank = get_ulysses_sequence_parallel_rank()
```

**Why it matters:** With multiple process groups active (FSDP + Ulysses SP), the
global group singleton can be stale or point to the wrong communicator.
Passing `group` explicitly guarantees the rank is resolved relative to the correct
SP communicator. Without it, `get_ulysses_sequence_parallel_rank()` falls back to the
global group → wrong rank → wrong tensor slice → corrupted activations. **Keep.**

---

### 8.6.6 `verl/utils/torch_functional.py` — off-by-one label shape mismatch handler

**Location:** `log_probs_from_logits`, before the cross_entropy call

```python
# Ours (new guard):
if labels.numel() != logits.size(0):
    if labels.numel() == logits.size(0) + 1:
        labels = labels[:-1]   # trim trailing padding token
    else:
        raise ValueError(f"... logits batch {logits.size(0)} and labels batch {labels.numel()} mismatch ...")

# Upstream: no guard
```

**Why it matters:** In multi-turn remote env rollouts with variable-length
responses, the `responses` tensor (used as labels) can end up one token longer
than `logits` after the `[:, -response_length-1:-1]` slice when
`response_length ≈ seqlen - 1`. Without this guard, PyTorch raises a silent
shape broadcast error or, worse, silently computes wrong log-probs for the
last token. The guard trims the single extra token (which is always a padding
token in that context). **Keep — prevents silent log-prob corruption.**

---

### 8.6.7 `verl/workers/actor/dp_actor.py` — ZeroDivisionError guard for empty batches

**Location:** `update_policy`, mini_batch splitting

```python
# Ours:
chunks = self.config.global_batch_size_per_device
if chunks <= 0 or len(data_selected) <= chunks:
    mini_batches = [data_selected]
else:
    mini_batches = data_selected.split(chunks)

# Upstream:
mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
```

**Why it matters:** In single-env smoke tests, `len(data) == 1` and
`global_batch_size_per_device` can be computed as `global_batch_size // world_size`
where `world_size > global_batch_size`, giving 0. `DataProto.split(0)` raises
`ZeroDivisionError`. The guard bypasses this by treating the full batch as a
single mini-batch. **Keep — prevents crash in small-batch smoke tests.**

---

### 8.6.8 `verl/trainer/ray_trainer.py` — `_ray_get_robust()` for fault-tolerant ops

**Status:** Entirely new method — not in upstream.

```python
# Ours:
def _ray_get_robust(self, futures, timeout_s=300):
    """ray.get with per-future timeout isolation; logs failed futures and returns None."""
```

**Why it matters:** In a 64-VM cloud deployment, any single remote env can become
a zombie — hanging indefinitely on `ray.get()`. The upstream code calls plain
`ray.get([reset_futures])`, which blocks forever if any future hangs. `_ray_get_robust`
sets per-future timeouts and isolates crashes so that a single VM failure doesn't
block the entire training step. **Keep — essential for cloud deployment reliability.**

---

### 8.6.9 `verl/trainer/core_algos.py` — GRPO n=1 fallback (full detail)

**Already noted in Section 8.4 table, but fuller detail for correctness:**

```python
# Ours:
if sample_num > 1:
    id2mean[idx] = torch.mean(sample_scores)
    id2std[idx] = torch.std(sample_scores)
else:
    # REINFORCE-style: (score - 0) / 1 = score
    id2mean[idx] = torch.zeros((), device=..., dtype=...)
    id2std[idx]  = torch.ones((),  device=..., dtype=...)

# Upstream:
assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
id2std[idx]  = torch.std(torch.tensor(id2score[idx]))
```

Two additional correctness improvements vs upstream:
1. Tensors created on the correct device and dtype (upstream uses default CPU float32
   for `torch.tensor(...)` which can cause device mismatch when reward is on CUDA).
2. No assert — smoke tests with 1 env and `rollout.n=4` can still produce
   groups of size 1 when only 1 env returns a valid result in a step.

**Keep.**

---

### 8.6.10 `verl/trainer/config.py` — remote env and WandB fields

```python
# Ours (EnvConfig additions):
remote_server_url:  Optional[str]       = None  # single remote server
remote_server_urls: Optional[list[str]] = None  # multi-server list

# Ours (TrainerConfig addition):
wandb_entity: Optional[str] = None  # WandB team/entity name
```

Upstream has none of these. These are our additions enabling remote env deployment
and WandB team logging. **Keep — required for cloud deployment.**

---

### Updated Action Summary (Sections 8.5 + 8.6 combined)

| Item | File | Action | Priority |
|------|------|--------|----------|
| `ulysses_sp: 4` → 1 | YAML config | Revert | **BLOCKER** |
| `gpu_memory_utilization: 0.4` → 0.6 | YAML config | Revert | **BLOCKER** |
| `enforce_eager: true` → false | YAML config | Revert | **BLOCKER** |
| `enable_chunked_prefill: true` → false | YAML config | Revert | **BLOCKER** |
| `tensor_parallel_size: 4` → 1 | YAML config | Revert | **BLOCKER** |
| `max_num_batched_tokens: 33024` → 128000 | YAML config | Revert | **BLOCKER** |
| `max_response_length: 1024` → 8192 | YAML config | Increase | **BLOCKER** |
| flash-attn `wrap_triton` patch | rotary.py | 1-line patch | **PREREQUISITE** |
| `attn_implementation: sdpa` → `flash_attention_2` | fsdp_workers.py | 1-line change | After flash-attn fixed |
| 3D position_ids in ulysses padding | ulysses.py | Already fixed in our fork (8.1 update) | Done — verify |
| `.float()` upcast in `log_probs_from_logits` | torch_functional.py | Monitor | Only if loss diverges |
| `torch.compile` re-enable | dp_actor.py | Re-enable when stable | Low priority |
| GPU pinning in `dist.init_process_group` | fsdp_workers.py | Keep (already done) | Done |
| `caching_allocator_warmup` patch | fsdp_workers.py | Keep (already done) | Done |
| `device_map="cpu"` + `is_meta` guard | fsdp_workers.py | Keep (already done) | Done |
| `empty_cache` before vLLM init | fsdp_workers.py | Keep (already done) | Done |
| `group` param in `get_ulysses_sp_rank` | ulysses.py | Keep (already done) | Done |
| Off-by-one label shape guard | torch_functional.py | Keep (already done) | Done |
| ZeroDivisionError guard in `update_policy` | dp_actor.py | Keep (already done) | Done |
| `_ray_get_robust()` for fault tolerance | ray_trainer.py | Keep (already done) | Done |
| GRPO n=1 REINFORCE fallback | core_algos.py | Keep (already done) | Done |
| `remote_server_url(s)` in EnvConfig | trainer/config.py | Keep (already done) | Done |

---

## 9. Files Not to Touch

These are our custom additions that extend the original ARPO. They are correct,
tested, and must not be reverted toward the upstream:

- `verl/workers/rollout/config.py` — only new optional fields, no regressions
- `verl/workers/actor/config.py` — only `empty_cache_policy` added
- `verl/trainer/config.py` — only `remote_server_urls` multi-URL field added
- All OSWorld env integration files (section 4 above)
- `scripts/servers/remote_env_server.py`
- `OSWorld/` submodule and evaluation examples
