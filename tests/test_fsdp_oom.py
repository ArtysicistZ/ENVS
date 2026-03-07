"""
Accurate FSDP OOM test using the REAL dp_actor._forward_micro_batch + PPO loss.
Exactly replicates what update_policy() does per micro-batch.

Config being tested:
  max_prompt_length=56000, max_response_length=6144
  gpu_memory_utilization=0.42  -> VLLM_GB=34

Run:
  torchrun --nproc_per_node=8 test_fsdp_oom.py
"""
import os
import sys
import ctypes
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_PROMPT_LEN = 56000   # max_prompt_length in config
MAX_RESP_LEN   = 6144    # max_response_length in config
SEQLEN         = MAX_PROMPT_LEN + MAX_RESP_LEN   # 62144 — actual input_ids length in training
VLLM_GB        = 34      # floor(0.42 x 80)
MODEL_PATH     = "ByteDance-Seed/UI-TARS-1.5-7B"
TEMPERATURE    = 1.0


def log(msg, rank=None):
    r = dist.get_rank()
    if rank is None or r == rank:
        alloc = torch.cuda.memory_allocated() / 1024**3
        peak  = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[rank{r}] {msg}  alloc={alloc:.2f}G peak={peak:.2f}G", flush=True)


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats()

    if rank == 0:
        print("=" * 60)
        print(f"FSDP OOM test (real dp_actor code): {world_size} GPUs")
        print(f"  seqlen={SEQLEN} (prompt={MAX_PROMPT_LEN} + resp={MAX_RESP_LEN})")
        print(f"  vLLM sim={VLLM_GB} GiB/GPU (0.42 x 80)")
        print("=" * 60)

    # 1. Simulate sleeping vLLM via cudaMalloc (outside PyTorch pool, no fragmentation)
    libcudart = ctypes.CDLL("libcudart.so")
    vllm_ptr  = ctypes.c_void_p()
    ret = libcudart.cudaMalloc(ctypes.byref(vllm_ptr), ctypes.c_size_t(VLLM_GB * 1024**3))
    if ret != 0:
        raise RuntimeError(f"cudaMalloc failed: error {ret}")
    dist.barrier()
    if rank == 0:
        print(f"[rank0] vLLM placeholder: {VLLM_GB} GiB via cudaMalloc (torch_alloc=0.0G)", flush=True)

    # 2. Load model and FSDP-shard (same as fsdp_workers.py)
    from transformers import AutoModelForVision2Seq
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

    if rank == 0:
        print(f"\nLoading {MODEL_PATH}...", flush=True)

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2_5_VLDecoderLayer},
    )
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        device_id=local_rank,
    )
    model.train()

    # Optimizer states live on CPU (offload_optimizer: true in config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-2)

    dist.barrier()
    log("FSDP model + optimizer ready", rank=0)

    # 3. Build micro_batch — exact fields that dp_actor._forward_micro_batch expects
    dev = f"cuda:{local_rank}"
    input_ids      = torch.randint(0, 5000, (1, SEQLEN), device=dev)
    attention_mask = torch.ones(1, SEQLEN, dtype=torch.long, device=dev)
    position_ids   = torch.arange(SEQLEN, device=dev).unsqueeze(0)
    responses      = torch.randint(0, 5000, (1, MAX_RESP_LEN), device=dev)
    response_length = responses.size(-1)

    old_log_probs = torch.zeros(1, response_length, device=dev)
    advantages    = torch.ones(1, response_length, device=dev)
    response_mask = torch.ones(1, response_length, dtype=torch.bool, device=dev)

    dist.barrier()

    # 4. Forward — exact copy of dp_actor._forward_micro_batch (padding_free=False path)
    if rank == 0:
        print(f"\nForward (real _forward_micro_batch, logits_to_keep={response_length+1})...", flush=True)
    torch.cuda.reset_peak_memory_stats()

    try:
        from verl.utils import torch_functional as VF
        log_probs_from_logits = VF.log_probs_from_logits

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=response_length + 1,
        )
        logits = output.logits                             # [1, resp_len+1, vocab]
        logits.div_(TEMPERATURE)
        logits = logits[:, -response_length - 1 : -1, :]  # [1, resp_len, vocab]
        log_probs = log_probs_from_logits(logits, responses)  # [1, resp_len]

        dist.barrier()
        log("after forward", rank=0)

        # 5. Real PPO loss — exact same as update_policy()
        from verl.trainer import core_algos
        pg_loss, _, _, _ = core_algos.compute_policy_loss(
            old_log_probs=old_log_probs,
            log_probs=log_probs,
            advantages=advantages,
            response_mask=response_mask,
            clip_ratio_low=0.2,
            clip_ratio_high=0.3,
            clip_ratio_dual=3.0,
        )
        loss = pg_loss  # gradient_accumulation=1

        if rank == 0:
            print("\nBackward (real PPO loss)...", flush=True)
        loss.backward()
        dist.barrier()
        log("after backward", rank=0)

        # 6. Optimizer step — same as _optimizer_step()
        model.clip_grad_norm_(1.0)
        optimizer.step()
        optimizer.zero_grad()
        dist.barrier()
        log("after optimizer step", rank=0)

        peak = torch.cuda.max_memory_allocated() / 1024**3
        if rank == 0:
            print(f"\n[PASS] peak VRAM = {peak:.2f} GiB  (limit=79.25, headroom={79.25-peak:.2f} GiB)")

    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n[rank{rank}] [FAIL] OOM peak={peak:.2f} GiB: {e}", flush=True)
        libcudart.cudaFree(vllm_ptr)
        dist.destroy_process_group()
        sys.exit(1)

    libcudart.cudaFree(vllm_ptr)

    if rank == 0:
        print("\n" + "=" * 60)
        print("ALL CHECKS PASSED")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
