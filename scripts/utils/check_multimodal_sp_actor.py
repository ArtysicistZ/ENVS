"""
Real 8-GPU validation for the current multimodal + Ulysses SP actor path.

This exercises the same failure point as `actor_rollout_compute_log_probs()`:
- Qwen2.5-VL / UI-TARS model
- FSDP sharding
- frozen vision tower + use_orig_params=True
- padding_free=True
- ulysses_sequence_parallel_size=4 within an 8-rank launch (2 DP groups x 4 SP ranks)
- multimodal `pixel_values` + `image_grid_thw`
- real forward + backward through `DataParallelPPOActor._forward_micro_batch`

Run:
  torchrun --nproc_per_node=8 scripts/check_multimodal_sp_actor.py
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoConfig, AutoModelForVision2Seq

from verl.models.monkey_patch import apply_ulysses_patch
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.utils.ulysses import set_ulysses_sequence_parallel_group
from verl.workers.actor.dp_actor import DataParallelPPOActor


MODEL_PATH = "ByteDance-Seed/UI-TARS-1.5-7B"
SP_SIZE = 4
SEQ_LEN = 32


def log(msg: str, rank: int = 0):
    if dist.get_rank() == rank:
        alloc = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[rank{dist.get_rank()}] {msg} alloc={alloc:.2f}G peak={peak:.2f}G", flush=True)


def build_micro_batch(model_config):
    image_token_id = model_config.image_token_id
    patch_dim = (
        model_config.vision_config.in_channels
        * model_config.vision_config.temporal_patch_size
        * model_config.vision_config.patch_size
        * model_config.vision_config.patch_size
    )

    # One image with grid [1, 4, 4] produces 16 vision patches and 4 merged image tokens.
    input_ids = torch.arange(1, SEQ_LEN + 1, dtype=torch.long, device="cuda").unsqueeze(0)
    input_ids[:, 8:12] = image_token_id
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long, device="cuda").repeat(3, 1).unsqueeze(0)
    responses = input_ids[:, 1:].contiguous()
    pixel_values = torch.zeros(16, patch_dim, dtype=torch.float32, device="cuda")
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device="cuda")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
        "multi_modal_inputs": [{"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}],
    }


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size % SP_SIZE == 0, f"Expected world_size to be divisible by SP_SIZE={SP_SIZE}, got {world_size}"

    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats()
    sp_group_ranks = list(range((rank // SP_SIZE) * SP_SIZE, ((rank // SP_SIZE) + 1) * SP_SIZE))
    sp_group = dist.new_group(ranks=sp_group_ranks, backend="nccl")
    set_ulysses_sequence_parallel_group(sp_group)

    model_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    apply_ulysses_patch(model_config.model_type)

    if rank == 0:
        print("=" * 60, flush=True)
        print("Multimodal SP actor check", flush=True)
        print(f"  model={MODEL_PATH}", flush=True)
        print(f"  world_size={world_size}  ulysses_sp={SP_SIZE}  dp_groups={world_size // SP_SIZE}", flush=True)
        print("=" * 60, flush=True)

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)

    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=get_fsdp_wrap_policy(model),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=torch.cuda.current_device(),
        forward_prefetch=False,
        use_orig_params=True,
    )
    fsdp_model.train()

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-6)
    actor = DataParallelPPOActor(
        config=SimpleNamespace(
            padding_free=True,
            ulysses_sequence_parallel_size=SP_SIZE,
            max_grad_norm=1.0,
        ),
        actor_module=fsdp_model,
        actor_optimizer=optimizer,
    )

    micro_batch = build_micro_batch(model_config)
    dist.barrier()
    log("starting multimodal SP forward/backward", rank=0)

    log_probs = actor._forward_micro_batch(micro_batch, temperature=1.0)
    loss = -log_probs.mean()
    loss.backward()
    fsdp_model.clip_grad_norm_(1.0)
    optimizer.step()
    optimizer.zero_grad()

    dist.barrier()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    log(f"PASS loss={loss.item():.6f} log_probs_shape={tuple(log_probs.shape)} peak={peak:.2f}G", rank=0)
    dist.destroy_process_group(sp_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
