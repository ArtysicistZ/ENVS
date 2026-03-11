# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import os
from typing import Literal, Optional, Union
from contextlib import nullcontext

import numpy as np
import psutil
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import AnyPrecisionAdamW, get_constant_schedule_with_warmup
from .actor import DataParallelPPOActor
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, RefConfig, WorkerConfig
from .critic import DataParallelPPOCritic
from .rollout import vLLMRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


def _validate_head_parallel_size(model_config: AutoConfig, parallel_size: int, parallelism_name: str) -> None:
    if parallel_size <= 1:
        return

    num_attention_heads = getattr(model_config, "num_attention_heads", None)
    num_key_value_heads = getattr(model_config, "num_key_value_heads", num_attention_heads)
    if num_attention_heads is None:
        return

    incompatible = []
    if num_attention_heads % parallel_size != 0:
        incompatible.append(f"num_attention_heads={num_attention_heads}")
    if num_key_value_heads is not None and num_key_value_heads % parallel_size != 0:
        incompatible.append(f"num_key_value_heads={num_key_value_heads}")

    if incompatible:
        raise ValueError(
            f"{parallelism_name}={parallel_size} is incompatible with model_type={model_config.model_type}: "
            f"{', '.join(incompatible)}. The current Ulysses/vLLM sharding path requires "
            f"{parallelism_name} to divide both head counts."
        )


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role

        if not dist.is_initialized():
            # Pin each rank to its GPU so NCCL doesn't guess (avoids hang/OOM on heterogeneous mapping)
            local_rank = int(
                os.environ.get("LOCAL_RANK", os.environ.get("RAY_LOCAL_RANK", os.environ.get("RANK", "0"))))
            # PyTorch 2.x init_process_group expects device_id as torch.device, not int
            device_id = torch.device("cuda", local_rank)
            dist.init_process_group(backend="nccl", device_id=device_id)

        # improve numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_critic = self.role == "critic"
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._use_param_offload = False
        self._use_optimizer_offload = False
        if self._is_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_config(self.config.actor, "actor")
        elif self._is_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_config(self.config.critic, "critic")
        elif self._is_ref:  # NOTE: it seems that manual offload is slower than FSDP offload
            self._use_param_offload = self.config.ref.offload.offload_params
            self._init_config(self.config.ref, "ref")

    def _init_config(
        self, config: Union[ActorConfig, CriticConfig, RefConfig], role: Literal["actor", "critic", "ref"]
    ):
        world_size = dist.get_world_size()
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:  # hsdp
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp")
            )

        if config.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(
                    world_size // config.ulysses_sequence_parallel_size,
                    config.ulysses_sequence_parallel_size,
                ),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        if not hasattr(config, "global_batch_size"):  # ref model
            return

        if self.config.rollout.n > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")

        config.global_batch_size_per_device = (
            config.global_batch_size // self.device_mesh.size() * config.ulysses_sequence_parallel_size
        )
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")

        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")

        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool = False,
        sequence_parallel_size: int = 1,
    ) -> None:
        self.tokenizer = get_tokenizer(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        self.processor = get_processor(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        self.model_config = AutoConfig.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_config.override_config,
        )

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
        except Exception:
            self.generation_config = GenerationConfig.from_model_config(self.model_config)

        self.print_rank0(f"Model config: {self.model_config}")

        _validate_head_parallel_size(
            self.model_config,
            parallel_size=sequence_parallel_size,
            parallelism_name="ulysses_sequence_parallel_size",
        )

        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor or self._is_critic else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if self._is_critic:
            auto_class = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForVision2Seq._model_mapping.keys():
            auto_class = AutoModelForVision2Seq
        else:
            auto_class = AutoModelForCausalLM

        print_gpu_memory_usage("BEFORE model loading", per_rank=True)
        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            # Avoid Transformers' caching_allocator_warmup on CUDA, which can allocate large
            # temporary buffers and trigger OOM during model load on multi-GPU smoke tests.
            try:
                from transformers import modeling_utils as _hf_modeling_utils

                if hasattr(_hf_modeling_utils, "caching_allocator_warmup"):
                    _hf_modeling_utils.caching_allocator_warmup = (  # type: ignore[assignment]
                        lambda *args, **kwargs: None
                    )
            except Exception:
                # Best-effort: if the API changes, we just skip patching and fall back to default behavior.
                pass

            # When enable_rank0_init=False, load onto CPU first to avoid OOM during checkpoint loading.
            # FSDP will handle moving to CUDA and sharding after the model is loaded.
            # When enable_rank0_init=True, rank 0 loads full model on CPU, then FSDP shards it.
            print_gpu_memory_usage("BEFORE from_pretrained", per_rank=True)
            model = auto_class.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",  # flash-attn: O(n) memory vs O(n²) for SDPA at long context
                device_map="cpu",  # Always load to CPU; FSDP will move to CUDA and shard
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = auto_class.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",  # flash-attn: O(n) memory vs O(n²) for SDPA at long context
                    trust_remote_code=model_config.trust_remote_code,
                )

        assert isinstance(model, PreTrainedModel)  # lint
        print_gpu_memory_usage("AFTER from_pretrained, BEFORE tie_weights", per_rank=True)
        model.tie_weights()  # avoid hanging
        print_gpu_memory_usage("AFTER tie_weights, BEFORE to(device='cpu')", per_rank=True)
        # Keep model on CPU and only change dtype; FSDP will move to CUDA and shard.
        # Non-rank-0 workers with enable_rank0_init use meta tensors and must not call .to().
        is_meta = any(p.is_meta for p in model.parameters())
        if not is_meta:
            model = model.to(device="cpu", dtype=torch_dtype)
        print_gpu_memory_usage("AFTER to(device='cpu')", per_rank=True)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if not (self._is_actor or self._is_critic):
            model.requires_grad_(False)

        if model_config.freeze_vision_tower:
            if hasattr(model, "visual"):
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")

        # Sync all ranks after loading model to CPU (no CUDA work immediately before
        # barrier to avoid extra allocations; ref/actor_rollout use same GPUs so ref
        # should use enable_cpu_offload in smoke configs to leave headroom).
        dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init and barrier", per_rank=True)
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")
        print_gpu_memory_usage("BEFORE FSDP wrapping", per_rank=True)

        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        self.fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init", per_rank=True)

        if self._is_actor or self._is_critic:
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    self.fsdp_module.parameters(),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    self.fsdp_module.parameters(),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
            print_gpu_memory_usage("After optimizer init")
        else:
            self.optimizer, self.lr_scheduler = None, None
        

    def _build_rollout(self) -> None:
        tp_size = self.config.rollout.tensor_parallel_size
        _validate_head_parallel_size(
            self.model_config,
            parallel_size=tp_size,
            parallelism_name="rollout.tensor_parallel_size",
        )
        dp_size = self.world_size // tp_size
        assert self.world_size % tp_size == 0, (
            f"rollout world size: {self.world_size} is not divisible by tp size: {tp_size}"
        )
        print_gpu_memory_usage("BEFORE vLLM rollout init", per_rank=True)
        # Clear CUDA cache before vLLM init to free memory for KV cache allocation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print_gpu_memory_usage("AFTER empty_cache, BEFORE vLLM init", per_rank=True)
        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
        )
        print_gpu_memory_usage("After vllm init")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._is_critic:
            model_config = self.config.critic.model
            fsdp_config = self.config.critic.fsdp
            optim_config = self.config.critic.optim
            padding_free = self.config.critic.padding_free
            sequence_parallel_size = self.config.critic.ulysses_sequence_parallel_size
            role = "critic"
        elif self._is_actor:
            model_config = self.config.actor.model
            fsdp_config = self.config.actor.fsdp
            optim_config = self.config.actor.optim
            padding_free = self.config.actor.padding_free
            sequence_parallel_size = self.config.actor.ulysses_sequence_parallel_size
            role = "actor"
        elif self._is_ref:
            model_config = self.config.actor.model
            fsdp_config = self.config.ref.fsdp
            optim_config = None
            padding_free = self.config.ref.padding_free
            sequence_parallel_size = self.config.ref.ulysses_sequence_parallel_size
            role = "ref"
        else:
            raise ValueError(f"Unknown role {self.role}.")

        if self._is_actor or self._is_critic or self._is_ref:
            self._build_model_optimizer(
                model_config=model_config,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                padding_free=padding_free,
                sequence_parallel_size=sequence_parallel_size,
            )
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")

        if self._is_actor:
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._is_critic:
            self.critic = DataParallelPPOCritic(
                config=self.config.critic,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._is_rollout:
            self._build_rollout()
            # only for actor_rollout_ref

        if self._is_ref:
            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.fsdp_module,
            )

        if self._is_actor or self._is_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str):
        assert self._is_actor or self._is_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(path)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:  # avoid OOM in resuming
            offload_fsdp_optimizer(self.optimizer)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor
        # Reclaim all cached CUDA blocks before actor update so we start with maximum headroom.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if self.rank == 0:
            _alloc = torch.cuda.memory_allocated() / 1024**3
            _resv = torch.cuda.memory_reserved() / 1024**3
            _free = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"[MEM] update_actor entry (after gc+empty_cache, before data.to(cuda)): allocated={_alloc:.2f} GiB, reserved={_resv:.2f} GiB, cuda_free={_free:.2f} GiB")
        data = data.to(torch.cuda.current_device())
        if self.rank == 0:
            _alloc = torch.cuda.memory_allocated() / 1024**3
            print(f"[MEM] after data.to(cuda): allocated={_alloc:.2f} GiB, data_len={len(data)}")

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            # Each rank reports its own peak — after DP_COMPUTE_PROTO concat,
            # the trainer gets an array of all per-GPU peaks.
            metrics["perf/gpu_peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)

            # Print per-GPU summary from rank 0 via all_gather
            local_peak = torch.tensor(
                [torch.cuda.max_memory_allocated() / (1024**3)],
                device=torch.cuda.current_device(),
            )
            gathered = [torch.zeros_like(local_peak) for _ in range(self.world_size)]
            torch.distributed.all_gather(gathered, local_peak)
            if self.rank == 0:
                per_gpu_peaks = [t.item() for t in gathered]
                print(
                    f"[VRAM] per-GPU peak (GiB): "
                    + " | ".join(f"GPU{i}={v:.2f}" for i, v in enumerate(per_gpu_peaks))
                    + f"  max={max(per_gpu_peaks):.2f}"
                )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info.
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        # Free training data and reclaim cached CUDA blocks after actor update
        # so the next rollout (vLLM wake_up) starts with maximum free memory.
        del data
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if self.rank == 0:
            _alloc = torch.cuda.memory_allocated() / 1024**3
            _free = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"[MEM] update_actor exit (after gc+empty_cache): allocated={_alloc:.2f} GiB, cuda_free={_free:.2f} GiB")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_generate_sequences(self):
        assert self._is_rollout
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        
        self.rollout_sharding_manager.__enter__()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_generate_sequences(self):
        assert self._is_rollout
        self.rollout_sharding_manager.__exit__(None, None, None)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        # if use_rollout_sharding_manager:
        #     context_manager = self.rollout_sharding_manager
        # else:
        #     context_manager = nullcontext()

        # if use_rollout_sharding_manager:
        if False:
            if self._use_param_offload:
                load_fsdp_model(self.fsdp_module)

            with self.rollout_sharding_manager:
                # after parameters sync with rollout, offload actor model to CPU
                if self._use_param_offload:
                    offload_fsdp_model(self.fsdp_module)

                if self._use_optimizer_offload:
                    offload_fsdp_optimizer(optimizer=self.optimizer)

                prompts = self.rollout_sharding_manager.preprocess_data(prompts)
                output = self.rollout.generate_sequences(prompts=prompts)
                output = self.rollout_sharding_manager.postprocess_data(output)
        else:
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._is_actor
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output}, meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        # Free batch data and force PyTorch to release cached CUDA blocks so that
        # subsequent update_actor has maximum available memory.
        del data
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        assert self._is_ref
        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._is_critic
        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info.
            output = DataProto(
                non_tensor_batch={
                    metric: np.array([value] if np.isscalar(value) else value) for metric, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output
