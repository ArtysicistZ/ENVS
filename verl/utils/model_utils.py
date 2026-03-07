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
Utilities to create common models
"""

from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn


@lru_cache
def is_rank0() -> int:
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def print_gpu_memory_usage(prefix: str = "GPU memory usage", per_rank: bool = False) -> None:
    """Report the current GPU VRAM usage with detailed breakdown.
    Set VERL_VERBOSE_GPU_MEM=1 to enable; otherwise no-op to reduce log volume.
    
    Args:
        prefix: Label for the memory report
        per_rank: If True, print memory for all ranks; if False, only rank 0
    """
    if not torch.cuda.is_available():
        return
    import os
    if os.environ.get("VERL_VERBOSE_GPU_MEM", "") != "1":
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Always print for rank 0, or print for all ranks if per_rank=True
    should_print = (rank == 0) or per_rank

    if should_print:
        device = torch.cuda.current_device()
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        allocated_mem = torch.cuda.memory_allocated(device)
        reserved_mem = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
        
        used_mem = total_mem - free_mem
        print(
            f"[Rank {rank}/{world_size}] {prefix}:\n"
            f"  Used: {used_mem / (1024**3):.2f} GB / {total_mem / (1024**3):.2f} GB ({used_mem/total_mem*100:.1f}%)\n"
            f"  Allocated: {allocated_mem / (1024**3):.2f} GB\n"
            f"  Reserved: {reserved_mem / (1024**3):.2f} GB\n"
            f"  Max Allocated: {max_allocated / (1024**3):.2f} GB\n"
            f"  Max Reserved: {max_reserved / (1024**3):.2f} GB"
        )


def _get_model_size(model: nn.Module, scale: str = "auto") -> Tuple[float, str]:
    """Compute the model size."""
    n_params = sum(p.numel() for p in model.parameters())

    if scale == "auto":
        if n_params > 1e9:
            scale = "B"
        elif n_params > 1e6:
            scale = "M"
        elif n_params > 1e3:
            scale = "K"
        else:
            scale = ""

    if scale == "B":
        n_params = n_params / 1e9
    elif scale == "M":
        n_params = n_params / 1e6
    elif scale == "K":
        n_params = n_params / 1e3
    elif scale == "":
        pass
    else:
        raise NotImplementedError(f"Unknown scale {scale}.")

    return n_params, scale


def print_model_size(model: nn.Module, name: Optional[str] = None) -> None:
    """Print the model size."""
    if is_rank0():
        n_params, scale = _get_model_size(model, scale="auto")
        if name is None:
            name = model.__class__.__name__

        print(f"{name} contains {n_params:.2f}{scale} parameters.")
