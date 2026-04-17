"""MCTS configuration dataclass."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MCTSConfig:
    """Configuration for the MCTS trajectory collection system.

    Key design: 40 VMs pre-setup per task, claimed on demand at branch points.
    The tree grows organically — VMs are allocated where the model is uncertain.

    Hard VM cap (`max_active_vms`) is the SINGLE place to control max utilization.
    """

    # ---- VMs ----
    vms_per_task: int = 80
    max_active_vms: int = 80           # HARD CAP — 1 task uses all 80 VMs
    tasks_per_batch: int = 1           # 1 task at a time, all 80 VMs

    # ---- Probing ----
    probe_temperature: float = 1.0     # Standard temperature — no garbage
    total_probe_budget: int = 32       # Hard cap on total prompts per step (~4/GPU)
    k_max: int = 32                    # Ceiling K per node (1 active → full budget)

    # ---- Branching ----
    spatial_grid_size: int = 50        # px grid for action fingerprinting
    min_cluster_size: int = 1          # singletons count as clusters
    max_branch_per_explorer: int = 5   # root: 5 branching opportunities
    child_branch_budget: int = 2       # children: 2 sub-explorations
    max_branches_per_step: int = 2     # top 2 minorities per node per step
    never_branch_after: int = 15       # effectively disabled — probe at all steps
    late_step_threshold: int = 10      # deferred branching (require 2 consecutive)

    # ---- Replay ----
    replay_pause_sec: float = 1.0

    # ---- Pruning ----
    stuck_repeat_limit: int = 3
    stuck_wait_limit: int = 2

    # ---- Phase 2 (Hindsight) ----
    enable_phase2: bool = True
    max_hindsight_per_success: int = 3

    # ---- Pipeline ----
    max_steps: int = 15

    # ---- GPU / Inference ----
    model_path: str = "ByteDance-Seed/UI-TARS-1.5-7B"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 32768
    limit_images: int = 3
    generation_max_tokens: int = 512

    # ---- Env servers ----
    remote_server_urls: List[str] = field(default_factory=list)

    # ---- Task file ----
    task_file: str = "OSWorld/evaluation_examples/test_all_300tasks_noproxy_softreset_clean.json"

    # ---- Noise (v3: per-branch random noise) ----
    enable_noise: bool = False
    noise_mode: str = "runtime_library"
    noise_branch_probability: float = 0.8   # 80% noisy, 20% clean control
    noise_min_fire_step: int = 3            # no noise in first N steps
    noise_min_task_buffer: int = 4          # min steps after recovery for task completion
    noise_sr_file: str = ""                 # collection_results.json with per-task clean SR
    # Fire count per clean SR:
    #   SR < 0.15  → 0 fires (protect very hard tasks)
    #   SR 0.15–0.60 → 1 fire (any element, feasibility-constrained placement)
    #   SR > 0.60  → 2 fires (sequential, non-overlapping recovery windows)

    # ---- Output ----
    output_dir: str = "checkpoints/mcts_trajectories"
    save_full_tree: bool = False      # v2: save full tree JSON (all nodes + Q-values)
