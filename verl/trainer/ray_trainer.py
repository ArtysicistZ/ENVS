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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import re
import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import copy
import numpy as np
import random
import ray
import torch
from codetiming import Timer
from tqdm import tqdm  # standard tqdm to avoid segfault in ray.experimental.tqdm_ray teardown
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from collections import defaultdict

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.dataset import collate_fn as collate_fn_raw
from ..utils.osworld import OSWorldDataset, OSWorldTaskConfigDataset, OSWorldGRPODataset, collate_fn, collate_fn_dataproto, collate_fn_fake, GRPODatasetProcessor
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str
from ..utils.rollout_logprobs import get_old_log_probs_with_fallback
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics

from .gui_agent import EnvWorker, RemoteEnvWorker, parse_action_to_structure_output
from .replay_buffer import ReplayBuffer
import datetime

from collections import defaultdict
from qwen_vl_utils import process_vision_info

import time
from concurrent.futures import ThreadPoolExecutor


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            per_node = ", ".join(f"{node}: {g} GPU(s)" for node, g in node_available_gpus.items())
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}. "
                f"Per-node (Ray view): {per_node}. "
                f"Fix: start Ray with GPUs (e.g. ray start --head --num-gpus=8) or unset RAY_ADDRESS to start a new cluster that auto-detects GPUs."
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    if "ref_log_probs" in data.batch.keys():
        kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
        kld = kld * response_mask  # (batch_size, response_length)
    else:
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last


def _ray_get_robust(futures: List, timeout: float, fallback_fn=None, label: str = "ray.get") -> List:
    """ray.get() with timeout and per-future crash isolation for 64-VM robustness.

    Args:
        futures: list of Ray ObjectRefs (one per worker)
        timeout: total wall-clock seconds to wait for all futures
        fallback_fn: callable(idx) -> value to use for timed-out/crashed futures
        label: log label for diagnostics
    Returns:
        list of results, same length as futures; failed entries use fallback_fn(idx)
    """
    if fallback_fn is None:
        fallback_fn = lambda idx: None  # noqa: E731

    try:
        return ray.get(futures, timeout=timeout)
    except ray.exceptions.GetTimeoutError:
        print(f"[{label}] ray.get timed out after {timeout:.0f}s; collecting partial results from {len(futures)} futures")
    except Exception as exc:
        print(f"[{label}] ray.get error: {exc!r}; collecting partial results from {len(futures)} futures")

    # Collect whatever completed; fill the rest with fallback
    results = []
    for idx, f in enumerate(futures):
        try:
            results.append(ray.get(f, timeout=0))
        except ray.exceptions.GetTimeoutError:
            print(f"[{label}] future[{idx}] timed out; using fallback")
            results.append(fallback_fn(idx))
        except Exception as exc:
            print(f"[{label}] future[{idx}] error: {exc!r}; using fallback")
            results.append(fallback_fn(idx))
    return results


def normalize_remote_server_urls(remote_server_url: Optional[str], remote_server_urls: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    if remote_server_url:
        urls.append(remote_server_url)
    if remote_server_urls:
        urls.extend(url for url in remote_server_urls if url)

    deduped: List[str] = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def build_rollout_jobs(task_configs: List[dict], rollout_n: int) -> List[dict]:
    return [task_config for task_config in task_configs for _ in range(rollout_n)]


def chunk_rollout_jobs(rollout_jobs: List[dict], num_envs: int) -> List[List[dict]]:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    return [rollout_jobs[i : i + num_envs] for i in range(0, len(rollout_jobs), num_envs)]


def merge_timing_raw(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0.0) + value


def reshape_rollout_metric(values: torch.Tensor, rollout_n: int) -> torch.Tensor:
    if rollout_n > 0 and values.numel() >= rollout_n and values.numel() % rollout_n == 0:
        return values.reshape(-1, rollout_n)
    return values.reshape(-1, 1)





class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
        val_reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        
        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        remote_urls = normalize_remote_server_urls(
            getattr(config.env, "remote_server_url", None),
            getattr(config.env, "remote_server_urls", None),
        )
        use_remote_env = bool(remote_urls)
        if not use_remote_env and config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if not use_remote_env and config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")
        
        print(config)

        self.task_config_single = None
        self.fake_dataset = None
        self._create_dataloader()
        self._create_envs()
        self._load_replay_data()
        self._init_noise_state()

    def _init_noise_state(self):
        """Initialize opt-in noise curriculum (see docs/noise_generating/RESEARCH_FRAMING.md).

        Safe to call even when enable_noise=False — sets self.noise_curriculum=None
        and self._noise_sr_by_task={} so downstream code can check a single flag.
        """
        self.noise_curriculum = None
        self._noise_sr_by_task: Dict[str, float] = {}
        algo = self.config.algorithm
        if not getattr(algo, "enable_noise", False):
            return

        # Lazy import so non-noise runs never load the curriculum module.
        from verl.workers.noise_curriculum import NoiseCurriculum

        self.noise_curriculum = NoiseCurriculum(
            target_success_rate=algo.noise_target_sr,
            lr=algo.noise_lr,
            p_min=algo.noise_probability_min,
            p_max=algo.noise_probability_max,
            tau_up=algo.noise_tau_up,
            tau_down=algo.noise_tau_down,
            k_up=algo.noise_k_up,
            ema_alpha=algo.noise_ema_alpha,
            recovery_ema_alpha=algo.noise_recovery_ema_alpha,
            initial_tier=algo.noise_initial_tier,
            initial_p=algo.noise_probability,
            tier_min=algo.noise_tier_min,
            tier_max=algo.noise_tier_max,
        )

        # Load per-task clean/MCTS success rates if available. We deliberately
        # do NOT seed tiers here from success rate alone; the actual initial
        # tier is computed later in `_annotate_task_configs_with_noise()` from a
        # broader static prior (success rate + horizon/domain/UI fragility).
        mcts_path = os.path.join(
            os.getcwd(), "checkpoints", "mcts_trajectories_v2",
            "combined_all", "collection_results.json",
        )
        if os.path.exists(mcts_path):
            try:
                import json as _json
                with open(mcts_path, "r", encoding="utf-8") as _f:
                    _data = _json.load(_f)
                _results = _data.get("results") if isinstance(_data, dict) else None
                if isinstance(_results, list):
                    n_loaded = 0
                    for entry in _results:
                        tid = entry.get("task_id")
                        sr = float(entry.get("success_rate", 0.0))
                        if not tid:
                            continue
                        self._noise_sr_by_task[tid] = sr
                        n_loaded += 1
                    print(f"[noise] Loaded success-rate priors for {n_loaded} tasks from MCTS SR file")
            except Exception as _e:
                print(f"[noise] Failed to seed from MCTS SR file: {_e}")
        else:
            print(f"[noise] MCTS SR file not found ({mcts_path}); curriculum starts at initial_tier={algo.noise_initial_tier} for all tasks")

    def _annotate_task_configs_with_noise(self, task_configs: List[dict], is_val: bool = False) -> None:
        """Decorate each task_config dict in-place with noise_* fields that
        `desktop_env._build_noise_meta()` will read. No-op when noise is
        disabled. Tasks that appear multiple times (e.g. rollout.n copies) are
        annotated consistently from the single curriculum state.

        Also stamps `noise_step_seed = self.global_step` so all `n` rollouts of
        the same `(task_id, training_step)` see an identical noise sample
        (Common Random Numbers / PAIRED-style fixed-env-instance baseline).
        Different training steps draw different schedules — diversity preserved
        across the curriculum, locked within each GRPO group.
        """
        algo = self.config.algorithm
        _eval_flag = bool(getattr(algo, "noise_validate_with_noise", False))
        print(f"[noise-eval][DEBUG] _annotate_task_configs_with_noise entry: "
              f"is_val={is_val} validate_flag={_eval_flag} n_tasks={len(task_configs)}", flush=True)
        if is_val and _eval_flag:
            # Deterministic eval path: build the noise schedule client-side and
            # ship it pre-assembled as task_config["noise_meta"]. The remote env
            # server's _build_noise_meta() returns it verbatim (bypasses the
            # server's own sampler, which doesn't know about fires_for_task_eval).
            #
            # Schedule shape:
            #   - fire count: fires_for_task_eval(task_id)  -> 0 (20%) or 1 (80%)
            #   - noise catalog: held-out (24 templates never seen in training)
            #   - RNG seed: md5(f"{task_id}|0")  -> identical across every run
            import hashlib
            from OSWorld.evaluation_examples.noise_generation.runtime_sampler import (
                RuntimeNoiseSampler,
            )
            max_steps = int(getattr(self.config.env, "max_steps", 15))
            fire_count = 0
            noise_count = 0
            for tc in task_configs:
                if not isinstance(tc, dict):
                    continue
                tid = tc.get("id") or tc.get("task_id")
                if not tid:
                    continue
                rng_seed = int(hashlib.md5(f"{tid}|0".encode("utf-8")).hexdigest()[:8], 16)
                sampler = RuntimeNoiseSampler(rng_seed=rng_seed)
                schedule = sampler.sample_fire_schedule(
                    task_json=tc, sr=0.0, max_steps=max_steps,
                    use_heldout=True, is_eval=True,
                )
                if schedule:
                    total_cost = sum(int(e.get("recovery_cost", 0)) for e in schedule)
                    tc["enable_noise"] = True
                    tc["noise_mode"] = "runtime_library"
                    tc["noise_meta"] = {
                        "trigger_mode": "deterministic_schedule_v4_eval",
                        "success_rate_used": 0.0,
                        "observed_sr_used": 0.0,
                        "fires_count": len(schedule),
                        "fire_steps": [int(e["fire_step"]) for e in schedule],
                        "total_recovery_cost": total_cost,
                        "elements": schedule,
                    }
                    tc["noise_max_steps"] = max_steps
                    noise_count += 1
                    fire_count += len(schedule)
                else:
                    tc["enable_noise"] = False
            print(
                f"[noise-eval] stamped {noise_count} noisy / "
                f"{len(task_configs) - noise_count} clean tasks, "
                f"{fire_count} total fires (heldout catalog, seed=0)"
            )
            return
        if self.noise_curriculum is None:
            return
        if is_val:
            # Run validation with zero noise to get clean success-rate signal.
            return
        from OSWorld.evaluation_examples.noise_generation.difficulty import compute_static_noise_seed
        algo = self.config.algorithm
        for tc in task_configs:
            if not isinstance(tc, dict):
                continue
            tid = tc.get("id") or tc.get("task_id")
            if not tid:
                continue
            success_rate = float(self._noise_sr_by_task.get(tid, 0.0))
            # When MCTS SR is missing for this task, fall back to the configured
            # `noise_initial_tier` as the explicit tier — otherwise the seed
            # collapses to `tier=very_hard=0` (cost cap 0 → sampler returns []
            # → `_build_noise_meta` returns None → noise NEVER FIRES).
            explicit_tier = tc.get("noise_tier_override")
            if explicit_tier is None and tid not in self._noise_sr_by_task:
                explicit_tier = int(algo.noise_initial_tier)
            static_seed = compute_static_noise_seed(
                success_rate=success_rate,
                task_json=tc,
                explicit_tier=explicit_tier,
            )
            static_seed_tier = int(static_seed["tier"])
            static_seed_p = max(
                float(algo.noise_probability_min),
                min(
                    float(algo.noise_probability_max),
                    float(algo.noise_probability) * (0.50 + 0.50 * (static_seed_tier / max(1, int(algo.noise_tier_max)))),
                ),
            )
            if algo.noise_use_curriculum:
                if not self.noise_curriculum.has_task(tid):
                    # Seed curriculum state with all three priors at once so
                    # `get_sr` returns the MCTS SR (not 0.0) before any
                    # rollouts have been observed.
                    self.noise_curriculum.seed_tier(
                        tid, static_seed_tier, static_seed_p,
                        initial_sr=success_rate,
                    )
                p = float(self.noise_curriculum.get_p(tid))
                tier = int(self.noise_curriculum.get_tier(tid))
                # v4: live EMA SR drives the per-rollout noise budget.
                observed_sr = float(self.noise_curriculum.get_sr(tid))
            else:
                p = static_seed_p
                tier = static_seed_tier
                observed_sr = success_rate
            tc["enable_noise"] = True
            tc["noise_mode"] = algo.noise_mode if algo.noise_mode != "none" else "runtime_library"
            tc["noise_probability"] = p   # v4: vestigial; scheduler ignores it
            tc["noise_tier"] = tier       # v4: tier still bounds eligible pool
            tc["noise_success_rate"] = success_rate
            tc["noise_observed_sr"] = observed_sr  # v4: live EMA SR → fires count
            tc["noise_use_heldout"] = bool(algo.noise_use_heldout or is_val)
            # CRN seed: same (task_id, training_step) across all n rollouts of
            # this group → identical noise sample → clean GRPO advantage.
            # Reads `self.global_step` (incremented in fit() before this call).
            tc["noise_step_seed"] = int(getattr(self, "global_step", 0))
            tc["noise_static_seed_tier"] = static_seed_tier
            tc["noise_static_seed_probability"] = static_seed_p
            tc["noise_horizon_proxy"] = float(static_seed.get("horizon_proxy", 0.0))
            tc["noise_domain_fragility"] = float(static_seed.get("domain_fragility", 0.0))
            tc["noise_ui_fragility"] = float(static_seed.get("ui_fragility", 0.0))
            tc["noise_fragility_penalty"] = float(static_seed.get("fragility_penalty", 0.0))

    def _update_noise_curriculum(self, task_configs: List[dict], eval_results: List[float], process_results: Optional[List[dict]] = None) -> None:
        """Update per-task curriculum state from this step's observed success.
        Aggregates duplicate task_ids (rollout.n copies) before updating."""
        if self.noise_curriculum is None:
            return
        from collections import defaultdict as _dd
        bucket: Dict[str, List[float]] = _dd(list)
        burden_bucket: Dict[str, List[float]] = _dd(list)
        process_results = process_results or [None] * len(task_configs)
        for tc, er, pr in zip(task_configs, eval_results, process_results):
            if tc is None:
                continue
            tid = tc.get("id") or tc.get("task_id")
            if not tid:
                continue
            bucket[tid].append(float(er) if er is not None else 0.0)
            step_meta_list = []
            if isinstance(pr, dict):
                step_meta_list = pr.get("rollout_step_metadata") or []
            total_recovery = 0.0
            total_interruptions = 0.0
            total_occlusion = 0.0
            total_focus_loss = 0.0
            for sm in step_meta_list:
                if not isinstance(sm, dict):
                    continue
                nb = sm.get("noise_burden")
                if not isinstance(nb, dict):
                    continue
                total_recovery += float(nb.get("step_recovery_cost", 0.0))
                total_interruptions += float(nb.get("events_fired_this_step", 0.0))
                total_occlusion += float(nb.get("occlusion_events", 0.0))
                total_focus_loss += float(nb.get("focus_loss_events", 0.0))
            realized_burden = (
                0.50 * total_recovery
                + 0.20 * total_interruptions
                + 0.20 * total_occlusion
                + 0.10 * total_focus_loss
            )
            burden_bucket[tid].append(realized_burden)
        for tid, vals in bucket.items():
            if not vals:
                continue
            sr = sum(1 for v in vals if v > 0.5) / len(vals)
            burden_vals = burden_bucket.get(tid, [])
            mean_burden = sum(burden_vals) / len(burden_vals) if burden_vals else 0.0
            self.noise_curriculum.update(tid, sr, mean_burden)

    def _collect_noise_metrics(self) -> Dict[str, float]:
        if self.noise_curriculum is None:
            return {}
        snapshot = self.noise_curriculum.snapshot()
        if not snapshot:
            return {}
        ps = [float(v.get("p_noise", 0.0)) for v in snapshot.values()]
        tiers = [float(v.get("tier", 0.0)) for v in snapshot.values()]
        sr_emas = [float(v.get("sr_ema", 0.0)) for v in snapshot.values()]
        recovery_emas = [float(v.get("recovery_ema", 0.0)) for v in snapshot.values()]
        return {
            "noise/p_mean": sum(ps) / len(ps),
            "noise/tier_mean": sum(tiers) / len(tiers),
            "noise/sr_ema_mean": sum(sr_emas) / len(sr_emas),
            "noise/recovery_ema_mean": sum(recovery_emas) / len(recovery_emas),
        }
    
    def _load_replay_data(self):
        self.replay = ReplayBuffer(None, 8)
        replay_path = self.config.trainer.replay_data_path
        if replay_path and os.path.exists(replay_path):
            import torch as _torch
            saved = _torch.load(replay_path, map_location="cpu", weights_only=False)
            count = 0
            for item in saved:
                task_id = item["task_id"]
                train_dict = item["train_dict"]
                eval_result = item["eval_result"]
                # Collate single item into DataProto
                batch = collate_fn_dataproto([train_dict], pad_token_id=getattr(self.tokenizer, "pad_token_id", 0))
                batch_proto = DataProto.from_single_dict(batch)
                batch_proto.batch["eval_results"] = torch.tensor([eval_result], dtype=torch.float32)
                self.replay.update_replay_buffer({"task_id": task_id}, batch_proto[0], eval_result)
                count += 1
            print(f"[replay] Loaded {count} trajectories for {len(self.replay.pos_dataset)} tasks from {replay_path}")



    def _create_envs(self) -> None:
        """
        Create env workers and data-processor workers, 
        and pin each EnvWorker to a different node (round-robin).
        """
        print('Start to create env_worker for OSWorld Environment')
        max_steps = self.config.env.max_steps
        num_envs = self.config.env.num_envs
        remote_urls = normalize_remote_server_urls(
            getattr(self.config.env, "remote_server_url", None),
            getattr(self.config.env, "remote_server_urls", None),
        )

        if remote_urls:
            # Distribute workers across servers proportionally if env_counts given,
            # otherwise round-robin.
            env_counts = getattr(self.config.env, "remote_server_env_counts", None)
            if env_counts and len(env_counts) == len(remote_urls):
                assert sum(env_counts) >= num_envs, (
                    f"Sum of remote_server_env_counts ({sum(env_counts)}) < num_envs ({num_envs})"
                )
                # Build (url, local_slot_id) assignment:
                # Workers 0..counts[0]-1 → (url[0], 0..counts[0]-1)
                # Workers counts[0]..counts[0]+counts[1]-1 → (url[1], 0..counts[1]-1)
                url_assignment = []
                slot_assignment = []
                for url, count in zip(remote_urls, env_counts):
                    url_assignment.extend([url] * count)
                    slot_assignment.extend(range(count))
            else:
                url_assignment = [remote_urls[i % len(remote_urls)] for i in range(num_envs)]
                slot_assignment = [i % len(remote_urls) for i in range(num_envs)]
                # For round-robin, count how many workers per server to compute local slot
                _rr_counters = {url: 0 for url in remote_urls}
                slot_assignment = []
                for i in range(num_envs):
                    url = url_assignment[i]
                    slot_assignment.append(_rr_counters[url])
                    _rr_counters[url] += 1

            self.env_workers = []
            for i in range(num_envs):
                w = RemoteEnvWorker.options(name=f"remote_env_worker_{i}", num_cpus=0).remote(
                    i, max_steps, self.config, url_assignment[i], slot_assignment[i]
                )
                self.env_workers.append(w)
            print(f"RemoteEnvWorker created (urls={remote_urls}, counts={env_counts}), num_envs={num_envs}, total: {len(self.env_workers)}")
        else:
            # Local Docker env workers pinned to nodes with docker resource
            all_res = ray.cluster_resources().keys()
            ip_labels = [r for r in all_res if re.match(r"^docker:\d+\.\d+\.\d+\.\d+$", r)]
            if not ip_labels:
                raise RuntimeError("没找到任何 IP 资源标签，请检查 ray start 时 --resources 参数")

            self.env_workers = []
            for i in range(num_envs):
                ip_label = ip_labels[i % len(ip_labels)]
                w = EnvWorker.options(
                    resources={ip_label: 1},
                    name=f"env_worker_{i}"
                ).remote(i, max_steps, self.config)
                self.env_workers.append(w)
            print(f'Env_worker for OSWorld Environment created!  total: {len(self.env_workers)}')

        # 3) 数据预处理器，放在 driver 或随意放一个节点上都行
        self.data_processor_workers = [
            GRPODatasetProcessor.remote(
                self.processor,
                self.tokenizer,
                max_prompt_length=self.config.data.max_prompt_length
            )
            for _ in range(len(self.env_workers))
        ] 
            
    def _create_dataloader(self) -> None:
        self.train_dataset = OSWorldTaskConfigDataset(
            data_path=self.config.data.train_files,
        )
        # data = self.train_dataset[0]
        # breakpoint()
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

        self.val_dataset = OSWorldTaskConfigDataset(
            data_path=self.config.data.val_files,
            n_repeats=self.config.data.val_n_repeats,
        )
        # Val batch size = number of envs (env_workers not created yet; use config)
        remote_urls = normalize_remote_server_urls(
            getattr(self.config.env, "remote_server_url", None),
            getattr(self.config.env, "remote_server_urls", None),
        )
        num_envs = self.config.env.num_envs
        val_batch_size = min(num_envs, len(self.val_dataset))
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")


        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = len(self.train_dataloader) * self.config.trainer.total_episodes

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        task_configs_total = []
        eval_results_total = []

        # --- Resume: load previous incremental results if they exist ---
        skip_batches = 0
        if self.config.trainer.val_only:
            _resume_path = os.path.join(self.config.trainer.save_checkpoint_path, f"eval_results_at_{self.global_step}.json")
            if os.path.exists(_resume_path) and os.path.getsize(_resume_path) > 0:
                with open(_resume_path, 'r') as _f:
                    _prev = json.load(_f)
                if _prev:
                    _completed_entries = sum(v["n_attempts"] for v in _prev.values())
                    _batch_size = self.config.env.num_envs
                    skip_batches = _completed_entries // _batch_size
                    for _tid, _v in _prev.items():
                        for _r in _v["results"]:
                            task_configs_total.append({"task_id": _tid})
                            eval_results_total.append(_r)
                    print(f"[val] Resuming: found {len(_prev)} tasks ({_completed_entries} entries) from previous run, skipping {skip_batches} batches")

        for batch_idx, batch_dict in enumerate(self.val_dataloader):
            if batch_idx < skip_batches:
                continue
            task_configs = batch_dict
            num_tasks = len(task_configs)
            assert num_tasks <= len(self.env_workers)
            # Deterministic eval-noise stamping (no-op unless
            # algorithm.noise_validate_with_noise=True). Must run BEFORE reset
            # so each env reset sees the pre-built noise_meta.
            self._annotate_task_configs_with_noise(task_configs, is_val=True)
            task_configs_total.extend(task_configs) # record task

            futures = [
                worker.reset.remote(task_config) for worker, task_config in
                zip(self.env_workers[:num_tasks], task_configs)
            ]
            reset_outputs = _ray_get_robust(futures, timeout=600, label="val_reset",
                                            fallback_fn=lambda idx: {"env_idx": idx, "obs_messages": None, "is_done": True, "format_reward": 0.0})
            print(f"[val] Batch {batch_idx}: all {num_tasks} resets completed, preparing vLLM engine...", flush=True)

            self.actor_rollout_wg.prepare_generate_sequences()
            print(f"[val] vLLM engine ready.", flush=True)
            generate_sequences_started = True
            batch_all_failed = False  # tracks if all envs failed (vllm_batch is None)

            env_outputs = reset_outputs

            try:
                for step_idx in range(self.config.env.max_steps):
                    is_done_futures = [worker.is_done.remote() for worker in self.env_workers[:num_tasks]]
                    is_done_results = _ray_get_robust(is_done_futures, timeout=30, label="val_is_done",
                                                      fallback_fn=lambda idx: True)
                    print(f"Step {step_idx} of {self.config.env.max_steps}: {is_done_results}")
                    world_size = self.actor_rollout_wg.world_size

                    vllm_batch, valid_env_idx, overlong_env_idx = self.prepare_vllm_inputs_full(env_outputs)

                    # Mark overlong-prompt envs as done so they don't break generation
                    for eidx in overlong_env_idx:
                        env_outputs = [x for x in env_outputs if x['env_idx'] != eidx]

                    if vllm_batch is None:
                        # No envs produced valid observations (e.g. all failed to start).
                        batch_all_failed = True
                        break

                    vllm_batch_pad, pad_size = pad_dataproto_to_divisor(vllm_batch, world_size)

                    gen_batch = vllm_batch_pad.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )

                    # override val config
                    gen_batch.meta_info = self.config.worker.rollout.val_override_config
                    self._apply_task_family_decoding_if_single(gen_batch, valid_env_idx, task_configs, is_val=True)

                    # predict actions
                    action_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    action_batch_output = unpad_dataproto(action_batch_output, pad_size=pad_size)

                    response_texts = self.tokenizer.batch_decode(action_batch_output.batch['responses'], skip_special_tokens=True)
                    response_texts, _, _, _ = self._retry_invalid_actions_once(
                        gen_batch, action_batch_output, response_texts, pad_size
                    )

                    cur_valid_envs = [self.env_workers[i] for i in valid_env_idx]

                    futures = [worker.step.remote(action_text) for worker, action_text in zip(cur_valid_envs, response_texts)]
                    env_outputs = _ray_get_robust(futures, timeout=300, label="val_step",
                                                  fallback_fn=lambda idx: {"env_idx": valid_env_idx[idx], "obs_messages": None, "is_done": True, "format_reward": 0.0})

                    is_all_done = all([x['is_done'] for x in env_outputs])
                    if is_all_done:
                        break
            finally:
                if generate_sequences_started:
                    self.actor_rollout_wg.finish_generate_sequences()
                    generate_sequences_started = False

            if batch_all_failed:
                # All envs failed — record zeros and skip evaluate
                eval_results_total.extend([0.0] * num_tasks)
                reward_tensor_lst.append(torch.zeros(num_tasks, 1, dtype=torch.float32))
                sample_inputs.extend([tc.get("instruction", "") for tc in task_configs])
                sample_outputs.extend([""] * num_tasks)
                sample_labels.extend(["none"] * num_tasks)
                sample_scores.extend([0.0] * num_tasks)
                continue

            futures = [worker.evaluate.remote() for worker in self.env_workers[:num_tasks]]
            eval_results = _ray_get_robust(futures, timeout=300, label="val_evaluate",
                                           fallback_fn=lambda idx: 0.0)
            eval_results_total.extend(eval_results)

            # --- Trajectory saving (if enabled) ---
            # Default: only successful episodes (eval_result > 0).
            # If trainer.save_all_trajectories=True, save failed episodes too (for diagnosis).
            if getattr(self.config.trainer, 'save_trajectories', False):
                from verl.utils.trajectory_io import TrajectoryWriter
                if not hasattr(self, '_traj_writer'):
                    _traj_path = os.path.join(self.config.trainer.save_checkpoint_path,
                                              f"trajectories_at_{self.global_step}.jsonl")
                    # On resume, delete old trajectory file to avoid duplicates
                    # (eval_results are resumed from JSON, but trajectories are not)
                    if skip_batches > 0 and os.path.exists(_traj_path):
                        os.remove(_traj_path)
                        print(f"[val] Removed stale trajectory file for clean resume: {_traj_path}")
                    self._traj_writer = TrajectoryWriter(_traj_path)
                _save_all = getattr(self.config.trainer, 'save_all_trajectories', False)
                if _save_all:
                    _fetch_indices = list(range(len(eval_results)))
                else:
                    _fetch_indices = [i for i, score in enumerate(eval_results) if score > 0]
                if _fetch_indices:
                    _limit_images = getattr(self.config.worker.rollout, 'limit_images', 8)
                    _traj_futures = [self.env_workers[i].get_compact_trajectory.remote(_limit_images)
                                     for i in _fetch_indices]
                    _trajs = _ray_get_robust(_traj_futures, timeout=120, label="val_traj",
                                             fallback_fn=lambda idx: None)
                    _saved = 0
                    for _traj in _trajs:
                        if _traj is not None and _traj.get("steps"):
                            self._traj_writer.write(_traj)
                            _saved += 1
                    _label = "all" if _save_all else "successful"
                    print(f"[val] Batch {batch_idx}: saved {_saved}/{len(_fetch_indices)} {_label} trajectories")

            # Store scores
            scores = eval_results
            reward_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

            # Only fetch history messages (which contain huge base64 screenshots) if we
            # actually need them for logging. Skip to avoid accumulating GBs of RAM in
            # the Runner process, which causes OOM on large evals (300+ tasks × n=8).
            _n_to_log = getattr(self.config.trainer, 'val_generations_to_log', 0)
            if _n_to_log > 0 and len(sample_inputs) < _n_to_log:
                history_futures = [worker.get_history_messages.remote() for worker in self.env_workers[:num_tasks]]
                history_messages = _ray_get_robust(history_futures, timeout=60, label="val_history",
                                                   fallback_fn=lambda idx: [])
                sample_inputs.extend([task_config.get('instruction', '') for task_config in task_configs])
                prompts = []
                for history_message in history_messages:
                    if not history_message or (isinstance(history_message, list) and len(history_message) == 0):
                        prompts.append("")
                    else:
                        try:
                            prompts.append(self.processor.apply_chat_template(history_message))
                        except (IndexError, KeyError, TypeError):
                            prompts.append("")
                sample_outputs.extend(prompts)
                sample_labels.extend(['none']*len(prompts))
                sample_scores.extend(scores)
                del history_messages, prompts
            else:
                sample_inputs.extend([task_config.get('instruction', '') for task_config in task_configs])
                sample_outputs.extend([''] * num_tasks)
                sample_labels.extend(['none'] * num_tasks)
                sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)

            # --- Incremental save after each batch (inference-only) ---
            if self.config.trainer.val_only:
                _inc_save_path = os.path.join(self.config.trainer.save_checkpoint_path, f"eval_results_at_{self.global_step}.json")
                os.makedirs(os.path.dirname(_inc_save_path), exist_ok=True)
                from collections import defaultdict as _dd
                _rpt = _dd(list)
                for _tc, _er in zip(task_configs_total, eval_results_total):
                    _rpt[_tc['task_id']].append(_er)
                _inc_dict = {}
                for _tid, _res in _rpt.items():
                    _ns = sum(1 for r in _res if r > 0.5)
                    _inc_dict[_tid] = {"success_rate": _ns / len(_res), "n_success": _ns, "n_attempts": len(_res), "results": _res}
                with open(_inc_save_path, 'w') as _f:
                    json.dump(_inc_dict, _f, indent=4)
                _done = len(_rpt)
                _doable = sum(1 for v in _inc_dict.values() if v["n_success"] > 0)
                print(f"[val] Batch done — {_done} tasks evaluated so far, {_doable} doable, saved to {_inc_save_path}")

        # Store eval_results — aggregate duplicate task_ids (e.g. n=8 inference)
        save_path = os.path.join(self.config.trainer.save_checkpoint_path, f"eval_results_at_{self.global_step}.json")
        from collections import defaultdict as _defaultdict
        results_per_task = _defaultdict(list)
        for task_config, eval_result in zip(task_configs_total, eval_results_total):
            task_id = task_config['task_id']
            results_per_task[task_id].append(eval_result)

        save_dict = {}
        for task_id, results in results_per_task.items():
            n_attempts = len(results)
            n_success = sum(1 for r in results if r > 0.5)
            save_dict[task_id] = {
                "success_rate": n_success / n_attempts,
                "n_success": n_success,
                "n_attempts": n_attempts,
                "results": results,
            }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=4)
        print(f"[val] Saved eval results for {len(save_dict)} tasks to {save_path}")
        doable = sum(1 for v in save_dict.values() if v["n_success"] > 0)
        print(f"[val] Doable tasks (>=1 success): {doable}/{len(save_dict)}")

        # Always close trajectory writer (even if _validate had errors earlier)
        if hasattr(self, '_traj_writer'):
            _traj_path = self._traj_writer.path
            try:
                self._traj_writer.close()
            except Exception:
                pass
            del self._traj_writer
            # Count lines to report total saved
            try:
                with open(_traj_path, 'r') as _f:
                    _n_saved = sum(1 for _ in _f)
            except Exception:
                _n_saved = "?"
            _label_close = "all (success+failed)" if getattr(self.config.trainer, 'save_all_trajectories', False) else "successful"
            print(f"[val] Trajectory file closed: {_n_saved} {_label_close} episodes saved to {_traj_path}")

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()

        # Free accumulated validation data to prevent OOM in Runner process.
        # With 300+ tasks × n=8, these lists can hold GBs of data.
        del sample_inputs, sample_outputs, sample_labels, sample_scores
        del reward_tensor_lst, task_configs_total, eval_results_total
        import gc; gc.collect()

        return {"val/reward_score": reward_score}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _looks_parseable_gui_action(self, text: str) -> bool:
        if not text or "Action:" not in text:
            return False
        try:
            screen_w, screen_h = 1920, 1080
            if getattr(self.config, "env", None) is not None and getattr(self.config.env, "screen_size", None):
                screen_w, screen_h = self.config.env.screen_size
            max_px = self.config.data.max_pixels
            min_px = self.config.data.min_pixels
            parse_action_to_structure_output(
                text,
                1000,  # matches gui_agent worker parse factor
                screen_h,
                screen_w,
                "qwen25vl",
                max_px,
                min_px,
            )
            return True
        except Exception:
            return False

    def _retry_invalid_actions_once(self, gen_batch, action_batch_output: DataProto, response_texts, pad_size: int):
        invalid_idx = [i for i, t in enumerate(response_texts) if not self._looks_parseable_gui_action(t)]
        metadata_entries = list(action_batch_output.non_tensor_batch.get("rollout_step_metadata", []))
        if len(metadata_entries) < len(response_texts):
            metadata_entries.extend({} for _ in range(len(response_texts) - len(metadata_entries)))
        if not invalid_idx:
            return response_texts, metadata_entries, 0, 0
        retry_count = len(invalid_idx)
        print(f"prestep_parser_retry: invalid actions before env.step at idx={invalid_idx}; regenerating once.")
        retry_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        retry_output = unpad_dataproto(retry_output, pad_size=pad_size)
        retry_texts = self.tokenizer.batch_decode(retry_output.batch['responses'], skip_special_tokens=True)
        retry_metadata_entries = list(retry_output.non_tensor_batch.get("rollout_step_metadata", []))
        recovered = 0
        for i in invalid_idx:
            if i < len(retry_texts) and self._looks_parseable_gui_action(retry_texts[i]):
                response_texts[i] = retry_texts[i]
                if i < len(retry_metadata_entries):
                    metadata_entries[i] = retry_metadata_entries[i]
                recovered += 1
        if recovered:
            print(f"prestep_parser_retry: recovered {recovered}/{retry_count} invalid actions before env.step.")
        return response_texts, metadata_entries, retry_count, recovered

    def _attach_rollout_step_metadata(self, process_results: List[dict], step_metadata_by_env: List[List[dict]]) -> List[dict]:
        for process_result, step_metadata in zip(process_results, step_metadata_by_env):
            process_result["rollout_step_metadata"] = copy.deepcopy(step_metadata)
        return process_results

    @staticmethod
    def _attach_env_noise_metadata(step_metadata_by_job: List[List[dict]], slot: int, result: dict) -> None:
        noise_burden = result.get("noise_burden")
        if noise_burden is None:
            return
        if not step_metadata_by_job[slot]:
            step_metadata_by_job[slot].append({})
        step_metadata_by_job[slot][-1]["noise_burden"] = copy.deepcopy(noise_burden)

    def _task_family_decoding_overrides(self, task_config: Optional[dict], is_val: bool = False) -> Dict[str, Any]:
        """Minimal task-family-conditioned decoding presets (safe for smoke runs)."""
        if not task_config:
            return {}
        text = str(task_config.get("instruction") or "").lower()
        domain = str(task_config.get("domain") or "").lower()

        # Defaults come from YAML; only override high-risk families.
        overrides: Dict[str, Any] = {}
        if is_val:
            # Keep validation mostly deterministic; don't loosen it.
            return overrides

        # Browser settings / shortcuts / search are the most hallucination-prone for 2B.
        is_browser_search = any(k in text for k in ("search", "discussions", "reddit", "community"))
        is_browser_setting = ("bing" in text and "search" in text) or ("default search" in text)
        is_web_shortcut = "shortcut" in text and any(k in text for k in ("site", "website", "page"))
        if is_browser_setting or is_web_shortcut:
            overrides.update({"temperature": 0.10, "top_p": 0.8, "top_k": 20})
        elif is_browser_search:
            overrides.update({"temperature": 0.15, "top_p": 0.8, "top_k": 20})
        # File manager / trash flows benefit from mild but not high exploration.
        elif "trash" in text or "deleted" in text or domain in {"os", "files"}:
            overrides.update({"temperature": 0.18, "top_p": 0.8, "top_k": 30})
        # GIMP/VLC tasks often need precise menu navigation.
        elif domain in {"gimp", "vlc"} or "gimp" in text or "vlc" in text or "music video" in text:
            overrides.update({"temperature": 0.15, "top_p": 0.8, "top_k": 20})

        return overrides

    def _apply_task_family_decoding_if_single(
        self,
        gen_batch: DataProto,
        valid_env_idx: List[int],
        task_configs: List[dict],
        is_val: bool = False,
        env_to_task_index: Optional[Dict[int, int]] = None,
    ) -> None:
        """Apply family-conditioned decoding only when batch maps to a single env/task (current smoke path)."""
        if len(valid_env_idx) != 1:
            return
        env_i = valid_env_idx[0]
        if env_to_task_index is not None:
            env_i = env_to_task_index.get(env_i, env_i)
        if env_i < 0 or env_i >= len(task_configs):
            return
        overrides = self._task_family_decoding_overrides(task_configs[env_i], is_val=is_val)
        if not overrides:
            return
        if getattr(gen_batch, "meta_info", None) is None:
            gen_batch.meta_info = {}
        # Preserve existing rollout/val overrides and only specialize decoding knobs.
        merged = dict(gen_batch.meta_info)
        merged.update(overrides)
        gen_batch.meta_info = merged
        print(f"task_family_decoding: env_idx={env_i} overrides={overrides} task_id={task_configs[env_i].get('id')}")

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        world_size = self.actor_rollout_wg.world_size
        if batch_size < world_size:
            return
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    
    def prepare_vllm_inputs_full(self, env_outputs: List):
        # NOTE: processor will be very slow
        obs_messages = [x['obs_messages'] for x in env_outputs]
        env_idx = [x['env_idx'] for x in env_outputs]

        valid_obs_messages = [x['obs_messages'] for x in env_outputs if x['obs_messages'] is not None and not x.get('is_done', False)]
        valid_env_idx = [x['env_idx'] for x in env_outputs if x['obs_messages'] is not None and not x.get('is_done', False)]

        if not valid_obs_messages:
            return None, [], []

        dataset = OSWorldDataset(
            valid_obs_messages,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            format_prompt=self.config.data.format_prompt,
            max_pixels=self.config.data.max_pixels,
            min_pixels=self.config.data.min_pixels,
            fast_rollout=True,
            limit_images=getattr(self.config.worker.rollout, "limit_images", 10),
        )

        # batch_dict = [dataset[i] for i in range(len(dataset))]
        def get_dataset_item(index):
            return dataset[index]

        with ThreadPoolExecutor(max_workers=64) as executor:
            batch_dict = list(executor.map(get_dataset_item, range(len(dataset))))

        # batch_dict = ray.get([get_dataset_item.remote(i) for i in range(len(dataset))])

        # Filter out items whose prompt exceeds max_prompt_length
        max_len = self.config.data.max_prompt_length
        keep_indices = []
        overlong_env_idx = []
        for i, item in enumerate(batch_dict):
            prompt_len = len(item["raw_prompt_ids"])
            if prompt_len > max_len:
                print(
                    f"[prepare_vllm] Skipping env {valid_env_idx[i]}: "
                    f"prompt length {prompt_len} > max {max_len}, marking as failed"
                )
                overlong_env_idx.append(valid_env_idx[i])
            else:
                keep_indices.append(i)

        if not keep_indices:
            return None, [], overlong_env_idx

        batch_dict = [batch_dict[i] for i in keep_indices]
        valid_env_idx = [valid_env_idx[i] for i in keep_indices]

        batch_dict = collate_fn_dataproto(batch_dict)
        batch = DataProto.from_single_dict(batch_dict)

        return batch, valid_env_idx, overlong_env_idx


    def prepare_grpo_inputs(self, messages, eval_results, task_configs):
        eval_result_flatten = eval_results
        messages_flatten = messages

        dataset = OSWorldGRPODataset(
            messages_flatten,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            format_prompt=self.config.data.format_prompt,
            max_pixels=self.config.data.max_pixels,
            min_pixels=self.config.data.min_pixels,
        )
        def get_dataset_item(index):
            return dataset[index]

        with ThreadPoolExecutor(max_workers=64) as executor:
            batch_dict = list(executor.map(get_dataset_item, range(len(dataset))))
        # batch_dict = [get_dataset_item(i) for i in range(len(dataset))]
        
        batch_dict = collate_fn_dataproto(batch_dict)
        batch = DataProto.from_single_dict(batch_dict)

        # uid
        # use batch to compute norm reward
        batch.non_tensor_batch["uid"] = np.array([x['id'] for x in task_configs], dtype=object)
        batch.non_tensor_batch["task_id"] = np.array([x['id'] for x in task_configs], dtype=object)

        batch.batch["rewards"] = torch.tensor([float(x) for x in eval_result_flatten], dtype=torch.float32)

        return batch


            

    def save_rollout_trajectories(self, action_batch_output, history_messages, eval_results, task_configs):
        visual_trajs = dict()
        visual_trajs['history_messages'] = history_messages
        visual_trajs['eval_results'] = eval_results
        visual_trajs['task_configs'] = task_configs
    
        # os.makedirs(self.config.trainer.save_checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.trainer.save_checkpoint_path, "trajs"), exist_ok=True)
        visual_folder_path = os.path.join(self.config.trainer.save_checkpoint_path, "trajs", f"global_step_{self.global_step}.pth")
        torch.save(visual_trajs, visual_folder_path)
        action_batch_output.save_to_disk(os.path.join(self.config.trainer.save_checkpoint_path, "trajs", f"global_step_{self.global_step}_batch.pkl"))

    def _build_rollout_chunks(self, batch_dict: List[dict]) -> List[List[dict]]:
        rollout_jobs = build_rollout_jobs(batch_dict, self.config.worker.rollout.n)
        # Opt-in: annotate per-task noise fields from current curriculum state
        # so workers (local or remote via HTTP) read a self-contained task_config.
        self._annotate_task_configs_with_noise(rollout_jobs, is_val=False)
        return chunk_rollout_jobs(rollout_jobs, len(self.env_workers))

    def _launch_prefetch_resets(self, batch_dict: List[dict]):
        """Launch VM resets for next step's chunk-0 while GPU is busy with current step.

        Returns (batch_dict, rollout_chunks, reset_futures) where reset_futures are
        ObjectRefs from worker.reset.remote() for chunk 0 only.  Workers are reused
        across chunks so only chunk 0 can be prefetched.
        """
        rollout_chunks = self._build_rollout_chunks(batch_dict)
        chunk0 = rollout_chunks[0]
        workers = self.env_workers[:len(chunk0)]
        futures = [w.reset.remote(tc) for w, tc in zip(workers, chunk0)]
        print(f"[prefetch] launched {len(futures)} resets for next step")
        return (batch_dict, rollout_chunks, futures)

    def _run_rollout_chunk(
        self,
        task_configs: List[dict],
        timing_raw: Dict[str, float],
        chunk_idx: int,
        total_chunks: int,
        prefetch_reset_futures=None,
    ) -> Tuple[List[dict], List[float], List[float], List[dict]]:
        active_workers = self.env_workers[: len(task_configs)]
        local_timing: Dict[str, float] = {}

        format_rewards = [0.0] * len(task_configs)
        eval_results_objects = [None] * len(task_configs)
        rollout_step_metadata_by_job = [[] for _ in task_configs]
        batch_skipped = False

        with _timer("env_reset", local_timing):
            # Timeout: RemoteEnvWorker reset does up to 3 attempts × 1200s each; give enough room
            # for one full reset attempt per worker. If a worker hangs, _ray_get_robust returns
            # obs_messages=None for it, which prepare_vllm_inputs_full handles gracefully.
            _reset_timeout = float(os.environ.get("ROLLOUT_RESET_TIMEOUT", "1300"))
            if prefetch_reset_futures is not None:
                # Resets were launched during previous step's GPU training phase.
                # If resets already finished, _ray_get_robust returns instantly (full savings).
                # If GPU finished before resets, _ray_get_robust blocks for the remaining
                # reset time (partial savings — still faster than no prefetch).
                reset_outputs = _ray_get_robust(
                    prefetch_reset_futures,
                    timeout=_reset_timeout,
                    fallback_fn=lambda idx: {"env_idx": idx, "obs_messages": None, "is_done": True, "format_reward": 0.0},
                    label=f"chunk{chunk_idx+1}/reset(prefetched)",
                )
            else:
                reset_outputs = _ray_get_robust(
                    [worker.reset.remote(task_config) for worker, task_config in zip(active_workers, task_configs)],
                    timeout=_reset_timeout,
                    fallback_fn=lambda idx: {"env_idx": idx, "obs_messages": None, "is_done": True, "format_reward": 0.0},
                    label=f"chunk{chunk_idx+1}/reset",
                )
        print(
            f"[chunk={chunk_idx + 1}/{total_chunks}] reset_time: {local_timing['env_reset']} "
            f"| task_ids={[task_config['id'] for task_config in task_configs]}"
        )

        active_worker_by_env_idx = {output["env_idx"]: worker for output, worker in zip(reset_outputs, active_workers)}
        slot_by_env_idx = {output["env_idx"]: slot for slot, output in enumerate(reset_outputs)}
        env_outputs = reset_outputs

        done_slots = set()  # Track workers that finished and launched evaluate
        had_successful_generate = False
        with _timer("gen", local_timing):
            # ── Per-GPU-group async step loop ──
            # Instead of waiting for ALL VMs to finish each step before generating,
            # groups of VMs (matching the DP split) advance independently.  When a
            # group's VMs all finish their current step, that group gets its next
            # generate call without waiting for other groups' slower VMs.
            world_size = self.actor_rollout_wg.world_size
            num_workers = len(active_workers)
            max_steps = self.config.env.max_steps
            _step_timeout = float(os.environ.get("ROLLOUT_STEP_TIMEOUT", "330"))

            # Partition workers into groups matching the DP split (world_size groups).
            # With DP=8 and 32 VMs, each group has 4 VMs.
            base_group_size = num_workers // world_size
            _remainder = num_workers % world_size
            groups = {}
            _offset_g = 0
            for g in range(world_size):
                _sz = base_group_size + (1 if g < _remainder else 0)
                groups[g] = list(range(_offset_g, _offset_g + _sz))
                _offset_g += _sz

            group_step = {g: 0 for g in groups}   # completed generates per group
            slot_to_group = {}
            for g, _slots in groups.items():
                for s in _slots:
                    slot_to_group[s] = g
            env_idx_by_slot = {slot: output["env_idx"] for slot, output in enumerate(reset_outputs)}

            ref_to_slot = {}        # ObjectRef -> slot index
            in_flight_refs = []     # all pending step ObjectRefs
            group_completed = {g: {} for g in groups}  # g -> {slot: env_output}

            # ── Phase 1: First generate from reset outputs (step 0) ──
            print(
                f"[chunk={chunk_idx + 1}/{total_chunks}] step_idx: 0 (initial), "
                f"finished: {len(done_slots)}/{num_workers}"
            )

            # Verify obs at step 0 (debugging)
            _obs = next((x["obs_messages"] for x in env_outputs if x.get("obs_messages")), None)
            if _obs is not None:
                _n_msg = len(_obs)
                _n_img = sum(
                    1
                    for m in _obs
                    for c in (m.get("content") or [])
                    if isinstance(c, dict) and c.get("type") == "image"
                )
                _txt_len = 0
                for m in _obs:
                    for c in (m.get("content") or []):
                        if isinstance(c, dict) and "text" in c:
                            _txt_len += len(c.get("text", ""))
                print(
                    f"verify_obs: chunk={chunk_idx + 1}/{total_chunks} step=0 "
                    f"messages={_n_msg} images={_n_img} instruction_text_len={_txt_len}"
                )
            elif _obs is None:
                _t = getattr(self, "_last_remote_fail_log", 0)
                if _t == 0 or time.time() - _t >= 30:
                    print(
                        "verify_obs: step=0 obs_messages is None (reset/step failed). "
                        "Remote env 503? (this message rate-limited to every 30s)"
                    )
                    self._last_remote_fail_log = time.time()

            with _timer("prepare_vllm_inputs", local_timing):
                vllm_batch, valid_env_idx, overlong_env_idx = self.prepare_vllm_inputs_full(env_outputs)

            # Mark overlong-prompt slots as done with score 0
            for eidx in overlong_env_idx:
                slot = slot_by_env_idx.get(eidx)
                if slot is not None and slot not in done_slots:
                    format_rewards[slot] = 0.0
                    done_slots.add(slot)
                    eval_results_objects[slot] = active_workers[slot].evaluate.remote()

            if vllm_batch is None or not isinstance(vllm_batch, DataProto):
                # All reset outputs failed — no valid obs to generate from
                for i in range(len(task_configs)):
                    if i not in done_slots:
                        format_rewards[i] = 0.0
                _t = getattr(self, "_last_remote_fail_log", 0)
                if _t == 0 or time.time() - _t >= 30:
                    print(
                        "prepare_vllm_inputs: no valid obs_messages (all envs returned None). "
                        "Remote reset must return obs_messages with screenshot; check remote server. (rate-limited 30s)"
                    )
                    self._last_remote_fail_log = time.time()
            else:
                had_successful_generate = True
                vllm_batch_pad, pad_size = pad_dataproto_to_divisor(vllm_batch, world_size)
                gen_batch = vllm_batch_pad.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                )
                self._apply_task_family_decoding_if_single(
                    gen_batch,
                    valid_env_idx,
                    task_configs,
                    is_val=False,
                    env_to_task_index=slot_by_env_idx,
                )
                with _timer("actor_rollout_wg", local_timing):
                    action_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                action_batch_output = unpad_dataproto(action_batch_output, pad_size=pad_size)

                response_texts = self.tokenizer.batch_decode(action_batch_output.batch["responses"], skip_special_tokens=True)
                response_texts, rollout_step_metadata, _, _ = self._retry_invalid_actions_once(
                    gen_batch, action_batch_output, response_texts, pad_size
                )
                for env_idx, step_meta in zip(valid_env_idx, rollout_step_metadata):
                    rollout_step_metadata_by_job[slot_by_env_idx[env_idx]].append(step_meta)

                if response_texts:
                    sample = (response_texts[0] or "")[:80].replace("\n", " ")
                    print(f"[chunk={chunk_idx + 1}/{total_chunks}] on_policy output preview: {sample!r}")

                # Dispatch step.remote() to all active VMs
                cur_valid_envs = [active_worker_by_env_idx[env_idx] for env_idx in valid_env_idx]
                for worker, action_text, env_idx in zip(cur_valid_envs, response_texts, valid_env_idx):
                    ref = worker.step.remote(action_text)
                    slot = slot_by_env_idx[env_idx]
                    ref_to_slot[ref] = slot
                    in_flight_refs.append(ref)

                # All groups with active VMs have completed their first generate
                for g in groups:
                    if any(s not in done_slots for s in groups[g]):
                        group_step[g] = 1

            # ── Phase 2: Async group loop ──
            # Each group advances independently.  When all active VMs in a group
            # finish their current step, that group gets its next generate call
            # without waiting for other groups' slower VMs.
            while in_flight_refs:
                # Wait for at least 1 future to complete
                newly_ready, still_pending = ray.wait(
                    in_flight_refs, num_returns=1, timeout=_step_timeout
                )
                # Greedily collect ALL other ready futures (non-blocking)
                # to batch multiple groups into a single generate call
                if still_pending:
                    more_ready, still_pending = ray.wait(
                        still_pending, num_returns=len(still_pending), timeout=0
                    )
                    newly_ready.extend(more_ready)
                in_flight_refs = list(still_pending)

                if not newly_ready:
                    # Hard timeout — all remaining futures stuck
                    print(
                        f"[chunk={chunk_idx+1}/{total_chunks}] async step loop: "
                        f"ray.wait timed out after {_step_timeout}s; "
                        f"marking {len(in_flight_refs)} remaining futures as failed"
                    )
                    for ref in in_flight_refs:
                        slot = ref_to_slot.pop(ref, None)
                        if slot is not None and slot not in done_slots:
                            done_slots.add(slot)
                            if eval_results_objects[slot] is None:
                                eval_results_objects[slot] = active_workers[slot].evaluate.remote()
                    in_flight_refs = []
                    break

                # Process each completed step future
                for ref in newly_ready:
                    slot = ref_to_slot.pop(ref)
                    group = slot_to_group[slot]
                    try:
                        result = ray.get(ref, timeout=0)
                    except Exception as e:
                        print(
                            f"[chunk={chunk_idx+1}/{total_chunks}] async step: "
                            f"ray.get failed for slot {slot} ({e!r}); using fallback"
                        )
                        result = {
                            "env_idx": env_idx_by_slot[slot],
                            "obs_messages": None,
                            "is_done": True,
                            "format_reward": 0.0,
                        }

                    self._attach_env_noise_metadata(rollout_step_metadata_by_job, slot, result)
                    format_rewards[slot] += float(result.get("format_reward", 0.0))
                    if result["is_done"] or result.get("obs_messages") is None:
                        done_slots.add(slot)
                        if eval_results_objects[slot] is None:
                            eval_results_objects[slot] = active_workers[slot].evaluate.remote()
                    else:
                        group_completed[group][slot] = result

                # Determine which groups are ready for their next generate
                ready_groups = {}
                for g in groups:
                    active_in_group = [s for s in groups[g] if s not in done_slots]
                    if not active_in_group:
                        continue  # group fully done
                    pending_in_group = [
                        s for s in active_in_group if s not in group_completed[g]
                    ]
                    if not pending_in_group:
                        # All active VMs in this group finished their current step
                        if group_step[g] < max_steps:
                            ready_groups[g] = [group_completed[g][s] for s in active_in_group]
                        else:
                            # Max steps reached — launch evaluate for remaining VMs
                            for s in active_in_group:
                                done_slots.add(s)
                                if eval_results_objects[s] is None:
                                    eval_results_objects[s] = active_workers[s].evaluate.remote()
                        group_completed[g] = {}  # clear for next step

                # Batch generate for all ready groups
                if ready_groups:
                    all_ready_obs = []
                    ready_group_ids = sorted(ready_groups.keys())
                    for g in ready_group_ids:
                        all_ready_obs.extend(ready_groups[g])

                    print(
                        f"[chunk={chunk_idx+1}/{total_chunks}] async generate: "
                        f"{len(ready_group_ids)} groups ({len(all_ready_obs)} VMs), "
                        f"steps: {{{', '.join(f'{g}:{group_step[g]}' for g in ready_group_ids)}}}, "
                        f"done: {len(done_slots)}/{num_workers}"
                    )

                    with _timer("prepare_vllm_inputs", local_timing):
                        vllm_batch, valid_env_idx, overlong_env_idx = self.prepare_vllm_inputs_full(all_ready_obs)

                    # Mark overlong-prompt slots as done with score 0
                    for eidx in overlong_env_idx:
                        slot = slot_by_env_idx.get(eidx)
                        if slot is not None and slot not in done_slots:
                            format_rewards[slot] = 0.0
                            done_slots.add(slot)
                            if eval_results_objects[slot] is None:
                                eval_results_objects[slot] = active_workers[slot].evaluate.remote()

                    if vllm_batch is not None and isinstance(vllm_batch, DataProto):
                        had_successful_generate = True
                        vllm_batch_pad, pad_size = pad_dataproto_to_divisor(vllm_batch, world_size)
                        gen_batch = vllm_batch_pad.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                        )
                        self._apply_task_family_decoding_if_single(
                            gen_batch,
                            valid_env_idx,
                            task_configs,
                            is_val=False,
                            env_to_task_index=slot_by_env_idx,
                        )
                        with _timer("actor_rollout_wg", local_timing):
                            action_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        action_batch_output = unpad_dataproto(action_batch_output, pad_size=pad_size)

                        response_texts = self.tokenizer.batch_decode(
                            action_batch_output.batch["responses"], skip_special_tokens=True
                        )
                        response_texts, rollout_step_metadata, _, _ = self._retry_invalid_actions_once(
                            gen_batch, action_batch_output, response_texts, pad_size
                        )
                        for env_idx, step_meta in zip(valid_env_idx, rollout_step_metadata):
                            rollout_step_metadata_by_job[slot_by_env_idx[env_idx]].append(step_meta)

                        # Dispatch step.remote() for ready VMs
                        cur_valid_envs = [active_worker_by_env_idx[env_idx] for env_idx in valid_env_idx]
                        for worker, action_text, env_idx in zip(
                            cur_valid_envs, response_texts, valid_env_idx
                        ):
                            ref = worker.step.remote(action_text)
                            slot = slot_by_env_idx[env_idx]
                            ref_to_slot[ref] = slot
                            in_flight_refs.append(ref)
                    else:
                        # All obs in ready groups are invalid
                        print(
                            f"[chunk={chunk_idx+1}/{total_chunks}] async generate: "
                            f"prepare_vllm_inputs returned None for {len(all_ready_obs)} VMs; "
                            "marking as failed"
                        )
                        for g in ready_group_ids:
                            for s in groups[g]:
                                if s not in done_slots:
                                    format_rewards[s] = 0.0
                                    done_slots.add(s)
                                    if eval_results_objects[s] is None:
                                        eval_results_objects[s] = active_workers[s].evaluate.remote()

                    # Update group step counters
                    for g in ready_group_ids:
                        group_step[g] += 1

                # Check termination
                if len(done_slots) >= num_workers:
                    break

            # Set batch_skipped based on whether any generate succeeded
            batch_skipped = not had_successful_generate

        if not batch_skipped:
            missing_eval_idx = [i for i, output in enumerate(eval_results_objects) if output is None]
            if missing_eval_idx:
                print(
                    f"[chunk={chunk_idx + 1}/{total_chunks}] max_steps reached before done for env_idx={missing_eval_idx}; "
                    "scheduling evaluate at cutoff."
                )
                for slot in missing_eval_idx:
                    eval_results_objects[slot] = active_workers[slot].evaluate.remote()

        # Launch get_train_dict on all workers BEFORE waiting on evaluate.
        # Since evaluate and get_train_dict are methods on the same single-threaded
        # Ray actor, get_train_dict queues behind evaluate on each actor.  By launching
        # these futures now, each actor processes evaluate -> get_train_dict back-to-back
        # with no driver round-trip in between.  This eliminates the old sequential wait
        # pattern (wait eval -> launch get_train_dict -> wait get_train_dict).
        train_dict_futures = [worker.get_train_dict.remote() for worker in active_workers]

        _eval_timeout = float(os.environ.get("ROLLOUT_EVAL_TIMEOUT", "350"))
        with _timer("evaluate_env", local_timing):
            if batch_skipped:
                # Collect real eval results for workers that already completed
                # (in done_slots with valid ObjectRefs); default 0.0 for the rest.
                eval_results = [0.0] * len(task_configs)
                for slot in done_slots:
                    if eval_results_objects[slot] is not None:
                        try:
                            eval_results[slot] = ray.get(eval_results_objects[slot], timeout=_eval_timeout)
                        except Exception:
                            eval_results[slot] = 0.0
            else:
                eval_results = _ray_get_robust(
                    eval_results_objects,
                    timeout=_eval_timeout,
                    fallback_fn=lambda _: 0.0,
                    label=f"chunk{chunk_idx+1}/evaluate",
                )
        print(
            f"[chunk={chunk_idx + 1}/{total_chunks}] evaluate_env_time: {local_timing['evaluate_env']} "
            f"| eval_results: {eval_results} | format_rewards: {format_rewards}"
        )

        # get_train_dict futures were launched before the eval wait above; by now each
        # actor has finished evaluate -> get_train_dict sequentially.  This wait should
        # be near-instant for workers whose evaluate completed during the eval wait.
        # Timeout accounts for evaluate potentially still running on the actor (if eval
        # timed out on the driver side, get_train_dict is still queued behind it).
        _train_dict_timeout = float(os.environ.get("ROLLOUT_TRAIN_DICT_TIMEOUT", str(_eval_timeout + 60)))
        process_results = _ray_get_robust(
            train_dict_futures,
            timeout=_train_dict_timeout,
            fallback_fn=lambda _: {
                "input_ids": torch.zeros((1,), dtype=torch.int64),
                "labels": torch.full((1,), -100, dtype=torch.int64),
                "attention_mask": torch.zeros((1,), dtype=torch.int64),
                "position_ids": torch.zeros((3, 1), dtype=torch.int64),
            },
            label=f"chunk{chunk_idx+1}/get_train_dict",
        )
        process_results = self._attach_rollout_step_metadata(process_results, rollout_step_metadata_by_job)
        merge_timing_raw(timing_raw, local_timing)
        return process_results, eval_results, format_rewards, task_configs
    
    @staticmethod
    def _align_replay_seq_len(pos_batch, target_seq_len, pad_token_id=0):
        """Left-pad or left-truncate replay batch tensors to match target_seq_len.

        Replay items come from previous steps with different collated max lengths.
        Since collate_fn_dataproto uses left-padding, the left side is always pad
        tokens. Left-truncation removes padding; left-padding adds it.
        """
        _SEQ_PAD = {"input_ids": pad_token_id, "attention_mask": 0, "position_ids": 0, "labels": -100}
        for key in list(pos_batch.batch.keys()):
            tensor = pos_batch.batch[key]
            if tensor.dim() < 2 or key not in _SEQ_PAD:
                continue
            cur_len = tensor.size(-1)
            if cur_len == target_seq_len:
                continue
            if cur_len > target_seq_len:
                # Left-truncate: remove left-padding
                excess = cur_len - target_seq_len
                pos_batch.batch[key] = tensor[..., excess:].contiguous()
            else:
                # Left-pad: prepend padding
                pad_size = target_seq_len - cur_len
                pad_shape = list(tensor.shape)
                pad_shape[-1] = pad_size
                pad_val = _SEQ_PAD[key]
                pad_t = torch.full(pad_shape, pad_val, dtype=tensor.dtype, device=tensor.device)
                pos_batch.batch[key] = torch.cat([pad_t, tensor], dim=-1)
        return pos_batch

    @staticmethod
    def _summarize_noise_burden(step_metadata):
        summary = {
            "interruptions": 0.0,
            "recovery_cost": 0.0,
            "occlusion": 0.0,
            "focus_loss": 0.0,
        }
        if not step_metadata:
            return summary

        for step in step_metadata:
            burden = (step or {}).get("noise_burden") or {}
            summary["interruptions"] += float(burden.get("interruptions", 0.0) or 0.0)
            summary["recovery_cost"] += float(burden.get("total_recovery", 0.0) or 0.0)
            summary["occlusion"] += float(burden.get("occlusion", 0.0) or 0.0)
            summary["focus_loss"] += float(burden.get("focus_loss", 0.0) or 0.0)
        return summary

    def _build_replay_tags(self, task_configs, batch):
        eval_results = batch.batch["eval_results"].tolist()
        metadata_entries = list(batch.non_tensor_batch.get("rollout_step_metadata", []))
        if len(metadata_entries) < len(eval_results):
            metadata_entries.extend([[] for _ in range(len(eval_results) - len(metadata_entries))])

        replay_tags = []
        for task_config, eval_result, step_metadata in zip(task_configs, eval_results, metadata_entries):
            if eval_result <= 0.1:
                replay_tags.append(None)
                continue

            burden = self._summarize_noise_burden(step_metadata)
            noise_enabled = bool(task_config.get("enable_noise"))
            realized_noise = any(value > 0 for value in burden.values())
            if realized_noise:
                replay_tags.append("recovery_success")
            elif noise_enabled:
                replay_tags.append("noisy_success")
            else:
                replay_tags.append("clean_success")
        return replay_tags

    @staticmethod
    def _preferred_replay_tags(task_config):
        if bool(task_config.get("enable_noise")):
            return ["recovery_success", "noisy_success", "clean_success"]
        return ["clean_success", "noisy_success", "recovery_success"]

    def apply_replay(self, task_configs, batch):
        eval_results = batch.batch["eval_results"].tolist()
        assert len(task_configs) == len(batch)

        rollout_n = self.config.worker.rollout.n
        bsz = len(task_configs) // rollout_n
        if bsz == 0:
            return batch  # e.g. 1 remote env with 1 rollout, no replay grouping

        cur_seq_len = batch.batch["input_ids"].size(-1)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        replay_tags = self._build_replay_tags(task_configs, batch)

        final_batch = []
        for i in range(bsz):
            cur_task_config = task_configs[i * rollout_n:(i + 1) * rollout_n]
            assert len(set([x['id'] for x in cur_task_config])) == 1
            task_id = cur_task_config[0]['id']
            instruction = cur_task_config[0]['instruction']

            cur_rewards = np.array(eval_results[i * rollout_n:(i + 1) * rollout_n], dtype=float)
            cur_batch = batch[i * rollout_n:(i + 1) * rollout_n]
            cur_reward_std = np.std(cur_rewards)
            cur_reward_mean = np.mean(cur_rewards)
            preferred_tags = self._preferred_replay_tags(cur_task_config[0])
            if cur_reward_std < 0.05 and cur_reward_mean < 0.2:  # all negative group
                pos_batch = self.replay.get_pos(cur_task_config[0]['id'], num_samples=1, preferred_tags=preferred_tags)
            else:
                pos_batch = []

            if len(pos_batch) > 0:
                pos_batch = self._align_replay_seq_len(pos_batch, cur_seq_len, pad_token_id)
                final_batch.append(pos_batch)
                final_batch.append(cur_batch[len(pos_batch):])
            else:
                final_batch.append(cur_batch)

            print(
                f'Task {task_id} {instruction} replay_buffer: {len(pos_batch)} | '
                f'preferred_tags: {preferred_tags} | rewards: {cur_rewards}'
            )

        self.replay.update_replay_buffer_batch(task_configs, batch, replay_tags=replay_tags)
        print('Update replay buffer done')
        final_batch = DataProto.concat(final_batch)
        return final_batch

        
        

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # Initialize JSONL training log
        log_dir = self.config.trainer.save_checkpoint_path
        os.makedirs(log_dir, exist_ok=True)
        self._training_log_path = os.path.join(log_dir, "training_log.jsonl")
        print(f"Training log: {self._training_log_path}")

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                print("[val_only] Validation complete.")
                return

        if self.config.trainer.val_only:
            print("[val_only] val_before_train is False; nothing to do.")
            return

        for _ in tqdm(range(self.config.trainer.total_episodes), desc="Episode", position=0):
            prefetched = None  # (batch_dict, rollout_chunks, reset_futures) or None
            iterator = iter(tqdm(self.train_dataloader, desc="Running step", position=1))

            while True:
                # Use prefetched data if available (resets launched during previous step's GPU training)
                if prefetched is not None:
                    batch_dict, rollout_chunks, prefetch_reset_futures_chunk0 = prefetched
                    prefetched = None
                else:
                    try:
                        batch_dict = next(iterator)
                    except StopIteration:
                        break
                    rollout_chunks = self._build_rollout_chunks(batch_dict)
                    prefetch_reset_futures_chunk0 = None  # first step of episode: no prefetch

                self.global_step += 1
                if self.global_step > self.training_steps:
                    break

                metrics, timing_raw = {}, {}
                required_rollouts = len(batch_dict) * self.config.worker.rollout.n
                print([config["id"] for config in batch_dict])
                print(
                    f"task_num: {len(batch_dict)}, env_num: {len(self.env_workers)}, "
                    f"required_rollouts={required_rollouts}, rollout_chunks={len(rollout_chunks)}"
                )
                print([config["instruction"] for config in batch_dict])

                all_process_results = [None] * required_rollouts
                all_eval_results = [0.0] * required_rollouts
                all_format_rewards = [0.0] * required_rollouts
                all_task_configs = [None] * required_rollouts

                with _timer("step", timing_raw):
                    self.actor_rollout_wg.prepare_generate_sequences()
                    try:
                        for chunk_idx, task_configs_chunk in enumerate(rollout_chunks):
                            try:
                                # Only chunk 0 can use prefetched resets (workers are reused across chunks)
                                chunk_prefetch = prefetch_reset_futures_chunk0 if chunk_idx == 0 else None
                                process_results, eval_results, format_rewards, chunk_task_configs = self._run_rollout_chunk(
                                    task_configs_chunk, timing_raw, chunk_idx, len(rollout_chunks),
                                    prefetch_reset_futures=chunk_prefetch,
                                )
                            except Exception as chunk_exc:
                                # Log and skip this chunk rather than crashing the training loop.
                                # This handles unexpected errors (e.g. RayActorError, RayTaskError) not
                                # already caught by _ray_get_robust. The training step will have
                                # partial data; num_valid_tokens==0 check below will skip the update.
                                print(
                                    f"[chunk={chunk_idx + 1}/{len(rollout_chunks)}] "
                                    f"CHUNK EXCEPTION — skipping chunk: {chunk_exc!r}"
                                )
                                import traceback as _tb
                                _tb.print_exc()
                                continue
                            start = chunk_idx * len(self.env_workers)
                            end = start + len(chunk_task_configs)
                            all_process_results[start:end] = process_results
                            all_eval_results[start:end] = eval_results
                            all_format_rewards[start:end] = format_rewards
                            all_task_configs[start:end] = chunk_task_configs
                    finally:
                        self.actor_rollout_wg.finish_generate_sequences()

                    # Opt-in: update per-task noise curriculum from this step's
                    # observed success rates. Must run before prefetch so the
                    # next step's annotation reflects updated tier/p.
                    try:
                        self._update_noise_curriculum(all_task_configs, all_eval_results, all_process_results)
                    except Exception as _ne:
                        print(f"[noise] curriculum update failed (non-fatal): {_ne}")

                    # Workers are idle after rollout. Peek at next batch and launch VM resets
                    # concurrently with GPU training (compute logprobs, advantages, updates).
                    # Resets are pure HTTP calls — no GPU/model/shared-state dependency.
                    try:
                        next_batch = next(iterator)
                    except StopIteration:
                        prefetched = None
                    else:
                        try:
                            prefetched = self._launch_prefetch_resets(next_batch)
                        except Exception as e:
                            # Worker died or other error — store batch without prefetch futures.
                            # Next step will launch resets synchronously (normal fallback path).
                            print(f"[prefetch] failed to launch resets ({e!r}); will reset synchronously next step")
                            prefetched = (next_batch, self._build_rollout_chunks(next_batch), None)

                    # Build the training batch from all sequential rollouts
                    with _timer("prepare_grpo_inputs", timing_raw):
                        if any(x is None for x in all_process_results):
                            raise RuntimeError("rollout scheduler produced incomplete process results.")
                        if any(x is None for x in all_task_configs):
                            raise RuntimeError("rollout scheduler produced incomplete task assignments.")
                        # Some rollouts can lose image features after truncation and omit
                        # multi_modal_* keys, while others still include them. Normalize
                        # keys across samples so DataProto non-tensor lengths stay aligned.
                        if all_process_results:
                            mm_keys = ("multi_modal_inputs", "multi_modal_data")
                            for k in mm_keys:
                                if any(k in x for x in all_process_results):
                                    for x in all_process_results:
                                        x.setdefault(k, {})
                        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
                        batch = collate_fn_dataproto(all_process_results, pad_token_id=pad_token_id)
                        batch = DataProto.from_single_dict(batch)

                        batch.batch["eval_results"] = torch.tensor([float(x) for x in all_eval_results], dtype=torch.float32)
                        batch.batch["format_rewards"] = torch.tensor([float(x) for x in all_format_rewards], dtype=torch.float32)
                        batch.non_tensor_batch["uid"] = np.array([x['id'] for x in all_task_configs], dtype=object)
                        batch.non_tensor_batch["task_id"] = np.array([x['id'] for x in all_task_configs], dtype=object)
                        task_configs = all_task_configs  # update for downstream use
                        
                    
                    with _timer("replay", timing_raw):
                        batch = self.apply_replay(task_configs, batch)

                    # responses = shifted input_ids (autoregressive: predict token i+1 from token i).
                    # _forward_micro_batch slices logits to [-response_length-1:-1], which gives
                    # exactly response_length elements only when response_length < seqlen.
                    # Setting responses = input_ids[:, 1:] ensures response_length = seqlen-1.
                    batch.batch["responses"] = batch.batch["input_ids"][:, 1:]
                    response_len = batch.batch["responses"].size(1)
                    seq_len = batch.batch["input_ids"].size(1)
                    # Truncate labels to match input_ids so no tensor has longer seq dim (avoids 8192 vs 45122 mismatch)
                    if batch.batch["labels"].size(1) > seq_len:
                        batch.batch["labels"] = batch.batch["labels"][:, :seq_len].contiguous()
                    # Truncate response_mask to match responses (labels can be longer than input_ids after collate)
                    labels_shifted = (batch.batch["labels"] != -100)[:, 1:]
                    batch.batch["response_mask"] = labels_shifted[:, :response_len].contiguous()

                    print('prepare_grpo_inputs_time: ', timing_raw['prepare_grpo_inputs'], '| batch size: ', len(batch),
                          '| input_ids:', batch.batch["input_ids"].shape,
                          '| responses:', batch.batch["responses"].shape,
                          '| response_mask:', batch.batch["response_mask"].shape)

                    # Skip update when batch has no valid tokens (e.g. remote reset returned obs_messages=None)
                    num_valid_tokens = batch.batch["attention_mask"].sum().item()
                    if num_valid_tokens == 0 or batch.batch["input_ids"].size(-1) == 0:
                        _t = getattr(self, "_last_remote_fail_log", 0)
                        if _t == 0 or time.time() - _t >= 30:
                            print(
                                "Skipping PPO update: batch has no valid tokens (remote env 503?). "
                                "Check remote server; start server on EC2 if needed. (rate-limited 30s)"
                            )
                            self._last_remote_fail_log = time.time()
                        continue

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()


                    # compute reward — paper-style: r = r_task + r_format
                    # r_task = eval_results (0 or 1), r_format = -1 if parse failure, 0 if valid
                    # Reference: ARPO paper Section 4.2 (Reward Design)
                    with _timer("reward", timing_raw):
                        # Convert accumulated server format_rewards to binary paper-style:
                        # negative accumulated value → had parse failures → -1; otherwise → 0
                        raw_fmt = batch.batch["format_rewards"]
                        format_rewards_clipped = torch.where(
                            raw_fmt < -0.01,
                            torch.tensor(-1.0, device=raw_fmt.device, dtype=raw_fmt.dtype),
                            torch.tensor(0.0, device=raw_fmt.device, dtype=raw_fmt.dtype),
                        )
                        rewards = batch.batch["eval_results"] + format_rewards_clipped
                        # Guard against NaN/Inf from corrupt evaluator results — replace with 0.0
                        if not torch.isfinite(rewards).all():
                            nan_count = (~torch.isfinite(rewards)).sum().item()
                            print(f"[WARNING] {nan_count} NaN/Inf rewards detected — replacing with 0.0")
                            rewards = torch.where(torch.isfinite(rewards), rewards, torch.zeros_like(rewards))
                        batch.batch["rewards"] = rewards

                        if self.use_reward_model:
                            raise NotImplementedError("Reward model is not supported yet.")

                        task_id_set = set(batch.non_tensor_batch["task_id"])
                        valid_task_id_set = set()

                        reward_stds_list = []
                        reward_n_per_task = []
                        for task_id in task_id_set:
                            reward_in_group = batch.batch["rewards"][batch.non_tensor_batch["task_id"] == task_id]
                            n = reward_in_group.numel()
                            reward_n_per_task.append(n)
                            # std undefined for 0 or 1 element; use 0.0 to avoid nan and UserWarning
                            reward_std = 0.0 if n <= 1 else reward_in_group.std().item()
                            reward_stds_list.append(reward_std)

                        # With 1 env/task we have 1 sample per task → reward_std=0 by definition; don't mark as invalid.
                        # Only mark invalid when we have multiple samples per task and std < 0.01 (no variance).
                        num_invalid_group = sum(
                            1 for i, x_std in enumerate(reward_stds_list)
                            if reward_n_per_task[i] > 1 and x_std < 0.01
                        )
                        print(
                            f"num_invalid_group: {num_invalid_group}/{len(reward_stds_list)} "
                            f"(n_per_task: {reward_n_per_task}) | reward_stds_list: {reward_stds_list}"
                        )

                        # we combine with rule-based rm
                        reward_tensor = batch.batch["rewards"]
                        reward_metrics = {
                            "reward_tensor": reward_tensor.tolist(),
                            "reward_std": reward_stds_list,
                            'num_invalid_group': num_invalid_group,
                            'traj_reward': all_eval_results,
                            'format_reward': all_format_rewards,
                            'format_reward_clipped': format_rewards_clipped.tolist(),
                        }

                        batch.batch["token_level_scores"] = reward_tensor.unsqueeze(-1)
                        reward_metrics = {
                            f"reward/{key}": value for key, value in reduce_metrics(reward_metrics).items()
                        }
                        metrics.update(reward_metrics)

                        rollout_n = self.config.worker.rollout.n
                        eval_results_global_np = reshape_rollout_metric(batch.batch["eval_results"], rollout_n)
                        format_rewards_np = reshape_rollout_metric(batch.batch["format_rewards"], rollout_n)
                        format_rewards_clipped_np = reshape_rollout_metric(format_rewards_clipped, rollout_n)
                        reward_tensor_np = reshape_rollout_metric(reward_tensor, rollout_n)
                        print(
                            f'Evaluation results:\n{eval_results_global_np}\n'
                            f'Format rewards (raw):\n{format_rewards_np}\n'
                            f'Format rewards (clipped):\n{format_rewards_clipped_np}\n'
                            f'Final rewards:\n{reward_tensor_np}'
                        )
                        print('Global eval_results: ', sum(reward_tensor.tolist())/len(batch))
                    

                    # ARPO = GRPO advantage + no KL (disable_kl) + experience replay. GRPO needs rollout.n>1 samples per task.
                    # For remote-env smoke tests we may have very small batches (e.g. 1 env, rollout.n=2) while world_size=4 FSDP ranks.
                    # Instead of skipping PPO when batch < world_size, rely on padding to safely distribute work across ranks.
                    num_dp_workers = self.actor_rollout_wg.world_size
                    if len(batch) == 0:
                        print("Skipping PPO update this step (empty batch).")
                        continue

                    # Pad batch so it is divisible by actor world_size (required for chunking across workers)
                    batch, update_pad_size = pad_dataproto_to_divisor(batch, num_dp_workers)

                    # recompute old_log_probs
                    with _timer("old", timing_raw):
                        old_log_probs, old_log_prob_stats = get_old_log_probs_with_fallback(
                            batch=batch,
                            source=self.config.worker.rollout.old_logprob_source,
                            recompute_fn=lambda: self.actor_rollout_wg.compute_log_probs(batch),
                        )
                        # Ensure old_log_probs tensor matches response_mask length (truncate if needed)
                        response_len = batch.batch["response_mask"].size(1)
                        t = old_log_probs.batch["old_log_probs"]
                        if t.size(1) != response_len:
                            old_log_probs.batch["old_log_probs"] = t[:, :response_len].contiguous()
                        batch = batch.union(old_log_probs)
                        metrics.update(
                            {
                                f"perf/{key}": value
                                for key, value in old_log_prob_stats.items()
                                if key != "rollout_old_logprob_error"
                            }
                        )
                        if "rollout_old_logprob_error" in old_log_prob_stats:
                            print(f"rollout_old_logprob_fallback: {old_log_prob_stats['rollout_old_logprob_error']}")

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                            # Ensure ref_log_probs tensor matches response_mask length (truncate if needed)
                            response_len = batch.batch["response_mask"].size(1)
                            t = ref_log_probs.batch["ref_log_probs"]
                            if t.size(1) != response_len:
                                ref_log_probs.batch["ref_log_probs"] = t[:, :response_len].contiguous()
                            batch = batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            # Ensure values tensor matches response_mask length (truncate if needed)
                            response_len = batch.batch["response_mask"].size(1)
                            t = values.batch["values"]
                            if t.size(1) != response_len:
                                values.batch["values"] = t[:, :response_len].contiguous()
                            batch = batch.union(values)

                    if update_pad_size > 0:
                        batch = unpad_dataproto(batch, pad_size=update_pad_size)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    with _timer("adv", timing_raw):
                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                        # Clamp extreme advantages (e.g., from GRPO with zero-variance groups)
                        adv_raw = batch.batch["advantages"]
                        if not torch.isfinite(adv_raw).all() or adv_raw.abs().max().item() > 10.0:
                            bad_count = (~torch.isfinite(adv_raw)).sum().item()
                            extreme_count = (adv_raw.abs() > 10.0).sum().item()
                            if bad_count > 0:
                                print(f"[WARNING] {bad_count} NaN/Inf advantages — replacing with 0.0")
                            if extreme_count > 0:
                                print(f"[WARNING] {extreme_count} extreme advantages (|adv|>10) — clamping to [-10, 10]")
                            adv_raw = torch.where(torch.isfinite(adv_raw), adv_raw, torch.zeros_like(adv_raw))
                            batch.batch["advantages"] = adv_raw.clamp(-10.0, 10.0)
                        # Log advantage stats to confirm non-zero and varying (needed for GRPO)
                        adv = batch.batch["advantages"]
                        resp_mask = batch.batch["response_mask"].bool()
                        valid_adv = torch.masked_select(adv, resp_mask) if resp_mask.any() else adv.flatten()
                        if valid_adv.numel() > 0:
                            a_mean, a_max, a_min = valid_adv.mean().item(), valid_adv.max().item(), valid_adv.min().item()
                            rew = batch.batch.get("rewards")
                            r_std = rew.std().item() if rew is not None and rew.numel() > 1 else 0.0
                            print(f"advantages: mean={a_mean:.4f} max={a_max:.4f} min={a_min:.4f} | reward_std={r_std:.4f}")
                        else:
                            print("advantages: (no valid tokens)")

                    # update critic (pad batch if needed for DP chunking)
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            if len(batch) % num_dp_workers != 0:
                                batch_dp, _ = pad_dataproto_to_divisor(batch, num_dp_workers)
                                critic_output = self.critic_wg.update_critic(batch_dp)
                            else:
                                critic_output = self.critic_wg.update_critic(batch)
                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)

                    # update actor (pad batch if needed for DP chunking)
                    if self.config.trainer.critic_warmup <= self.global_step:
                        with _timer("update_actor", timing_raw):
                            if len(batch) % num_dp_workers != 0:
                                batch_dp, _ = pad_dataproto_to_divisor(batch, num_dp_workers)
                                actor_output = self.actor_rollout_wg.update_actor(batch_dp)
                            else:
                                actor_output = self.actor_rollout_wg.update_actor(batch)

                        # Extract per-GPU peak VRAM before reduce_metrics averages it
                        per_gpu_peaks_raw = actor_output.non_tensor_batch.get("perf/gpu_peak_vram_gb")
                        if per_gpu_peaks_raw is not None:
                            metrics["perf/per_gpu_peak_vram_gb"] = per_gpu_peaks_raw.tolist() if hasattr(per_gpu_peaks_raw, 'tolist') else list(per_gpu_peaks_raw)
                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                    # validate
                    if (
                        self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                    ):
                        # Validation resets env_workers to different tasks, making any
                        # in-flight prefetch results invalid. Save the batch, discard
                        # stale futures (GC'd by Ray), then re-launch after validation.
                        saved_prefetch_batch = prefetched[0] if prefetched else None
                        prefetched = None

                        with _timer("validation", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)

                        # Re-launch prefetch resets after validation (env workers are free)
                        if saved_prefetch_batch is not None:
                            try:
                                prefetched = self._launch_prefetch_resets(saved_prefetch_batch)
                            except Exception as e:
                                print(f"[prefetch] failed to re-launch resets after validation ({e!r}); will reset synchronously")
                                prefetched = (saved_prefetch_batch, self._build_rollout_chunks(saved_prefetch_batch), None)

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                metrics.update(self._collect_noise_metrics())

                self.logger.log(data=metrics, step=self.global_step)

                # --- Detailed step summary (terminal + JSONL log) ---
                try:
                    rollout_n = self.config.worker.rollout.n
                    task_ids = list(batch.non_tensor_batch.get("task_id", []))
                    unique_tasks = list(dict.fromkeys(task_ids))  # preserve order, deduplicate
                    eval_results_t = batch.batch.get("eval_results")
                    format_rewards_t = batch.batch.get("format_rewards")
                    rewards_t = batch.batch.get("rewards")
                    prompt_lens = batch.batch["attention_mask"].sum(dim=-1).tolist()

                    # Build per-task summary
                    per_task = []
                    for tid in unique_tasks:
                        mask = np.array(task_ids) == tid
                        idxs = np.where(mask)[0]
                        t_eval = [float(eval_results_t[i]) for i in idxs] if eval_results_t is not None else []
                        t_fmt = [float(format_rewards_t[i]) for i in idxs] if format_rewards_t is not None else []
                        t_rew = [float(rewards_t[i]) for i in idxs] if rewards_t is not None else []
                        t_plen = [int(prompt_lens[i]) for i in idxs]
                        instr = ""
                        for tc in batch_dict:
                            if tc.get("id") == tid:
                                instr = tc.get("instruction", "")[:80]
                                break
                        per_task.append({
                            "task_id": str(tid),
                            "instruction": instr,
                            "n_rollouts": len(idxs),
                            "eval_scores": t_eval,
                            "format_rewards": t_fmt,
                            "final_rewards": t_rew,
                            "prompt_lengths": t_plen,
                        })

                    # Aggregate realized noise burden for observability.
                    step_metadata_entries = list(batch.non_tensor_batch.get("rollout_step_metadata", []))
                    noise_summary = {
                        "trajectories_with_noise": 0,
                        "interruptions": 0.0,
                        "recovery_cost": 0.0,
                        "occlusion": 0.0,
                        "focus_loss": 0.0,
                    }
                    for step_meta in step_metadata_entries:
                        burden = self._summarize_noise_burden(step_meta)
                        if any(v > 0 for v in burden.values()):
                            noise_summary["trajectories_with_noise"] += 1
                        noise_summary["interruptions"] += burden["interruptions"]
                        noise_summary["recovery_cost"] += burden["recovery_cost"]
                        noise_summary["occlusion"] += burden["occlusion"]
                        noise_summary["focus_loss"] += burden["focus_loss"]

                    # Print per-task summary
                    print(f"\n{'='*80}")
                    print(f"  Step {self.global_step}  |  {datetime.datetime.now().strftime('%H:%M:%S')}  |  "
                          f"{len(unique_tasks)} tasks × {rollout_n} rollouts = {len(batch)} trajectories")
                    print(f"{'='*80}")
                    for t in per_task:
                        avg_eval = sum(t["eval_scores"]) / max(len(t["eval_scores"]), 1)
                        avg_fmt = sum(t["format_rewards"]) / max(len(t["format_rewards"]), 1)
                        avg_plen = sum(t["prompt_lengths"]) / max(len(t["prompt_lengths"]), 1)
                        print(f"  {t['task_id'][:12]:12s}  eval={avg_eval:.3f}  fmt_r={avg_fmt:.3f}  "
                              f"plen={avg_plen:.0f}  | {t['instruction']}")
                    print(f"  Timing: " + "  ".join(f"{k}={v:.1f}s" for k, v in timing_raw.items()))
                    per_gpu_peaks = metrics.get("perf/per_gpu_peak_vram_gb")
                    if per_gpu_peaks is not None:
                        if isinstance(per_gpu_peaks, (list, np.ndarray)):
                            print(f"  VRAM peak: " + " | ".join(f"GPU{i}={v:.1f}G" for i, v in enumerate(per_gpu_peaks))
                                  + f"  MAX={max(per_gpu_peaks):.1f}G")
                    print(f"  pg_loss={metrics.get('actor/pg_loss', 'n/a')}  "
                          f"grad_norm={metrics.get('actor/grad_norm', 'n/a')}  "
                          f"entropy={metrics.get('actor/entropy_loss', 'n/a')}")
                    print(f"  noise_fired={noise_summary['trajectories_with_noise']}/{len(batch)}  "
                          f"interruptions={noise_summary['interruptions']:.1f}  "
                          f"recovery_cost={noise_summary['recovery_cost']:.1f}  "
                          f"occlusion={noise_summary['occlusion']:.1f}  "
                          f"focus_loss={noise_summary['focus_loss']:.1f}")
                    print(f"{'='*80}\n")

                    # Write JSONL log entry
                    def _json_safe(v):
                        if isinstance(v, (np.ndarray, np.generic)):
                            return v.tolist()
                        if isinstance(v, torch.Tensor):
                            return v.tolist()
                        return v

                    log_entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "global_step": self.global_step,
                        "num_trajectories": len(batch),
                        "per_task": per_task,
                        "timing_s": {k: round(v, 3) for k, v in timing_raw.items()},
                        "per_gpu_peak_vram_gb": _json_safe(per_gpu_peaks) if per_gpu_peaks is not None else None,
                        "max_gpu_vram_gb": round(max(per_gpu_peaks), 2) if per_gpu_peaks is not None and isinstance(per_gpu_peaks, (list, np.ndarray)) else None,
                        "pg_loss": _json_safe(metrics.get("actor/pg_loss")),
                        "grad_norm": _json_safe(metrics.get("actor/grad_norm")),
                        "entropy_loss": _json_safe(metrics.get("actor/entropy_loss")),
                        "prompt_length_mean": _json_safe(metrics.get("prompt_length/mean")),
                        "prompt_length_max": _json_safe(metrics.get("prompt_length/max")),
                        "response_length_mean": _json_safe(metrics.get("response_length/mean")),
                        "throughput": _json_safe(metrics.get("perf/throughput")),
                        "noise_summary": {k: _json_safe(v) for k, v in noise_summary.items()},
                    }
                    with open(self._training_log_path, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                except Exception as e:
                    print(f"[WARN] step logging failed: {e}")

        # perform validation after training (only when val_freq is enabled)
        if self.config.trainer.val_freq > 0 and (
            val_metrics is None
            or self.global_step % self.config.trainer.val_freq != 0
        ):
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)

        if val_metrics is not None:
            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()

        self.logger.finish()
