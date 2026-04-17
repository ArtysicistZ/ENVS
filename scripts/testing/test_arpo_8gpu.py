"""
ARPO 8-GPU Pipeline Benchmark Test
===================================
Tests the full GPU computation path (model init, compute_log_probs, update_actor)
with synthetic data on 8 GPUs. No remote env servers needed.

Validates:
1. Ray cluster + 8 GPU resource allocation
2. Model loading + FSDP sharding on all 8 GPUs
3. vLLM engine initialization on all 8 GPUs
4. compute_log_probs timing (forward pass)
5. update_actor timing (forward + backward + optimizer step)
6. Memory usage per GPU
7. Performance: GPU computation <= 40s for batch_size=96

Usage:
    python scripts/testing/test_arpo_8gpu.py
"""

import json
import os
import sys
import time

# Add OSWorld to path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_osworld = os.path.join(_repo_root, "OSWorld")
if os.path.isdir(_osworld) and _osworld not in sys.path:
    sys.path.insert(0, _osworld)
os.chdir(_repo_root)

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import PPOConfig
from verl.trainer.ray_trainer import Role, ResourcePoolManager
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
from verl.workers.fsdp_workers import FSDPWorker


def create_synthetic_batch(batch_size: int, seq_len: int, pad_token_id: int = 151643) -> DataProto:
    """Create a synthetic DataProto batch that mimics real ARPO training data."""
    # Simulate variable-length sequences with left-padding
    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    position_ids = torch.zeros((batch_size, 3, seq_len), dtype=torch.long)

    for i in range(batch_size):
        # Random actual length between 1000 and 8000 tokens (typical ARPO episode)
        actual_len = min(torch.randint(1000, 8000, (1,)).item(), seq_len)
        start = seq_len - actual_len

        # Fill non-padding region
        input_ids[i, start:] = torch.randint(0, 150000, (actual_len,))
        attention_mask[i, start:] = 1
        # Labels: last 200 tokens are response (action text)
        response_start = max(start, seq_len - 200)
        labels[i, response_start:] = input_ids[i, response_start:]
        # Position IDs (simplified)
        position_ids[i, 0, start:] = torch.arange(actual_len)

    # responses = shifted input_ids (autoregressive)
    responses = input_ids[:, 1:]
    response_len = responses.size(1)
    labels_shifted = (labels != -100)[:, 1:]
    response_mask = labels_shifted[:, :response_len].contiguous()

    # Token-level scores and rewards
    rewards = torch.zeros(batch_size, dtype=torch.float32)
    # Give ~20% of samples a positive reward (realistic ARPO success rate)
    n_positive = max(1, batch_size // 5)
    rewards[:n_positive] = 1.0

    token_level_scores = rewards.unsqueeze(-1)  # (B, 1)
    token_level_rewards = token_level_scores.clone()

    # Advantages: simulate GRPO-style
    advantages = torch.randn(batch_size, response_len)
    advantages = advantages * response_mask.float()
    returns = advantages.clone()

    # old_log_probs placeholder
    old_log_probs = torch.randn(batch_size, response_len) * 0.1

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_ids": position_ids,
        "responses": responses,
        "response_mask": response_mask,
        "rewards": rewards,
        "token_level_scores": token_level_scores,
        "token_level_rewards": token_level_rewards,
        "advantages": advantages,
        "returns": returns,
        "old_log_probs": old_log_probs,
    }

    from tensordict import TensorDict
    td = TensorDict(batch_dict, batch_size=[batch_size])
    batch = DataProto(batch=td)

    # uid for GRPO grouping
    uids = np.array([f"task_{i // 8}" for i in range(batch_size)], dtype=object)
    batch.non_tensor_batch["uid"] = uids
    batch.non_tensor_batch["task_id"] = uids

    # global_token_num for MFU computation
    batch.meta_info["global_token_num"] = attention_mask.sum(dim=-1).tolist()
    batch.meta_info["temperature"] = 1.0

    return batch


@ray.remote(num_cpus=1)
class BenchmarkRunner:
    """Runs the benchmark inside a Ray actor to match the real training pattern."""

    def run(self, config_path: str, batch_sizes: list):
        config = self._load_config(config_path)
        config.deep_post_init()

        from verl.utils.tokenizer import get_processor, get_tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            trust_remote_code=True, use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            trust_remote_code=True, use_fast=True,
        )

        # Set up resource pool (8 GPUs, single pool)
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
        }
        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {Role.ActorRollout: "global_pool"}
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping,
        )

        print("\n" + "=" * 70)
        print("  ARPO 8-GPU Pipeline Benchmark")
        print("=" * 70)

        # Step 1: Initialize workers
        print("\n[1/4] Initializing Ray resource pool and worker group...")
        t0 = time.time()
        resource_pool_manager.create_resource_pool()
        resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=role_worker_mapping[Role.ActorRollout],
            config=config.worker, role="actor_rollout",
        )
        class_dict = {"actor_rollout": actor_rollout_cls}
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        actor_rollout_wg = spawn_wg["actor_rollout"]
        print(f"    Workers spawned: {time.time() - t0:.1f}s")

        # Step 2: Initialize model (FSDP + vLLM)
        print("\n[2/4] Initializing model (FSDP + vLLM on 8 GPUs)...")
        t0 = time.time()
        actor_rollout_wg.init_model()
        init_time = time.time() - t0
        print(f"    Model initialized: {init_time:.1f}s")

        world_size = actor_rollout_wg.world_size
        print(f"    World size: {world_size}")
        pad_token_id = getattr(tokenizer, "pad_token_id", 0)

        # Step 3: Test vLLM generation (prepare + generate + finish)
        print("\n[3/4] Testing vLLM generation engine...")
        t0 = time.time()
        actor_rollout_wg.prepare_generate_sequences()
        prepare_time = time.time() - t0
        print(f"    vLLM prepare: {prepare_time:.1f}s")

        # Create a small synthetic prompt for generation test
        prompt_len = 2000
        gen_input_ids = torch.randint(0, 150000, (world_size, prompt_len))
        gen_attention_mask = torch.ones(world_size, prompt_len, dtype=torch.long)
        gen_position_ids = torch.zeros(world_size, 3, prompt_len, dtype=torch.long)
        for i in range(world_size):
            gen_position_ids[i, 0] = torch.arange(prompt_len)

        from tensordict import TensorDict
        gen_td = TensorDict({
            "input_ids": gen_input_ids,
            "attention_mask": gen_attention_mask,
            "position_ids": gen_position_ids,
        }, batch_size=[world_size])
        gen_batch = DataProto(batch=gen_td)
        gen_batch.non_tensor_batch["raw_prompt_ids"] = np.array(
            [gen_input_ids[i].numpy() for i in range(world_size)], dtype=object
        )
        gen_batch.meta_info = {
            "temperature": 1.0,
            "n": 1,
        }

        t0 = time.time()
        gen_output = actor_rollout_wg.generate_sequences(gen_batch)
        gen_time = time.time() - t0
        print(f"    vLLM generate ({world_size} seqs, {prompt_len} tokens): {gen_time:.1f}s")

        actor_rollout_wg.finish_generate_sequences()
        print(f"    vLLM engine OK")

        # Step 4: Benchmark compute_log_probs + update_actor
        print("\n[4/4] Benchmarking GPU training computation...")
        results = {}

        for batch_size in batch_sizes:
            seq_len = 4096  # typical ARPO episode length (max_prompt_length=12000, limit_images=3)
            print(f"\n  --- Batch size = {batch_size}, seq_len = {seq_len} ---")
            batch = create_synthetic_batch(batch_size, seq_len, pad_token_id)

            # Pad to world_size
            batch_padded, pad_size = pad_dataproto_to_divisor(batch, world_size)
            actual_batch_size = len(batch_padded)
            print(f"    Padded batch size: {actual_batch_size} (pad={pad_size})")
            print(f"    Samples per GPU: {actual_batch_size // world_size}")
            total_tokens = batch_padded.batch["attention_mask"].sum().item()
            print(f"    Total non-padding tokens: {total_tokens:,}")

            # --- compute_log_probs ---
            t0 = time.time()
            old_log_probs_output = actor_rollout_wg.compute_log_probs(batch_padded)
            logprob_time = time.time() - t0
            print(f"    compute_log_probs: {logprob_time:.1f}s")

            # Replace old_log_probs with recomputed values
            response_len = batch_padded.batch["response_mask"].size(1)
            t = old_log_probs_output.batch["old_log_probs"]
            if t.size(1) != response_len:
                t = t[:, :response_len].contiguous()
            batch_padded.batch["old_log_probs"] = t

            # Balance batch tokens across DP ranks
            attention_mask = batch_padded.batch["attention_mask"]
            bs = attention_mask.shape[0]
            global_seqlen_lst = attention_mask.view(bs, -1).sum(-1).tolist()
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
            global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
            batch_padded.reorder(global_idx)

            # --- update_actor ---
            t0 = time.time()
            actor_output = actor_rollout_wg.update_actor(batch_padded)
            update_time = time.time() - t0
            print(f"    update_actor: {update_time:.1f}s")

            total_gpu_time = logprob_time + update_time
            print(f"    TOTAL GPU TIME: {total_gpu_time:.1f}s")

            # Extract metrics
            pg_loss = actor_output.non_tensor_batch.get("actor/pg_loss")
            grad_norm = actor_output.non_tensor_batch.get("actor/grad_norm")
            peak_vram = actor_output.non_tensor_batch.get("perf/gpu_peak_vram_gb")
            print(f"    pg_loss: {pg_loss}")
            print(f"    grad_norm: {grad_norm}")
            if peak_vram is not None:
                if hasattr(peak_vram, 'tolist'):
                    peaks = peak_vram.tolist()
                else:
                    peaks = list(peak_vram)
                print(f"    GPU peak VRAM: {' | '.join(f'GPU{i}={v:.1f}G' for i, v in enumerate(peaks))}")
                print(f"    Max VRAM: {max(peaks):.1f}G / 80G")

            passed = total_gpu_time <= 40.0
            results[batch_size] = {
                "logprob_time": logprob_time,
                "update_time": update_time,
                "total_gpu_time": total_gpu_time,
                "passed": passed,
                "peak_vram": max(peaks) if peak_vram is not None else None,
            }

            status = "PASS" if passed else "FAIL"
            print(f"    Performance check (<= 40s): {status}")

        # Summary
        print("\n" + "=" * 70)
        print("  BENCHMARK SUMMARY")
        print("=" * 70)
        all_passed = True
        for bs, r in results.items():
            status = "PASS" if r["passed"] else "FAIL"
            if not r["passed"]:
                all_passed = False
            vram_str = f"{r['peak_vram']:.1f}G" if r["peak_vram"] else "N/A"
            print(
                f"  batch_size={bs:3d}: "
                f"logprob={r['logprob_time']:.1f}s + update={r['update_time']:.1f}s = "
                f"total={r['total_gpu_time']:.1f}s  VRAM={vram_str}  [{status}]"
            )

        if all_passed:
            print("\n  ALL TESTS PASSED - GPU pipeline ready for ARPO training!")
        else:
            print("\n  SOME TESTS FAILED - need to tune config (reduce batch/seq or increase parallelism)")
        print("=" * 70 + "\n")

        # Keep references alive to avoid segfault during cleanup
        self._wg_dict = wg_dict
        self._actor_rollout_wg = actor_rollout_wg

        return results

    def _load_config(self, config_path: str):
        default_config = OmegaConf.structured(PPOConfig())
        file_config = OmegaConf.load(config_path)
        merged = OmegaConf.merge(default_config, file_config)
        config = OmegaConf.to_object(merged)

        # Override for benchmark: no wandb, short training
        config.trainer.logger = ("console",)
        config.trainer.total_episodes = 1
        config.trainer.val_before_train = False

        # Resolve paths
        if hasattr(config.data, "train_files") and config.data.train_files and not os.path.isabs(config.data.train_files):
            config.data.train_files = os.path.abspath(config.data.train_files)
        if hasattr(config.data, "val_files") and config.data.val_files and not os.path.isabs(config.data.val_files):
            config.data.val_files = os.path.abspath(config.data.val_files)

        return config


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/arpo_smoke_8gpu.yaml"

    # Kill any existing Ray cluster
    for key in list(os.environ):
        if key.startswith("RAY_"):
            os.environ.pop(key, None)

    if not ray.is_initialized():
        ray.init(
            address="local",
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }},
        )

    print(f"Ray cluster resources: {ray.cluster_resources()}")

    runner = BenchmarkRunner.remote()
    _exit_code = 0
    try:
        # Test with batch sizes 64 and 96 (production ARPO training uses 96)
        results = ray.get(runner.run.remote(config_path, [64, 96]))
        if all(r["passed"] for r in results.values()):
            print("Benchmark PASSED!")
        else:
            print("Benchmark FAILED - some batch sizes exceeded 40s limit")
            _exit_code = 1
    except Exception as e:
        print(f"Benchmark CRASHED: {e}")
        import traceback
        traceback.print_exc()
        _exit_code = 1

    os._exit(_exit_code)


if __name__ == "__main__":
    main()
