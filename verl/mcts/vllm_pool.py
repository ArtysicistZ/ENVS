"""Pool of vLLM engines across multiple GPUs for parallel generation.

Each GPU runs an independent vLLM instance with TP=1. The pool distributes
generation requests round-robin across GPUs, achieving N× throughput.

For a 7B model on 8× A100 80GB:
  - Model: ~16GB per GPU
  - KV cache: ~52GB per GPU
  - 8 GPUs process 8 prompts simultaneously
  - K=16 candidates: 2 per GPU → ~4s total (vs ~32s with TP=4 sequential)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import ray

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1, num_cpus=1)
class VLLMWorker:
    """A single vLLM engine on one GPU."""

    def __init__(self, gpu_id: int, model_path: str, gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 32768, limit_images: int = 3,
                 max_pixels: int = 2116800, min_pixels: int = 256):
        # Don't set CUDA_VISIBLE_DEVICES — Ray's num_gpus=1 handles GPU assignment
        from vllm import LLM
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": limit_images},
            mm_processor_kwargs={"max_pixels": max_pixels, "min_pixels": min_pixels},
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.gpu_id = gpu_id
        logger.info("VLLMWorker on GPU %d ready.", gpu_id)

    def generate(self, vllm_inputs: List[Dict], temperature: float = 1.0,
                 max_tokens: int = 512) -> List[str]:
        """Generate responses for a batch of inputs.

        Args:
            vllm_inputs: List of {"prompt_token_ids": [...], "multi_modal_data": {...}}
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            List of generated text strings (one per input)
        """
        from vllm import SamplingParams

        if not vllm_inputs:
            return []

        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params)
        return [out.outputs[0].text for out in outputs]

    def health_check(self) -> bool:
        return True


class VLLMPool:
    """Pool of vLLM workers across multiple GPUs.

    Distributes generation requests round-robin for maximum throughput.
    """

    def __init__(self, n_gpus: int, model_path: str, gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 32768, limit_images: int = 3,
                 max_pixels: int = 2116800, min_pixels: int = 256):
        self.n_gpus = n_gpus
        self.workers: List[Any] = []

        logger.info("Creating %d VLLMWorker instances (TP=1 each)...", n_gpus)
        for gpu_id in range(n_gpus):
            w = VLLMWorker.remote(
                gpu_id=gpu_id,
                model_path=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                limit_images=limit_images,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
            )
            self.workers.append(w)

        # Wait for all workers to initialize
        health = ray.get([w.health_check.remote() for w in self.workers])
        assert all(health), f"Some VLLMWorkers failed to initialize: {health}"
        logger.info("All %d VLLMWorkers ready.", n_gpus)

    def generate_batch(self, vllm_inputs: List[Dict], temperature: float = 1.0,
                       max_tokens: int = 512) -> List[str]:
        """Generate responses for all inputs, distributed across GPUs.

        Args:
            vllm_inputs: List of N vllm input dicts
            temperature: Sampling temperature
            max_tokens: Max tokens per response

        Returns:
            List of N generated text strings (preserving input order)
        """
        n = len(vllm_inputs)
        if n == 0:
            return []

        # Distribute inputs round-robin across GPUs
        # GPU 0 gets inputs [0, 8, 16, ...], GPU 1 gets [1, 9, 17, ...], etc.
        per_gpu: List[List[Tuple[int, Dict]]] = [[] for _ in range(self.n_gpus)]
        for i, inp in enumerate(vllm_inputs):
            gpu_idx = i % self.n_gpus
            per_gpu[gpu_idx].append((i, inp))

        # Launch all GPU workers in parallel
        futures = []
        for gpu_idx, items in enumerate(per_gpu):
            if not items:
                continue
            worker_inputs = [inp for _, inp in items]
            future = self.workers[gpu_idx].generate.remote(
                worker_inputs, temperature=temperature, max_tokens=max_tokens,
            )
            futures.append((gpu_idx, items, future))

        # Collect results and reassemble in original order
        results = [None] * n
        for gpu_idx, items, future in futures:
            try:
                worker_results = ray.get(future, timeout=120)
                for (orig_idx, _), text in zip(items, worker_results):
                    results[orig_idx] = text
            except Exception as e:
                logger.error("VLLMWorker GPU %d failed: %s", gpu_idx, e)
                for orig_idx, _ in items:
                    results[orig_idx] = ""

        # Replace any None (shouldn't happen, but defensive)
        return [r if r is not None else "" for r in results]
