import os
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.workers.rollout.config import RolloutConfig
from verl.workers.rollout.vllm_rollout_spmd import vLLMRollout


class DummyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        self.last_encoded = text
        return [901, 902]


class DummySamplingParams:
    def __init__(self, **kwargs):
        self.n = kwargs.get("n", 1)
        self.logprobs = kwargs.get("logprobs")
        self.prompt_logprobs = kwargs.get("prompt_logprobs")
        for key, value in kwargs.items():
            setattr(self, key, value)


class DummyLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def sleep(self, level=1):
        self.sleep_level = level


class TestVLLMRolloutConfig(unittest.TestCase):
    @patch("verl.workers.rollout.vllm_rollout_spmd.torch.distributed.get_world_size", return_value=1)
    @patch("verl.workers.rollout.vllm_rollout_spmd.SamplingParams", DummySamplingParams)
    @patch("verl.workers.rollout.vllm_rollout_spmd.LLM", DummyLLM)
    def test_rollout_init_passes_memory_caps_and_logprob_knobs(self, _world_size):
        config = RolloutConfig()
        config.prompt_length = 128
        config.response_length = 32
        config.tensor_parallel_size = 1
        config.max_num_batched_tokens = 256
        config.max_num_seqs = 8
        config.kv_cache_memory_bytes = 123456
        config.old_logprob_source = "auto"
        rollout = vLLMRollout(model_path="dummy-model", config=config, tokenizer=DummyTokenizer())

        llm_kwargs = rollout.inference_engine.kwargs
        self.assertEqual(llm_kwargs["max_num_batched_tokens"], 256)
        self.assertEqual(llm_kwargs["max_num_seqs"], 8)
        self.assertEqual(llm_kwargs["kv_cache_memory_bytes"], 123456)
        self.assertNotIn("gpu_memory_utilization", llm_kwargs)
        self.assertEqual(rollout.sampling_params.logprobs, 1)
        self.assertEqual(rollout.sampling_params.prompt_logprobs, 1)
        self.assertEqual(rollout.sampling_params.n, 1)

    @patch("verl.workers.rollout.vllm_rollout_spmd.torch.distributed.get_world_size", return_value=1)
    @patch("verl.workers.rollout.vllm_rollout_spmd.SamplingParams", DummySamplingParams)
    @patch("verl.workers.rollout.vllm_rollout_spmd.LLM", DummyLLM)
    def test_recompute_mode_keeps_rollout_logprob_collection_off(self, _world_size):
        config = RolloutConfig()
        config.prompt_length = 64
        config.response_length = 16
        config.tensor_parallel_size = 1
        config.old_logprob_source = "recompute"
        rollout = vLLMRollout(model_path="dummy-model", config=config, tokenizer=DummyTokenizer())

        llm_kwargs = rollout.inference_engine.kwargs
        self.assertEqual(llm_kwargs["gpu_memory_utilization"], config.gpu_memory_utilization)
        self.assertIsNone(rollout.sampling_params.logprobs)
        self.assertIsNone(rollout.sampling_params.prompt_logprobs)


if __name__ == "__main__":
    unittest.main()
