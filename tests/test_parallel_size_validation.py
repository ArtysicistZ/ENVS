import os
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.workers.fsdp_workers import FSDPWorker, _validate_head_parallel_size


class TestHeadParallelSizeValidation(unittest.TestCase):
    def test_accepts_parallel_size_that_divides_query_and_kv_heads(self):
        model_config = SimpleNamespace(
            model_type="qwen2_5_vl",
            num_attention_heads=28,
            num_key_value_heads=4,
        )

        _validate_head_parallel_size(model_config, parallel_size=4, parallelism_name="ulysses_sequence_parallel_size")

    def test_rejects_parallel_size_that_does_not_divide_heads(self):
        model_config = SimpleNamespace(
            model_type="qwen2_5_vl",
            num_attention_heads=28,
            num_key_value_heads=4,
        )

        with self.assertRaisesRegex(
            ValueError,
            "ulysses_sequence_parallel_size=8.*num_attention_heads=28.*num_key_value_heads=4",
        ):
            _validate_head_parallel_size(
                model_config,
                parallel_size=8,
                parallelism_name="ulysses_sequence_parallel_size",
            )

    def test_init_model_uses_role_specific_sequence_parallel_size(self):
        class DummyWorker:
            def __init__(self):
                self.role = "actor"
                self._is_actor = True
                self._is_critic = False
                self._is_ref = False
                self._is_rollout = False
                self._use_param_offload = False
                self._use_optimizer_offload = False
                self.fsdp_module = object()
                self.optimizer = object()
                self.recorded = None
                self.config = SimpleNamespace(
                    actor=SimpleNamespace(
                        model=SimpleNamespace(model_path="actor-model"),
                        fsdp=SimpleNamespace(),
                        optim=SimpleNamespace(),
                        padding_free=True,
                        ulysses_sequence_parallel_size=4,
                    ),
                    critic=SimpleNamespace(
                        model=SimpleNamespace(model_path="critic-model"),
                        fsdp=SimpleNamespace(),
                        optim=SimpleNamespace(),
                        padding_free=False,
                        ulysses_sequence_parallel_size=1,
                    ),
                    ref=SimpleNamespace(
                        fsdp=SimpleNamespace(),
                        padding_free=True,
                        ulysses_sequence_parallel_size=4,
                    ),
                )

            def _build_model_optimizer(self, **kwargs):
                self.recorded = kwargs
                self.model_config = SimpleNamespace()
                self.lr_scheduler = object()
                self.processor = None
                self.tokenizer = object()

        worker = DummyWorker()

        with patch("verl.workers.fsdp_workers.FlopsCounter", return_value=object()):
            with patch("verl.workers.fsdp_workers.FSDPCheckpointManager", return_value=object()):
                FSDPWorker.init_model(worker)

        self.assertEqual(worker.recorded["sequence_parallel_size"], 4)
        self.assertTrue(hasattr(worker, "actor"))


if __name__ == "__main__":
    unittest.main()
