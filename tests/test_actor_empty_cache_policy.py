import os
import sys
import types
import unittest
from unittest.mock import patch

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.protocol import DataProto
from verl.workers.actor.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor


class TestActorEmptyCachePolicy(unittest.TestCase):
    def _build_actor(self, empty_cache_policy: str) -> DataParallelPPOActor:
        config = ActorConfig()
        config.empty_cache_policy = empty_cache_policy
        config.micro_batch_size_per_device_for_update = 1
        config.global_batch_size_per_device = 1
        config.use_kl_loss = False
        actor_module = torch.nn.Linear(1, 1, bias=False)
        actor = DataParallelPPOActor(config=config, actor_module=actor_module, actor_optimizer=torch.optim.SGD(actor_module.parameters(), lr=0.01))

        def fake_forward(self, model_inputs, temperature):
            response_len = model_inputs["responses"].size(1)
            base = self.actor_module.weight.sum()
            return base * torch.ones((model_inputs["responses"].size(0), response_len), dtype=torch.float32)

        actor._forward_micro_batch = types.MethodType(fake_forward, actor)
        return actor

    def _build_batch(self) -> DataProto:
        return DataProto.from_dict(
            tensors={
                "responses": torch.tensor([[3, 4]], dtype=torch.long),
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
                "old_log_probs": torch.zeros((1, 2), dtype=torch.float32),
                "advantages": torch.ones((1, 2), dtype=torch.float32),
                "response_mask": torch.ones((1, 2), dtype=torch.long),
            },
            meta_info={"temperature": 1.0, "global_token_num": [3]},
        )

    def test_boundary_only_skips_per_microbatch_empty_cache(self):
        actor = self._build_actor("boundary_only")
        batch = self._build_batch()
        with patch("verl.workers.actor.dp_actor.torch.cuda.empty_cache") as empty_cache:
            actor.update_policy(batch)
        self.assertEqual(empty_cache.call_count, 0)

    def test_aggressive_keeps_per_microbatch_empty_cache(self):
        actor = self._build_actor("aggressive")
        batch = self._build_batch()
        with patch("verl.workers.actor.dp_actor.torch.cuda.empty_cache") as empty_cache:
            actor.update_policy(batch)
        self.assertGreaterEqual(empty_cache.call_count, 2)


if __name__ == "__main__":
    unittest.main()
