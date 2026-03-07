import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.workers.config import WorkerConfig


class TestWorkerConfig(unittest.TestCase):
    def test_critic_defaults_inherit_actor_when_model_path_missing(self):
        config = WorkerConfig()
        config.actor.model.model_path = "actor-model"
        config.actor.padding_free = True
        config.actor.ulysses_sequence_parallel_size = 8
        config.actor.global_batch_size = 4
        config.actor.micro_batch_size_per_device_for_update = 1
        config.actor.micro_batch_size_per_device_for_experience = 1
        config.actor.optim.strategy = "adamw_bf16"
        config.actor.offload.offload_optimizer = True

        config.post_init()

        self.assertEqual(config.critic.model.model_path, "actor-model")
        self.assertEqual(config.critic.model.tokenizer_path, "actor-model")
        self.assertTrue(config.critic.padding_free)
        self.assertEqual(config.critic.ulysses_sequence_parallel_size, 8)
        self.assertEqual(config.critic.global_batch_size, 4)
        self.assertEqual(config.critic.micro_batch_size_per_device_for_update, 1)
        self.assertEqual(config.critic.micro_batch_size_per_device_for_experience, 1)
        self.assertEqual(config.critic.optim.strategy, "adamw_bf16")
        self.assertTrue(config.critic.offload.offload_optimizer)

    def test_explicit_critic_model_path_is_not_overwritten(self):
        config = WorkerConfig()
        config.actor.model.model_path = "actor-model"
        config.critic.model.model_path = "critic-model"
        config.critic.padding_free = False
        config.critic.ulysses_sequence_parallel_size = 1

        config.post_init()

        self.assertEqual(config.critic.model.model_path, "critic-model")
        self.assertFalse(config.critic.padding_free)
        self.assertEqual(config.critic.ulysses_sequence_parallel_size, 1)


if __name__ == "__main__":
    unittest.main()
