import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.trainer.ray_trainer import reshape_rollout_metric


class TestRolloutLogging(unittest.TestCase):
    def test_reshape_rollout_metric_groups_by_rollout_n(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        reshaped = reshape_rollout_metric(values, rollout_n=2)
        self.assertEqual(tuple(reshaped.shape), (2, 2))
        self.assertTrue(torch.equal(reshaped, torch.tensor([[1.0, 2.0], [3.0, 4.0]])))

    def test_reshape_rollout_metric_falls_back_to_column_when_not_divisible(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        reshaped = reshape_rollout_metric(values, rollout_n=2)
        self.assertEqual(tuple(reshaped.shape), (3, 1))
        self.assertTrue(torch.equal(reshaped.squeeze(-1), values))


if __name__ == "__main__":
    unittest.main()
