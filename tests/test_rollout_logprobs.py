import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.protocol import DataProto
from verl.utils.rollout_logprobs import (
    RolloutLogProbAlignmentError,
    assemble_old_log_probs_from_sample,
    build_rollout_step_metadata,
    get_old_log_probs_with_fallback,
)


def _lp(value: float):
    return SimpleNamespace(logprob=value)


class TestRolloutLogProbs(unittest.TestCase):
    def test_build_rollout_step_metadata_uses_prompt_suffix_and_response(self):
        assistant_prefix = [101, 102]
        metadata = build_rollout_step_metadata(
            prompt_token_ids=[11, 12, 101, 102],
            prompt_logprobs=[{11: _lp(-1.1)}, {12: _lp(-1.2)}, {101: _lp(-1.3)}, {102: _lp(-1.4)}],
            response_token_ids=[201, 202, 203],
            response_logprobs=[{201: _lp(-2.1)}, {202: _lp(-2.2)}, {203: _lp(-2.3)}],
            assistant_prefix_token_ids=assistant_prefix,
        )
        self.assertEqual(metadata["assistant_token_ids"], [101, 102, 201, 202, 203])
        self.assertEqual(metadata["assistant_token_logprobs"], [-1.3, -1.4, -2.1, -2.2, -2.3])

    def test_assemble_old_log_probs_handles_left_padding_multiturn_and_truncation(self):
        step_metadata = [
            {"assistant_token_ids": [41, 42], "assistant_token_logprobs": [-4.1, -4.2]},
            {"assistant_token_ids": [51, 52, 53], "assistant_token_logprobs": [-5.1, -5.2, -5.3]},
        ]
        input_ids = torch.tensor([0, 0, 7, 8, 41, 42, 9, 51, 52], dtype=torch.long)
        labels = torch.tensor([-100, -100, -100, -100, 41, 42, -100, 51, 52], dtype=torch.long)

        dense = assemble_old_log_probs_from_sample(
            input_ids=input_ids,
            labels=labels,
            rollout_step_metadata=step_metadata,
        )

        expected = torch.zeros(input_ids.size(0) - 1, dtype=torch.float32)
        expected[3] = -4.1
        expected[4] = -4.2
        expected[6] = -5.1
        expected[7] = -5.2
        self.assertTrue(torch.equal(dense, expected))

    def test_assemble_old_log_probs_detects_token_mismatch(self):
        with self.assertRaises(RolloutLogProbAlignmentError):
            assemble_old_log_probs_from_sample(
                input_ids=torch.tensor([3, 9, 5], dtype=torch.long),
                labels=torch.tensor([-100, 9, 5], dtype=torch.long),
                rollout_step_metadata=[
                    {"assistant_token_ids": [4, 5], "assistant_token_logprobs": [-1.0, -2.0]}
                ],
            )

    def test_get_old_log_probs_with_fallback_skips_recompute_when_rollout_alignment_succeeds(self):
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor([[1, 2, 11, 12]], dtype=torch.long),
                "labels": torch.tensor([[-100, -100, 11, 12]], dtype=torch.long),
            },
            non_tensors={
                "rollout_step_metadata": np.array(
                    [[{"assistant_token_ids": [11, 12], "assistant_token_logprobs": [-1.1, -1.2]}]], dtype=object
                )
            },
        )
        recompute = Mock()

        old_log_probs, stats = get_old_log_probs_with_fallback(batch=batch, source="auto", recompute_fn=recompute)

        self.assertEqual(recompute.call_count, 0)
        self.assertEqual(stats["used_rollout_old_logprobs"], 1)
        self.assertEqual(tuple(old_log_probs.batch["old_log_probs"].shape), (1, 3))
        self.assertAlmostEqual(old_log_probs.batch["old_log_probs"][0, 1].item(), -1.1, places=6)
        self.assertAlmostEqual(old_log_probs.batch["old_log_probs"][0, 2].item(), -1.2, places=6)

    def test_get_old_log_probs_with_fallback_recomputes_on_alignment_error_in_auto_mode(self):
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor([[1, 2, 77]], dtype=torch.long),
                "labels": torch.tensor([[-100, -100, 99]], dtype=torch.long),
            },
            non_tensors={
                "rollout_step_metadata": np.array(
                    [[{"assistant_token_ids": [11], "assistant_token_logprobs": [-1.1]}]], dtype=object
                )
            },
        )
        expected = DataProto.from_dict(tensors={"old_log_probs": torch.full((1, 2), -7.0)})
        recompute = Mock(return_value=expected)

        old_log_probs, stats = get_old_log_probs_with_fallback(batch=batch, source="auto", recompute_fn=recompute)

        self.assertEqual(recompute.call_count, 1)
        self.assertEqual(stats["rollout_old_logprob_fallback"], 1)
        self.assertTrue(torch.equal(old_log_probs.batch["old_log_probs"], expected.batch["old_log_probs"]))

    def test_get_old_log_probs_with_fallback_hard_fails_in_rollout_mode(self):
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor([[1, 2, 77]], dtype=torch.long),
                "labels": torch.tensor([[-100, -100, 99]], dtype=torch.long),
            },
            non_tensors={
                "rollout_step_metadata": np.array(
                    [[{"assistant_token_ids": [11], "assistant_token_logprobs": [-1.1]}]], dtype=object
                )
            },
        )

        with self.assertRaises(RolloutLogProbAlignmentError):
            get_old_log_probs_with_fallback(batch=batch, source="rollout", recompute_fn=lambda: None)


if __name__ == "__main__":
    unittest.main()
