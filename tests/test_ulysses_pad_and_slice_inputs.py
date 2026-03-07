import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.utils import ulysses


class TestUlyssesPadAndSliceInputs(unittest.TestCase):
    def setUp(self):
        self._orig_get_world_size = ulysses.dist.get_world_size
        self._orig_get_rank = ulysses.dist.get_rank
        self._orig_group = ulysses.get_ulysses_sequence_parallel_group()
        self._group = object()
        ulysses.dist.get_world_size = lambda group=None: 8
        ulysses.dist.get_rank = lambda group=None: 3
        ulysses.set_ulysses_sequence_parallel_group(self._group)

    def tearDown(self):
        ulysses.dist.get_world_size = self._orig_get_world_size
        ulysses.dist.get_rank = self._orig_get_rank
        ulysses.set_ulysses_sequence_parallel_group(self._orig_group)

    def test_slices_multimodal_position_ids_with_input_ids(self):
        input_ids = torch.arange(19760).view(1, -1)
        position_ids = torch.arange(19760).view(1, 1, -1).expand(3, 1, -1).clone()

        sliced_ids, sliced_pos, pad_size = ulysses.ulysses_pad_and_slice_inputs(
            input_ids, position_ids, sp_size=8
        )

        self.assertEqual(pad_size, 0)
        self.assertEqual(tuple(sliced_ids.shape), (1, 2470))
        self.assertEqual(tuple(sliced_pos.shape), (3, 1, 2470))
        self.assertTrue(torch.equal(sliced_pos, position_ids[..., 7410:9880]))

    def test_padding_keeps_last_dimension_aligned(self):
        input_ids = torch.arange(10).view(1, -1)
        position_ids = torch.arange(10).view(1, 1, -1).expand(3, 1, -1).clone()

        sliced_ids, sliced_pos, pad_size = ulysses.ulysses_pad_and_slice_inputs(
            input_ids, position_ids, sp_size=8
        )

        self.assertEqual(pad_size, 6)
        self.assertEqual(sliced_ids.shape[-1], sliced_pos.shape[-1])
        self.assertEqual(tuple(sliced_ids.shape), (1, 2))
        self.assertEqual(tuple(sliced_pos.shape), (3, 1, 2))


if __name__ == "__main__":
    unittest.main()
