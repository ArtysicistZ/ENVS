"""
Tests for _select_pixel_values_for_sp_rank in DataParallelPPOActor.

Run with:
    .venv/bin/python tests/test_select_pixel_values.py

No GPU or distributed process required — mocks the actor for pure logic testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import unittest


IMAGE_TOKEN_ID = 151655
TEXT_TOKEN_ID = 100
SP_SIZE = 8


# ---------------------------------------------------------------------------
# Minimal mock — replicates _select_pixel_values_for_sp_rank without dist
# ---------------------------------------------------------------------------

class MockConfig:
    image_token_id = IMAGE_TOKEN_ID


class MockActorModule:
    config = MockConfig()


class MockActor:
    def __init__(self, sp_rank):
        self.actor_module = MockActorModule()
        self._sp_rank = sp_rank

    def _select_pixel_values_for_sp_rank(self, input_ids_rmpad, multi_modal_inputs, sp_size):
        if not multi_modal_inputs or "pixel_values" not in multi_modal_inputs:
            return multi_modal_inputs

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        pixel_values = multi_modal_inputs["pixel_values"]
        if image_grid_thw is None or len(image_grid_thw) == 0:
            return multi_modal_inputs

        sp_rank = self._sp_rank
        total_nnz = input_ids_rmpad.shape[1]
        pad_size = (sp_size - total_nnz % sp_size) % sp_size
        slice_size = (total_nnz + pad_size) // sp_size
        rank_start = sp_rank * slice_size
        rank_end = min((sp_rank + 1) * slice_size, total_nnz)

        try:
            image_token_id = self.actor_module.config.image_token_id
        except AttributeError:
            image_token_id = IMAGE_TOKEN_ID

        flat_ids = input_ids_rmpad.squeeze(0)
        is_image = flat_ids == image_token_id

        if not is_image.any():
            return multi_modal_inputs

        false_pad = torch.zeros(1, dtype=torch.bool)
        padded = torch.cat([false_pad, is_image, false_pad])
        block_starts = (padded[1:] & ~padded[:-1]).nonzero(as_tuple=True)[0]
        block_ends = (~padded[1:] & padded[:-1]).nonzero(as_tuple=True)[0] - 1

        num_images = len(image_grid_thw)
        if len(block_starts) != num_images:
            return multi_modal_inputs

        pv_per_image = [
            int(image_grid_thw[i, 0] * image_grid_thw[i, 1] * image_grid_thw[i, 2])
            for i in range(num_images)
        ]

        rank_image_indices = [
            i for i in range(num_images)
            if int(block_starts[i]) < rank_end and int(block_ends[i]) >= rank_start
        ]

        if len(rank_image_indices) == num_images:
            return multi_modal_inputs

        if len(rank_image_indices) == 0:
            result = dict(multi_modal_inputs)
            result.pop("pixel_values", None)
            result.pop("image_grid_thw", None)
            return result

        rank_image_set = set(rank_image_indices)
        pv_chunks = []
        pv_offset = 0
        for i, n_pv in enumerate(pv_per_image):
            if i in rank_image_set:
                pv_chunks.append(pixel_values[pv_offset : pv_offset + n_pv])
            pv_offset += n_pv

        rank_pixel_values = torch.cat(pv_chunks, dim=0)
        rank_grid_thw = image_grid_thw[rank_image_indices]
        return {**multi_modal_inputs, "pixel_values": rank_pixel_values, "image_grid_thw": rank_grid_thw}


# ---------------------------------------------------------------------------
# Helper: build a synthetic multi-turn trajectory
# ---------------------------------------------------------------------------

def build_trajectory(num_images, img_seq_tokens, pv_patches_per_image,
                     text_prefix=300, text_between=5000, text_suffix=300):
    """
    Build synthetic input_ids_rmpad, pixel_values, image_grid_thw.

    img_seq_tokens: count of IMAGE_TOKEN_ID tokens per image in input_ids (post-merge).
    pv_patches_per_image: pixel_values rows per image (pre-merge = img_seq_tokens * merge_size^2).
    image_grid_thw: uses (1, 2, pv_patches_per_image//2) — a rectangle with t=1, h=2.
    """
    segments = [torch.full((text_prefix,), TEXT_TOKEN_ID, dtype=torch.long)]
    for i in range(num_images):
        segments.append(torch.full((img_seq_tokens,), IMAGE_TOKEN_ID, dtype=torch.long))
        if i < num_images - 1:
            segments.append(torch.full((text_between,), TEXT_TOKEN_ID, dtype=torch.long))
    segments.append(torch.full((text_suffix,), TEXT_TOKEN_ID, dtype=torch.long))

    input_ids = torch.cat(segments).unsqueeze(0)  # [1, total_nnz]

    # pixel_values: distinct float value per image for traceability
    pv_dim = 3
    pixel_values = torch.zeros(num_images * pv_patches_per_image, pv_dim)
    for i in range(num_images):
        pixel_values[i * pv_patches_per_image : (i + 1) * pv_patches_per_image] = float(i)

    # image_grid_thw: (t=1, h=2, w=pv_patches//2) — pre-merge dims, product = pv_patches_per_image
    assert pv_patches_per_image % 2 == 0, "pv_patches_per_image must be even"
    image_grid_thw = torch.tensor([[1, 2, pv_patches_per_image // 2]] * num_images, dtype=torch.long)

    return input_ids, {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}


def run_all_ranks(input_ids, mmi, sp_size=SP_SIZE):
    return [
        MockActor(sp_rank=r)._select_pixel_values_for_sp_rank(input_ids, mmi, sp_size)
        for r in range(sp_size)
    ]


def image_block_positions(input_ids_rmpad):
    """Return (start, end) inclusive for each contiguous image block."""
    flat = input_ids_rmpad.squeeze(0)
    is_img = flat == IMAGE_TOKEN_ID
    false_pad = torch.zeros(1, dtype=torch.bool)
    padded = torch.cat([false_pad, is_img, false_pad])
    starts = (padded[1:] & ~padded[:-1]).nonzero(as_tuple=True)[0]
    ends = (~padded[1:] & padded[:-1]).nonzero(as_tuple=True)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelectPixelValuesForSPRank(unittest.TestCase):

    def test_no_images_in_sequence(self):
        """No IMAGE_TOKEN_ID tokens → pixel_values returned unchanged."""
        ids = torch.full((1, 1000), TEXT_TOKEN_ID, dtype=torch.long)
        mmi = {"pixel_values": torch.zeros(10, 3), "image_grid_thw": torch.tensor([[1, 2, 5]])}
        actor = MockActor(sp_rank=0)
        out = actor._select_pixel_values_for_sp_rank(ids, mmi, SP_SIZE)
        self.assertIs(out, mmi)

    def test_no_pixel_values_key(self):
        """Missing pixel_values key → unchanged."""
        ids = torch.full((1, 500), TEXT_TOKEN_ID, dtype=torch.long)
        mmi = {"other": torch.zeros(3)}
        actor = MockActor(sp_rank=0)
        self.assertIs(actor._select_pixel_values_for_sp_rank(ids, mmi, SP_SIZE), mmi)

    def test_sp_size_1_returns_all(self):
        """sp_size=1 → rank 0 slice = full sequence → all images returned."""
        ids, mmi = build_trajectory(3, img_seq_tokens=100, pv_patches_per_image=400)
        actor = MockActor(sp_rank=0)
        out = actor._select_pixel_values_for_sp_rank(ids, mmi, sp_size=1)
        self.assertEqual(out["pixel_values"].shape[0], mmi["pixel_values"].shape[0])
        self.assertEqual(len(out["image_grid_thw"]), 3)

    def test_single_image_one_rank_gets_it(self):
        """One image in a long sequence → exactly 1 rank gets pixel_values, 7 get none."""
        # Image at [300, 599], total_nnz = 300 + 300 + 60000 = 60600 → slice ≈ 7575
        # Image falls in rank 0 [0, 7575)
        ids, mmi = build_trajectory(
            num_images=1, img_seq_tokens=300, pv_patches_per_image=1200,
            text_prefix=300, text_between=0, text_suffix=60000,
        )
        results = run_all_ranks(ids, mmi)

        ranks_with = [r for r, res in enumerate(results) if "pixel_values" in res]
        ranks_without = [r for r, res in enumerate(results) if "pixel_values" not in res]

        self.assertEqual(len(ranks_with), 1, f"Expected 1 rank with images, got {ranks_with}")
        self.assertEqual(len(ranks_without), SP_SIZE - 1)

        res = results[ranks_with[0]]
        self.assertEqual(res["pixel_values"].shape[0], 1200)
        self.assertEqual(len(res["image_grid_thw"]), 1)

    def test_correct_pv_rows_per_rank(self):
        """
        Each rank gets pixel_values for exactly the images in its sequence slice.
        We mark each image's pv rows with its index (float i) for traceability.
        We use enough spacing so no image straddles a rank boundary.
        """
        # 8 images, 1 per rank. Slice ≈ total/8.
        # img_seq_tokens=300, pv=1200, text_between=6000 → period=6300
        # total = 300 + 8*300 + 7*6000 + 300 = 300+2400+42000+300 = 45000, slice=5625
        # Image i starts at 300 + i*(300+6000) = 300 + i*6300. Image 0 at 300-600, in rank 0 [0,5625) ✓
        # Image 1 at 6600-6900, in rank 1 [5625,11250) ✓ etc.
        num_images = 8
        ids, mmi = build_trajectory(
            num_images=num_images, img_seq_tokens=300, pv_patches_per_image=1200,
            text_prefix=300, text_between=6000, text_suffix=300,
        )
        results = run_all_ranks(ids, mmi)

        blocks = image_block_positions(ids)
        total_nnz = ids.shape[1]
        pad_size = (SP_SIZE - total_nnz % SP_SIZE) % SP_SIZE
        slice_size = (total_nnz + pad_size) // SP_SIZE

        for rank, res in enumerate(results):
            if "pixel_values" not in res:
                # Verify no image blocks overlap with this rank's slice
                rs, re = rank * slice_size, min((rank + 1) * slice_size, total_nnz)
                for s, e in blocks:
                    self.assertFalse(
                        s < re and e >= rs,
                        f"Rank {rank} [{rs},{re}) should have image [{s},{e}] but got none"
                    )
                continue

            rank_pv = res["pixel_values"]
            # All rows in rank_pv should have the same float value (one image per rank)
            unique_vals = rank_pv[:, 0].unique()
            self.assertEqual(len(unique_vals), 1,
                             f"Rank {rank} got mixed images: {unique_vals.tolist()}")

            # That image's index tells us which image it is
            img_idx = int(unique_vals[0].item())
            self.assertEqual(rank_pv.shape[0], 1200,
                             f"Rank {rank} expected 1200 pv rows for image {img_idx}")

    def test_image_grid_thw_consistent_with_pixel_values(self):
        """
        For every rank, image_grid_thw[i].prod() == pixel_values rows for image i.
        This confirms the filtering keeps grid and pv in sync.
        """
        ids, mmi = build_trajectory(
            num_images=12, img_seq_tokens=256, pv_patches_per_image=1024,
            text_prefix=300, text_between=4000, text_suffix=300,
        )
        results = run_all_ranks(ids, mmi)

        for rank, res in enumerate(results):
            if "pixel_values" not in res:
                continue
            grid = res["image_grid_thw"]
            expected_pv = sum(int(grid[i, 0] * grid[i, 1] * grid[i, 2]) for i in range(len(grid)))
            self.assertEqual(
                expected_pv, res["pixel_values"].shape[0],
                f"Rank {rank}: grid product {expected_pv} != pv rows {res['pixel_values'].shape[0]}"
            )

    def test_no_images_in_rank_removes_pixel_values(self):
        """Ranks whose slice has no image tokens get no pixel_values key."""
        # Put 1 image at the very start; ranks 1-7 have no images
        ids, mmi = build_trajectory(
            num_images=1, img_seq_tokens=500, pv_patches_per_image=2000,
            text_prefix=100, text_between=0, text_suffix=80000,
        )
        results = run_all_ranks(ids, mmi)
        for rank in range(1, SP_SIZE):
            self.assertNotIn(
                "pixel_values", results[rank],
                f"Rank {rank} should have no pixel_values (no images in its slice)"
            )

    def test_qwen25_vl_realistic(self):
        """
        Realistic Qwen2.5-VL: 13 screenshots with img_seq_tokens=1296, pv_patches=5184.
        Spacing = 5000 text tokens between images (>> 1296) to avoid boundary straddling.
        Validates: each rank's pv count matches its image_grid_thw product.
        """
        num_images = 13
        img_seq_tokens = 1296    # t*(h//2)*(w//2) = 1*(48)*(27) = 1296
        pv_patches = 5184        # t*h*w = 1*96*54 = 5184 (pre-merge, h=96,w=54)

        # Use h=2, w=pv//2=2592 for image_grid_thw (product = 5184) ✓
        text_between = 5000      # >> 1296, minimises boundary straddling

        ids, mmi = build_trajectory(
            num_images=num_images, img_seq_tokens=img_seq_tokens, pv_patches_per_image=pv_patches,
            text_prefix=500, text_between=text_between, text_suffix=500,
        )

        # Use realistic image_grid_thw: (1, 96, 54) → product = 5184 ✓
        mmi["image_grid_thw"] = torch.tensor([[1, 96, 54]] * num_images, dtype=torch.long)

        results = run_all_ranks(ids, mmi)

        total_pv = mmi["pixel_values"].shape[0]

        # Compute which images actually straddle rank boundaries
        blocks = image_block_positions(ids)
        total_nnz = ids.shape[1]
        pad_size = (SP_SIZE - total_nnz % SP_SIZE) % SP_SIZE
        slice_size = (total_nnz + pad_size) // SP_SIZE
        straddling = set()
        for i, (s, e) in enumerate(blocks):
            for boundary in range(1, SP_SIZE):
                bp = boundary * slice_size
                if s < bp <= e:
                    straddling.add(i)

        total_assigned = sum(
            res["pixel_values"].shape[0] for res in results if "pixel_values" in res
        )
        # Each straddling image is counted in 2 ranks instead of 1
        expected_assigned = total_pv + len(straddling) * pv_patches
        self.assertEqual(
            total_assigned, expected_assigned,
            f"total_assigned={total_assigned}, expected={expected_assigned} "
            f"(straddling images: {straddling})"
        )

        # Validate grid/pv consistency for every rank
        for rank, res in enumerate(results):
            if "pixel_values" not in res:
                continue
            grid = res["image_grid_thw"]
            expected_pv = sum(int(grid[j, 0] * grid[j, 1] * grid[j, 2]) for j in range(len(grid)))
            self.assertEqual(
                expected_pv, res["pixel_values"].shape[0],
                f"Rank {rank}: grid product {expected_pv} != pv rows {res['pixel_values'].shape[0]}"
            )

    def test_all_images_in_rank_returns_original_dict(self):
        """When all images fall in one rank, the original dict object is returned (no copy)."""
        ids, mmi = build_trajectory(
            num_images=3, img_seq_tokens=100, pv_patches_per_image=400,
            text_prefix=100, text_between=200, text_suffix=60000,
        )
        # Rank 0 should contain all 3 images (they're at the start of a 60K+ sequence)
        actor = MockActor(sp_rank=0)
        out = actor._select_pixel_values_for_sp_rank(ids, mmi, SP_SIZE)
        self.assertIs(out, mmi, "Should return original dict when all images are in this rank")


if __name__ == "__main__":
    unittest.main(verbosity=2)
