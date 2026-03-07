"""
Diagnostic test for _select_pixel_values_for_sp_rank.
Runs the EXACT same logic as _forward_micro_batch on realistic synthetic data.
NO model weights, NO GPU, NO distributed setup needed. Runs in seconds.

Usage: python tests/test_sp_pixel_values_debug.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from flash_attn.bert_padding import unpad_input  # same as training

# --------------------------------------------------------------------------
# Qwen2.5-VL special token IDs (hardcoded, no model needed)
# --------------------------------------------------------------------------
IMAGE_PAD_ID   = 151655   # <|image_pad|> = image_token_id
VISION_START   = 151652   # <|vision_start|>
VISION_END     = 151653   # <|vision_end|>
IM_START       = 151644   # <|im_start|>
IM_END         = 151645   # <|im_end|>
PAD_TOKEN_ID   = 151643   # <|endoftext|> used as pad
MERGE_SIZE     = 2        # Qwen2.5-VL default spatial merge

# --------------------------------------------------------------------------
# Copy of the selection logic from dp_actor.py (no self/dist needed)
# --------------------------------------------------------------------------
def select_pixel_values_for_sp_rank(
    input_ids_rmpad: torch.Tensor,   # [1, total_nnz]
    multi_modal_inputs: dict,
    sp_rank: int,
    sp_size: int,
) -> tuple[dict, str]:
    """Returns (filtered_mm_inputs, reason_str)."""
    if not multi_modal_inputs or "pixel_values" not in multi_modal_inputs:
        return multi_modal_inputs, "NO_PIXEL_VALUES"

    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
    pixel_values = multi_modal_inputs["pixel_values"]
    if image_grid_thw is None or len(image_grid_thw) == 0:
        return multi_modal_inputs, "NO_GRID_THW"

    total_nnz = input_ids_rmpad.shape[1]
    pad_size = (sp_size - total_nnz % sp_size) % sp_size
    slice_size = (total_nnz + pad_size) // sp_size
    rank_start = sp_rank * slice_size
    rank_end = min((sp_rank + 1) * slice_size, total_nnz)

    flat_ids = input_ids_rmpad.squeeze(0)
    is_image = flat_ids == IMAGE_PAD_ID

    if not is_image.any():
        return multi_modal_inputs, "NO_IMAGE_TOKENS_IN_INPUT_IDS"

    false_pad = torch.zeros(1, dtype=torch.bool)
    padded = torch.cat([false_pad, is_image, false_pad])
    block_starts = (padded[1:] & ~padded[:-1]).nonzero(as_tuple=True)[0]
    block_ends   = (~padded[1:] & padded[:-1]).nonzero(as_tuple=True)[0] - 1

    num_images = len(image_grid_thw)
    if len(block_starts) != num_images:
        return multi_modal_inputs, f"FALLBACK: block_starts={len(block_starts)} != num_images={num_images}"

    pv_per_image = [
        int(image_grid_thw[i, 0] * image_grid_thw[i, 1] * image_grid_thw[i, 2])
        for i in range(num_images)
    ]

    rank_image_indices = [
        i for i in range(num_images)
        if int(block_starts[i]) < rank_end and int(block_ends[i]) >= rank_start
    ]

    if len(rank_image_indices) == num_images:
        reason = f"ALL_IMAGES_IN_RANK (all {num_images} images straddle or fit in rank)"
        return multi_modal_inputs, reason

    if len(rank_image_indices) == 0:
        result = dict(multi_modal_inputs)
        result.pop("pixel_values", None)
        result.pop("image_grid_thw", None)
        return result, "NO_IMAGES_IN_RANK"

    rank_image_set = set(rank_image_indices)
    pv_chunks = []
    pv_offset = 0
    for i, n_pv in enumerate(pv_per_image):
        if i in rank_image_set:
            pv_chunks.append(pixel_values[pv_offset : pv_offset + n_pv])
        pv_offset += n_pv

    rank_pixel_values = torch.cat(pv_chunks, dim=0)
    rank_grid_thw = image_grid_thw[rank_image_indices]
    filtered = {**multi_modal_inputs, "pixel_values": rank_pixel_values, "image_grid_thw": rank_grid_thw}
    return filtered, f"OK: {len(rank_image_indices)}/{num_images} images"


def count_image_tokens_in_slice(input_ids_rmpad, sp_rank, sp_size):
    """Count IMAGE_PAD_ID in this rank's slice of input_ids_rmpad."""
    total_nnz = input_ids_rmpad.shape[1]
    pad_size = (sp_size - total_nnz % sp_size) % sp_size
    slice_size = (total_nnz + pad_size) // sp_size
    rank_start = sp_rank * slice_size
    rank_end = min((sp_rank + 1) * slice_size, total_nnz)
    sliced = input_ids_rmpad[0, rank_start:rank_end]
    return int((sliced == IMAGE_PAD_ID).sum())


def compute_expected_features(image_grid_thw):
    """Compute post-merge feature count from image_grid_thw."""
    if image_grid_thw is None or len(image_grid_thw) == 0:
        return 0
    total = 0
    for i in range(len(image_grid_thw)):
        t, h, w = image_grid_thw[i].tolist()
        total += (t * h * w) // (MERGE_SIZE ** 2)
    return total


# --------------------------------------------------------------------------
# Build a synthetic multi-turn trajectory
# --------------------------------------------------------------------------
def make_trajectory(
    n_screenshots: int,
    tokens_per_screenshot: int,  # post-merge image tokens per screenshot
    text_len_per_turn: int = 50,
    response_len: int = 30,
):
    """
    Build synthetic input_ids for a K-step trajectory:
      system + (screenshot + action)*K
    Returns:
      input_ids  [total_len]  int64 tensor
      image_grid_thw  [n_screenshots, 3] tensor (t, h, w) pre-merge
      pixel_values  [total_pv_patches, 3, 14, 14] float tensor (mock)
    """
    tokens = []
    # system turn
    tokens += [IM_START, 9738, IM_END]  # <|im_start|>system<|im_end|>

    image_grid_thw = []
    pv_rows = 0

    for step in range(n_screenshots):
        # user turn: screenshot + instruction
        tokens += [IM_START, 882]  # <|im_start|>user
        tokens += [VISION_START]
        tokens += [IMAGE_PAD_ID] * tokens_per_screenshot
        tokens += [VISION_END]
        tokens += [269]  # newline
        tokens += [1234] * text_len_per_turn  # instruction text
        tokens += [IM_END]

        # image grid: t=1, h and w such that post-merge = tokens_per_screenshot
        # post-merge = (h//2)*(w//2) = tokens_per_screenshot
        # Choose h=w=2*sqrt(tokens_per_screenshot) (pre-merge)
        import math
        hw_post = int(math.sqrt(tokens_per_screenshot))
        assert hw_post * hw_post == tokens_per_screenshot, \
            f"tokens_per_screenshot={tokens_per_screenshot} must be a perfect square"
        h_pre = hw_post * MERGE_SIZE  # e.g. 2*hw_post
        w_pre = hw_post * MERGE_SIZE
        t = 1
        image_grid_thw.append([t, h_pre, w_pre])
        pv_rows += t * h_pre * w_pre

        # assistant turn
        tokens += [IM_START, 77091]  # <|im_start|>assistant
        tokens += [5678] * response_len  # action text
        tokens += [IM_END]

    input_ids = torch.tensor(tokens, dtype=torch.long)
    image_grid_thw_t = torch.tensor(image_grid_thw, dtype=torch.long)
    pixel_values = torch.zeros(pv_rows, 3, 14, 14)  # mock pixel values

    return input_ids, image_grid_thw_t, pixel_values


def run_test(name, input_ids, image_grid_thw, pixel_values, sp_size=8, max_len=None):
    """Run the exact same flow as _forward_micro_batch and check for mismatches."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")

    if max_len is not None and len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        print(f"  [truncated input_ids to max_len={max_len}]")

    # Simulate padding to batch_size=1
    input_ids_2d = input_ids.unsqueeze(0)  # [1, seqlen]
    attention_mask = (input_ids_2d != PAD_TOKEN_ID).long()

    total_img_tokens = int((input_ids == IMAGE_PAD_ID).sum())
    n_images = len(image_grid_thw)
    expected_features_total = compute_expected_features(image_grid_thw)

    print(f"  seqlen={len(input_ids)}, n_images={n_images}, "
          f"total img tokens={total_img_tokens}, total features={expected_features_total}")
    if total_img_tokens != expected_features_total:
        print(f"  *** WARNING: global mismatch BEFORE SP! "
              f"tokens={total_img_tokens} != features={expected_features_total} ***")

    # Simulate unpad_input (padding_free)
    input_ids_rmpad, indices, *_ = unpad_input(input_ids_2d.unsqueeze(-1), attention_mask)
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # [1, total_nnz]
    total_nnz = input_ids_rmpad.shape[1]
    print(f"  total_nnz (after unpad) = {total_nnz}")

    # Detect image blocks in the full unpadded sequence
    flat_ids = input_ids_rmpad.squeeze(0)
    is_image = flat_ids == IMAGE_PAD_ID
    false_pad = torch.zeros(1, dtype=torch.bool)
    padded = torch.cat([false_pad, is_image, false_pad])
    block_starts = (padded[1:] & ~padded[:-1]).nonzero(as_tuple=True)[0]
    block_ends   = (~padded[1:] & padded[:-1]).nonzero(as_tuple=True)[0] - 1
    print(f"  detected blocks={len(block_starts)}, image_grid_thw entries={n_images}")

    if len(block_starts) != n_images:
        print(f"  *** MISMATCH: detected {len(block_starts)} blocks but {n_images} images! ***")
        print(f"  block_starts[:10] = {block_starts[:10].tolist()}")
        print(f"  block_ends[:10]   = {block_ends[:10].tolist()}")
        # Show non-image, non-pad token positions near each block
        for bi in range(min(len(block_starts), 5)):
            s = int(block_starts[bi])
            e = int(block_ends[bi])
            print(f"    block {bi}: [{s}:{e}] len={e-s+1}")
        print(f"  Expected {n_images} blocks from image_grid_thw:")
        for i in range(min(n_images, 5)):
            t,h,w = image_grid_thw[i].tolist()
            print(f"    image {i}: t={t},h={h},w={w} -> {(t*h*w)//(MERGE_SIZE**2)} tokens")

    multi_modal_inputs = {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    print(f"\n  Per-rank analysis (sp_size={sp_size}):")
    pad_size = (sp_size - total_nnz % sp_size) % sp_size
    slice_size = (total_nnz + pad_size) // sp_size
    any_mismatch = False
    for rank in range(sp_size):
        rank_start = rank * slice_size
        rank_end = min((rank + 1) * slice_size, total_nnz)
        filtered, reason = select_pixel_values_for_sp_rank(
            input_ids_rmpad, multi_modal_inputs, rank, sp_size
        )
        token_count = count_image_tokens_in_slice(input_ids_rmpad, rank, sp_size)
        feature_count = compute_expected_features(filtered.get("image_grid_thw"))
        mismatch = (token_count != feature_count) if "pixel_values" in filtered else False
        flag = " *** MISMATCH ***" if mismatch else ""
        if mismatch:
            any_mismatch = True
        print(f"    rank {rank} [{rank_start}:{rank_end}]: "
              f"tokens={token_count}, features={feature_count}, "
              f"reason={reason}{flag}")

    if not any_mismatch:
        print(f"\n  [PASS] All ranks match tokens==features")
    else:
        print(f"\n  [FAIL] Token/feature mismatches found!")

    return not any_mismatch


# --------------------------------------------------------------------------
# Test scenarios
# --------------------------------------------------------------------------
if __name__ == "__main__":
    all_pass = True

    # --- Test 1: Small clean trajectory (should work) ---
    input_ids, igt, pv = make_trajectory(
        n_screenshots=5,
        tokens_per_screenshot=100,  # 10x10 post-merge = 20x20 pre-merge
        text_len_per_turn=20,
        response_len=20,
    )
    ok = run_test("5 screenshots x 100 tokens each, sp=8", input_ids, igt, pv, sp_size=8)
    all_pass &= ok

    # --- Test 2: Realistic OSWorld screenshot sizes ---
    # Typical 1920x1080 screenshot → ~2500 post-merge tokens
    # Use 2500 = 50x50 (post-merge), 100x100 (pre-merge)
    input_ids, igt, pv = make_trajectory(
        n_screenshots=10,
        tokens_per_screenshot=2500,
        text_len_per_turn=50,
        response_len=30,
    )
    ok = run_test("10 screenshots x 2500 tokens, sp=8", input_ids, igt, pv, sp_size=8)
    all_pass &= ok

    # --- Test 3: 15 screenshots (limit_images=15), realistic sizes ---
    input_ids, igt, pv = make_trajectory(
        n_screenshots=15,
        tokens_per_screenshot=1764,  # 42x42 post-merge
        text_len_per_turn=40,
        response_len=25,
    )
    ok = run_test("15 screenshots x 1764 tokens (realistic), sp=8", input_ids, igt, pv, sp_size=8)
    all_pass &= ok

    # --- Test 4: Truncation scenario ---
    # 20 screenshots but max_len=64000 truncates some
    input_ids_long, igt_long, pv_long = make_trajectory(
        n_screenshots=20,
        tokens_per_screenshot=2500,
        text_len_per_turn=50,
        response_len=30,
    )
    print(f"\n[INFO] Long trajectory: {len(input_ids_long)} tokens, {len(igt_long)} images")
    # Manually truncate (simulate what get_train_dict + postprocess_data do)
    max_len = 64000
    if len(input_ids_long) > max_len:
        input_ids_trunc = input_ids_long[:max_len]
        # Count how many image blocks fit in truncated
        n_blocks = 0
        pv_per = []
        t_img = int((input_ids_trunc == IMAGE_PAD_ID).sum())
        exp = 0
        for i in range(len(igt_long)):
            t, h, w = igt_long[i].tolist()
            n_tokens = (t*h*w) // (MERGE_SIZE**2)
            n_pv = t*h*w
            exp += n_tokens
            if exp <= t_img:
                n_blocks += 1
                pv_per.append(n_pv)
        igt_trunc = igt_long[:n_blocks]
        pv_trunc = pv_long[:sum(pv_per)]
        ok = run_test(f"20 screenshots truncated to {max_len} tokens → {n_blocks} images",
                      input_ids_trunc, igt_trunc, pv_trunc, sp_size=8)
        all_pass &= ok

    # --- Test 5: Straddle scenario (image at rank boundary) ---
    # Manually craft a sequence where an image exactly straddles a rank boundary
    print(f"\n{'='*70}")
    print("TEST: Manually crafted straddling image at rank boundary")
    print(f"{'='*70}")
    sp_size = 8
    # We'll put 1 image of 200 tokens that spans across the rank boundary
    # total_nnz target = 1600, so slice_size = 200
    # Put image from position 100 to 399 (spanning rank 0 [0:200] and rank 1 [200:400])
    pre_text = [1000] * 100         # 100 text tokens
    img_tokens = [IMAGE_PAD_ID] * 200  # 200 image tokens (straddling ranks 0 and 1)
    post_text = [1001] * (1600 - 100 - 200)  # fill to 1600 total
    straddle_ids = torch.tensor(pre_text + img_tokens + post_text, dtype=torch.long)
    # h_pre = w_pre such that t*h*w // 4 = 200 → h_pre = w_pre = 20, t=1
    # post-merge: 10*10 = 100... but we want 200. Try h=20,w=20 → (20//2)*(20//2) = 100 ≠ 200
    # Try t=1, h=20, w=40 → (20//2)*(40//2) = 10*20 = 200 ✓, pv = 1*20*40 = 800
    straddle_igt = torch.tensor([[1, 20, 40]], dtype=torch.long)  # t=1,h=20,w=40
    straddle_pv = torch.zeros(1*20*40, 3, 14, 14)  # 800 pixel patches

    straddle_ids_2d = straddle_ids.unsqueeze(0)
    attn_mask = torch.ones_like(straddle_ids_2d)
    ids_rmpad, _, *_ = unpad_input(straddle_ids_2d.unsqueeze(-1), attn_mask)
    ids_rmpad = ids_rmpad.transpose(0, 1)
    mm = {"pixel_values": straddle_pv, "image_grid_thw": straddle_igt}

    pad_size = (sp_size - ids_rmpad.shape[1] % sp_size) % sp_size
    slice_size = (ids_rmpad.shape[1] + pad_size) // sp_size
    any_mismatch = False
    print(f"  total_nnz={ids_rmpad.shape[1]}, slice_size={slice_size}")
    for rank in range(sp_size):
        rank_start = rank * slice_size
        rank_end = min((rank+1)*slice_size, ids_rmpad.shape[1])
        filtered, reason = select_pixel_values_for_sp_rank(ids_rmpad, mm, rank, sp_size)
        token_count = count_image_tokens_in_slice(ids_rmpad, rank, sp_size)
        feature_count = compute_expected_features(filtered.get("image_grid_thw"))
        mismatch = (token_count != feature_count) if "pixel_values" in filtered else False
        if mismatch:
            any_mismatch = True
        flag = " *** MISMATCH ***" if mismatch else ""
        print(f"    rank {rank} [{rank_start}:{rank_end}]: tokens={token_count}, features={feature_count}, reason={reason}{flag}")

    if any_mismatch:
        print("\n  [EXPECTED FAIL] Straddling image causes mismatch - this is the known bug!")
        print("  Straddling images cannot be split between ranks with the current approach.")
    else:
        print("\n  [PASS] No mismatches (maybe image doesn't straddle in this config)")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print(f"{'='*70}")
