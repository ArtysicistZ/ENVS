"""Single-cell VLMEvalKit runner: <model_path, bench_name> -> (predictions TSV, score JSON).

Why this layer (not VLMEvalKit's run.py directly):
  - VLMEvalKit's `Qwen2VLChat` hard-codes tp_size by GPU count; for UI-TARS-1.5-7B
    (28 attention heads) tp_size=8 fails. We need explicit TP=1.
  - VLMEvalKit hard-codes max_num_seqs=5 inside Qwen2VLChat — kills throughput.
  - We already have a tested `UITARSClient` (scripts/eval_visual/common.py) with
    the right ARPO-training-matched config (max_pixels=2.1M, max_num_seqs=8).

Strategy: drive VLMEvalKit only for (1) dataset loading and (2) scoring.
Inference goes through our own vLLM client. Predictions are written in the TSV
schema VLMEvalKit's `dataset.evaluate()` expects.

Resume-safe: partial TSV is read on startup, completed indices skipped.
Sharded: --shard-id N --num-shards M, indices i where i % M == N.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# VLMEvalKit must be importable.
_VLMEVALKIT_PATH = _REPO_ROOT / "cache_dirs" / "VLMEvalKit"
if str(_VLMEVALKIT_PATH) not in sys.path:
    sys.path.insert(0, str(_VLMEVALKIT_PATH))

from scripts.eval_visual.common import (  # noqa: E402
    GenerationConfig,
    UITARSClient,
    ensure_pil_rgb,
    selftest_coord_roundtrip,
    smart_resize_image,
)


# Qwen2.5-VL chat template (clean Q&A, NO UI-TARS GUI system prompt).
# Keeping the system prompt minimal avoids the model falling back into
# `Action: click(...)` GUI-grammar mode that would break MCQ/yes-no parsing.
QA_SYSTEM_PROMPT = "You are a helpful assistant."


def _build_chat_prompt(image: Image.Image, question: str) -> dict[str, Any]:
    """Single-image Qwen2.5-VL chat prompt for vLLM offline batched inference."""
    chat = (
        f"<|im_start|>system\n{QA_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {"prompt": chat, "multi_modal_data": {"image": image}}


def _load_image_from_msg_value(value: str) -> Image.Image:
    """VLMEvalKit returns image paths (or base64-decoded paths). Read as PIL."""
    if value.startswith("file://"):
        value = value[len("file://"):]
    return ensure_pil_rgb(Image.open(value))


def _build_dataset(bench_name: str):
    """Construct the VLMEvalKit dataset for the given bench.

    Auto-downloads to LMUDataRoot (defaults under HF_HOME / hf_cache).
    """
    from vlmeval.dataset import build_dataset

    ds = build_dataset(bench_name)
    if ds is None:
        raise RuntimeError(f"VLMEvalKit could not build dataset {bench_name!r}")
    return ds


def _msg_to_image_question(msgs: list[dict]) -> tuple[Image.Image, str]:
    """Extract the FIRST image and concatenated text from VLMEvalKit msgs.

    For our 4 benches (HallusionBench, POPE, MMVP, MMStar) every sample has
    exactly one image and one text question. We assert that here.
    """
    image: Image.Image | None = None
    text_parts: list[str] = []
    for m in msgs:
        if m["type"] == "image":
            if image is None:
                image = _load_image_from_msg_value(m["value"])
        elif m["type"] == "text":
            text_parts.append(m["value"])
    if image is None:
        raise ValueError("no image in msgs")
    if not text_parts:
        raise ValueError("no text in msgs")
    return image, "\n".join(text_parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--bench", required=True, help="VLMEvalKit dataset name (HallusionBench, POPE, MMVP, MMStar, ...)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--smoke", type=int, default=0, help="N>0 = only the first N global samples (sharded among num_shards)")
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if args.num_shards > 1:
        shard_dir = out_dir / f"shard_{args.shard_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = shard_dir / "predictions.tsv"
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = out_dir / "predictions.tsv"

    print(f"[vlmevalkit] bench={args.bench} model={args.model_path}", flush=True)
    print(f"[vlmevalkit] output={out_dir} shard={args.shard_id}/{args.num_shards}", flush=True)

    selftest_coord_roundtrip()

    print(f"[vlmevalkit] building dataset {args.bench}...", flush=True)
    ds = _build_dataset(args.bench)
    n = len(ds)
    if args.smoke > 0:
        n = min(n, args.smoke)
    print(f"[vlmevalkit] dataset size: {n} samples", flush=True)

    shard_indices = [i for i in range(n) if i % args.num_shards == args.shard_id]
    print(f"[vlmevalkit] shard owns {len(shard_indices)} of {n} samples", flush=True)

    # Resume: load any existing TSV and skip already-done indices.
    done_indices: set[str] = set()
    existing_rows: list[dict] = []
    if tsv_path.exists():
        try:
            df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
            for _, row in df.iterrows():
                done_indices.add(str(row["index"]))
                existing_rows.append({"index": row["index"], "prediction": row["prediction"]})
            print(f"[vlmevalkit] resume: {len(done_indices)} samples already in TSV", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[vlmevalkit] could not parse existing TSV: {e}; starting fresh", flush=True)
            done_indices = set()
            existing_rows = []

    todo_idx = []
    for i in shard_indices:
        line = ds.data.iloc[i]
        idx = str(line["index"])
        if idx in done_indices:
            continue
        todo_idx.append(i)

    print(f"[vlmevalkit] {len(todo_idx)} samples to run", flush=True)

    if not todo_idx:
        print("[vlmevalkit] nothing to do; exiting", flush=True)
        return 0

    # Build vLLM client.
    max_pixels_kw = {}
    if args.max_pixels is not None:
        max_pixels_kw["max_pixels"] = args.max_pixels
    if args.min_pixels is not None:
        max_pixels_kw["min_pixels"] = args.min_pixels
    client = UITARSClient(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        **max_pixels_kw,
    )
    cfg = GenerationConfig(max_tokens=args.max_tokens, temperature=0.0, seed=0)

    rows_for_this_run: list[dict] = []
    pending_prompts: list[dict] = []
    pending_indices: list[str] = []

    def flush_batch() -> None:
        if not pending_prompts:
            return
        try:
            outs = client.generate(pending_prompts, cfg=cfg)
        except Exception as e:  # noqa: BLE001
            err = f"VLLM_GEN_FAIL: {type(e).__name__}: {e}"
            print(f"[vlmevalkit] batch failed: {err}", flush=True)
            for idx in pending_indices:
                rows_for_this_run.append({"index": idx, "prediction": err})
            pending_prompts.clear()
            pending_indices.clear()
            return
        for raw, idx in zip(outs, pending_indices):
            rows_for_this_run.append({"index": idx, "prediction": raw.strip()})
        pending_prompts.clear()
        pending_indices.clear()

    def write_tsv() -> None:
        all_rows = existing_rows + rows_for_this_run
        df = pd.DataFrame(all_rows)
        df.to_csv(tsv_path, sep="\t", index=False)

    t0 = time.time()
    n_done = 0
    for i in todo_idx:
        try:
            line = ds.data.iloc[i]
            idx = str(line["index"])
            msgs = ds.build_prompt(line)
            image, question = _msg_to_image_question(msgs)
            # Resize image to ARPO-training cap (max_pixels=2.1M) so input
            # distribution matches what UI-TARS-1.5-7B was trained on.
            resized, _smart_h, _smart_w = smart_resize_image(
                image,
                max_pixels=client.max_pixels,
                min_pixels=client.min_pixels,
            )
            prompt = _build_chat_prompt(resized, question)
            pending_prompts.append(prompt)
            pending_indices.append(idx)
        except Exception as e:  # noqa: BLE001
            err = f"PREP_FAIL: {type(e).__name__}: {e}"
            print(f"[vlmevalkit] sample {i} prep failed: {err}", flush=True)
            try:
                idx = str(ds.data.iloc[i]["index"])
            except Exception:  # noqa: BLE001
                idx = str(i)
            rows_for_this_run.append({"index": idx, "prediction": err})
            continue

        if len(pending_prompts) >= args.batch_size:
            flush_batch()
            write_tsv()
            n_done += args.batch_size
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            print(f"[vlmevalkit] {n_done}/{len(todo_idx)} done in {elapsed:.0f}s ({rate:.2f}/s)", flush=True)

    flush_batch()
    write_tsv()

    elapsed = time.time() - t0
    print(f"[vlmevalkit] shard done in {elapsed:.0f}s; new={len(rows_for_this_run)}", flush=True)
    print(f"[vlmevalkit] wrote {tsv_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
