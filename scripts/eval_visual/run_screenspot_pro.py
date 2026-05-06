"""Run UI-TARS on ScreenSpot-Pro (high-resolution professional GUI grounding).

Source: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding (HF mirror
`likaixin/ScreenSpot-Pro` if available).

Differs from ScreenSpot-V2 in two ways:
  1. Images are 4K-ish; smart_resize will reduce them, but keep the inverse-
     mapping exact via the existing parser. We do NOT lower max_pixels by
     default — UI-TARS-1.5 was trained with the same 12.8M cap.
  2. Targets average 0.07% of the screen, so misses are common; sub-class
     breakdown by industry/OS is essential.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_visual.common import (
    GenerationConfig,
    ResultsLog,
    RunStats,
    UITARSClient,
    build_grounding_prompt,
    ensure_pil_rgb,
    image_sha256,
    normalize_bbox,
    parse_grounding_response,
    point_in_bbox,
    selftest_coord_roundtrip,
    smart_resize_image,
    write_summary,
)


def _load_dataset(dataset_repo: str, split: str = "test"):
    from datasets import load_dataset

    return load_dataset(dataset_repo, split=split)


def _canonicalize_sample(sample: dict, idx: int) -> dict | None:
    """ScreenSpot-Pro schema fields:
      image, instruction, bbox (pixel), application, group (industry), platform.
    """
    image = sample.get("image")
    instruction = sample.get("instruction") or sample.get("prompt")
    bbox = sample.get("bbox") or sample.get("ground_truth") or sample.get("box")
    application = sample.get("application") or sample.get("app") or "unknown"
    group = sample.get("group") or sample.get("industry") or sample.get("category") or "unknown"
    platform = sample.get("platform") or sample.get("os") or "unknown"
    sample_id = str(sample.get("id") or sample.get("file_name") or idx)
    if image is None or instruction is None or bbox is None:
        return None
    if not isinstance(image, Image.Image):
        return None
    return {
        "sample_id": sample_id,
        "image": image,
        "instruction": str(instruction),
        "bbox_raw": list(bbox),
        "application": str(application),
        "group": str(group),
        "platform": str(platform),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-repo", default="likaixin/ScreenSpot-Pro")
    ap.add_argument("--split", default="test")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)  # smaller — high-res images
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if args.num_shards > 1:
        shard_dir = out_dir / f"shard_{args.shard_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "results.jsonl"
        summary_path = shard_dir / "summary.json"
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "results.jsonl"
        summary_path = out_dir / "summary.json"

    print(f"[screenspot_pro] model={args.model_path}", flush=True)
    print(f"[screenspot_pro] output={out_dir} shard={args.shard_id}/{args.num_shards}", flush=True)

    selftest_coord_roundtrip()
    print("[screenspot_pro] coord round-trip self-test passed", flush=True)

    ds = _load_dataset(args.dataset_repo, args.split)
    n = len(ds)
    if args.smoke > 0:
        n = min(n, args.smoke)
    print(f"[screenspot_pro] loaded {n} samples", flush=True)

    shard_indices = [i for i in range(n) if i % args.num_shards == args.shard_id]
    print(f"[screenspot_pro] shard owns {len(shard_indices)} of {n} samples", flush=True)
    log = ResultsLog.open(log_path)
    n_already = sum(1 for i in shard_indices if log.has(_canonicalize_sample(ds[int(i)], i)["sample_id"]))
    print(f"[screenspot_pro] resume: {n_already}/{len(shard_indices)} already done", flush=True)
    n = len(shard_indices)

    if n_already >= n:
        print("[screenspot_pro] all samples scored, skipping inference", flush=True)
        _emit_summary(log_path, summary_path, args, n)
        return 0

    max_pixels_kw = {}
    if args.max_pixels is not None:
        max_pixels_kw["max_pixels"] = args.max_pixels
    if args.min_pixels is not None:
        max_pixels_kw["min_pixels"] = args.min_pixels
    client = UITARSClient(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **max_pixels_kw,
    )
    cfg = GenerationConfig(max_tokens=args.max_tokens, temperature=0.0, seed=0)

    t0 = time.time()
    n_done = 0
    pending_prompts: list[dict] = []
    pending_records: list[dict] = []

    def flush_batch() -> None:
        nonlocal n_done
        if not pending_prompts:
            return
        try:
            outs = client.generate(pending_prompts, cfg=cfg)
        except Exception as e:  # noqa: BLE001
            err = f"vllm_generate_failed: {type(e).__name__}: {e}"
            for rec in pending_records:
                rec["raw_response"] = ""
                rec["parse_method"] = "vllm_failure"
                rec["parse_error"] = err
                rec["predicted_xy"] = None
                rec["hit"] = False
                log.append(rec)
            pending_prompts.clear()
            pending_records.clear()
            print(f"[screenspot_pro] batch failed: {err}", flush=True)
            return
        for raw, rec in zip(outs, pending_records):
            try:
                pred = parse_grounding_response(
                    raw,
                    smart_h=rec["_smart_h"],
                    smart_w=rec["_smart_w"],
                    orig_h=rec["_orig_h"],
                    orig_w=rec["_orig_w"],
                    allow_refusal=False,
                )
                bbox_pix = normalize_bbox(rec["bbox_raw"], rec["_orig_w"], rec["_orig_h"])
                hit = point_in_bbox(pred.point_orig, bbox_pix)
                rec.update(
                    raw_response=raw,
                    parse_method=pred.parse_method,
                    parse_error=pred.parse_error,
                    predicted_xy=list(pred.point_orig) if pred.point_orig else None,
                    bbox_pix=list(bbox_pix),
                    hit=bool(hit),
                )
            except Exception as e:  # noqa: BLE001
                rec.update(
                    raw_response=raw,
                    parse_method="exception",
                    parse_error=f"{type(e).__name__}: {e}",
                    predicted_xy=None,
                    hit=False,
                )
            for k in list(rec.keys()):
                if k.startswith("_"):
                    rec.pop(k)
            log.append(rec)
            n_done += 1
        pending_prompts.clear()
        pending_records.clear()
        if n_done and n_done % (args.batch_size * 4) == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            print(f"[screenspot_pro] {n_done} done in {elapsed:.0f}s ({rate:.2f}/s)", flush=True)

    for i in shard_indices:
        try:
            raw_sample = ds[int(i)]
        except Exception as e:  # noqa: BLE001
            print(f"[screenspot_pro] dataset[{i}] load failed: {e}", flush=True)
            continue
        c = _canonicalize_sample(raw_sample, i)
        if c is None:
            continue
        if log.has(c["sample_id"]):
            continue

        try:
            image = ensure_pil_rgb(c["image"])
            orig_w, orig_h = image.size
            resized, smart_h, smart_w = smart_resize_image(
                image,
                max_pixels=client.max_pixels,
                min_pixels=client.min_pixels,
            )
            prompt = build_grounding_prompt(resized, c["instruction"])
            rec = {
                "sample_id": c["sample_id"],
                "instruction": c["instruction"],
                "bbox_raw": c["bbox_raw"],
                "application": c["application"],
                "group": c["group"],
                "platform": c["platform"],
                "image_sha256": image_sha256(image),
                "orig_h": orig_h,
                "orig_w": orig_w,
                "smart_h": smart_h,
                "smart_w": smart_w,
                "_orig_h": orig_h,
                "_orig_w": orig_w,
                "_smart_h": smart_h,
                "_smart_w": smart_w,
            }
            pending_prompts.append(prompt)
            pending_records.append(rec)
        except Exception as e:  # noqa: BLE001
            err = f"prep_failed: {type(e).__name__}: {e}"
            print(f"[screenspot_pro] sample {c['sample_id']} prep failed: {err}", flush=True)
            log.append(
                {
                    "sample_id": c["sample_id"],
                    "instruction": c.get("instruction", ""),
                    "bbox_raw": c.get("bbox_raw"),
                    "application": c.get("application", "unknown"),
                    "group": c.get("group", "unknown"),
                    "platform": c.get("platform", "unknown"),
                    "raw_response": "",
                    "parse_method": "prep_failure",
                    "parse_error": err,
                    "predicted_xy": None,
                    "hit": False,
                }
            )
            continue

        if len(pending_prompts) >= args.batch_size:
            flush_batch()

    flush_batch()
    elapsed = time.time() - t0
    print(f"[screenspot_pro] done in {elapsed:.0f}s; new samples: {n_done}", flush=True)
    _emit_summary(log_path, summary_path, args, n)
    return 0


def _emit_summary(log_path: Path, summary_path: Path, args, n_planned: int) -> None:
    stats = RunStats()
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sub_class = f"{rec.get('group', 'unknown')}|{rec.get('platform', 'unknown')}"
            scored = rec.get("predicted_xy") is not None
            hit = bool(rec.get("hit"))
            parse_error = rec.get("parse_error") is not None
            stats.record(sub_class, scored=scored, hit=hit, parse_error=parse_error)
    header = {
        "benchmark": "screenspot_pro",
        "model_path": args.model_path,
        "dataset_repo": args.dataset_repo,
        "split": args.split,
        "n_planned": n_planned,
        "smoke": args.smoke,
    }
    write_summary(summary_path, header, stats)
    print(
        f"[screenspot_pro] accuracy={stats.n_hit}/{stats.n_total}={stats.summary_dict()['accuracy']:.4f} "
        f"(parse_err={stats.n_parse_error})",
        flush=True,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
