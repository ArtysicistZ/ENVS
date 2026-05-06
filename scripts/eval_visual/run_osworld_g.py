"""Run UI-TARS on OSWorld-G (xlang-ai/OSWorld-G NeurIPS 2025 spotlight).

Source: https://github.com/xlang-ai/OSWorld-G  — cloned to
cache_dirs/osworld_g_repo. The benchmark is **not** on HuggingFace; it ships as
a JSON file plus images on disk.

Schema (from OSWorld-G_refined.json):
  - id              str   "0FOB4CLBT2-0"
  - image_path      str   "0FOB4CLBT2.png" (relative to benchmark/images/)
  - image_size      [w, h]
  - instruction     str
  - box_type        "bbox" | "polygon" | "refusal"
  - box_coordinates list  bbox=[x,y,w,h], polygon=[x1,y1,...], refusal=[0,0,0,0]
  - GUI_types       list[str]

Eval semantics (matching OSWorld-G evaluation/eval.py):
  - bbox:    center (cx,cy) inside [x, y, x+w, y+h]
  - polygon: center inside polygon (ray-casting)
  - refusal: ALL center coords are negative (UI-TARS expresses this via fail();
             we encode fail() as point (-1, -1)).

Uses the agent-style prompt (with fail() in the action space) so the model can
naturally refuse. Falls back to grounding-prompt parsing if the agent grammar
doesn't trigger.
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
    build_agent_prompt,
    ensure_pil_rgb,
    image_sha256,
    parse_grounding_response,
    selftest_coord_roundtrip,
    smart_resize_image,
    write_summary,
)


def _is_point_in_rectangle(point, rect):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def _is_point_in_polygon(point, polygon):
    """Ray-casting. Polygon is flat [x1,y1,x2,y2,...,xn,yn]."""
    x, y = point
    n = len(polygon) // 2
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i * 2], polygon[i * 2 + 1]
        xj, yj = polygon[j * 2], polygon[j * 2 + 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _score(point, box_type: str, box_coordinates: list) -> bool:
    """Direct port of OSWorld-G evaluation/eval.py:_eval."""
    if box_type == "refusal":
        # Model correctly declined iff both coords are negative.
        return all(c < 0 for c in point)
    # For non-refusal gold, refusal point (negatives) is wrong.
    if any(c < 0 for c in point):
        return False
    if box_type == "bbox":
        x, y, w, h = box_coordinates
        return _is_point_in_rectangle(point, [x, y, x + w, y + h])
    if box_type == "polygon":
        return _is_point_in_polygon(point, box_coordinates)
    return False


def _load_benchmark(repo_dir: Path) -> tuple[list[dict], Path]:
    bench_json = repo_dir / "benchmark" / "OSWorld-G_refined.json"
    images_dir = repo_dir / "benchmark" / "images"
    if not bench_json.exists():
        raise FileNotFoundError(f"OSWorld-G benchmark JSON not found at {bench_json}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"OSWorld-G images dir not found at {images_dir}")
    with bench_json.open() as f:
        data = json.load(f)
    return data, images_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--repo-dir",
        default=str(Path(__file__).resolve().parents[2] / "cache_dirs" / "osworld_g_repo"),
    )
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=12)
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

    print(f"[osworld_g] model={args.model_path}", flush=True)
    print(f"[osworld_g] output={out_dir} shard={args.shard_id}/{args.num_shards}", flush=True)

    selftest_coord_roundtrip()
    print("[osworld_g] coord round-trip self-test passed", flush=True)

    data, images_dir = _load_benchmark(Path(args.repo_dir))
    n = len(data)
    if args.smoke > 0:
        n = min(n, args.smoke)
    print(f"[osworld_g] loaded {n} samples from {args.repo_dir}", flush=True)

    # Shard the dataset by index modulo num_shards.
    sharded_data = [s for i, s in enumerate(data[:n]) if i % args.num_shards == args.shard_id]
    print(f"[osworld_g] shard owns {len(sharded_data)} of {n} samples", flush=True)

    log = ResultsLog.open(log_path)
    n_already = sum(1 for s in sharded_data if log.has(str(s["id"])))
    print(f"[osworld_g] resume: {n_already}/{len(sharded_data)} already done", flush=True)
    n = len(sharded_data)

    if n_already >= n:
        print("[osworld_g] all samples scored, skipping inference", flush=True)
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
            print(f"[osworld_g] batch failed: {err}", flush=True)
            return
        for raw, rec in zip(outs, pending_records):
            try:
                pred = parse_grounding_response(
                    raw,
                    smart_h=rec["_smart_h"],
                    smart_w=rec["_smart_w"],
                    orig_h=rec["_orig_h"],
                    orig_w=rec["_orig_w"],
                    allow_refusal=True,
                )
                if pred.is_refusal:
                    point = (-1.0, -1.0)
                elif pred.point_orig is not None:
                    point = pred.point_orig
                else:
                    # Unparseable — score as a miss with positive sentinel so it
                    # cannot accidentally satisfy a refusal gold.
                    point = (1e9, 1e9)
                hit = _score(list(point), rec["box_type"], rec["box_coordinates"])
                rec.update(
                    raw_response=raw,
                    parse_method=pred.parse_method + ("|refusal" if pred.is_refusal else ""),
                    parse_error=pred.parse_error,
                    predicted_xy=list(point),
                    is_refusal_pred=pred.is_refusal,
                    hit=bool(hit),
                )
            except Exception as e:  # noqa: BLE001
                rec.update(
                    raw_response=raw,
                    parse_method="exception",
                    parse_error=f"{type(e).__name__}: {e}",
                    predicted_xy=None,
                    is_refusal_pred=False,
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
            print(f"[osworld_g] {n_done} done in {elapsed:.0f}s ({rate:.2f}/s)", flush=True)

    for sample in sharded_data:
        sample_id = str(sample["id"])
        if log.has(sample_id):
            continue
        try:
            image_path = images_dir / sample["image_path"]
            if not image_path.exists():
                err = f"missing_image: {image_path}"
                print(f"[osworld_g] {sample_id}: {err}", flush=True)
                log.append(
                    {
                        "sample_id": sample_id,
                        "instruction": sample.get("instruction", ""),
                        "box_type": sample.get("box_type"),
                        "box_coordinates": sample.get("box_coordinates"),
                        "GUI_types": sample.get("GUI_types"),
                        "raw_response": "",
                        "parse_method": "missing_image",
                        "parse_error": err,
                        "predicted_xy": None,
                        "is_refusal_pred": False,
                        "hit": False,
                    }
                )
                continue
            image = ensure_pil_rgb(Image.open(image_path))
            orig_w, orig_h = image.size
            resized, smart_h, smart_w = smart_resize_image(
                image,
                max_pixels=client.max_pixels,
                min_pixels=client.min_pixels,
            )
            prompt = build_agent_prompt(resized, sample["instruction"])
            rec = {
                "sample_id": sample_id,
                "instruction": sample["instruction"],
                "box_type": sample["box_type"],
                "box_coordinates": sample["box_coordinates"],
                "GUI_types": sample.get("GUI_types") or [],
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
            print(f"[osworld_g] {sample_id} prep failed: {err}", flush=True)
            log.append(
                {
                    "sample_id": sample_id,
                    "instruction": sample.get("instruction", ""),
                    "box_type": sample.get("box_type"),
                    "box_coordinates": sample.get("box_coordinates"),
                    "GUI_types": sample.get("GUI_types"),
                    "raw_response": "",
                    "parse_method": "prep_failure",
                    "parse_error": err,
                    "predicted_xy": None,
                    "is_refusal_pred": False,
                    "hit": False,
                }
            )
            continue

        if len(pending_prompts) >= args.batch_size:
            flush_batch()

    flush_batch()
    elapsed = time.time() - t0
    print(f"[osworld_g] done in {elapsed:.0f}s; new samples: {n_done}", flush=True)
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
            sub_class = rec.get("box_type") or "unknown"
            scored = rec.get("predicted_xy") is not None
            hit = bool(rec.get("hit"))
            parse_error = rec.get("parse_error") is not None
            stats.record(sub_class, scored=scored, hit=hit, parse_error=parse_error)
            # Refusal accounting
            if rec.get("box_type") == "refusal":
                if hit:
                    stats.n_refusal_correct += 1
                else:
                    stats.n_refusal_wrong += 1
    header = {
        "benchmark": "osworld_g",
        "model_path": args.model_path,
        "repo_dir": args.repo_dir,
        "n_planned": n_planned,
        "smoke": args.smoke,
    }
    write_summary(summary_path, header, stats)
    print(
        f"[osworld_g] accuracy={stats.n_hit}/{stats.n_total}={stats.summary_dict()['accuracy']:.4f} "
        f"(parse_err={stats.n_parse_error}, refusal_correct={stats.n_refusal_correct})",
        flush=True,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
