"""After all shards write predictions.tsv, merge them and score via VLMEvalKit.

Reads:
  <output_dir>/shard_<i>/predictions.tsv  (or <output_dir>/predictions.tsv when num_shards=1)

Writes:
  <output_dir>/predictions.tsv     (merged, in VLMEvalKit's expected schema)
  <output_dir>/score.json           (VLMEvalKit's evaluation result)
  <output_dir>/results.jsonl        (per-sample (index, prediction, gold, correct) for our aggregator)
  <output_dir>/summary.json         (headline + per-class for our aggregator)

Robustness: per-bench scoring uses VLMEvalKit's strict-match (exact_matching) —
no LLM judge required for HallusionBench, POPE, MMVP, MMStar.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VLMEVALKIT_PATH = _REPO_ROOT / "cache_dirs" / "VLMEvalKit"
for p in (_REPO_ROOT, _VLMEVALKIT_PATH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _merge_shards(out_dir: Path) -> Path:
    shard_tsvs = sorted(out_dir.glob("shard_*/predictions.tsv"))
    merged_path = out_dir / "predictions.tsv"
    if not shard_tsvs:
        if merged_path.exists():
            return merged_path
        raise FileNotFoundError(f"no shard predictions in {out_dir}")
    dfs = []
    for f in shard_tsvs:
        try:
            dfs.append(pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False))
        except Exception as e:  # noqa: BLE001
            print(f"[score] could not read {f}: {e}", flush=True)
    if not dfs:
        raise RuntimeError("all shard TSVs unreadable")
    merged = pd.concat(dfs, ignore_index=True)
    # Dedup on index, keep last.
    merged = merged.drop_duplicates(subset=["index"], keep="last")
    merged.to_csv(merged_path, sep="\t", index=False)
    print(f"[score] merged {len(shard_tsvs)} shards -> {merged_path} ({len(merged)} rows)", flush=True)
    return merged_path


def _augment_with_dataset_columns(merged_tsv: Path, bench: str) -> Path:
    """VLMEvalKit's evaluators expect TSV columns from the source dataset (index,
    question, answer, image, ...). Our prediction TSV only has (index, prediction).
    We left-join the source TSV on `index` so evaluators have all columns.
    """
    from vlmeval.dataset import build_dataset

    ds = build_dataset(bench)
    src = ds.data.copy()
    src["index"] = src["index"].astype(str)
    pred = pd.read_csv(merged_tsv, sep="\t", dtype=str, keep_default_na=False)
    pred["index"] = pred["index"].astype(str)
    # Keep VLMEvalKit's expected column order: dataset columns first, then prediction.
    if "prediction" in src.columns:
        src = src.drop(columns=["prediction"])
    joined = src.merge(pred, on="index", how="inner")
    augmented_path = merged_tsv.parent / "predictions_full.tsv"
    joined.to_csv(augmented_path, sep="\t", index=False)
    print(f"[score] augmented TSV ({len(joined)} rows) -> {augmented_path}", flush=True)
    return augmented_path


def _score(augmented_tsv: Path, bench: str) -> dict[str, Any]:
    from vlmeval.dataset import build_dataset

    # Wipe stale VLMEvalKit intermediate files before scoring. Without this,
    # the evaluator reuses _auxmatch.xlsx etc. from a prior run with a smaller
    # predictions set (e.g. 10% smoke), and the headline scores stay frozen at
    # the smaller-N values even when the merged predictions.tsv has grown.
    parent = augmented_tsv.parent
    stem = augmented_tsv.stem
    for pat in (f"{stem}_auxmatch*", f"{stem}_score*", f"{stem}_tmp*",
                f"{stem}_acc*", f"{stem}_extract*"):
        for f in parent.glob(pat):
            try:
                f.unlink()
            except Exception:  # noqa: BLE001
                pass

    ds = build_dataset(bench)
    # Force exact-matching judge so we never call out to an LLM judge.
    judge_kwargs = {"model": "exact_matching", "nproc": 1}
    result = ds.evaluate(str(augmented_tsv), **judge_kwargs)
    if isinstance(result, pd.DataFrame):
        out = result.to_dict(orient="list" if len(result) > 1 else "records")
    elif isinstance(result, dict):
        out = result
    else:
        out = {"raw": str(result)}
    return out


def _emit_jsonl_and_summary(out_dir: Path, augmented_tsv: Path, bench: str, score: dict, model_path: str) -> None:
    """Write per-sample JSONL and a summary.json compatible with our aggregator."""
    from vlmeval.dataset import build_dataset
    from vlmeval.dataset.utils.yorn import YOrN_Extraction
    from vlmeval.smp import load

    ds = build_dataset(bench)
    df = pd.read_csv(augmented_tsv, sep="\t", dtype=str, keep_default_na=False)

    # Try the evaluator's auxmatch file if present (most accurate per-sample correctness).
    auxmatch_path = augmented_tsv.parent / (augmented_tsv.stem + "_auxmatch.xlsx")
    if not auxmatch_path.exists():
        # YORN benches write _auxmatch.xlsx; MCQ benches write different names. Try the most common patterns.
        for cand in augmented_tsv.parent.glob("*_auxmatch*"):
            auxmatch_path = cand
            break

    extracted_map: dict[str, str] = {}
    if auxmatch_path.exists():
        try:
            aux = load(str(auxmatch_path))
            if "extracted" in aux.columns:
                for _, r in aux.iterrows():
                    extracted_map[str(r["index"])] = str(r["extracted"])
        except Exception:  # noqa: BLE001
            pass

    jsonl_path = out_dir / "results.jsonl"
    n_total = 0
    n_hit = 0
    n_parse_err = 0
    by_class_hit: dict[str, int] = {}
    by_class_total: dict[str, int] = {}
    with jsonl_path.open("w") as f:
        for _, row in df.iterrows():
            idx = str(row["index"])
            pred = row.get("prediction", "")
            gold = str(row.get("answer", ""))
            extracted = extracted_map.get(idx, "")
            # Strict-match correctness: prefer evaluator's extracted vs gold; fall back to direct gold-in-pred.
            correct = False
            if extracted:
                correct = extracted.strip().lower() == gold.strip().lower()
            else:
                # Best-effort: yes/no detection
                pl = pred.lower()
                if gold.lower() in ("yes", "no") and gold.lower() in pl:
                    # crude: look for the answer word, prefer matching first char A-Z for MCQ
                    correct = pl.startswith(gold.lower()) or f" {gold.lower()}" in pl
                else:
                    correct = gold.strip().lower() in pl
            n_total += 1
            if pred.startswith("VLLM_GEN_FAIL") or pred.startswith("PREP_FAIL"):
                n_parse_err += 1
                correct = False
            if correct:
                n_hit += 1
            sub = str(row.get("category", row.get("subset", row.get("question_type", "all"))))
            by_class_total[sub] = by_class_total.get(sub, 0) + 1
            if correct:
                by_class_hit[sub] = by_class_hit.get(sub, 0) + 1
            f.write(json.dumps({
                "sample_id": idx,
                "prediction": pred,
                "gold": gold,
                "extracted": extracted,
                "correct": bool(correct),
                "sub_class": sub,
            }, ensure_ascii=False) + "\n")

    summary = {
        "header": {
            "benchmark": bench,
            "model_path": model_path,
            "n_planned": n_total,
        },
        "stats": {
            "n_total": n_total,
            "n_hit": n_hit,
            "n_parse_error": n_parse_err,
            "accuracy": (n_hit / n_total) if n_total else 0.0,
            "by_class": {
                c: {
                    "hit": by_class_hit.get(c, 0),
                    "total": by_class_total[c],
                    "accuracy": by_class_hit.get(c, 0) / by_class_total[c],
                }
                for c in sorted(by_class_total)
            },
            "vlmevalkit_score": score,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"[score] {bench}: accuracy={n_hit}/{n_total}={summary['stats']['accuracy']:.4f}, parse_err={n_parse_err}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--model-path", default="<unknown>")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    merged = _merge_shards(out_dir)
    augmented = _augment_with_dataset_columns(merged, args.bench)
    score = _score(augmented, args.bench)
    (out_dir / "score.json").write_text(json.dumps(score, ensure_ascii=False, default=str, indent=2))
    _emit_jsonl_and_summary(out_dir, augmented, args.bench, score, args.model_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
