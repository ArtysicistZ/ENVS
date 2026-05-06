"""Aggregate visual-eval JSONL results into a 3x3 markdown table.

Walks `logs/eval_visual/<checkpoint_tag>/<benchmark>/results.jsonl` for every
combination present and emits a single markdown summary suitable for inclusion
in docs/ARPO_EXPERIMENTS_REPORT.md.

Usage:
  python scripts/eval_visual/summary.py --root logs/eval_visual --out logs/eval_visual/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCHES = (
    # grounding sweep
    "screenspot",
    "screenspot_pro",
    "osworld_g",
    # perception/hallucination sweep wave 1 (VLMEvalKit)
    "HallusionBench",
    "POPE",
    "MMVP",
    "MMStar",
    # perception/hallucination sweep wave 2 (VLMEvalKit)
    "BLINK",
    "MathVerse_MINI_Vision_Only",
    "AMBER",
)
BENCH_LABELS = {
    "screenspot": "ScreenSpot (V1, 1272)",
    "screenspot_pro": "ScreenSpot-Pro (1581)",
    "osworld_g": "OSWorld-G (564)",
    "HallusionBench": "HallusionBench (951)",
    "POPE": "POPE (5127)",
    "MMVP": "MMVP (300 pair)",
    "MMStar": "MMStar (1500)",
    "BLINK": "BLINK (1901)",
    "MathVerse_MINI_Vision_Only": "MathVerse-VO (788)",
    "AMBER": "AMBER (3000 of 14216)",
}


def _aggregate_jsonl(path: Path) -> dict:
    n_total = 0
    n_hit = 0
    n_parse_err = 0
    n_refusal_correct = 0
    n_refusal_total = 0
    by_class_total: dict[str, int] = {}
    by_class_hit: dict[str, int] = {}
    if not path.exists():
        return {"present": False}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            # Perception benches use `correct` instead of `hit`.
            is_correct = rec.get("hit") if "hit" in rec else rec.get("correct", False)
            if is_correct:
                n_hit += 1
            if rec.get("parse_error") or (
                isinstance(rec.get("prediction"), str)
                and (rec["prediction"].startswith("VLLM_GEN_FAIL") or rec["prediction"].startswith("PREP_FAIL"))
            ):
                n_parse_err += 1
            # ScreenSpot-V2: data_source + data_type
            if "data_source" in rec and "data_type" in rec:
                cls = f"{rec['data_source']}_{rec['data_type']}"
            elif "group" in rec:
                cls = f"{rec.get('group', 'unknown')}|{rec.get('platform', 'unknown')}"
            elif "box_type" in rec:
                cls = rec["box_type"]
                if cls == "refusal":
                    n_refusal_total += 1
                    if rec.get("hit"):
                        n_refusal_correct += 1
            elif "sub_class" in rec:
                cls = str(rec["sub_class"])
            else:
                cls = "all"
            by_class_total[cls] = by_class_total.get(cls, 0) + 1
            if is_correct:
                by_class_hit[cls] = by_class_hit.get(cls, 0) + 1
    return {
        "present": True,
        "n_total": n_total,
        "n_hit": n_hit,
        "n_parse_err": n_parse_err,
        "accuracy": n_hit / n_total if n_total else 0.0,
        "by_class": {
            c: {
                "hit": by_class_hit.get(c, 0),
                "total": by_class_total[c],
                "accuracy": by_class_hit.get(c, 0) / by_class_total[c],
            }
            for c in sorted(by_class_total)
        },
        "n_refusal_correct": n_refusal_correct,
        "n_refusal_total": n_refusal_total,
    }


def _scan(root: Path) -> dict[str, dict[str, dict]]:
    """Returns {checkpoint_tag: {benchmark: aggregated_dict}}."""
    out: dict[str, dict[str, dict]] = {}
    if not root.exists():
        return out
    for ck_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        bench_results: dict[str, dict] = {}
        for bench in BENCHES:
            jsonl = ck_dir / bench / "results.jsonl"
            bench_results[bench] = _aggregate_jsonl(jsonl)
        out[ck_dir.name] = bench_results
    return out


def _fmt_acc(d: dict) -> str:
    if not d.get("present") or d["n_total"] == 0:
        return "—"
    pe = f", parse_err={d['n_parse_err']}" if d["n_parse_err"] else ""
    return f"{d['n_hit']}/{d['n_total']} = {d['accuracy']*100:.1f}%{pe}"


def _emit_md(by_ck: dict[str, dict[str, dict]]) -> str:
    lines: list[str] = []
    lines.append("# Visual VLM Generalization — 3×N Results\n")
    lines.append("Eval is **clean** (no runtime noise injected). Greedy decode, seed=0.\n")
    headers = ["Checkpoint"] + [BENCH_LABELS[b] for b in BENCHES]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for ck, bench_results in by_ck.items():
        row = [ck]
        for b in BENCHES:
            row.append(_fmt_acc(bench_results.get(b, {})))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Per-class breakdown\n")
    for ck, bench_results in by_ck.items():
        lines.append(f"### {ck}\n")
        for b in BENCHES:
            d = bench_results.get(b, {})
            lines.append(f"**{BENCH_LABELS[b]}** — {_fmt_acc(d)}")
            if d.get("present") and d.get("by_class"):
                lines.append("")
                lines.append("| sub-class | hit | total | acc |")
                lines.append("|---|---|---|---|")
                for cls, cd in sorted(d["by_class"].items(), key=lambda kv: -kv[1]["total"]):
                    lines.append(f"| {cls} | {cd['hit']} | {cd['total']} | {cd['accuracy']*100:.1f}% |")
                lines.append("")
            else:
                lines.append("(no data)\n")
        if any(b == "osworld_g" and bench_results.get(b, {}).get("n_refusal_total") for b in BENCHES):
            d = bench_results.get("osworld_g", {})
            lines.append(
                f"OSWorld-G refusal: {d['n_refusal_correct']}/{d['n_refusal_total']} correct\n"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/eval_visual")
    ap.add_argument("--out", default="logs/eval_visual/SUMMARY.md")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    by_ck = _scan(root)
    md = _emit_md(by_ck)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"\nWrote summary to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
