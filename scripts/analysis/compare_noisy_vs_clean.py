#!/usr/bin/env python3
"""Analyze a noisy-eval output against the deterministic fires_for_task_eval
partition. Optionally diffs against a clean baseline for the same model.

Usage:
    python scripts/analysis/compare_noisy_vs_clean.py \
        --noisy checkpoints/noisy_eval/base_uitars_n1_t0_heldout/eval_results_at_0.json \
        [--clean <clean_baseline_results.json>] \
        [--label "base UI-TARS n=1 t=0"]
"""
import argparse
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "OSWorld"))

from OSWorld.evaluation_examples.noise_generation.runtime_sampler import fires_for_task_eval


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for tid, rec in data.items():
        if isinstance(rec, dict) and "success_rate" in rec:
            out[tid] = float(rec["success_rate"])
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--clean", default=None)
    ap.add_argument("--label", default="model")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    noisy_sr = load_results(args.noisy)
    print(f"=== {args.label} ===")
    print(f"noisy eval file: {args.noisy}   ({len(noisy_sr)} tasks)")

    noisy_ids = [t for t in noisy_sr if fires_for_task_eval(t) == 1]
    clean_ids = [t for t in noisy_sr if fires_for_task_eval(t) == 0]
    print(f"partition: {len(noisy_ids)} noisy (fires=1) + {len(clean_ids)} clean (fires=0)")
    print()

    overall = mean([noisy_sr[t] for t in noisy_sr])
    noisy_mean = mean([noisy_sr[t] for t in noisy_ids])
    clean_mean = mean([noisy_sr[t] for t in clean_ids])
    print(f"Overall SR (all 300):                {overall:.3f}  (n={len(noisy_sr)})")
    print(f"Clean subset SR (fires=0, baseline): {clean_mean:.3f}  (n={len(clean_ids)})")
    print(f"Noisy subset SR (fires=1, robust):   {noisy_mean:.3f}  (n={len(noisy_ids)})")
    print(f"In-run delta (noisy - clean):        {noisy_mean - clean_mean:+.3f}")

    if args.clean:
        base = load_results(args.clean)
        print()
        print(f"clean baseline file: {args.clean}   ({len(base)} tasks)")
        matched = [t for t in noisy_sr if t in base]

        def pair(ids):
            ps = [(noisy_sr[t], base[t]) for t in ids if t in base]
            if not ps:
                return 0.0, 0.0, 0.0, 0
            nm = mean([a for a, _ in ps]); cm = mean([b for _, b in ps])
            return nm, cm, nm - cm, len(ps)

        nm, cm, d, n = pair(matched)
        print(f"Paired Δ (matched={n}):           noisy={nm:.3f}  clean={cm:.3f}  Δ={d:+.3f}")
        nm, cm, d, n = pair(noisy_ids)
        print(f"  noisy subset (n={n}):           noisy={nm:.3f}  clean={cm:.3f}  Δ={d:+.3f}")
        nm, cm, d, n = pair(clean_ids)
        print(f"  clean subset (n={n}, expect~0): noisy={nm:.3f}  clean={cm:.3f}  Δ={d:+.3f}")

        print()
        print(f"Top {args.top_k} worst noisy-subset collapses (clean>0, noisy<clean):")
        collapses = []
        for t in noisy_ids:
            if t not in base:
                continue
            cl = base[t]; ns = noisy_sr[t]
            if cl > 0 and ns < cl:
                collapses.append((ns - cl, t, ns, cl))
        for d, t, ns, cl in sorted(collapses)[: args.top_k]:
            print(f"  Δ={d:+.2f}  {t[:12]}  clean={cl:.2f}  noisy={ns:.2f}")


if __name__ == "__main__":
    main()
