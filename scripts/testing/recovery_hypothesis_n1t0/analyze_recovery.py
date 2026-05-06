"""
Tests three hypotheses about why ARPO is unstable vs MCTS-SFT init:

  H1 (NEW main): ARPO can't explore mid-trajectory.
                 SFT trajectories include detours / corrective actions / mid-point exploration;
                 ARPO is rewarded only at the trajectory level so mid-trajectory recovery is
                 not credit-assigned, and the policy collapses onto the argmax action.
  H2: ARPO wanders too long.
  H3: ARPO's first few key steps are wrong.

Inputs:
  SFT_PATH  = sft_init/trajectories_at_0.jsonl
  ARPO_PATH = arpo_step35/trajectories_at_35.jsonl
  (both n=1, t=0, the only paired runs available)

Outputs in the same directory:
  summary_table.csv         per-cohort stats
  pairs_table.csv           paired-by-task SFT vs ARPO stats
  fig_*.png                 distribution / bar charts
  REPORT.md                 written interpretation
"""
import json, re, os, statistics, collections, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
SFT_PATH  = "/mnt/kevinzyz/arpo_local/checkpoints/clean_eval/arpo_86tasks_clean_from_v2.1_no_mask_diagnosis/sft_init/trajectories_at_0.jsonl"
ARPO_PATH = "/mnt/kevinzyz/arpo_local/checkpoints/clean_eval/arpo_86tasks_clean_from_v2.1_no_mask_diagnosis/arpo_step35/trajectories_at_35.jsonl"

# ------------- parsing -------------
ACTION_FN_RE = re.compile(r"^\s*(\w+)\s*\(")
COORD_RE     = re.compile(r"\((\d+),\s*(\d+)\)")
THOUGHT_RE   = re.compile(r"Thought:\s*(.*?)(?:\nAction:|\Z)", re.S)
ACTION_RE    = re.compile(r"Action:\s*(.*)", re.S)
SELF_CORRECT_RE = re.compile(
    r"wrong spot|wrong turn|let me start over|start over|going in circles|"
    r"isn['’]t working|reassess|change my (?:strategy|approach)|didn['’]t work|"
    r"different approach|different way|took a wrong",
    re.I,
)

def parse_step(s):
    th = THOUGHT_RE.search(s);  th = th.group(1).strip() if th else ""
    ac = ACTION_RE.search(s);   ac = ac.group(1).strip() if ac else s.strip()
    fn = ACTION_FN_RE.match(ac); fn = fn.group(1) if fn else "NA"
    coords = [(int(x), int(y)) for x, y in COORD_RE.findall(ac)]
    return {"fn": fn, "ac": ac, "th": th, "coords": coords}

def cluster_id(coord, clusters, tol=30):
    """Bucket a coordinate against existing cluster centers within tol."""
    for cid, c in enumerate(clusters):
        if abs(coord[0] - c[0]) < tol and abs(coord[1] - c[1]) < tol:
            return cid
    clusters.append(coord)
    return len(clusters) - 1

# ------------- features per trajectory -------------
def features(steps):
    parsed = [parse_step(s["action"]) for s in steps]
    fns = [p["fn"] for p in parsed]
    n = len(fns)

    # max consecutive same-fn run
    max_run = 1; cur = 1
    for i in range(1, n):
        if fns[i] == fns[i-1]:
            cur += 1; max_run = max(max_run, cur)
        else:
            cur = 1

    # click coord clusters (any near-duplicate within 30 px counts as same cluster)
    clusters = []
    cluster_ids = []
    for p in parsed:
        for c in p["coords"]:
            cluster_ids.append(cluster_id(c, clusters))
    n_unique_clusters = len(clusters)
    n_clicks = len(cluster_ids)

    # Action-type diversity
    n_unique_fns = len(set(fns))

    # Self-correction language
    n_self_correct = sum(1 for p in parsed if SELF_CORRECT_RE.search(p["th"]))

    # Recovery: after a self-correction thought OR a duplicate-of-prev action,
    # did the NEXT action differ in fn or in click cluster?
    recover_attempts = 0
    recover_success  = 0
    for i in range(n - 1):
        prev_fn = fns[i-1] if i >= 1 else None
        is_dup_of_prev = (i >= 1 and fns[i] == prev_fn)
        is_self_correct = bool(SELF_CORRECT_RE.search(parsed[i]["th"]))
        if not (is_dup_of_prev or is_self_correct):
            continue
        recover_attempts += 1
        # Compare step i to step i+1
        a, b = parsed[i], parsed[i+1]
        if a["fn"] != b["fn"]:
            recover_success += 1; continue
        # Same fn — check click cluster
        if a["coords"] and b["coords"]:
            ca = cluster_id(a["coords"][0], clusters)
            cb = cluster_id(b["coords"][0], clusters)
            if ca != cb:
                recover_success += 1
    recover_rate = recover_success / recover_attempts if recover_attempts else None

    # "Back-and-forth" / non-monotone exploration: count fn transitions A->B->A (B != A)
    bandf = 0
    for i in range(2, n):
        if fns[i] == fns[i-2] and fns[i-1] != fns[i]:
            bandf += 1

    # Distinct fn transitions (unique 2-grams), normalized
    bigrams = set(zip(fns[:-1], fns[1:])) if n > 1 else set()
    n_unique_bigrams = len(bigrams)

    # First-3-step bad signals
    first3_bad = 0; prev = None
    for i, p in enumerate(parsed[:3]):
        bad = (p["fn"] == "wait") \
              or (i >= 1 and p["fn"] == prev) \
              or bool(SELF_CORRECT_RE.search(p["th"]))
        first3_bad += int(bad)
        prev = p["fn"]

    return dict(
        n=n,
        n_unique_fns=n_unique_fns,
        n_unique_coord_clusters=n_unique_clusters,
        n_clicks=n_clicks,
        max_run=max_run,
        n_self_correct=n_self_correct,
        recover_attempts=recover_attempts,
        recover_success=recover_success,
        recover_rate=recover_rate,
        bandf=bandf,
        n_unique_bigrams=n_unique_bigrams,
        first3_bad=first3_bad,
    )

# ------------- load -------------
def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]

sft_records = load_jsonl(SFT_PATH)
arpo_records = load_jsonl(ARPO_PATH)
sft  = {r["task_id"]: r for r in sft_records}
arpo = {r["task_id"]: r for r in arpo_records}
common = sorted(set(sft) & set(arpo))

# attach features
for r in sft_records + arpo_records:
    r["feat"] = features(r["steps"])

# ------------- cohort summary -------------
def cohort_stats(recs, predicate):
    sub = [r for r in recs if predicate(r)]
    if not sub: return {}
    keys = ["n", "n_unique_fns", "n_unique_coord_clusters", "n_clicks",
            "max_run", "n_self_correct", "bandf", "n_unique_bigrams", "first3_bad"]
    out = {"count": len(sub)}
    for k in keys:
        vals = [r["feat"][k] for r in sub]
        out[f"mean_{k}"]   = float(np.mean(vals))
        out[f"median_{k}"] = float(np.median(vals))
    rrs = [r["feat"]["recover_rate"] for r in sub if r["feat"]["recover_rate"] is not None]
    out["mean_recover_rate"] = float(np.mean(rrs)) if rrs else float("nan")
    out["n_with_recover_attempts"] = len(rrs)
    out["mean_recover_attempts"] = float(np.mean([r["feat"]["recover_attempts"] for r in sub]))
    return out

cohorts = {
    "SFT_success":  (sft_records, lambda r: r["eval_result"] > 0.0),
    "SFT_fail":     (sft_records, lambda r: r["eval_result"] == 0.0),
    "ARPO_success": (arpo_records, lambda r: r["eval_result"] > 0.0),
    "ARPO_fail":    (arpo_records, lambda r: r["eval_result"] == 0.0),
}
summary = {label: cohort_stats(rs, p) for label, (rs, p) in cohorts.items()}

with open(os.path.join(OUT, "summary_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    cols = ["metric"] + list(cohorts)
    w.writerow(cols)
    metric_keys = [k for k in summary["SFT_success"] if k != "count"]
    w.writerow(["count"] + [summary[c]["count"] for c in cohorts])
    for k in metric_keys:
        w.writerow([k] + [f"{summary[c].get(k, float('nan')):.3f}" for c in cohorts])

# ------------- per-task paired table (SFT vs ARPO) -------------
with open(os.path.join(OUT, "pairs_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "task_id", "sft_pass", "arpo_pass", "transition",
        "sft_n", "arpo_n",
        "sft_unique_fns", "arpo_unique_fns",
        "sft_unique_clusters", "arpo_unique_clusters",
        "sft_max_run", "arpo_max_run",
        "sft_self_correct", "arpo_self_correct",
        "sft_recover_rate", "arpo_recover_rate",
        "sft_bandf", "arpo_bandf",
        "sft_first3_bad", "arpo_first3_bad",
    ])
    for tid in common:
        s, a = sft[tid], arpo[tid]
        sp = s["eval_result"] > 0; ap = a["eval_result"] > 0
        trans = ("PP" if sp and ap else "PF" if sp else "FP" if ap else "FF")
        rs = s["feat"]["recover_rate"]; ra = a["feat"]["recover_rate"]
        w.writerow([
            tid, int(sp), int(ap), trans,
            s["feat"]["n"], a["feat"]["n"],
            s["feat"]["n_unique_fns"], a["feat"]["n_unique_fns"],
            s["feat"]["n_unique_coord_clusters"], a["feat"]["n_unique_coord_clusters"],
            s["feat"]["max_run"], a["feat"]["max_run"],
            s["feat"]["n_self_correct"], a["feat"]["n_self_correct"],
            f"{rs:.3f}" if rs is not None else "",
            f"{ra:.3f}" if ra is not None else "",
            s["feat"]["bandf"], a["feat"]["bandf"],
            s["feat"]["first3_bad"], a["feat"]["first3_bad"],
        ])

# ------------- figures -------------
def vals(recs, pred, key):
    return [r["feat"][key] for r in recs if pred(r)]

def boxset(metric, ylabel, fname):
    data = [
        vals(sft_records,  lambda r: r["eval_result"] > 0,  metric),
        vals(sft_records,  lambda r: r["eval_result"] == 0, metric),
        vals(arpo_records, lambda r: r["eval_result"] > 0,  metric),
        vals(arpo_records, lambda r: r["eval_result"] == 0, metric),
    ]
    labels = ["SFT pass", "SFT fail", "ARPO pass", "ARPO fail"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    colors = ["#7fbf7f", "#ff8c8c", "#4f9bff", "#c87fff"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by cohort")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)

boxset("n_unique_fns", "Distinct action-fn types per trajectory", "fig_unique_fns.png")
boxset("n_unique_coord_clusters", "Distinct click clusters (~30 px) per trajectory", "fig_unique_clusters.png")
boxset("max_run", "Max consecutive same-fn run", "fig_max_run.png")
boxset("n_self_correct", "Self-correction phrases (count)", "fig_self_correct.png")
boxset("bandf", "Back-and-forth fn transitions A->B->A", "fig_bandf.png")
boxset("n", "Trajectory length (steps)", "fig_length.png")
boxset("first3_bad", "Bad-signal count in first 3 steps", "fig_first3.png")

# Recovery-rate bar chart (mean recovery rate by cohort, with N and recover-attempt sample size)
def mean_or_nan(vs): return float(np.mean(vs)) if vs else float("nan")
recover = {}
for label, (rs, pred) in cohorts.items():
    sub = [r for r in rs if pred(r)]
    rates = [r["feat"]["recover_rate"] for r in sub if r["feat"]["recover_rate"] is not None]
    recover[label] = (mean_or_nan(rates), len(rates))
fig, ax = plt.subplots(figsize=(6.5, 4.0))
labels = list(recover)
means  = [recover[l][0] for l in labels]
ns     = [recover[l][1] for l in labels]
bars = ax.bar(labels, means,
              color=["#7fbf7f","#ff8c8c","#4f9bff","#c87fff"], alpha=0.7,
              edgecolor="black")
for bar, n, m in zip(bars, ns, means):
    ax.text(bar.get_x() + bar.get_width()/2, m + 0.01,
            f"n={n}", ha="center", fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Recovery rate (different action after duplicate / self-correction)")
ax.set_title("Mid-trajectory recovery rate")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_recovery_rate.png"), dpi=150)
plt.close(fig)

# Outcome-transition pie/bar
trans_counter = collections.Counter()
for tid in common:
    sp = sft[tid]["eval_result"] > 0; ap = arpo[tid]["eval_result"] > 0
    trans_counter[("P" if sp else "F", "P" if ap else "F")] += 1
labels_t = ["SFT_pass\n→ARPO_pass", "SFT_pass\n→ARPO_FAIL\n(regression)",
            "SFT_fail\n→ARPO_pass\n(gain)", "SFT_fail\n→ARPO_fail"]
counts_t = [trans_counter[("P","P")], trans_counter[("P","F")],
            trans_counter[("F","P")], trans_counter[("F","F")]]
fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.bar(labels_t, counts_t, color=["#5fa15f","#cc4444","#3a7adf","#999999"], alpha=0.85)
for i, c in enumerate(counts_t):
    ax.text(i, c + 0.5, str(c), ha="center")
ax.set_ylabel("# tasks")
ax.set_title(f"Outcome transition (paired by task, N={len(common)})")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_transitions.png"), dpi=150)
plt.close(fig)

# Paired delta plots: SFT minus ARPO on key metrics, restricted to regression cohort
regress = [tid for tid in common if sft[tid]["eval_result"] > 0 and arpo[tid]["eval_result"] == 0]
gains   = [tid for tid in common if sft[tid]["eval_result"] == 0 and arpo[tid]["eval_result"] > 0]
both_p  = [tid for tid in common if sft[tid]["eval_result"] > 0 and arpo[tid]["eval_result"] > 0]
both_f  = [tid for tid in common if sft[tid]["eval_result"] == 0 and arpo[tid]["eval_result"] == 0]

def paired_means(tids, key):
    sft_v  = [sft[t]["feat"][key]  for t in tids]
    arpo_v = [arpo[t]["feat"][key] for t in tids]
    return (float(np.mean(sft_v)) if sft_v else float("nan"),
            float(np.mean(arpo_v)) if arpo_v else float("nan"))

paired_keys = [
    ("n_unique_fns",            "Unique action-fn types"),
    ("n_unique_coord_clusters", "Unique click clusters"),
    ("max_run",                 "Max consec same-fn run"),
    ("n_self_correct",          "Self-correction phrases"),
    ("bandf",                   "Back-and-forth A→B→A"),
    ("first3_bad",              "First-3 bad-signal count"),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
groups = [("Regress\n(SFT pass→ARPO fail)", regress),
          ("Gain\n(SFT fail→ARPO pass)", gains),
          ("Both pass", both_p),
          ("Both fail", both_f)]
for ax, (k, label) in zip(axes.ravel(), paired_keys):
    sft_means, arpo_means, names = [], [], []
    for gname, tids in groups:
        sm, am = paired_means(tids, k)
        sft_means.append(sm); arpo_means.append(am); names.append(f"{gname}\n(n={len(tids)})")
    x = np.arange(len(groups))
    w = 0.38
    ax.bar(x - w/2, sft_means,  w, label="SFT",  color="#7fbf7f", alpha=0.85)
    ax.bar(x + w/2, arpo_means, w, label="ARPO", color="#4f9bff", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
fig.suptitle("Paired SFT vs ARPO trajectory features by outcome group", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_paired_features.png"), dpi=150)
plt.close(fig)

# ------------- Hypothesis verdict computations -------------
def mean(vs): return float(np.mean(vs)) if vs else float("nan")

# H1 — exploration / recovery
H1_unique_fns_sft_pass  = mean([sft[t]["feat"]["n_unique_fns"]  for t in common if sft[t]["eval_result"] > 0])
H1_unique_fns_arpo_pass = mean([arpo[t]["feat"]["n_unique_fns"] for t in common if arpo[t]["eval_result"] > 0])
H1_unique_fns_sft_fail  = mean([sft[t]["feat"]["n_unique_fns"]  for t in common if sft[t]["eval_result"] == 0])
H1_unique_fns_arpo_fail = mean([arpo[t]["feat"]["n_unique_fns"] for t in common if arpo[t]["eval_result"] == 0])

H1_recover_sft_pass = mean([sft[t]["feat"]["recover_rate"] for t in common
                             if sft[t]["eval_result"] > 0 and sft[t]["feat"]["recover_rate"] is not None])
H1_recover_arpo_pass = mean([arpo[t]["feat"]["recover_rate"] for t in common
                             if arpo[t]["eval_result"] > 0 and arpo[t]["feat"]["recover_rate"] is not None])
H1_recover_sft_fail = mean([sft[t]["feat"]["recover_rate"] for t in common
                             if sft[t]["eval_result"] == 0 and sft[t]["feat"]["recover_rate"] is not None])
H1_recover_arpo_fail = mean([arpo[t]["feat"]["recover_rate"] for t in common
                             if arpo[t]["eval_result"] == 0 and arpo[t]["feat"]["recover_rate"] is not None])

# H1 paired on regressions
reg_sft_uniq  = mean([sft[t]["feat"]["n_unique_fns"]  for t in regress])
reg_arpo_uniq = mean([arpo[t]["feat"]["n_unique_fns"] for t in regress])
reg_sft_clu   = mean([sft[t]["feat"]["n_unique_coord_clusters"]  for t in regress])
reg_arpo_clu  = mean([arpo[t]["feat"]["n_unique_coord_clusters"] for t in regress])
reg_sft_self  = mean([sft[t]["feat"]["n_self_correct"] for t in regress])
reg_arpo_self = mean([arpo[t]["feat"]["n_self_correct"] for t in regress])
reg_sft_recov = mean([sft[t]["feat"]["recover_rate"]  for t in regress
                       if sft[t]["feat"]["recover_rate"] is not None])
reg_arpo_recov = mean([arpo[t]["feat"]["recover_rate"] for t in regress
                       if arpo[t]["feat"]["recover_rate"] is not None])
reg_sft_bandf  = mean([sft[t]["feat"]["bandf"]  for t in regress])
reg_arpo_bandf = mean([arpo[t]["feat"]["bandf"] for t in regress])

# H2 — wandering
H2_steps_sft_fail  = mean([sft[t]["feat"]["n"]  for t in common if sft[t]["eval_result"] == 0])
H2_steps_arpo_fail = mean([arpo[t]["feat"]["n"] for t in common if arpo[t]["eval_result"] == 0])
H2_steps_sft_pass  = mean([sft[t]["feat"]["n"]  for t in common if sft[t]["eval_result"] > 0])
H2_steps_arpo_pass = mean([arpo[t]["feat"]["n"] for t in common if arpo[t]["eval_result"] > 0])
H2_paired_both_fail_sft  = mean([sft[t]["feat"]["n"]  for t in both_f])
H2_paired_both_fail_arpo = mean([arpo[t]["feat"]["n"] for t in both_f])

# H3 — first-3 bad signal predictiveness
clean3 = [tid for tid in common
          if sft[tid]["feat"]["first3_bad"] == 0 or arpo[tid]["feat"]["first3_bad"] == 0]  # placeholder
# proper test: across both runs combined, success rate when first3_bad==0 vs >=1
all_traj = [(r, r["feat"]["first3_bad"], int(r["eval_result"] > 0)) for r in sft_records + arpo_records]
clean = [s for _, b, s in all_traj if b == 0]
dirty = [s for _, b, s in all_traj if b >= 1]
H3_clean_rate = mean(clean); H3_dirty_rate = mean(dirty)

reg_sft_first3  = mean([sft[t]["feat"]["first3_bad"]  for t in regress])
reg_arpo_first3 = mean([arpo[t]["feat"]["first3_bad"] for t in regress])
gain_sft_first3 = mean([sft[t]["feat"]["first3_bad"]  for t in gains])
gain_arpo_first3 = mean([arpo[t]["feat"]["first3_bad"] for t in gains])

# ------------- write report -------------
report = []
report.append(f"# Recovery hypothesis: ARPO step35 vs SFT init (n=1)\n")
report.append(f"Tasks: {len(common)} paired. SFT pass = "
              f"{sum(1 for t in common if sft[t]['eval_result']>0)}, "
              f"ARPO pass = {sum(1 for t in common if arpo[t]['eval_result']>0)}.\n")
report.append(f"Outcome transitions: PP={trans_counter[('P','P')]}, "
              f"PF (regression)={trans_counter[('P','F')]}, "
              f"FP (gain)={trans_counter[('F','P')]}, "
              f"FF={trans_counter[('F','F')]}.\n")

report.append("## H1 (main): ARPO can't explore mid-trajectory\n")
report.append("| metric | SFT pass | ARPO pass | SFT fail | ARPO fail |")
report.append("|---|---|---|---|---|")
report.append(f"| unique action-fn types | {H1_unique_fns_sft_pass:.2f} | {H1_unique_fns_arpo_pass:.2f} | {H1_unique_fns_sft_fail:.2f} | {H1_unique_fns_arpo_fail:.2f} |")
report.append(f"| recovery rate after duplicate/self-correct | {H1_recover_sft_pass:.2f} | {H1_recover_arpo_pass:.2f} | {H1_recover_sft_fail:.2f} | {H1_recover_arpo_fail:.2f} |")
report.append("")
report.append("**Paired regressions (14 tasks: SFT pass → ARPO fail on the same task):**")
report.append("")
report.append("| metric | SFT (passed) | ARPO (failed) |")
report.append("|---|---|---|")
report.append(f"| unique action-fn types | {reg_sft_uniq:.2f} | {reg_arpo_uniq:.2f} |")
report.append(f"| unique click clusters | {reg_sft_clu:.2f} | {reg_arpo_clu:.2f} |")
report.append(f"| self-correction phrases | {reg_sft_self:.2f} | {reg_arpo_self:.2f} |")
report.append(f"| recovery rate | {reg_sft_recov:.2f} | {reg_arpo_recov:.2f} |")
report.append(f"| back-and-forth A→B→A | {reg_sft_bandf:.2f} | {reg_arpo_bandf:.2f} |")
report.append("")

report.append("## H2: ARPO wanders too long\n")
report.append("| metric | SFT | ARPO |")
report.append("|---|---|---|")
report.append(f"| mean steps (success) | {H2_steps_sft_pass:.2f} | {H2_steps_arpo_pass:.2f} |")
report.append(f"| mean steps (failure) | {H2_steps_sft_fail:.2f} | {H2_steps_arpo_fail:.2f} |")
report.append(f"| mean steps on SAME failed tasks (paired both-fail, N={len(both_f)}) | {H2_paired_both_fail_sft:.2f} | {H2_paired_both_fail_arpo:.2f} |")
report.append("")

report.append("## H3: first few key steps wrong\n")
report.append(f"Marginal success rate when first-3-bad==0: {H3_clean_rate:.2f} ({len(clean)} traj). When >=1: {H3_dirty_rate:.2f} ({len(dirty)} traj).\n")
report.append("On regressions and gains the early-step quality is similar across SFT/ARPO:\n")
report.append("| group | SFT first3_bad | ARPO first3_bad | SFT outcome | ARPO outcome |")
report.append("|---|---|---|---|---|")
report.append(f"| Regressions (N={len(regress)}) | {reg_sft_first3:.2f} | {reg_arpo_first3:.2f} | pass | fail |")
report.append(f"| Gains (N={len(gains)})       | {gain_sft_first3:.2f} | {gain_arpo_first3:.2f} | fail | pass |")
report.append("")

report_path = os.path.join(OUT, "REPORT.md")
with open(report_path, "w") as f:
    f.write("\n".join(report))

# Also dump raw numbers as JSON for downstream
raw = {
    "n_paired_tasks": len(common),
    "transitions": {f"{a}->{b}": v for (a,b),v in trans_counter.items()},
    "H1": {
        "cohort_unique_fns": {
            "SFT_pass": H1_unique_fns_sft_pass, "ARPO_pass": H1_unique_fns_arpo_pass,
            "SFT_fail": H1_unique_fns_sft_fail, "ARPO_fail": H1_unique_fns_arpo_fail,
        },
        "cohort_recover_rate": {
            "SFT_pass": H1_recover_sft_pass, "ARPO_pass": H1_recover_arpo_pass,
            "SFT_fail": H1_recover_sft_fail, "ARPO_fail": H1_recover_arpo_fail,
        },
        "regression_pair": {
            "unique_fns_sft": reg_sft_uniq, "unique_fns_arpo": reg_arpo_uniq,
            "unique_clusters_sft": reg_sft_clu, "unique_clusters_arpo": reg_arpo_clu,
            "self_correct_sft": reg_sft_self, "self_correct_arpo": reg_arpo_self,
            "recover_rate_sft": reg_sft_recov, "recover_rate_arpo": reg_arpo_recov,
            "bandf_sft": reg_sft_bandf, "bandf_arpo": reg_arpo_bandf,
        },
    },
    "H2": {
        "steps_sft_fail": H2_steps_sft_fail, "steps_arpo_fail": H2_steps_arpo_fail,
        "steps_sft_pass": H2_steps_sft_pass, "steps_arpo_pass": H2_steps_arpo_pass,
        "steps_paired_both_fail_sft": H2_paired_both_fail_sft,
        "steps_paired_both_fail_arpo": H2_paired_both_fail_arpo,
    },
    "H3": {
        "marginal_success_clean3": H3_clean_rate, "marginal_success_dirty3": H3_dirty_rate,
        "regress_first3_sft": reg_sft_first3, "regress_first3_arpo": reg_arpo_first3,
        "gain_first3_sft": gain_sft_first3, "gain_first3_arpo": gain_arpo_first3,
        "n_clean": len(clean), "n_dirty": len(dirty),
    },
}
with open(os.path.join(OUT, "raw_metrics.json"), "w") as f:
    json.dump(raw, f, indent=2)

print("Wrote outputs to", OUT)
print(json.dumps(raw, indent=2))
