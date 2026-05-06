"""
n=8, t=1 version of the recovery / exploration hypothesis test.

Inputs (8 rollouts per task, 86 paired tasks):
  SFT_PATH  = sft_init_n8_t1/trajectories_at_0.jsonl
  ARPO_PATH = arpo_step35_n8_t1/trajectories_at_35.jsonl

Hypotheses tested:
  H1 (main): ARPO can't explore mid-trajectory, MCTS-SFT inherits detour patterns
             from successful MCTS rollouts.
             Strongest n=8 form: among rollouts that SUCCEED, SFT trajectories
             show more recovery / detour structure than ARPO trajectories.
  H2:        ARPO wanders too long.                       (re-test)
  H3:        ARPO's first few key steps are wrong.        (re-test)
  H4 (new):  ARPO has collapsed cross-rollout diversity at fixed task
             (lower action-fn entropy across the 8 rollouts).
  H5 (new):  ARPO's pass@8 - pass@1 gap is small relative to SFT — the policy
             is too tight to "luck into" success across resamples.
"""
import json, re, os, math, csv, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
SFT_PATH  = "/mnt/kevinzyz/arpo_local/checkpoints/clean_eval/arpo_86tasks_clean_from_v2.1_no_mask_diagnosis/sft_init_n8_t1/trajectories_at_0.jsonl"
ARPO_PATH = "/mnt/kevinzyz/arpo_local/checkpoints/clean_eval/arpo_86tasks_clean_from_v2.1_no_mask_diagnosis/arpo_step35_n8_t1/trajectories_at_35.jsonl"

# ------------- parsing (identical to n1t0 version) -------------
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
    for cid, c in enumerate(clusters):
        if abs(coord[0] - c[0]) < tol and abs(coord[1] - c[1]) < tol:
            return cid
    clusters.append(coord)
    return len(clusters) - 1

def features(steps):
    parsed = [parse_step(s["action"]) for s in steps]
    fns = [p["fn"] for p in parsed]
    n = len(fns)
    max_run = 1; cur = 1
    for i in range(1, n):
        if fns[i] == fns[i-1]:
            cur += 1; max_run = max(max_run, cur)
        else:
            cur = 1
    clusters = []
    for p in parsed:
        for c in p["coords"]:
            cluster_id(c, clusters)
    n_unique_clusters = len(clusters)
    n_unique_fns = len(set(fns))
    n_self_correct = sum(1 for p in parsed if SELF_CORRECT_RE.search(p["th"]))
    recover_attempts = 0; recover_success = 0
    for i in range(n - 1):
        prev_fn = fns[i-1] if i >= 1 else None
        is_dup_of_prev = (i >= 1 and fns[i] == prev_fn)
        is_self_correct = bool(SELF_CORRECT_RE.search(parsed[i]["th"]))
        if not (is_dup_of_prev or is_self_correct):
            continue
        recover_attempts += 1
        a, b = parsed[i], parsed[i+1]
        if a["fn"] != b["fn"]:
            recover_success += 1; continue
        if a["coords"] and b["coords"]:
            ca = cluster_id(a["coords"][0], clusters)
            cb = cluster_id(b["coords"][0], clusters)
            if ca != cb:
                recover_success += 1
    recover_rate = recover_success / recover_attempts if recover_attempts else None
    bandf = 0
    for i in range(2, n):
        if fns[i] == fns[i-2] and fns[i-1] != fns[i]:
            bandf += 1
    bigrams = set(zip(fns[:-1], fns[1:])) if n > 1 else set()
    first3_bad = 0; prev = None
    for i, p in enumerate(parsed[:3]):
        bad = (p["fn"] == "wait") or (i >= 1 and p["fn"] == prev) \
              or bool(SELF_CORRECT_RE.search(p["th"]))
        first3_bad += int(bad)
        prev = p["fn"]
    return dict(
        n=n, fns=fns,
        n_unique_fns=n_unique_fns,
        n_unique_coord_clusters=n_unique_clusters,
        max_run=max_run,
        n_self_correct=n_self_correct,
        recover_attempts=recover_attempts,
        recover_success=recover_success,
        recover_rate=recover_rate,
        bandf=bandf,
        n_unique_bigrams=len(bigrams),
        first3_bad=first3_bad,
    )

# ------------- load -------------
def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]

sft_recs  = load_jsonl(SFT_PATH)
arpo_recs = load_jsonl(ARPO_PATH)
for r in sft_recs + arpo_recs:
    r["feat"] = features(r["steps"])

def group_by_task(recs):
    g = collections.defaultdict(list)
    for r in recs:
        g[r["task_id"]].append(r)
    return dict(g)

sft_by  = group_by_task(sft_recs)
arpo_by = group_by_task(arpo_recs)
common  = sorted(set(sft_by) & set(arpo_by))
assert all(len(sft_by[t]) == 8 and len(arpo_by[t]) == 8 for t in common), "expected 8 rollouts per task"

# ------------- task-level outcomes -------------
def n_success(recs): return sum(1 for r in recs if r["eval_result"] > 0.0)
def pass1(recs):     return n_success(recs) / len(recs)
def passk(recs):     return 1.0 if n_success(recs) > 0 else 0.0  # pass@8 (any)

def cohort_of(sft_pass, arpo_pass):
    return ("PP" if sft_pass and arpo_pass else
            "PF" if sft_pass and not arpo_pass else
            "FP" if not sft_pass and arpo_pass else "FF")

task_meta = {}
for t in common:
    sp1 = pass1(sft_by[t]); ap1 = pass1(arpo_by[t])
    sp8 = passk(sft_by[t]); ap8 = passk(arpo_by[t])
    task_meta[t] = dict(
        sft_pass1=sp1, arpo_pass1=ap1,
        sft_pass8=sp8, arpo_pass8=ap8,
        sft_n_succ=n_success(sft_by[t]), arpo_n_succ=n_success(arpo_by[t]),
        cohort=cohort_of(sp8 > 0, ap8 > 0),
    )

trans = collections.Counter(m["cohort"] for m in task_meta.values())
total_sft_succ  = sum(m["sft_n_succ"]  for m in task_meta.values())
total_arpo_succ = sum(m["arpo_n_succ"] for m in task_meta.values())
print(f"Tasks: {len(common)}.  Rollouts each side: {8*len(common)}.")
print(f"SFT  pass@1 (avg) = {total_sft_succ /(8*len(common)):.3f}, pass@8 = {sum(1 for m in task_meta.values() if m['sft_pass8']>0)}/{len(common)}")
print(f"ARPO pass@1 (avg) = {total_arpo_succ/(8*len(common)):.3f}, pass@8 = {sum(1 for m in task_meta.values() if m['arpo_pass8']>0)}/{len(common)}")
print("Transitions (pass@8 basis):", dict(trans))

# ------------- helpers -------------
def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")

def aggregate_feature(recs_subset, key):
    """mean over a list of rollouts for a feature; recover_rate filters None."""
    vals = []
    for r in recs_subset:
        v = r["feat"][key]
        if v is None: continue
        vals.append(v)
    return mean(vals) if vals else float("nan")

def split_succ_fail(recs):
    s = [r for r in recs if r["eval_result"] > 0.0]
    f = [r for r in recs if r["eval_result"] == 0.0]
    return s, f

# ------------- H1 main: features within-success and within-failure -------------
H1_keys = ["n_unique_fns", "n_unique_coord_clusters", "max_run",
           "n_self_correct", "bandf", "first3_bad", "recover_rate", "n"]

def cohort_feature_means(by_task, side="success"):
    """Average per-rollout features over all rollouts of given side across all tasks."""
    sub = []
    for t in common:
        s, f = split_succ_fail(by_task[t])
        sub.extend(s if side == "success" else f)
    return {k: aggregate_feature(sub, k) for k in H1_keys}, len(sub)

sft_succ_means,  n_sft_succ_roll  = cohort_feature_means(sft_by,  "success")
sft_fail_means,  n_sft_fail_roll  = cohort_feature_means(sft_by,  "fail")
arpo_succ_means, n_arpo_succ_roll = cohort_feature_means(arpo_by, "success")
arpo_fail_means, n_arpo_fail_roll = cohort_feature_means(arpo_by, "fail")

# Paired form: only for tasks where BOTH SFT and ARPO have at least one success.
both_succ_tasks = [t for t in common if task_meta[t]["sft_n_succ"] > 0 and task_meta[t]["arpo_n_succ"] > 0]
def paired_succ_means(by_task, tids):
    sub = []
    for t in tids:
        s, _ = split_succ_fail(by_task[t])
        sub.extend(s)
    return {k: aggregate_feature(sub, k) for k in H1_keys}, len(sub)
psm_sft, n_psm_sft   = paired_succ_means(sft_by,  both_succ_tasks)
psm_arpo, n_psm_arpo = paired_succ_means(arpo_by, both_succ_tasks)

# Regression cohort: SFT pass@8 -> ARPO 0/8 (the strongest "ARPO got worse").
regress = [t for t in common if task_meta[t]["cohort"] == "PF"]
gains   = [t for t in common if task_meta[t]["cohort"] == "FP"]
both_p  = [t for t in common if task_meta[t]["cohort"] == "PP"]
both_f  = [t for t in common if task_meta[t]["cohort"] == "FF"]

# On regression tasks, compare SFT successful rollouts to ARPO failure rollouts.
def slice_means(by_task, tids, side):
    sub = []
    for t in tids:
        s, f = split_succ_fail(by_task[t])
        sub.extend(s if side == "success" else f)
    return {k: aggregate_feature(sub, k) for k in H1_keys}, len(sub)

reg_sft_succ, n_reg_sft_succ = slice_means(sft_by,  regress, "success")
reg_arpo_fail, n_reg_arpo_fail = slice_means(arpo_by, regress, "fail")

# ------------- H4: cross-rollout diversity per task -------------
def fn_entropy(rollouts, max_step=12):
    """Across the 8 rollouts of one task, what's the mean per-step entropy of action-fn?
    Step i contributes H(p_i) where p_i is the distribution of action-fn at step i across rollouts.
    Average over the first max_step steps that have >=2 rollouts present."""
    H_vals = []
    for i in range(max_step):
        bag = [r["feat"]["fns"][i] for r in rollouts if len(r["feat"]["fns"]) > i]
        if len(bag) < 2: continue
        c = collections.Counter(bag)
        p = np.array(list(c.values()), dtype=float); p /= p.sum()
        H = -float((p * np.log(p)).sum())
        H_vals.append(H)
    return mean(H_vals) if H_vals else float("nan"), len(H_vals)

per_task_entropy = {}
for t in common:
    sH, sN = fn_entropy(sft_by[t])
    aH, aN = fn_entropy(arpo_by[t])
    per_task_entropy[t] = (sH, aH, sN, aN)

H4_sft_mean  = mean([sH for (sH, _, _, _) in per_task_entropy.values()])
H4_arpo_mean = mean([aH for (_, aH, _, _) in per_task_entropy.values()])

# Paired diff (only tasks where both have valid entropy)
H4_pairs = [(sH, aH) for (sH, aH, _, _) in per_task_entropy.values()
            if not (math.isnan(sH) or math.isnan(aH))]
H4_paired_diff = mean([s - a for s, a in H4_pairs])  # SFT - ARPO

# By cohort
def entropy_for_cohort(tids):
    pairs = [(per_task_entropy[t][0], per_task_entropy[t][1]) for t in tids
             if not (math.isnan(per_task_entropy[t][0]) or math.isnan(per_task_entropy[t][1]))]
    if not pairs: return (float("nan"), float("nan"), 0)
    return (mean([s for s, _ in pairs]), mean([a for _, a in pairs]), len(pairs))

H4_by_cohort = {
    "PP (both pass)":   entropy_for_cohort(both_p),
    "PF (regression)":  entropy_for_cohort(regress),
    "FP (gain)":        entropy_for_cohort(gains),
    "FF (both fail)":   entropy_for_cohort(both_f),
}

# ------------- H5: pass@8 - pass@1 gap (exploration headroom) -------------
def headroom(p1, p8):
    return p8 - p1 if (p1 is not None and p8 is not None) else None

H5_sft  = mean([task_meta[t]["sft_pass8"]  - task_meta[t]["sft_pass1"]  for t in common])
H5_arpo = mean([task_meta[t]["arpo_pass8"] - task_meta[t]["arpo_pass1"] for t in common])
# Subset: tasks where pass@8 > 0 (otherwise gap is trivially 0)
sft_solvable  = [t for t in common if task_meta[t]["sft_pass8"]  > 0]
arpo_solvable = [t for t in common if task_meta[t]["arpo_pass8"] > 0]
H5_sft_solv   = mean([task_meta[t]["sft_pass8"]  - task_meta[t]["sft_pass1"]  for t in sft_solvable])
H5_arpo_solv  = mean([task_meta[t]["arpo_pass8"] - task_meta[t]["arpo_pass1"] for t in arpo_solvable])

# ------------- H2: wandering -------------
H2_steps_sft_succ  = aggregate_feature([r for t in common for r in split_succ_fail(sft_by[t])[0]],  "n")
H2_steps_arpo_succ = aggregate_feature([r for t in common for r in split_succ_fail(arpo_by[t])[0]], "n")
H2_steps_sft_fail  = aggregate_feature([r for t in common for r in split_succ_fail(sft_by[t])[1]],  "n")
H2_steps_arpo_fail = aggregate_feature([r for t in common for r in split_succ_fail(arpo_by[t])[1]], "n")
# Paired both-fail rollouts (per task: every SFT-fail rollout vs every ARPO-fail rollout — averaged within task then across)
def per_task_mean_n(by_task, tids, side):
    out = []
    for t in tids:
        s, f = split_succ_fail(by_task[t])
        sub = s if side == "success" else f
        if not sub: continue
        out.append(mean([r["feat"]["n"] for r in sub]))
    return out
H2_paired_both_fail_sft  = mean(per_task_mean_n(sft_by,  both_f, "fail"))
H2_paired_both_fail_arpo = mean(per_task_mean_n(arpo_by, both_f, "fail"))

# ------------- H3: first-3 bad signals -------------
all_roll = sft_recs + arpo_recs
clean = [int(r["eval_result"] > 0) for r in all_roll if r["feat"]["first3_bad"] == 0]
dirty = [int(r["eval_result"] > 0) for r in all_roll if r["feat"]["first3_bad"] >= 1]
H3_clean_rate = mean(clean); H3_dirty_rate = mean(dirty)

# ------------- write CSVs -------------
with open(os.path.join(OUT, "summary_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    cols = ["metric", "SFT_success_rolls", "ARPO_success_rolls",
            "SFT_fail_rolls", "ARPO_fail_rolls"]
    w.writerow(cols)
    w.writerow(["count", n_sft_succ_roll, n_arpo_succ_roll, n_sft_fail_roll, n_arpo_fail_roll])
    for k in H1_keys:
        w.writerow([k,
                    f"{sft_succ_means[k]:.3f}",  f"{arpo_succ_means[k]:.3f}",
                    f"{sft_fail_means[k]:.3f}",  f"{arpo_fail_means[k]:.3f}"])

with open(os.path.join(OUT, "tasks_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task_id", "sft_n_succ", "arpo_n_succ",
                "sft_pass1", "arpo_pass1", "sft_pass8", "arpo_pass8",
                "cohort", "sft_fn_entropy", "arpo_fn_entropy"])
    for t in common:
        m = task_meta[t]; sH, aH, _, _ = per_task_entropy[t]
        w.writerow([t, m["sft_n_succ"], m["arpo_n_succ"],
                    f"{m['sft_pass1']:.3f}", f"{m['arpo_pass1']:.3f}",
                    f"{m['sft_pass8']:.3f}", f"{m['arpo_pass8']:.3f}",
                    m["cohort"],
                    "" if math.isnan(sH) else f"{sH:.3f}",
                    "" if math.isnan(aH) else f"{aH:.3f}"])

# ------------- figures -------------

# H1: paired feature bars on regression cohort, and within-success bars across all tasks
def grouped_bar(ax, names, sft_vals, arpo_vals, title, ylabel=""):
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w/2, sft_vals,  w, label="SFT",  color="#7fbf7f", alpha=0.85)
    ax.bar(x + w/2, arpo_vals, w, label="ARPO", color="#4f9bff", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_title(title, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, axis="y"); ax.legend(fontsize=8)

# fig: within-success rollouts SFT vs ARPO (the H1 cleanest test)
fig, ax = plt.subplots(figsize=(8.5, 4.5))
keys_for_plot = ["n_unique_fns", "n_unique_coord_clusters", "max_run",
                 "n_self_correct", "bandf", "recover_rate"]
labels = ["uniq fn", "uniq clust", "max run", "self-corr",
          "A→B→A", "recover rate"]
sft_v  = [sft_succ_means[k]  for k in keys_for_plot]
arpo_v = [arpo_succ_means[k] for k in keys_for_plot]
grouped_bar(ax, labels, sft_v, arpo_v,
            f"Features of SUCCESSFUL rollouts (SFT n={n_sft_succ_roll}, ARPO n={n_arpo_succ_roll})")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_success_rollouts.png"), dpi=150); plt.close(fig)

# fig: within-success on tasks where BOTH SFT and ARPO solve (paired succ)
fig, ax = plt.subplots(figsize=(8.5, 4.5))
sft_v  = [psm_sft[k]  for k in keys_for_plot]
arpo_v = [psm_arpo[k] for k in keys_for_plot]
grouped_bar(ax, labels, sft_v, arpo_v,
            f"Successful rollouts on tasks where BOTH solve (N={len(both_succ_tasks)} tasks; "
            f"SFT n={n_psm_sft}, ARPO n={n_psm_arpo} rollouts)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_paired_success.png"), dpi=150); plt.close(fig)

# fig: cohort feature bars (4-panel)
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
cohorts4 = [("Both pass\n(PP)", both_p),
            ("Regression\nSFT pass / ARPO fail", regress),
            ("Gain\nSFT fail / ARPO pass", gains),
            ("Both fail\n(FF)", both_f)]
for ax, (title, tids) in zip(axes, cohorts4):
    sft_v, arpo_v = [], []
    for k in keys_for_plot:
        # average per-rollout feature across all rollouts in cohort, per side
        sft_sub = [r for t in tids for r in sft_by[t]]
        arpo_sub = [r for t in tids for r in arpo_by[t]]
        sft_v.append(aggregate_feature(sft_sub, k))
        arpo_v.append(aggregate_feature(arpo_sub, k))
    grouped_bar(ax, labels, sft_v, arpo_v, f"{title} (N={len(tids)})")
fig.suptitle("Per-rollout features by paired-task outcome cohort", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_paired_features.png"), dpi=150); plt.close(fig)

# fig: cross-rollout entropy (H4)
fig, ax = plt.subplots(figsize=(7, 4))
groups = list(H4_by_cohort.keys())
sft_e  = [H4_by_cohort[g][0] for g in groups]
arpo_e = [H4_by_cohort[g][1] for g in groups]
ns     = [H4_by_cohort[g][2] for g in groups]
x = np.arange(len(groups)); w = 0.38
b1 = ax.bar(x - w/2, sft_e,  w, label="SFT",  color="#7fbf7f", alpha=0.85)
b2 = ax.bar(x + w/2, arpo_e, w, label="ARPO", color="#4f9bff", alpha=0.85)
for i, n in enumerate(ns):
    ax.text(i, max((sft_e[i] if not math.isnan(sft_e[i]) else 0),
                   (arpo_e[i] if not math.isnan(arpo_e[i]) else 0)) + 0.03,
            f"N={n}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
ax.set_ylabel("Mean per-step action-fn entropy across the 8 rollouts (nats)")
ax.set_title("H4: Cross-rollout diversity (higher = more exploration across resamples)")
ax.legend(); ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_entropy.png"), dpi=150); plt.close(fig)

# fig: pass@1 vs pass@8 scatter, paired by task
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sft_p1 = [task_meta[t]["sft_pass1"]  for t in common]
sft_p8 = [task_meta[t]["sft_pass8"]  for t in common]
arpo_p1 = [task_meta[t]["arpo_pass1"] for t in common]
arpo_p8 = [task_meta[t]["arpo_pass8"] for t in common]
# Plot pass@1 (x) vs pass@1 ARPO (y), color by gap
for i, t in enumerate(common):
    ax.scatter(sft_p1[i], arpo_p1[i], s=30, alpha=0.6,
               c="#cc4444" if task_meta[t]["cohort"]=="PF" else
                 "#3a7adf" if task_meta[t]["cohort"]=="FP" else
                 "#5fa15f" if task_meta[t]["cohort"]=="PP" else "#999999")
ax.plot([0,1],[0,1],"k--",alpha=0.4)
ax.set_xlabel("SFT pass@1 (n_success / 8)")
ax.set_ylabel("ARPO pass@1 (n_success / 8)")
ax.set_title("Per-task pass@1: ARPO vs SFT  (red=regression, blue=gain, green=both-pass)")
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pass1_scatter.png"), dpi=150); plt.close(fig)

# fig: pass@8 - pass@1 headroom histogram
fig, ax = plt.subplots(figsize=(7, 4))
sft_gap  = [task_meta[t]["sft_pass8"]  - task_meta[t]["sft_pass1"]  for t in common]
arpo_gap = [task_meta[t]["arpo_pass8"] - task_meta[t]["arpo_pass1"] for t in common]
bins = np.linspace(0, 1, 11)
ax.hist([sft_gap, arpo_gap], bins=bins, label=["SFT", "ARPO"],
        color=["#7fbf7f", "#4f9bff"], alpha=0.75)
ax.set_xlabel("pass@8 − pass@1 (per task)")
ax.set_ylabel("# tasks")
ax.set_title(f"H5: Exploration headroom.  SFT mean={H5_sft:.3f},  ARPO mean={H5_arpo:.3f}")
ax.legend(); ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_headroom.png"), dpi=150); plt.close(fig)

# fig: trajectory length boxplot
fig, ax = plt.subplots(figsize=(7, 4))
data = [
    [r["feat"]["n"] for t in common for r in split_succ_fail(sft_by[t])[0]],
    [r["feat"]["n"] for t in common for r in split_succ_fail(sft_by[t])[1]],
    [r["feat"]["n"] for t in common for r in split_succ_fail(arpo_by[t])[0]],
    [r["feat"]["n"] for t in common for r in split_succ_fail(arpo_by[t])[1]],
]
bp = ax.boxplot(data, labels=["SFT pass","SFT fail","ARPO pass","ARPO fail"],
                patch_artist=True, showmeans=True)
for patch, c in zip(bp["boxes"], ["#7fbf7f","#ff8c8c","#4f9bff","#c87fff"]):
    patch.set_facecolor(c); patch.set_alpha(0.55)
ax.set_ylabel("Trajectory length (steps)")
ax.set_title(f"H2: trajectory length by rollout outcome  "
             f"(succ_sft={H2_steps_sft_succ:.1f}, succ_arpo={H2_steps_arpo_succ:.1f})")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_length.png"), dpi=150); plt.close(fig)

# fig: first-3 bad signal predictiveness
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["clean opener\n(first3_bad=0)", "shaky opener\n(first3_bad≥1)"],
       [H3_clean_rate, H3_dirty_rate], color=["#5fa15f", "#cc4444"], alpha=0.85)
ax.text(0, H3_clean_rate + 0.02, f"n={len(clean)}", ha="center", fontsize=9)
ax.text(1, H3_dirty_rate + 0.02, f"n={len(dirty)}", ha="center", fontsize=9)
ax.set_ylim(0, max(H3_clean_rate, H3_dirty_rate) + 0.1)
ax.set_ylabel("Marginal success rate")
ax.set_title("H3: opener quality vs success rate (pooled across runs)")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_first3.png"), dpi=150); plt.close(fig)

# fig: outcome transitions
fig, ax = plt.subplots(figsize=(7, 4))
labels_t = ["PP\nboth pass", "PF\nregression", "FP\ngain", "FF\nboth fail"]
counts_t = [trans["PP"], trans["PF"], trans["FP"], trans["FF"]]
ax.bar(labels_t, counts_t, color=["#5fa15f","#cc4444","#3a7adf","#999999"], alpha=0.85)
for i, c in enumerate(counts_t):
    ax.text(i, c + 0.5, str(c), ha="center")
ax.set_ylabel("# tasks")
ax.set_title(f"Outcome transitions on pass@8 basis (N={len(common)})")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_transitions.png"), dpi=150); plt.close(fig)

# ------------- raw metrics dump -------------
raw = {
    "n_paired_tasks": len(common),
    "rollouts_per_task": 8,
    "totals": {
        "sft_pass1_mean":  total_sft_succ /(8*len(common)),
        "arpo_pass1_mean": total_arpo_succ/(8*len(common)),
        "sft_pass8_count":  sum(1 for m in task_meta.values() if m["sft_pass8"]>0),
        "arpo_pass8_count": sum(1 for m in task_meta.values() if m["arpo_pass8"]>0),
    },
    "transitions_pass8": dict(trans),
    "H1_within_success_rollouts": {
        "sft":  sft_succ_means,  "n_sft_rollouts": n_sft_succ_roll,
        "arpo": arpo_succ_means, "n_arpo_rollouts": n_arpo_succ_roll,
    },
    "H1_within_fail_rollouts": {
        "sft":  sft_fail_means,  "n_sft_rollouts": n_sft_fail_roll,
        "arpo": arpo_fail_means, "n_arpo_rollouts": n_arpo_fail_roll,
    },
    "H1_paired_success_tasks_only": {
        "n_tasks": len(both_succ_tasks),
        "sft":  psm_sft,  "arpo": psm_arpo,
        "n_sft_rollouts": n_psm_sft, "n_arpo_rollouts": n_psm_arpo,
    },
    "H1_regression_cohort": {
        "n_tasks": len(regress),
        "sft_success_rollouts":  reg_sft_succ,  "n_sft_rollouts": n_reg_sft_succ,
        "arpo_fail_rollouts":    reg_arpo_fail, "n_arpo_rollouts": n_reg_arpo_fail,
    },
    "H2_wandering": {
        "steps_sft_success":  H2_steps_sft_succ,  "steps_arpo_success":  H2_steps_arpo_succ,
        "steps_sft_fail":     H2_steps_sft_fail,  "steps_arpo_fail":     H2_steps_arpo_fail,
        "paired_both_fail_sft":  H2_paired_both_fail_sft,
        "paired_both_fail_arpo": H2_paired_both_fail_arpo,
    },
    "H3_first3": {
        "marginal_clean":  H3_clean_rate,  "n_clean": len(clean),
        "marginal_dirty":  H3_dirty_rate,  "n_dirty": len(dirty),
    },
    "H4_cross_rollout_entropy": {
        "sft_mean":  H4_sft_mean, "arpo_mean": H4_arpo_mean,
        "paired_sft_minus_arpo": H4_paired_diff,
        "by_cohort": {k: {"sft": v[0], "arpo": v[1], "n": v[2]} for k, v in H4_by_cohort.items()},
    },
    "H5_pass8_minus_pass1": {
        "sft_all":  H5_sft,  "arpo_all":  H5_arpo,
        "sft_solvable_only":  H5_sft_solv,  "arpo_solvable_only":  H5_arpo_solv,
        "n_sft_solvable": len(sft_solvable),  "n_arpo_solvable": len(arpo_solvable),
    },
}
with open(os.path.join(OUT, "raw_metrics.json"), "w") as f:
    json.dump(raw, f, indent=2, default=str)

print(json.dumps(raw, indent=2, default=str))
print("\nWrote outputs to", OUT)
