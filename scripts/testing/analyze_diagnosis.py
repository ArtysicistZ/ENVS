"""Diagnosis analysis: SFT init vs ARPO step_35 on 86-trainable, n=1 T=0.

Outputs (under checkpoints/clean_eval/arpo_86tasks_clean_from_v2.1_no_mask_diagnosis/analysis/):
  - bucket_table.csv         : per-task bucket assignment
  - summary.md               : top-line numbers across all 6 sections
  - qualitative_examples.md  : 2-3 qualitative examples per pattern
"""

import json
import os
import re
from collections import defaultdict, Counter
from statistics import mean, median

ROOT = "/mnt/kevinzyz/arpo_local/checkpoints/clean_eval/arpo_86tasks_clean_from_v2.1_no_mask_diagnosis"
SFT_PATH = os.path.join(ROOT, "sft_init/trajectories_at_0.jsonl")
ARPO_PATH = os.path.join(ROOT, "arpo_step35/trajectories_at_35.jsonl")
OUT_DIR = "/mnt/kevinzyz/arpo_local/scripts/testing/diagnosis_analysis_out"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_STEPS = 15  # from env config


def load_jsonl(path):
    eps = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            eps[ep["task_id"]] = ep
    return eps


sft = load_jsonl(SFT_PATH)
arpo = load_jsonl(ARPO_PATH)
common = sorted(set(sft) & set(arpo))
assert len(common) == 86, f"expected 86 task overlap, got {len(common)}"


# --------- ACTION PARSING ---------
ACTION_RE = re.compile(r"Action\s*:\s*(\w+)\s*\(", re.IGNORECASE)
FINISH_RE = re.compile(r"Action\s*:\s*(finish|FINISH|done|DONE|terminate)\s*\(", re.IGNORECASE)


def parse_action_type(action_text):
    """Return the primary action verb from a model output, or 'parse_error'."""
    if not action_text:
        return "empty"
    m = ACTION_RE.search(action_text)
    if m:
        return m.group(1).lower()
    return "parse_error"


def is_finish(action_text):
    return bool(FINISH_RE.search(action_text or ""))


def classify_failure(ep):
    """Heuristic failure-mode classification. Only meaningful when eval_result==0."""
    steps = ep.get("steps") or []
    n = len(steps)
    if n == 0:
        return "empty_trajectory"
    last = steps[-1].get("action", "") or ""
    parse_errs = sum(1 for s in steps if parse_action_type(s.get("action", "")) == "parse_error")
    finishes = [i for i, s in enumerate(steps) if is_finish(s.get("action", ""))]
    if parse_errs / max(n, 1) > 0.3:
        return "parse_error_heavy"
    if finishes and finishes[0] < MAX_STEPS - 1:
        # ended via finish() before hitting max steps -> "early finish but eval=0"
        return "early_finish_eval_failed"
    if n >= MAX_STEPS:
        # Hit max steps -> wandering
        return "max_steps_no_progress"
    # Trajectory ended early without a finish -> likely env error or premature termination
    return "premature_termination"


# --------- 1. TASK-LEVEL BUCKETS ---------
buckets_4 = Counter()  # SFT_win / ARPO_win / both_solve / both_fail
buckets_directional = Counter()  # improvement / regression / same (with n=1 these collapse)
rows = []
for tid in common:
    s_score = sft[tid]["eval_result"]
    a_score = arpo[tid]["eval_result"]
    s_succ = 1 if s_score > 0 else 0
    a_succ = 1 if a_score > 0 else 0
    if s_succ and not a_succ:
        b = "SFT_win"
    elif not s_succ and a_succ:
        b = "ARPO_win"
    elif s_succ and a_succ:
        b = "both_solve"
    else:
        b = "both_fail"
    buckets_4[b] += 1
    # n=1 directional bucket
    if a_succ > s_succ:
        d = "improvement"
    elif a_succ < s_succ:
        d = "regression"
    else:
        d = "same"
    buckets_directional[d] += 1
    rows.append({
        "task_id": tid,
        "instruction": sft[tid]["instruction"][:120],
        "sft_score": s_score,
        "arpo_score": a_score,
        "sft_n_steps": len(sft[tid]["steps"]),
        "arpo_n_steps": len(arpo[tid]["steps"]),
        "bucket_4": b,
        "bucket_dir": d,
    })

# Write bucket CSV
with open(os.path.join(OUT_DIR, "bucket_table.csv"), "w") as f:
    f.write("task_id,sft_score,arpo_score,sft_steps,arpo_steps,bucket_4,bucket_dir,instruction\n")
    for r in rows:
        instr = r["instruction"].replace(",", " ").replace("\n", " ")
        f.write(f"{r['task_id']},{r['sft_score']},{r['arpo_score']},{r['sft_n_steps']},{r['arpo_n_steps']},{r['bucket_4']},{r['bucket_dir']},{instr}\n")

sft_succ_total = sum(1 for r in rows if r["sft_score"] > 0)
arpo_succ_total = sum(1 for r in rows if r["arpo_score"] > 0)


# --------- 2. TRAJECTORY LENGTH (success-conditioned) ---------
def length_stats(eps_dict, only_success=True):
    lens = []
    for tid, ep in eps_dict.items():
        if only_success and ep["eval_result"] <= 0:
            continue
        if not only_success and ep["eval_result"] > 0:
            continue
        lens.append(len(ep["steps"]))
    return lens


sft_succ_lens = length_stats(sft, only_success=True)
arpo_succ_lens = length_stats(arpo, only_success=True)
sft_fail_lens = length_stats(sft, only_success=False)
arpo_fail_lens = length_stats(arpo, only_success=False)


def bucket_lens(lens):
    short = sum(1 for x in lens if 1 <= x <= 5)
    med = sum(1 for x in lens if 6 <= x <= 9)
    long_ = sum(1 for x in lens if 10 <= x <= 15)
    return short, med, long_


def safe_stats(lens):
    if not lens:
        return None, None, 0, 0
    return (mean(lens), median(lens),
            sum(1 for x in lens if x >= 10) / len(lens) * 100,
            sum(1 for x in lens if x >= MAX_STEPS) / len(lens) * 100)


# --------- 3. FAILURE MODES ---------
def failure_mode_breakdown(eps_dict):
    modes = Counter()
    fail_lens = []
    for tid, ep in eps_dict.items():
        if ep["eval_result"] > 0:
            continue
        modes[classify_failure(ep)] += 1
        fail_lens.append(len(ep["steps"]))
    return modes, fail_lens


sft_fail_modes, sft_fail_step_lens = failure_mode_breakdown(sft)
arpo_fail_modes, arpo_fail_step_lens = failure_mode_breakdown(arpo)


# --------- 4. FIRST-STEPS COMPARISON ---------
def first_actions(ep, k=3):
    types = []
    for s in ep["steps"][:k]:
        types.append(parse_action_type(s.get("action", "")))
    return types


def first_action_text(ep, k=3, max_len=300):
    """Return abbreviated first-k action lines (just the Action: ... line)."""
    out = []
    for s in ep["steps"][:k]:
        a = s.get("action", "") or ""
        m = re.search(r"Action\s*:.*", a)
        line = m.group(0) if m else "(no Action: line)"
        if len(line) > max_len:
            line = line[:max_len] + "..."
        out.append(line)
    return out


# --------- 5. ACTION DISTRIBUTION ---------
def action_distribution(eps_dict):
    cnt = Counter()
    total = 0
    finish_step_positions = []  # at what step does FINISH first appear?
    repeated_actions_total = 0
    actions_total_for_rep = 0
    for tid, ep in eps_dict.items():
        steps = ep.get("steps") or []
        actions_in_ep = [parse_action_type(s.get("action", "")) for s in steps]
        for a in actions_in_ep:
            cnt[a] += 1
            total += 1
        # finish step position
        for i, s in enumerate(steps):
            if is_finish(s.get("action", "")):
                finish_step_positions.append(i + 1)  # 1-indexed
                break
        # repeated actions: consecutive identical action *texts* (full string)
        for i in range(1, len(steps)):
            actions_total_for_rep += 1
            a_prev = (steps[i - 1].get("action") or "").strip()
            a_cur = (steps[i].get("action") or "").strip()
            # Strip the Thought: ... prefix; compare just the Action: line
            def get_act(s):
                m = re.search(r"Action\s*:.*", s)
                return m.group(0).strip() if m else s
            if get_act(a_prev) == get_act(a_cur):
                repeated_actions_total += 1
    return cnt, total, finish_step_positions, repeated_actions_total, actions_total_for_rep


sft_act, sft_total, sft_fin_pos, sft_rep, sft_rep_tot = action_distribution(sft)
arpo_act, arpo_total, arpo_fin_pos, arpo_rep, arpo_rep_tot = action_distribution(arpo)


# --------- 6. WRITE SUMMARY ---------
lines = []
lines.append("# Diagnosis report: SFT init vs ARPO step_35 (86 trainable tasks, n=1, T=0)\n")
lines.append("**Caveat:** n=1 deterministic. Bucket counts are over single rollouts, not probability mass. ")
lines.append("Numbers like 'SFT solved 44, ARPO solved X' reflect what the **modal** policy does, not full success rate.\n")

lines.append("\n## 1. Task-level buckets\n")
lines.append(f"- Tasks: {len(common)}")
lines.append(f"- SFT solved: **{sft_succ_total}/86** ({sft_succ_total/86*100:.1f}%)")
lines.append(f"- ARPO step_35 solved: **{arpo_succ_total}/86** ({arpo_succ_total/86*100:.1f}%)")
lines.append(f"- Net coverage change: **{arpo_succ_total - sft_succ_total:+d} tasks**\n")
lines.append("Four-way bucket:")
lines.append("| Bucket | Count | %% |")
lines.append("|---|---|---|")
for b in ["SFT_win", "ARPO_win", "both_solve", "both_fail"]:
    n = buckets_4.get(b, 0)
    lines.append(f"| {b} | {n} | {n/86*100:.1f}% |")
lines.append("")
lines.append("Directional (with n=1, regression == SFT_win, improvement == ARPO_win):")
for d in ["improvement", "regression", "same"]:
    n = buckets_directional.get(d, 0)
    lines.append(f"- {d}: {n}")

lines.append("\n## 2. Trajectory length (success-conditioned)\n")
lines.append("| Model | success rate | avg success steps | median | %% steps>=10 | %% hit max(15) |")
lines.append("|---|---|---|---|---|---|")
for name, lens, total in [("SFT init", sft_succ_lens, 86), ("ARPO step_35", arpo_succ_lens, 86)]:
    sr = len(lens) / total * 100
    avg, med, pct_long, pct_max = safe_stats(lens)
    lines.append(f"| {name} | {sr:.1f}% | {avg:.2f if avg is not None else '-'}{'' if avg is None else ''} | {med if med is not None else '-'} | {pct_long:.1f}% | {pct_max:.1f}% |".replace("{med if med is not None else '-'}", str(med) if med is not None else '-'))

# Cleaner re-render
lines = lines[:-3]
for name, lens, total in [("SFT init", sft_succ_lens, 86), ("ARPO step_35", arpo_succ_lens, 86)]:
    sr = len(lens) / total * 100
    if not lens:
        lines.append(f"| {name} | {sr:.1f}% | - | - | - | - |")
    else:
        avg = mean(lens); med = median(lens)
        pct_long = sum(1 for x in lens if x >= 10) / len(lens) * 100
        pct_max = sum(1 for x in lens if x >= MAX_STEPS) / len(lens) * 100
        lines.append(f"| {name} | {sr:.1f}% | {avg:.2f} | {med} | {pct_long:.1f}% | {pct_max:.1f}% |")

lines.append("")
lines.append("Length buckets among **successful** trajectories:")
lines.append("| Model | short(1-5) | medium(6-9) | long(10-15) |")
lines.append("|---|---|---|---|")
for name, lens in [("SFT init", sft_succ_lens), ("ARPO step_35", arpo_succ_lens)]:
    if not lens:
        lines.append(f"| {name} | - | - | - |")
        continue
    s, m, l = bucket_lens(lens)
    n = len(lens)
    lines.append(f"| {name} | {s} ({s/n*100:.0f}%) | {m} ({m/n*100:.0f}%) | {l} ({l/n*100:.0f}%) |")

lines.append("\n## 3. Failure-mode breakdown\n")
lines.append("| Mode | SFT (n=" + str(86 - sft_succ_total) + ") | ARPO (n=" + str(86 - arpo_succ_total) + ") |")
lines.append("|---|---|---|")
all_modes = sorted(set(sft_fail_modes) | set(arpo_fail_modes))
for m in all_modes:
    s = sft_fail_modes.get(m, 0)
    a = arpo_fail_modes.get(m, 0)
    s_pct = s / max(86 - sft_succ_total, 1) * 100
    a_pct = a / max(86 - arpo_succ_total, 1) * 100
    lines.append(f"| {m} | {s} ({s_pct:.0f}%) | {a} ({a_pct:.0f}%) |")

if sft_fail_step_lens:
    lines.append(f"\nAvg failure trajectory length: SFT {mean(sft_fail_step_lens):.2f}, ARPO {mean(arpo_fail_step_lens):.2f}")

lines.append("\n## 5. Action-type distribution (all steps, all tasks)\n")
lines.append("| action | SFT count | SFT %% | ARPO count | ARPO %% | delta(pp) |")
lines.append("|---|---|---|---|---|---|")
all_acts = sorted(set(sft_act) | set(arpo_act), key=lambda a: -(sft_act.get(a,0)+arpo_act.get(a,0)))
for a in all_acts[:15]:
    sc = sft_act.get(a, 0); ac = arpo_act.get(a, 0)
    sp = sc / max(sft_total, 1) * 100
    ap = ac / max(arpo_total, 1) * 100
    delta = ap - sp
    lines.append(f"| {a} | {sc} | {sp:.1f}% | {ac} | {ap:.1f}% | {delta:+.1f} |")

lines.append("")
lines.append(f"- Repeated-action rate (consecutive same Action): SFT **{sft_rep/max(sft_rep_tot,1)*100:.1f}%** ({sft_rep}/{sft_rep_tot}), ARPO **{arpo_rep/max(arpo_rep_tot,1)*100:.1f}%** ({arpo_rep}/{arpo_rep_tot})")
lines.append(f"- First FINISH step position (avg): SFT {mean(sft_fin_pos):.2f if sft_fin_pos else '-'}, ARPO {mean(arpo_fin_pos):.2f if arpo_fin_pos else '-'} (n_finish: SFT={len(sft_fin_pos)}, ARPO={len(arpo_fin_pos)})".replace("{mean(sft_fin_pos):.2f if sft_fin_pos else '-'}", f"{mean(sft_fin_pos):.2f}" if sft_fin_pos else '-').replace("{mean(arpo_fin_pos):.2f if arpo_fin_pos else '-'}", f"{mean(arpo_fin_pos):.2f}" if arpo_fin_pos else '-'))

# --------- 4 + 6. QUALITATIVE EXAMPLES (separate file) ---------
qual = []
qual.append("# Qualitative examples — first-3-actions diff per bucket\n")

# Pattern A: SFT_win (RL forgot)
sft_wins = [r for r in rows if r["bucket_4"] == "SFT_win"]
qual.append(f"\n## Pattern A: SFT_win ({len(sft_wins)} tasks) — RL forgot SFT-solved task\n")
for r in sft_wins[:5]:
    tid = r["task_id"]
    qual.append(f"\n### {tid}")
    qual.append(f"**Instruction:** {sft[tid]['instruction']}")
    qual.append(f"**SFT eval={sft[tid]['eval_result']}, steps={len(sft[tid]['steps'])}**  |  **ARPO eval={arpo[tid]['eval_result']}, steps={len(arpo[tid]['steps'])}**")
    qual.append("\n**SFT first 3 actions:**")
    for i, line in enumerate(first_action_text(sft[tid], k=3)):
        qual.append(f"  {i+1}. {line}")
    qual.append("\n**ARPO first 3 actions:**")
    for i, line in enumerate(first_action_text(arpo[tid], k=3)):
        qual.append(f"  {i+1}. {line}")

# Pattern B: both_solve but ARPO much longer
both = [r for r in rows if r["bucket_4"] == "both_solve"]
both_arpo_longer = [r for r in both if r["arpo_n_steps"] > r["sft_n_steps"] + 3]
qual.append(f"\n\n## Pattern B: both_solve, ARPO much longer (>{3} extra steps) ({len(both_arpo_longer)} tasks) — RL added wandering\n")
for r in sorted(both_arpo_longer, key=lambda x: -(x["arpo_n_steps"] - x["sft_n_steps"]))[:5]:
    tid = r["task_id"]
    qual.append(f"\n### {tid} (SFT={r['sft_n_steps']} steps, ARPO={r['arpo_n_steps']} steps, +{r['arpo_n_steps']-r['sft_n_steps']})")
    qual.append(f"**Instruction:** {sft[tid]['instruction']}")
    qual.append("\n**SFT first 3 actions:**")
    for i, line in enumerate(first_action_text(sft[tid], k=3)):
        qual.append(f"  {i+1}. {line}")
    qual.append("\n**ARPO first 3 actions:**")
    for i, line in enumerate(first_action_text(arpo[tid], k=3)):
        qual.append(f"  {i+1}. {line}")

# Pattern C: ARPO_win (RL improved)
arpo_wins = [r for r in rows if r["bucket_4"] == "ARPO_win"]
qual.append(f"\n\n## Pattern C: ARPO_win ({len(arpo_wins)} tasks) — RL improved over SFT\n")
for r in arpo_wins[:5]:
    tid = r["task_id"]
    qual.append(f"\n### {tid}")
    qual.append(f"**Instruction:** {sft[tid]['instruction']}")
    qual.append(f"**SFT eval={sft[tid]['eval_result']}, steps={len(sft[tid]['steps'])}**  |  **ARPO eval={arpo[tid]['eval_result']}, steps={len(arpo[tid]['steps'])}**")
    qual.append("\n**SFT first 3 actions:**")
    for i, line in enumerate(first_action_text(sft[tid], k=3)):
        qual.append(f"  {i+1}. {line}")
    qual.append("\n**ARPO first 3 actions:**")
    for i, line in enumerate(first_action_text(arpo[tid], k=3)):
        qual.append(f"  {i+1}. {line}")

with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
    f.write("\n".join(lines))
with open(os.path.join(OUT_DIR, "qualitative_examples.md"), "w") as f:
    f.write("\n".join(qual))

print("Wrote:")
print(f"  {OUT_DIR}/bucket_table.csv")
print(f"  {OUT_DIR}/summary.md")
print(f"  {OUT_DIR}/qualitative_examples.md")
print()
print("="*70)
print("\n".join(lines))
EOF
