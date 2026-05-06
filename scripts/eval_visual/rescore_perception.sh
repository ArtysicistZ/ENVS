#!/usr/bin/env bash
# Re-run scoring on all completed perception cells.
# Deletes stale VLMEvalKit auxmatch files first so the evaluator re-extracts
# yes/no / MCQ-letter answers from the (now-full) predictions.tsv.
#
# Use this when predictions.tsv was extended (e.g. 10% → full sweep) but the
# score.json / summary.json still reflects the old auxmatch.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/kevinzyz/arpo_main}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
LOGS_ROOT="${LOGS_ROOT:-${REPO_ROOT}/logs/eval_visual}"
HF_CACHE="${HF_CACHE:-${REPO_ROOT}/cache_dirs/hf_cache}"

cd "${REPO_ROOT}"

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1
export LMUData="${REPO_ROOT}/cache_dirs/LMUData_real"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/cache_dirs/VLMEvalKit:${REPO_ROOT}/cache_dirs/extra_pkgs:${PYTHONPATH:-}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

CHECKPOINTS=("base_uitars_1.5_7b" "clean_sft_v2.1_no_mask" "noisy_sft_v3")
BENCHES=("HallusionBench" "POPE" "MMVP" "MMStar")

for ck in "${CHECKPOINTS[@]}"; do
  for bench in "${BENCHES[@]}"; do
    out_dir="${LOGS_ROOT}/${ck}/${bench}"
    if [[ ! -f "${out_dir}/predictions.tsv" ]]; then
      echo "[$(stamp)] [rescore] skip ${ck}/${bench} — no predictions.tsv"
      continue
    fi
    echo "[$(stamp)] [rescore] ${ck}/${bench}"
    # Remove every kind of stale intermediate file so VLMEvalKit re-extracts.
    rm -f "${out_dir}"/predictions_full*_auxmatch* \
          "${out_dir}"/predictions_full*_score* \
          "${out_dir}"/predictions_full*_tmp* \
          "${out_dir}"/predictions_full*_acc* \
          "${out_dir}"/predictions_full*_extract* 2>/dev/null || true

    "${PYTHON_BIN}" scripts/eval_visual/score_vlmevalkit.py \
      --output-dir "${out_dir}" \
      --bench "${bench}" \
      --model-path "<${ck}>" 2>&1 | tail -3
  done
done

echo ""
echo "[$(stamp)] [rescore] aggregating SUMMARY.md"
"${PYTHON_BIN}" scripts/eval_visual/summary.py --root "${LOGS_ROOT}" --out "${LOGS_ROOT}/SUMMARY.md" || true

echo "[$(stamp)] [rescore] done"
