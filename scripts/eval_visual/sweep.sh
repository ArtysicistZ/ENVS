#!/usr/bin/env bash
# =============================================================================
# Visual VLM generalization sweep: 3 checkpoints × 3 benchmarks.
#
# Pipeline:
#   1. Run parser regression tests (gates everything; no GPU needed).
#   2. For each checkpoint:
#        a. For each benchmark: run runner (resume-safe, single vLLM load).
#   3. Aggregate all results into a 3×3 markdown summary.
#
# Eval is CLEAN — no runtime noise injected. Greedy decode, seed=0.
# Uses TP=8 to occupy all 8 GPUs.
#
# Resume: every runner is JSONL append-resume; rerunning skips done samples.
# Failure isolation: if one (checkpoint, benchmark) cell fails, others still run.
#
# Usage:
#   bash scripts/eval_visual/sweep.sh                    # full sweep
#   SMOKE=16 bash scripts/eval_visual/sweep.sh           # 16-sample smoke per cell
#   ONLY_CHECKPOINTS="base_uitars_1.5_7b" \
#       bash scripts/eval_visual/sweep.sh                # filter to one ckpt
#   ONLY_BENCHES="screenspot_v2" \
#       bash scripts/eval_visual/sweep.sh                # filter to one bench
# =============================================================================

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/kevinzyz/arpo_local}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
LOGS_ROOT="${LOGS_ROOT:-${REPO_ROOT}/logs/eval_visual}"
HF_CACHE="${HF_CACHE:-${REPO_ROOT}/cache_dirs/hf_cache}"
SMOKE="${SMOKE:-0}"  # >0 = run only N samples per cell
TP="${TP:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# max-tokens is per-bench (set in BENCH_EXTRA below):
#   ScreenSpot V2/Pro emit short "(x,y)" — 64 tokens is plenty.
#   OSWorld-G uses the full UI-TARS agent grammar (Thought:.../Action: click(...)
#   plus optional fail()) — 64 truncates the action half. Bumped to 256.

ONLY_CHECKPOINTS="${ONLY_CHECKPOINTS:-}"
ONLY_BENCHES="${ONLY_BENCHES:-}"

cd "${REPO_ROOT}"

mkdir -p "${LOGS_ROOT}" "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export PYTHONUNBUFFERED=1

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# -----------------------------------------------------------------------------
# Checkpoints (tag, model_path)
# -----------------------------------------------------------------------------
CHECKPOINT_TAGS=(
  "base_uitars_1.5_7b"
  "clean_sft_v2.1"
  "noisy_sft_v3"
)
CHECKPOINT_PATHS=(
  "ByteDance-Seed/UI-TARS-1.5-7B"
  "${REPO_ROOT}/checkpoints/mcts_sft_v2.1/beta05_2e-6/epoch_1"
  "${REPO_ROOT}/checkpoints/mcts_sft_v3_noisy_combined/beta05_1.5e-6/epoch_1"
)

# -----------------------------------------------------------------------------
# Benchmarks (name, runner_script, extra_args)
# -----------------------------------------------------------------------------
BENCH_NAMES=("screenspot" "screenspot_pro" "osworld_g")
BENCH_RUNNERS=(
  "scripts/eval_visual/run_screenspot_v2.py"
  "scripts/eval_visual/run_screenspot_pro.py"
  "scripts/eval_visual/run_osworld_g.py"
)
# Per-benchmark extra args (space-separated).
# NOTE on ScreenSpot: OS-Copilot/ScreenSpot-v2 fails to load on HF (broken
# parquet — DatasetGenerationError). Using rootsautomation/ScreenSpot (V1,
# 1272 samples, same task, canonical published numbers). V2 differs only in
# annotation cleanup; V1 still measures the exact same generalization signal.
BENCH_EXTRA=(
  "--dataset-repo rootsautomation/ScreenSpot --split test --batch-size 16 --max-tokens 64"
  "--dataset-repo lmms-lab/ScreenSpot-Pro --split train --batch-size 8 --max-tokens 64"
  "--batch-size 12 --max-tokens 256"
)

filter_in() {
  # $1=needle  $2=csv_haystack (empty=match all)
  local needle="$1" csv="$2"
  [[ -z "${csv}" ]] && return 0
  local IFS=','
  for v in ${csv}; do
    [[ "${v}" == "${needle}" ]] && return 0
  done
  return 1
}

# -----------------------------------------------------------------------------
# Step 1: parser regression tests (no GPU)
# -----------------------------------------------------------------------------
echo "[$(stamp)] [sweep] step 1: parser regression tests"
PARSER_TEST_LOG="${LOGS_ROOT}/parser_tests.log"
if "${PYTHON_BIN}" -m pytest tests/eval_visual/test_parser.py -v \
    > "${PARSER_TEST_LOG}" 2>&1; then
  echo "[$(stamp)] [sweep] parser tests PASSED -> ${PARSER_TEST_LOG}"
else
  echo "[$(stamp)] [sweep] parser tests FAILED — see ${PARSER_TEST_LOG}"
  tail -40 "${PARSER_TEST_LOG}"
  exit 2
fi

# -----------------------------------------------------------------------------
# Step 2: 3 × 3 sweep
# -----------------------------------------------------------------------------
N_CK=${#CHECKPOINT_TAGS[@]}
N_BENCH=${#BENCH_NAMES[@]}

declare -A CELL_STATUS
TOTAL_CELLS=0
FAILED_CELLS=0

for ((i=0; i<N_CK; i++)); do
  ck_tag="${CHECKPOINT_TAGS[$i]}"
  ck_path="${CHECKPOINT_PATHS[$i]}"
  if ! filter_in "${ck_tag}" "${ONLY_CHECKPOINTS}"; then
    echo "[$(stamp)] [sweep] skip checkpoint ${ck_tag} (not in ONLY_CHECKPOINTS)"
    continue
  fi

  for ((j=0; j<N_BENCH; j++)); do
    bench="${BENCH_NAMES[$j]}"
    runner="${BENCH_RUNNERS[$j]}"
    extra="${BENCH_EXTRA[$j]}"
    if ! filter_in "${bench}" "${ONLY_BENCHES}"; then
      continue
    fi

    TOTAL_CELLS=$((TOTAL_CELLS + 1))
    out_dir="${LOGS_ROOT}/${ck_tag}/${bench}"
    cell_log="${out_dir}/run.log"
    mkdir -p "${out_dir}"

    smoke_arg=""
    if [[ "${SMOKE}" != "0" ]]; then
      smoke_arg="--smoke ${SMOKE}"
    fi

    echo ""
    echo "============================================================"
    echo "[$(stamp)] [sweep] cell: ${ck_tag} / ${bench}"
    echo "  model: ${ck_path}"
    echo "  out:   ${out_dir}"
    echo "  log:   ${cell_log}"
    echo "============================================================"

    set +e
    "${PYTHON_BIN}" "${runner}" \
      --model-path "${ck_path}" \
      --output-dir "${out_dir}" \
      --tensor-parallel-size "${TP}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      ${extra} \
      ${smoke_arg} \
      > "${cell_log}" 2>&1
    rc=$?
    set -e

    if [[ ${rc} -eq 0 ]]; then
      CELL_STATUS["${ck_tag}/${bench}"]="OK"
      echo "[$(stamp)] [sweep] cell OK"
      tail -3 "${cell_log}"
    else
      CELL_STATUS["${ck_tag}/${bench}"]="FAIL(rc=${rc})"
      FAILED_CELLS=$((FAILED_CELLS + 1))
      echo "[$(stamp)] [sweep] cell FAILED (rc=${rc}) — see ${cell_log}"
      tail -20 "${cell_log}"
    fi

    # Aggressive between-cell GPU cleanup so the next vLLM init starts fresh.
    pkill -9 -f "from vllm" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 4
  done
done

# -----------------------------------------------------------------------------
# Step 3: aggregate
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[$(stamp)] [sweep] aggregating results"
echo "============================================================"
SUMMARY_MD="${LOGS_ROOT}/SUMMARY.md"
"${PYTHON_BIN}" scripts/eval_visual/summary.py \
  --root "${LOGS_ROOT}" \
  --out "${SUMMARY_MD}" || true

echo ""
echo "============================================================"
echo "[$(stamp)] [sweep] FINAL CELL STATUS (${TOTAL_CELLS} cells, ${FAILED_CELLS} failed)"
echo "============================================================"
for key in "${!CELL_STATUS[@]}"; do
  printf "  %-50s %s\n" "${key}" "${CELL_STATUS[$key]}"
done

echo ""
echo "[$(stamp)] [sweep] summary: ${SUMMARY_MD}"

if [[ ${FAILED_CELLS} -gt 0 ]]; then
  exit 1
fi
exit 0
