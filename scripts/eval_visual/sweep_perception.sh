#!/usr/bin/env bash
# =============================================================================
# Perception/hallucination/calibration sweep — 3 ckpts × 4 benches = 12 cells.
# Eval is CLEAN (no runtime noise). Greedy decode, seed=0.
#
# Architecture: single-replica per cell, TP=1, max_num_seqs=32.
#   - These benches have SHORT outputs (yes/no, MCQ letter), ~100 tokens max.
#   - DP=8 model-load overhead would dominate (~12 cells × 8 reloads × 1 min).
#   - One vLLM instance per cell at TP=1 with high max_num_seqs gives the best
#     throughput shape for short-output workloads.
#
# Datasets are downloaded by VLMEvalKit on first cell; subsequent cells reuse
# the cache. Cache root: $HF_HOME (defaults to repo cache_dirs/hf_cache).
#
# Resume: predictions.tsv per shard is append-resume. Re-running skips done
# samples. Failed cells can be re-run individually via ONLY_CHECKPOINTS / ONLY_BENCHES.
# =============================================================================

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/kevinzyz/arpo_main}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
LOGS_ROOT="${LOGS_ROOT:-${REPO_ROOT}/logs/eval_visual}"
HF_CACHE="${HF_CACHE:-${REPO_ROOT}/cache_dirs/hf_cache}"
SMOKE="${SMOKE:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
TP="${TP:-1}"

ONLY_CHECKPOINTS="${ONLY_CHECKPOINTS:-}"
ONLY_BENCHES="${ONLY_BENCHES:-}"

cd "${REPO_ROOT}"

mkdir -p "${LOGS_ROOT}" "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1

# 4.6 root partition is at 100% — keep all caches off /.
EXTRA_CACHE_ROOT="${REPO_ROOT}/cache_dirs/runtime_cache"
mkdir -p "${EXTRA_CACHE_ROOT}/xdg" "${EXTRA_CACHE_ROOT}/tmp" \
         "${EXTRA_CACHE_ROOT}/vllm" "${EXTRA_CACHE_ROOT}/triton" "${EXTRA_CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${EXTRA_CACHE_ROOT}/xdg"
export TMPDIR="${EXTRA_CACHE_ROOT}/tmp"
export VLLM_CACHE_ROOT="${EXTRA_CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${EXTRA_CACHE_ROOT}/triton"
export TORCH_HOME="${EXTRA_CACHE_ROOT}/torch"

# LMUDataRoot for VLMEvalKit dataset cache (TSVs + images).
# 4.6 root partition is at 100% — VLMEvalKit's default of ~/LMUData would fill /.
# Use /mnt explicitly. (We pre-moved any existing ~/LMUData to here as a symlink target.)
export LMUData="${REPO_ROOT}/cache_dirs/LMUData_real"
mkdir -p "${LMUData}"

# VLMEvalKit + extra_pkgs (--target install) on PYTHONPATH.
# extra_pkgs first so missing libs (tabulate, decord, etc) resolve, but AFTER
# REPO_ROOT so our own modules win conflicts; AFTER VLMEvalKit so the kit
# resolves its own internal imports first.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/cache_dirs/VLMEvalKit:${REPO_ROOT}/cache_dirs/extra_pkgs:${PYTHONPATH:-}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

CHECKPOINT_TAGS=(
  "base_uitars_1.5_7b"
  "clean_sft_v2.1_no_mask"
  "noisy_sft_v3"
)
CHECKPOINT_PATHS=(
  "ByteDance-Seed/UI-TARS-1.5-7B"
  "${REPO_ROOT}/checkpoints/mcts_sft_v2.1_no_mask/beta05_1.886e-6/epoch_1"
  "${REPO_ROOT}/checkpoints/mcts_sft_v3_noisy_combined/beta05_1.5e-6/epoch_1"
)

# 7 benches across two waves:
#   Wave 1 (already complete): HallusionBench, POPE, MMVP, MMStar
#   Wave 2 (extends Wave 1): BLINK, MathVerse_MINI_Vision_Only, AMBER
# All yes/no or MCQ; canonical VLMEvalKit metric per bench.
BENCH_NAMES=("HallusionBench" "POPE" "MMVP" "MMStar" "BLINK" "MathVerse_MINI_Vision_Only" "AMBER")
# Batch and max-tokens. Yes/no needs ~32; MCQ needs ~64-128; MathVerse is free-form numeric so 256.
BENCH_BATCH_SIZE=(16 16 16 16 16 16 16)
BENCH_MAX_TOKENS=(64 64 96 128 96 256 64)
# Per-bench SMOKE caps for 10% pilot (set SMOKE_FRAC=0 to use full size).
BENCH_SMOKE_10PCT=(113 900 60 150 190 78 1421)
# Per-bench HARD cap (always applied regardless of SMOKE_FRAC). Set to 0 for no cap.
# AMBER capped at 3000 (full is 14216, would dominate wall-clock for marginal extra signal).
BENCH_HARD_CAP=(0 0 0 0 0 0 3000)
SMOKE_FRAC="${SMOKE_FRAC:-0.10}"

filter_in() {
  local needle="$1" csv="$2"
  [[ -z "${csv}" ]] && return 0
  local IFS=','
  for v in ${csv}; do
    [[ "${v}" == "${needle}" ]] && return 0
  done
  return 1
}

# -----------------------------------------------------------------------------
# Step 1: parser regression tests (gates everything; fast, no GPU)
# -----------------------------------------------------------------------------
echo "[$(stamp)] [perception] step 1: parser regression tests"
PARSER_TEST_LOG="${LOGS_ROOT}/parser_tests_perception.log"
if "${PYTHON_BIN}" -m pytest tests/eval_visual/test_parser.py -q \
    > "${PARSER_TEST_LOG}" 2>&1; then
  echo "[$(stamp)] [perception] parser tests PASSED"
else
  echo "[$(stamp)] [perception] parser tests FAILED — see ${PARSER_TEST_LOG}"
  tail -40 "${PARSER_TEST_LOG}"
  exit 2
fi

# -----------------------------------------------------------------------------
# Step 2: outer-loop bench, inner-loop checkpoint sweep.
# Per user directive: for each bench, run all 3 models, then move to next bench.
# This means the bench dataset cache is hot (loaded once, reused 3 times) and
# per-bench Δ is a directly-comparable 3-row vertical strip.
# -----------------------------------------------------------------------------
N_CK=${#CHECKPOINT_TAGS[@]}
N_BENCH=${#BENCH_NAMES[@]}

declare -A CELL_STATUS
TOTAL_CELLS=0
FAILED_CELLS=0

for ((j=0; j<N_BENCH; j++)); do
  bench="${BENCH_NAMES[$j]}"
  bs="${BENCH_BATCH_SIZE[$j]}"
  mt="${BENCH_MAX_TOKENS[$j]}"
  per_bench_smoke="${BENCH_SMOKE_10PCT[$j]}"
  if ! filter_in "${bench}" "${ONLY_BENCHES}"; then
    continue
  fi

  echo ""
  echo "############################################################"
  echo "[$(stamp)] [perception] BENCH ${bench} — sweeping ${N_CK} checkpoints"
  echo "############################################################"

  for ((i=0; i<N_CK; i++)); do
    ck_tag="${CHECKPOINT_TAGS[$i]}"
    ck_path="${CHECKPOINT_PATHS[$i]}"
    if ! filter_in "${ck_tag}" "${ONLY_CHECKPOINTS}"; then
      echo "[$(stamp)] [perception] skip checkpoint ${ck_tag}"
      continue
    fi

    TOTAL_CELLS=$((TOTAL_CELLS + 1))
    out_dir="${LOGS_ROOT}/${ck_tag}/${bench}"
    cell_log="${out_dir}/run.log"
    mkdir -p "${out_dir}"

    # SMOKE precedence: explicit SMOKE env var > BENCH_SMOKE_10PCT > BENCH_HARD_CAP > full.
    per_bench_cap="${BENCH_HARD_CAP[$j]}"
    if [[ "${SMOKE}" != "0" ]]; then
      smoke_arg="--smoke ${SMOKE}"
    elif [[ "${SMOKE_FRAC}" != "0" ]]; then
      smoke_arg="--smoke ${per_bench_smoke}"
    elif [[ "${per_bench_cap}" != "0" ]]; then
      smoke_arg="--smoke ${per_bench_cap}"
    else
      smoke_arg=""
    fi

    echo ""
    echo "============================================================"
    echo "[$(stamp)] [perception] cell: ${ck_tag} / ${bench} ${smoke_arg}"
    echo "  model: ${ck_path}"
    echo "  out:   ${out_dir}"
    echo "============================================================"

    # Run inference (single-replica, TP=1, num_shards=1).
    set +e
    "${PYTHON_BIN}" scripts/eval_visual/run_vlmevalkit.py \
      --model-path "${ck_path}" \
      --bench "${bench}" \
      --output-dir "${out_dir}" \
      --tensor-parallel-size "${TP}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-tokens "${mt}" \
      --batch-size "${bs}" \
      --num-shards 1 \
      --shard-id 0 \
      ${smoke_arg} \
      > "${cell_log}" 2>&1
    rc_infer=$?

    # Run scoring (only if inference succeeded).
    rc_score=0
    if [[ ${rc_infer} -eq 0 ]]; then
      "${PYTHON_BIN}" scripts/eval_visual/score_vlmevalkit.py \
        --output-dir "${out_dir}" \
        --bench "${bench}" \
        --model-path "${ck_path}" \
        >> "${cell_log}" 2>&1
      rc_score=$?
    fi
    set -e

    if [[ ${rc_infer} -eq 0 && ${rc_score} -eq 0 ]]; then
      CELL_STATUS["${ck_tag}/${bench}"]="OK"
      echo "[$(stamp)] [perception] cell OK"
      tail -3 "${cell_log}"
    else
      CELL_STATUS["${ck_tag}/${bench}"]="FAIL(infer=${rc_infer},score=${rc_score})"
      FAILED_CELLS=$((FAILED_CELLS + 1))
      echo "[$(stamp)] [perception] cell FAILED — see ${cell_log}"
      tail -25 "${cell_log}"
    fi

    # Per-cell GPU cleanup (only by PID lineage of THIS sweep).
    sleep 6
  done
done

# -----------------------------------------------------------------------------
# Step 3: aggregate (existing summary.py handles both grounding + perception)
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[$(stamp)] [perception] aggregating results"
echo "============================================================"
SUMMARY_MD="${LOGS_ROOT}/SUMMARY.md"
"${PYTHON_BIN}" scripts/eval_visual/summary.py \
  --root "${LOGS_ROOT}" \
  --out "${SUMMARY_MD}" || true

echo ""
echo "============================================================"
echo "[$(stamp)] [perception] FINAL CELL STATUS (${TOTAL_CELLS} cells, ${FAILED_CELLS} failed)"
echo "============================================================"
for key in "${!CELL_STATUS[@]}"; do
  printf "  %-50s %s\n" "${key}" "${CELL_STATUS[$key]}"
done

echo ""
echo "[$(stamp)] [perception] summary: ${SUMMARY_MD}"

if [[ ${FAILED_CELLS} -gt 0 ]]; then
  exit 1
fi
exit 0
