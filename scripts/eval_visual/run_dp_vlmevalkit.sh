#!/usr/bin/env bash
# DP=8 wrapper for run_vlmevalkit.py.
# Fans out 8 single-GPU vLLM replicas, each processing 1/8 of the bench, then
# merges shard predictions and scores via score_vlmevalkit.py.
#
# Usage:
#   run_dp_vlmevalkit.sh <bench> <model_path> <output_dir> [extra_args...]
#
# Env:
#   NUM_SHARDS  (default: 8)            — number of GPUs / data-parallel replicas
#   GPU_OFFSET  (default: 0)            — first GPU index to use
#   GPU_MEM_UTIL (default: 0.5)         — vLLM gpu_memory_utilization (per replica)
#   MAX_TOKENS  (default: 96)           — generation cap (override per-bench)
#   BATCH_SIZE  (default: 16)
#   PYTHON_BIN  (default: ./.venv/bin/python)
#   REPO_ROOT   (default: pwd)

set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <bench> <model_path> <output_dir> [extra args]" >&2
  exit 2
fi

BENCH="$1"; shift
MODEL_PATH="$1"; shift
OUTPUT_DIR="$1"; shift
EXTRA_ARGS=("$@")

NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
MAX_TOKENS="${MAX_TOKENS:-96}"
BATCH_SIZE="${BATCH_SIZE:-16}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

mkdir -p "${OUTPUT_DIR}/shard_logs"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] [run_dp_vmek] launching ${NUM_SHARDS} replicas (GPUs ${GPU_OFFSET}..$((GPU_OFFSET+NUM_SHARDS-1)))"
echo "[$(stamp)] [run_dp_vmek] bench=${BENCH} model=${MODEL_PATH} out=${OUTPUT_DIR}"

PIDS=()
for ((s=0; s<NUM_SHARDS; s++)); do
  gpu=$((GPU_OFFSET + s))
  log="${OUTPUT_DIR}/shard_logs/shard_${s}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_visual/run_vlmevalkit.py" \
    --model-path "${MODEL_PATH}" \
    --bench "${BENCH}" \
    --output-dir "${OUTPUT_DIR}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-num-seqs 32 \
    --max-tokens "${MAX_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --shard-id "${s}" \
    --num-shards "${NUM_SHARDS}" \
    "${EXTRA_ARGS[@]}" \
    > "${log}" 2>&1 &
  PIDS+=($!)
  echo "[$(stamp)]  shard ${s} -> GPU ${gpu}, pid ${PIDS[-1]}, log ${log}"
  sleep 1
done

ANY_FAIL=0
for ((s=0; s<NUM_SHARDS; s++)); do
  if wait "${PIDS[$s]}"; then
    echo "[$(stamp)]  shard ${s} OK"
  else
    rc=$?
    echo "[$(stamp)]  shard ${s} FAILED (rc=${rc})"
    tail -15 "${OUTPUT_DIR}/shard_logs/shard_${s}.log" || true
    ANY_FAIL=1
  fi
done

# Merge shard TSVs into the canonical predictions.tsv. score_vlmevalkit.py
# already expects shard_<i>/predictions.tsv layout.
echo "[$(stamp)] [run_dp_vmek] scoring..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_visual/score_vlmevalkit.py" \
  --output-dir "${OUTPUT_DIR}" \
  --bench "${BENCH}" \
  --model-path "${MODEL_PATH}" 2>&1 | tail -5

if [[ ${ANY_FAIL} -ne 0 ]]; then
  exit 1
fi
exit 0
