#!/usr/bin/env bash
# Data-parallel launcher: fans out one Python process per GPU, each loads vLLM
# on a single GPU (TP=1) and processes 1/N of the dataset. Merges shard JSONLs
# into a single results.jsonl when all done.
#
# Usage:
#   run_dp.sh <runner.py> <output_dir> <model_path> -- <bench_extra_args...>
#
# Env:
#   NUM_SHARDS  (default: 8)            — number of GPUs / data-parallel replicas
#   GPU_OFFSET  (default: 0)            — first GPU index to use
#   GPU_MEM_UTIL (default: 0.5)         — vLLM gpu_memory_utilization (per replica)
#   SMOKE       (default: 0)            — N>0 = only run the first N global samples
#                                          (each shard sees ~N/NUM_SHARDS)
#
# Failure handling:
#   - Each shard runs independently; one shard's crash does not abort others.
#   - sweep.sh treats a cell as failed if any shard's exit code is nonzero or
#     fewer than expected samples are merged.

set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <runner.py> <output_dir> <model_path> [-- <extra_args>]" >&2
  exit 2
fi

RUNNER="$1"; shift
OUTPUT_DIR="$1"; shift
MODEL_PATH="$1"; shift
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")

NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
SMOKE="${SMOKE:-0}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/kevinzyz/arpo_main/.venv/bin/python}"

mkdir -p "${OUTPUT_DIR}"
SHARDS_LOG_DIR="${OUTPUT_DIR}/shard_logs"
mkdir -p "${SHARDS_LOG_DIR}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] [run_dp] launching ${NUM_SHARDS} replicas (GPUs ${GPU_OFFSET}..$((GPU_OFFSET+NUM_SHARDS-1)))"
echo "[$(stamp)] [run_dp] runner=${RUNNER}  model=${MODEL_PATH}  out=${OUTPUT_DIR}"

PIDS=()
for ((s=0; s<NUM_SHARDS; s++)); do
  gpu=$((GPU_OFFSET + s))
  log="${SHARDS_LOG_DIR}/shard_${s}.log"
  smoke_arg=""
  if [[ "${SMOKE}" != "0" ]]; then smoke_arg="--smoke ${SMOKE}"; fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${RUNNER}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --shard-id "${s}" \
    --num-shards "${NUM_SHARDS}" \
    ${smoke_arg} \
    "${EXTRA_ARGS[@]}" \
    > "${log}" 2>&1 &
  PIDS+=($!)
  echo "[$(stamp)] [run_dp]  shard ${s} -> GPU ${gpu}, pid ${PIDS[-1]}, log ${log}"
  # Tiny stagger to avoid simultaneous HF cache contention on first run.
  sleep 1
done

# Wait for all shards. Track per-shard exit codes.
echo "[$(stamp)] [run_dp] waiting for ${#PIDS[@]} shards to finish..."
ANY_FAIL=0
for ((s=0; s<NUM_SHARDS; s++)); do
  if wait "${PIDS[$s]}"; then
    echo "[$(stamp)] [run_dp] shard ${s} OK"
  else
    rc=$?
    echo "[$(stamp)] [run_dp] shard ${s} FAILED (rc=${rc})"
    tail -20 "${SHARDS_LOG_DIR}/shard_${s}.log" || true
    ANY_FAIL=1
  fi
done

# Merge all shard JSONLs into a single results.jsonl.
echo "[$(stamp)] [run_dp] merging shard JSONLs"
MERGED="${OUTPUT_DIR}/results.jsonl"
: > "${MERGED}"
for ((s=0; s<NUM_SHARDS; s++)); do
  shard_jsonl="${OUTPUT_DIR}/shard_${s}/results.jsonl"
  if [[ -f "${shard_jsonl}" ]]; then
    cat "${shard_jsonl}" >> "${MERGED}"
  else
    echo "[$(stamp)] [run_dp] WARN: missing ${shard_jsonl}"
  fi
done
TOTAL=$(wc -l < "${MERGED}" 2>/dev/null || echo 0)
echo "[$(stamp)] [run_dp] merged ${TOTAL} samples into ${MERGED}"

if [[ ${ANY_FAIL} -ne 0 ]]; then
  exit 1
fi
exit 0
