#!/bin/bash
# =============================================================================
# SFT Training with Per-Epoch Evaluation on OSWorld (306 tasks, n=1)
#
# Pipeline:
#   1. Baseline evaluation (original model)
#   2. For each epoch 1..N: train → cleanup GPU → evaluate → cleanup GPU
#
# All results are logged to {OUTPUT_DIR}/training_log.json
#
# Usage:
#   bash scripts/run_sft_with_eval.sh
#
# Environment variables (override defaults):
#   MODEL_PATH    — base model (default: ByteDance-Seed/UI-TARS-1.5-7B)
#   DATA_PATH     — trajectory JSONL (default: checkpoints/arpo-inference/sft_86tasks_trajectories.jsonl)
#   OUTPUT_DIR    — output directory (default: checkpoints/sft_86tasks)
#   NUM_EPOCHS    — number of epochs (default: 5)
#   NUM_GPUS      — GPUs for training (default: 8)
#   SKIP_BASELINE — set to 1 to skip baseline eval (default: 0)
#   SKIP_TRAINING — set to 1 to skip training, only eval existing checkpoints (default: 0)
#   LR            — learning rate (default: 1e-5)
#   LR_SCHEDULER  — lr schedule type (default: cosine)
#   WARMUP_RATIO  — warmup ratio (default: 0.03)
#   MAX_GRAD_NORM — gradient clipping (default: 1.0)
#   BATCH_SIZE    — per-device batch size (default: 1)
#   GRAD_ACCUM    — gradient accumulation steps (default: 4)
#   MAX_LENGTH    — max sequence length (default: 12000)
#   LIMIT_IMAGES  — max screenshots in context (default: 3)
# =============================================================================

set -euo pipefail

# --- Configuration ---
MODEL_PATH="${MODEL_PATH:-ByteDance-Seed/UI-TARS-1.5-7B}"
DATA_PATH="${DATA_PATH:-checkpoints/arpo-inference/sft_86tasks_trajectories.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_86tasks}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
NUM_GPUS="${NUM_GPUS:-8}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
LR="${LR:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LENGTH="${MAX_LENGTH:-12000}"
LIMIT_IMAGES="${LIMIT_IMAGES:-3}"

EVAL_CONFIG="configs/sft_eval_306tasks_n1.yaml"
LOG_FILE="${OUTPUT_DIR}/training_log.json"
PYTHON_BIN="${PYTHON_BIN:-$(which python3)}"

echo "============================================================"
echo "SFT Training with Per-Epoch Evaluation"
echo "============================================================"
echo "Model:         ${MODEL_PATH}"
echo "Data:          ${DATA_PATH}"
echo "Output:        ${OUTPUT_DIR}"
echo "Epochs:        ${NUM_EPOCHS}"
echo "GPUs:          ${NUM_GPUS}"
echo "LR:            ${LR} (${LR_SCHEDULER}, warmup=${WARMUP_RATIO})"
echo "Max grad norm: ${MAX_GRAD_NORM}"
echo "Batch size:    ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS} = $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "Max length:    ${MAX_LENGTH}"
echo "Limit images:  ${LIMIT_IMAGES}"
echo "Eval config:   ${EVAL_CONFIG}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

# --- Helper: kill lingering Ray/vLLM processes and free GPU memory ---
# verl.trainer.main uses os._exit(0) which bypasses ray.shutdown(), leaving
# orphaned Ray workers holding GPU memory. This function forces cleanup.
cleanup_gpu() {
    echo ""
    echo ">>> Cleaning up GPU processes..."

    # Stop the Ray cluster (kills all Ray workers including vLLM engines)
    sudo -E env "PATH=${PATH}" ray stop --force 2>/dev/null || true

    # Kill any remaining python processes holding GPU memory (safety net)
    # Only kill processes with "ray::" or "vllm" in their cmdline
    sudo pkill -9 -f "ray::" 2>/dev/null || true
    sudo pkill -9 -f "from vllm" 2>/dev/null || true
    sudo pkill -9 -f "raylet" 2>/dev/null || true
    sudo pkill -9 -f "gcs_server" 2>/dev/null || true

    # Wait for GPU memory to be fully released
    sleep 5

    # Verify GPUs are free
    local gpu_procs
    gpu_procs=$(sudo nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l) || gpu_procs=0
    if [ "${gpu_procs}" -gt 0 ]; then
        echo ">>> WARNING: ${gpu_procs} processes still using GPU after cleanup"
        sudo nvidia-smi 2>/dev/null || true
        sleep 5
    else
        echo ">>> GPUs are free"
    fi
}

# --- Helper: run evaluation and record results ---
run_eval() {
    local model_path="$1"
    local eval_name="$2"
    local save_path="${OUTPUT_DIR}/eval_${eval_name}"

    echo ""
    echo ">>> Evaluating: ${eval_name} (model: ${model_path})"
    echo ">>> Save path: ${save_path}"
    echo ""

    sudo -E env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH:-}" "${PYTHON_BIN}" -m verl.trainer.main \
        config="${EVAL_CONFIG}" \
        worker.actor.model.model_path="${model_path}" \
        trainer.experiment_name="sft_eval_${eval_name}" \
        trainer.save_checkpoint_path="${save_path}"

    # Parse eval results and update training_log.json
    local eval_results_file="${save_path}/eval_results_at_0.json"
    if [ -f "${eval_results_file}" ]; then
        sudo -E env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH:-}" "${PYTHON_BIN}" << PYEOF
import json, os, sys, tempfile

eval_results_file = "${eval_results_file}"
log_file = "${LOG_FILE}"
eval_name = "${eval_name}"

with open(eval_results_file) as f:
    eval_results = json.load(f)

# Compute summary
n_tasks = len(eval_results)
n_doable = sum(1 for v in eval_results.values() if v.get("n_success", 0) > 0)
total_attempts = sum(v.get("n_attempts", 0) for v in eval_results.values())
total_successes = sum(v.get("n_success", 0) for v in eval_results.values())

summary = {
    "n_tasks_evaluated": n_tasks,
    "n_tasks_doable": n_doable,
    "doable_rate": round(n_doable / n_tasks, 4) if n_tasks > 0 else 0,
    "total_attempts": total_attempts,
    "total_successes": total_successes,
    "overall_success_rate": round(total_successes / total_attempts, 4) if total_attempts > 0 else 0,
    "per_domain": {},
}

# Per-domain breakdown
domain_stats = {}
for tid, v in eval_results.items():
    # Domain is not stored in eval_results; just aggregate overall
    pass

# Load and update log
try:
    with open(log_file) as f:
        log = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    log = {"experiment": "sft_86tasks", "epochs": []}

eval_entry = {
    "eval_results_file": eval_results_file,
    "summary": summary,
    "per_task_results": eval_results,
}

if eval_name == "baseline":
    log["baseline_eval"] = eval_entry
    log["baseline_eval"]["model_path"] = "${model_path}"
else:
    # Find the epoch entry and update it
    epoch_num = eval_name.replace("epoch_", "")
    try:
        epoch_num = int(epoch_num)
    except ValueError:
        epoch_num = -1
    for ep in log.get("epochs", []):
        if ep.get("epoch") == epoch_num:
            ep["eval_results"] = eval_entry["per_task_results"]
            ep["eval_summary"] = summary
            ep["eval_results_file"] = eval_results_file
            break
    else:
        # Epoch entry not found (training log not written yet), add standalone
        log.setdefault("standalone_evals", {})[eval_name] = eval_entry

import tempfile
tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(log_file) or ".")
with os.fdopen(tmp_fd, "w") as f:
    json.dump(log, f, indent=2)
os.replace(tmp_path, log_file)

print(f"[eval] {eval_name}: {n_doable}/{n_tasks} doable ({summary['doable_rate']*100:.1f}%), "
      f"overall success rate: {summary['overall_success_rate']*100:.2f}%")
PYEOF
    else
        echo "[eval] WARNING: eval results file not found: ${eval_results_file}"
    fi
}

# --- Step 0: Baseline evaluation ---
if [ "${SKIP_BASELINE}" != "1" ]; then
    echo ""
    echo "============================================================"
    echo "Step 0: Baseline Evaluation"
    echo "============================================================"
    run_eval "${MODEL_PATH}" "baseline"
    cleanup_gpu
else
    echo ""
    echo ">>> Skipping baseline evaluation (SKIP_BASELINE=1)"
fi

# --- Steps 1-N: Train one epoch at a time, evaluate after each ---
if [ "${SKIP_TRAINING}" != "1" ]; then
    for epoch in $(seq 1 "${NUM_EPOCHS}"); do
        EPOCH_MODEL="${OUTPUT_DIR}/epoch_${epoch}"
        EVAL_DIR="${OUTPUT_DIR}/eval_epoch_${epoch}"

        # --- Skip already-completed epochs (safe restart) ---
        if [ -d "${EPOCH_MODEL}" ] && [ -f "${EVAL_DIR}/eval_results_at_0.json" ]; then
            echo ""
            echo ">>> Epoch ${epoch} already trained + evaluated, skipping"
            continue
        fi

        # --- Train (skip if model already exists from a previous run) ---
        if [ -d "${EPOCH_MODEL}" ]; then
            echo ""
            echo ">>> Epoch ${epoch} model exists at ${EPOCH_MODEL}, skipping training"
        else
            echo ""
            echo "============================================================"
            echo "Step ${epoch}a: Training Epoch ${epoch}/${NUM_EPOCHS}"
            echo "============================================================"

            # Resume from latest checkpoint (works for ALL epochs, including epoch 1 on restart)
            RESUME_ARG=""
            PREV_CKPT=$(ls -td "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | head -1 || true)
            if [ -n "${PREV_CKPT}" ]; then
                RESUME_ARG="--resume_from_checkpoint ${PREV_CKPT}"
                echo ">>> Resuming from: ${PREV_CKPT}"
            fi

            sudo -E env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH:-}" \
            torchrun --nproc_per_node="${NUM_GPUS}" scripts/train_sft.py \
                --model_path "${MODEL_PATH}" \
                --data_path "${DATA_PATH}" \
                --output_dir "${OUTPUT_DIR}" \
                --num_epochs "${NUM_EPOCHS}" \
                --stop_after_epoch "${epoch}" \
                --per_device_batch_size "${BATCH_SIZE}" \
                --gradient_accumulation_steps "${GRAD_ACCUM}" \
                --learning_rate "${LR}" \
                --lr_scheduler_type "${LR_SCHEDULER}" \
                --warmup_ratio "${WARMUP_RATIO}" \
                --max_grad_norm "${MAX_GRAD_NORM}" \
                --max_length "${MAX_LENGTH}" \
                --limit_images "${LIMIT_IMAGES}" \
                --logging_steps 1 \
                ${RESUME_ARG}

            # Free GPU memory from training before starting eval
            cleanup_gpu
        fi

        # --- Evaluate ---
        echo ""
        echo "============================================================"
        echo "Step ${epoch}b: Evaluating Epoch ${epoch}"
        echo "============================================================"

        if [ -d "${EPOCH_MODEL}" ]; then
            run_eval "${EPOCH_MODEL}" "epoch_${epoch}"
            cleanup_gpu
        else
            echo "[ERROR] Epoch ${epoch} model not found at ${EPOCH_MODEL}"
            echo "  Available dirs: $(ls -d ${OUTPUT_DIR}/epoch_* 2>/dev/null || echo 'none')"
        fi
    done
else
    echo ""
    echo ">>> Skipping training (SKIP_TRAINING=1)"
    echo ">>> Evaluating existing checkpoints..."
    for epoch in $(seq 1 "${NUM_EPOCHS}"); do
        EPOCH_MODEL="${OUTPUT_DIR}/epoch_${epoch}"
        if [ -d "${EPOCH_MODEL}" ]; then
            run_eval "${EPOCH_MODEL}" "epoch_${epoch}"
            cleanup_gpu
        else
            echo "[SKIP] No checkpoint for epoch ${epoch} at ${EPOCH_MODEL}"
        fi
    done
fi

echo ""
echo "============================================================"
echo "All Done!"
echo "============================================================"
echo "Training log: ${LOG_FILE}"
echo ""

# Print final summary
sudo -E env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH:-}" "${PYTHON_BIN}" << PYEOF
import json

try:
    with open("${LOG_FILE}") as f:
        log = json.load(f)
except Exception:
    print("  (no log file found)")
    exit(0)

print("Summary:")
print(f"  Model: {log.get('config', {}).get('model_path', 'N/A')}")
print(f"  Dataset: {log.get('dataset', {}).get('num_sft_examples', 'N/A')} examples, "
      f"{log.get('dataset', {}).get('num_unique_tasks', 'N/A')} tasks")
print()

baseline = log.get("baseline_eval", {}).get("summary", {})
if baseline:
    print(f"  Baseline: {baseline.get('n_tasks_doable', '?')}/{baseline.get('n_tasks_evaluated', '?')} doable "
          f"({baseline.get('doable_rate', 0)*100:.1f}%), "
          f"success rate: {baseline.get('overall_success_rate', 0)*100:.2f}%")

for ep in log.get("epochs", []):
    loss = ep.get("training_metrics", {}).get("avg_loss")
    summary = ep.get("eval_summary", {})
    loss_str = f"loss={loss:.4f}" if loss is not None else "loss=N/A"
    if summary:
        print(f"  Epoch {ep['epoch']}: {loss_str}, "
              f"{summary.get('n_tasks_doable', '?')}/{summary.get('n_tasks_evaluated', '?')} doable "
              f"({summary.get('doable_rate', 0)*100:.1f}%), "
              f"success rate: {summary.get('overall_success_rate', 0)*100:.2f}%")
    else:
        print(f"  Epoch {ep['epoch']}: {loss_str}, eval=pending")
PYEOF
