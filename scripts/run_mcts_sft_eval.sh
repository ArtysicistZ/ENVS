#!/bin/bash
# =============================================================================
# MCTS SFT: Train 1 epoch + Evaluate
#
# Pipeline:
#   1. Train MCTS SFT for 1 epoch (step-masked, task-balanced resampling)
#   2. Cleanup GPU
#   3. Evaluate on 300 tasks (n=1)
#   4. Record results in training_log.json
#
# Usage:
#   bash scripts/run_mcts_sft_eval.sh
#
# To skip training (eval only on existing checkpoint):
#   SKIP_TRAINING=1 bash scripts/run_mcts_sft_eval.sh
# =============================================================================

set -euo pipefail

if [ -f .env ]; then
    set -a; source .env; set +a
fi

# --- Configuration ---
MODEL_PATH="${MODEL_PATH:-ByteDance-Seed/UI-TARS-1.5-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/mcts_sft/v1}"
CONFIG="${CONFIG:-configs/mcts_sft.yaml}"
EVAL_CONFIG="configs/sft_eval_300tasks_clean_n1.yaml"
NUM_GPUS="${NUM_GPUS:-8}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
PYTHON_BIN="${PYTHON_BIN:-$(which python3)}"

EPOCH_MODEL="${OUTPUT_DIR}/epoch_1"
EVAL_DIR="${OUTPUT_DIR}/eval_epoch_1"
LOG_FILE="${OUTPUT_DIR}/training_log.json"

echo "============================================================"
echo "MCTS SFT: Train + Evaluate"
echo "============================================================"
echo "Model:       ${MODEL_PATH}"
echo "Output:      ${OUTPUT_DIR}"
echo "Config:      ${CONFIG}"
echo "Eval config: ${EVAL_CONFIG}"
echo "GPUs:        ${NUM_GPUS}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

# --- Helper: cleanup GPU ---
cleanup_gpu() {
    echo ""
    echo ">>> Cleaning up GPU processes..."
    sudo -E env "PATH=${PATH}" ray stop --force 2>/dev/null || true
    sudo pkill -9 -f "ray::" 2>/dev/null || true
    sudo pkill -9 -f "from vllm" 2>/dev/null || true
    sudo pkill -9 -f "raylet" 2>/dev/null || true
    sudo pkill -9 -f "gcs_server" 2>/dev/null || true
    sleep 5
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

# ==========================================================================
# Step 1: Training
# ==========================================================================
if [ "${SKIP_TRAINING}" != "1" ]; then
    if [ -d "${EPOCH_MODEL}" ]; then
        N_SHARDS=$(ls "${EPOCH_MODEL}"/model-*.safetensors 2>/dev/null | wc -l)
        if [ "${N_SHARDS}" -ge 7 ]; then
            echo ""
            echo ">>> Epoch 1 model already exists at ${EPOCH_MODEL} (${N_SHARDS} shards), skipping training"
        else
            echo ""
            echo ">>> Incomplete checkpoint found (${N_SHARDS}/7 shards), removing and retraining..."
            rm -rf "${EPOCH_MODEL}"
        fi
    fi

    if [ ! -d "${EPOCH_MODEL}" ]; then
        echo ""
        echo "============================================================"
        echo "Step 1: Training MCTS SFT (1 epoch)"
        echo "============================================================"

        sudo -E env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH:-}" \
            "WANDB_API_KEY=${WANDB_API_KEY:-}" \
            "WANDB_PROJECT=ARPO" "WANDB_ENTITY=artysicistz-university-of-pennsylvania" \
            "WANDB_RUN_NAME=mcts_sft_v1" \
        torchrun --nproc_per_node="${NUM_GPUS}" scripts/train_mcts_sft.py \
            --config "${CONFIG}" \
            --output_dir "${OUTPUT_DIR}"

        cleanup_gpu
    fi
else
    echo ""
    echo ">>> Skipping training (SKIP_TRAINING=1)"
fi

# ==========================================================================
# Step 2: Verify checkpoint
# ==========================================================================
if [ ! -d "${EPOCH_MODEL}" ]; then
    echo "[ERROR] Checkpoint not found: ${EPOCH_MODEL}"
    exit 1
fi

N_SHARDS=$(ls "${EPOCH_MODEL}"/model-*.safetensors 2>/dev/null | wc -l)
if [ "${N_SHARDS}" -lt 7 ]; then
    echo "[ERROR] Incomplete checkpoint: only ${N_SHARDS}/7 shards found"
    exit 1
fi
echo ">>> Checkpoint OK (${N_SHARDS} shards)"

# ==========================================================================
# Step 3: Evaluation
# ==========================================================================

# Skip if already evaluated
if [ -f "${EVAL_DIR}/eval_results_at_0.json" ]; then
    echo ""
    echo ">>> Evaluation already exists at ${EVAL_DIR}, skipping"
else
    cleanup_gpu

    echo ""
    echo "============================================================"
    echo "Step 2: Evaluating epoch_1 (n=1, 300 tasks)"
    echo "============================================================"
    echo ">>> Model: ${EPOCH_MODEL}"
    echo ">>> Save path: ${EVAL_DIR}"
    echo ""

    sudo -E env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH:-}" "${PYTHON_BIN}" -m verl.trainer.main \
        config="${EVAL_CONFIG}" \
        worker.actor.model.model_path="${EPOCH_MODEL}" \
        trainer.experiment_name="mcts_sft_v1_eval_epoch_1" \
        trainer.save_checkpoint_path="${EVAL_DIR}"

    cleanup_gpu
fi

# ==========================================================================
# Step 4: Parse and record results
# ==========================================================================
EVAL_RESULTS_FILE="${EVAL_DIR}/eval_results_at_0.json"
if [ -f "${EVAL_RESULTS_FILE}" ]; then
    sudo -E env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH:-}" "${PYTHON_BIN}" << PYEOF
import json, os, tempfile

eval_results_file = "${EVAL_RESULTS_FILE}"
log_file = "${LOG_FILE}"

with open(eval_results_file) as f:
    eval_results = json.load(f)

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
}

try:
    with open(log_file) as f:
        log = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    log = {"experiment": "mcts_sft_v1", "epochs": []}

for ep in log.get("epochs", []):
    if ep.get("epoch") == 1:
        ep["eval_results"] = eval_results
        ep["eval_summary"] = summary
        ep["eval_results_file"] = eval_results_file
        break
else:
    log.setdefault("standalone_evals", {})["epoch_1"] = {
        "eval_results_file": eval_results_file,
        "summary": summary,
        "per_task_results": eval_results,
    }

tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(log_file) or ".")
with os.fdopen(tmp_fd, "w") as f:
    json.dump(log, f, indent=2)
os.replace(tmp_path, log_file)

print(f"[eval] epoch_1: {n_doable}/{n_tasks} doable ({summary['doable_rate']*100:.1f}%), "
      f"overall success rate: {summary['overall_success_rate']*100:.2f}%")
PYEOF
else
    echo "[eval] WARNING: eval results file not found: ${EVAL_RESULTS_FILE}"
fi

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"
echo "Checkpoint: ${EPOCH_MODEL}"
echo "Eval results: ${EVAL_DIR}/"
echo "Training log: ${LOG_FILE}"
echo "============================================================"
