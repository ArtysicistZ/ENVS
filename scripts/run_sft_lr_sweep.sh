#!/bin/bash
# =============================================================================
# SFT LR Sweep: 1 epoch each, 300 tasks n=1 eval, batch=32
#
# Sweeps learning rates to find the optimal value.
# Known: 5e-6 works, 1e-5 overfits.
# Sweep: 3e-6, 5e-6, 7e-6
#
# Each LR: train 1 epoch → eval → delete model to free SSD → next LR
#
# Usage:
#   bash scripts/run_sft_lr_sweep.sh
# =============================================================================

set -euo pipefail

# Load .env if present (for WANDB_API_KEY etc.)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# --- Configuration ---
MODEL_PATH="${MODEL_PATH:-ByteDance-Seed/UI-TARS-1.5-7B}"
DATA_PATH="${DATA_PATH:-checkpoints/arpo-inference/sft_trajectories_selected.jsonl}"
SWEEP_DIR="${SWEEP_DIR:-checkpoints/sft_lr_sweep}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-12000}"
LIMIT_IMAGES="${LIMIT_IMAGES:-3}"

EVAL_CONFIG="configs/sft_eval_300tasks_clean_n1.yaml"
PYTHON_BIN="${PYTHON_BIN:-$(which python3)}"

# Learning rates to sweep
LR_VALUES=("5e-6" "3e-6" "7e-6")

echo "============================================================"
echo "SFT Learning Rate Sweep"
echo "============================================================"
echo "Model:         ${MODEL_PATH}"
echo "Data:          ${DATA_PATH}"
echo "Sweep dir:     ${SWEEP_DIR}"
echo "LR values:     ${LR_VALUES[*]}"
echo "Batch size:    ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS} = $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "Eval config:   ${EVAL_CONFIG}"
echo "============================================================"

mkdir -p "${SWEEP_DIR}"

# --- Helper: kill lingering Ray/vLLM processes and free GPU memory ---
cleanup_gpu() {
    echo ""
    echo ">>> Cleaning up GPU processes..."

    sudo -E env "PATH=${PATH}" ray stop --force 2>/dev/null || true
    sleep 2
    sudo pkill -9 -f "ray::" 2>/dev/null || true
    sudo pkill -9 -f "from vllm" 2>/dev/null || true
    sudo pkill -9 -f "raylet" 2>/dev/null || true
    sudo pkill -9 -f "gcs_server" 2>/dev/null || true

    # Wait for disk-sleep processes to actually die (they can't be killed instantly)
    for i in $(seq 1 12); do
        local stuck
        stuck=$(ps aux 2>/dev/null | grep -E "raylet|gcs_server" | grep -v grep | wc -l) || stuck=0
        if [ "${stuck}" -eq 0 ]; then
            break
        fi
        echo ">>> Waiting for ${stuck} lingering Ray processes to die (attempt ${i}/12)..."
        sleep 5
    done

    # Clear Ray temp state to prevent stale cluster issues
    sudo rm -rf /tmp/ray/session_latest 2>/dev/null || true

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

# --- Sweep loop ---
for LR in "${LR_VALUES[@]}"; do
    OUTPUT_DIR="${SWEEP_DIR}/lr_${LR}"
    LOG_FILE="${OUTPUT_DIR}/training_log.json"
    EPOCH_MODEL="${OUTPUT_DIR}/epoch_1"
    EVAL_DIR="${OUTPUT_DIR}/eval_epoch_1"

    echo ""
    echo "============================================================"
    echo "LR = ${LR}"
    echo "============================================================"

    mkdir -p "${OUTPUT_DIR}"

    # --- Skip if already evaluated ---
    if [ -f "${EVAL_DIR}/eval_results_at_0.json" ]; then
        echo ">>> LR=${LR} already trained + evaluated, skipping"
        continue
    fi

    # --- Train (skip if model already exists) ---
    if [ -d "${EPOCH_MODEL}" ]; then
        echo ">>> LR=${LR} model exists, skipping training"
    else
        echo ">>> Training with LR=${LR}..."

        sudo -E env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH:-}" \
            "WANDB_API_KEY=${WANDB_API_KEY:-}" \
            "WANDB_PROJECT=ARPO" "WANDB_ENTITY=artysicistz-university-of-pennsylvania" \
            "WANDB_RUN_NAME=sft_lr_sweep_${LR}" \
        torchrun --nproc_per_node="${NUM_GPUS}" scripts/train_sft.py \
            --model_path "${MODEL_PATH}" \
            --data_path "${DATA_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --num_epochs 1 \
            --per_device_batch_size "${BATCH_SIZE}" \
            --gradient_accumulation_steps "${GRAD_ACCUM}" \
            --learning_rate "${LR}" \
            --lr_scheduler_type "${LR_SCHEDULER}" \
            --warmup_ratio "${WARMUP_RATIO}" \
            --max_grad_norm "${MAX_GRAD_NORM}" \
            --max_length "${MAX_LENGTH}" \
            --limit_images "${LIMIT_IMAGES}" \
            --logging_steps 1 \
            --fsdp_offload

        cleanup_gpu
    fi

    # --- Evaluate ---
    if [ -d "${EPOCH_MODEL}" ]; then
        echo ">>> Evaluating LR=${LR}..."

        sudo -E env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH:-}" "${PYTHON_BIN}" -m verl.trainer.main \
            config="${EVAL_CONFIG}" \
            worker.actor.model.model_path="${EPOCH_MODEL}" \
            trainer.experiment_name="sft_lr_sweep_eval_${LR}" \
            trainer.save_checkpoint_path="${EVAL_DIR}"

        cleanup_gpu

        # Record eval summary in training log
        EVAL_RESULTS_FILE="${EVAL_DIR}/eval_results_at_0.json"
        if [ -f "${EVAL_RESULTS_FILE}" ]; then
            sudo -E env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH:-}" "${PYTHON_BIN}" << PYEOF
import json, os, tempfile

with open("${EVAL_RESULTS_FILE}") as f:
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
    with open("${LOG_FILE}") as f:
        log = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    log = {"experiment": "sft_lr_sweep", "epochs": []}

for ep in log.get("epochs", []):
    if ep.get("epoch") == 1:
        ep["eval_summary"] = summary
        break

tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname("${LOG_FILE}") or ".")
with os.fdopen(tmp_fd, "w") as f:
    json.dump(log, f, indent=2)
os.replace(tmp_path, "${LOG_FILE}")

print(f"[LR={os.path.basename('${OUTPUT_DIR}')}] {n_doable}/{n_tasks} doable ({summary['doable_rate']*100:.1f}%), "
      f"SR={summary['overall_success_rate']*100:.2f}%")
PYEOF
        fi

        # Keep model — user will delete manually if needed
        echo ">>> Keeping model at: ${EPOCH_MODEL}"
    else
        echo "[ERROR] Model not found at ${EPOCH_MODEL}"
    fi
done

# --- Final summary ---
echo ""
echo "============================================================"
echo "LR Sweep Complete!"
echo "============================================================"

sudo -E env "PATH=${PATH}" "PYTHONPATH=${PYTHONPATH:-}" "SWEEP_DIR=${SWEEP_DIR}" "${PYTHON_BIN}" << 'PYEOF'
import json, os, glob

sweep_dir = os.environ.get("SWEEP_DIR", "checkpoints/sft_lr_sweep")
results = []

# Also include v2.1 (LR=5e-6) for comparison
for eval_path in sorted(glob.glob(f"{sweep_dir}/lr_*/eval_epoch_1/eval_results_at_0.json")):
    lr_dir = os.path.basename(os.path.dirname(os.path.dirname(eval_path)))
    lr = lr_dir.replace("lr_", "")
    with open(eval_path) as f:
        data = json.load(f)
    n_tasks = len(data)
    n_doable = sum(1 for v in data.values() if v.get("n_success", 0) > 0)
    total_succ = sum(v.get("n_success", 0) for v in data.values())
    total_att = sum(v.get("n_attempts", 0) for v in data.values())
    sr = total_succ / total_att * 100 if total_att > 0 else 0
    results.append((lr, n_doable, n_tasks, sr))

# Check v2.1 results
v21_path = "checkpoints/sft_v2.1/eval_epoch_1/eval_results_at_0.json"
if os.path.exists(v21_path):
    with open(v21_path) as f:
        data = json.load(f)
    n_tasks = len(data)
    n_doable = sum(1 for v in data.values() if v.get("n_success", 0) > 0)
    total_succ = sum(v.get("n_success", 0) for v in data.values())
    total_att = sum(v.get("n_attempts", 0) for v in data.values())
    sr = total_succ / total_att * 100 if total_att > 0 else 0
    results.append(("5e-6 (v2.1)", n_doable, n_tasks, sr))

# Check v1 results (LR=1e-5)
v1_path = "checkpoints/sft_v1/eval_epoch1_300clean_n8/eval_results_at_0.json"
if os.path.exists(v1_path):
    with open(v1_path) as f:
        data = json.load(f)
    n_tasks = len(data)
    n_doable = sum(1 for v in data.values() if v.get("n_success", 0) > 0)
    total_succ = sum(v.get("n_success", 0) for v in data.values())
    total_att = sum(v.get("n_attempts", 0) for v in data.values())
    sr = total_succ / total_att * 100 if total_att > 0 else 0
    results.append(("1e-5 (v1, n=8)", n_doable, n_tasks, sr))

print(f"{'LR':>16s}  {'Doable':>10s}  {'SR':>8s}")
print("-" * 40)
for lr, doable, total, sr in sorted(results, key=lambda x: x[0]):
    print(f"{lr:>16s}  {doable:>3d}/{total:<3d}      {sr:>5.2f}%")
PYEOF
