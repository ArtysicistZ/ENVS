#!/bin/bash
# ARPO Training Script for UI-TARS-2B on Mac CPU
# Adapted from examples/osworld_subset32.sh

set -x

echo "=============================================="
echo "ARPO Training - UI-TARS-2B on Mac CPU"
echo "=============================================="

# Check if UI-TARS-2B server is running
if ! curl -s http://localhost:9000/v1/models > /dev/null 2>&1; then
    echo ""
    echo "❌ ERROR: UI-TARS-2B server not running!"
    echo ""
    echo "Please start the server in another terminal:"
    echo "  conda activate arpo"
    echo "  cd /Users/hanszhu/Desktop/ARPO_replicate"
    echo "  python uitars_2b_server.py"
    echo ""
    exit 1
fi

echo "✓ UI-TARS-2B server is running"
echo ""

# Check if Ray is running
if ! ray status > /dev/null 2>&1; then
    echo "Starting Ray cluster..."
    ray start --head --port=2468 --resources='{"docker:127.0.0.1": 128}'
    sleep 2
fi

echo "✓ Ray cluster is running"
echo ""

# Configuration
MODEL_PATH="ByteDance-Seed/UI-TARS-2B-SFT"
SYSTEM_PROMPT="You are a helpful GUI agent assistant."

# Mac CPU settings (ultra-light)
NUM_GPUS=0  # CPU only
NUM_ENVS=1  # Single VMware VM
ROLLOUT_N=1  # Single rollout

# CPU-specific environment
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Create directories
mkdir -p checkpoints_2b/ results_2b/ logs/

# Generate 8-task subset if not exists
if [ ! -f "evaluation_examples/train_subset8.json" ]; then
    python3 << 'EOF'
import json
with open('evaluation_examples/test_all.json', 'r') as f:
    all_tasks = json.load(f)

# Get first 8 tasks
subset = {}
count = 0
for domain, tasks in all_tasks.items():
    subset[domain] = tasks[:min(2, len(tasks))]  # 2 per domain
    count += len(subset[domain])
    if count >= 8:
        break

with open('evaluation_examples/train_subset8.json', 'w') as f:
    json.dump(subset, f, indent=2)

print(f"Created train_subset8.json with {sum(len(v) for v in subset.values())} tasks")
EOF
fi

echo ""
echo "Starting ARPO training..."
echo "Configuration:"
echo "  Model: UI-TARS-2B"
echo "  Tasks: 8"
echo "  Environments: 1 (VMware VM)"
echo "  Rollouts: 1 per task"
echo "  Epochs: 5"
echo "  Device: CPU"
echo ""

# ARPO Training with VERL
python3 -m verl.trainer.main \
    config=config_uitars_2b_mac.yaml \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.train_files=evaluation_examples/train_subset8.json \
    data.val_files=evaluation_examples/train_subset8.json \
    data.max_prompt_length=32768 \
    data.max_response_length=4096 \
    data.rollout_batch_size=1 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.optim.strategy=adamw \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.3 \
    worker.actor.global_batch_size=2 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.temperature=0.7 \
    worker.rollout.n=$ROLLOUT_N \
    worker.rollout.limit_images=10 \
    worker.rollout.use_external_server=true \
    worker.rollout.server_url="http://localhost:9000/v1" \
    algorithm.disable_kl=true \
    algorithm.kl_coef=0 \
    algorithm.enable_replay=true \
    env.num_envs=$NUM_ENVS \
    env.max_steps=10 \
    env.provider=vmware \
    trainer.experiment_name=uitars_2b_cpu_mac \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.save_limit=3 \
    trainer.val_before_train=true \
    trainer.val_freq=2 \
    trainer.total_episodes=5 \
    trainer.use_cpu=true

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Results saved to: results_2b/"
echo "Checkpoints saved to: checkpoints_2b/"
