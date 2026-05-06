#!/usr/bin/env bash
# 4.6 chain for the data-quantity sweep (2026-04-28 overnight).
#
# Phase 0: wait for lingbot-3, restart 4.6 pool containers.
# Phase 1: wait for rsync of 45% dataset 4.4 → 4.6 to complete.
# Phase 2: train 45% MCTS-SFT (no mask) on 4.6.
# Phase 3: eval 45% clean 300 with both n=8 t=1 and n=1 t=0.
# Phase 4: greedy backfills for 4.6-resident checkpoints (v3_noisy_combined).
#
# Pool: 4.6 only (64 envs) throughout.
# Disjoint from 4.4 chain ({4.4, 4.7}).
#
# Triggered remotely from 4.4 by `tmux new-session -d -s sweep_46 ssh 10.100.4.6 ...`
# OR manually run on 4.6:
#   tmux new-session -d -s sweep_46 \
#     "bash /mnt/kevinzyz/arpo_main/scripts/orchestrate_46_sweep_2026_04_28.sh \
#      2>&1 | tee /mnt/kevinzyz/arpo_main/logs/orchestrate_46_sweep.log"

set -uo pipefail
cd /mnt/kevinzyz/arpo_main

LOG_DIR=/mnt/kevinzyz/arpo_main/logs
mkdir -p "$LOG_DIR"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Common env for verl
ENV_VERL=(
  RAY_TMPDIR=/mnt/ray_tmp
  PYTHONUNBUFFERED=1
  HYDRA_FULL_ERROR=1
  MKL_SERVICE_FORCE_INTEL=1
  MKL_THREADING_LAYER=GNU
  RAY_memory_usage_threshold=0.95
  RAY_memory_monitor_refresh_ms=0
)

# ---- Phase 0: wait for lingbot-3 + restart 4.6 pool ----
echo "[46 $(stamp)] waiting for lingbot-3 to finish (no jiajun python convert_to_lerobot)"
while pgrep -fu jiajun "convert_to_lerobot" > /dev/null; do sleep 60; done
echo "[46 $(stamp)] lingbot-3 done; restarting 4.6 pool containers"

# Restart pool containers (we own pool 4.6 since no GPU work running on 4.6 right now)
tmux kill-session -t env_server 2>/dev/null
sudo pkill -9 -f remote_env_server.py 2>/dev/null
sleep 3
sudo docker ps --filter ancestor=osworld:latest -q | xargs -r sudo docker rm -f 2>&1 | tail -3
echo "[46 $(stamp)] containers force-removed; relaunching env_server with POOL_SIZE=64"

set -a; [ -f /mnt/kevinzyz/arpo_main/.env ] && source /mnt/kevinzyz/arpo_main/.env; set +a
tmux new-session -d -s env_server "cd /mnt/kevinzyz/arpo_main && \
  sudo -E env PATH=/mnt/kevinzyz/arpo_main/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  VIRTUAL_ENV=/mnt/kevinzyz/arpo_main/.venv \
  PROVIDER=docker OSWORLD_POOL_SIZE=64 OSWORLD_DOCKER_IMAGE=osworld:latest \
  bash -c '.venv/bin/python scripts/servers/remote_env_server.py 2>&1 | tee ~/osworld_pool_logs/osworld_server.log'"

# Wait for pool 64/64 ready
echo "[46 $(stamp)] waiting for 64/64 pool ready"
until [ "$(curl -s --max-time 3 http://127.0.0.1:15001/health 2>/dev/null | jq -r '.pool_size')" = "64" ] && \
      tail -10 ~/osworld_pool_logs/osworld_server.log 2>/dev/null | grep -q "health_monitor.*64/64"; do
  sleep 30
done
echo "[46 $(stamp)] pool 64/64 ready"

# ---- Phase 1: wait for 45% dataset on 4.6 ----
PREP45=/mnt/kevinzyz/arpo_main/checkpoints/mcts_trajectories_v2_subsample_45pct/task_index.json
echo "[46 $(stamp)] waiting for 45% dataset rsync ($PREP45)"
while [ ! -f "$PREP45" ]; do sleep 30; done
echo "[46 $(stamp)] 45% dataset present"

# ---- Phase 2: train 45% no-mask SFT ----
TRAIN45_OUT=/mnt/kevinzyz/arpo_main/checkpoints/mcts_sft_v2_subsample_45pct/beta05_4.190e-6/epoch_1
if [ -f "$TRAIN45_OUT/model.safetensors.index.json" ]; then
  echo "[46 $(stamp)] 45% SFT already trained, skipping"
else
  echo "[46 $(stamp)] LAUNCH 45% SFT training"
  sudo -E env \
    HF_HOME=/home/kevinzyz/.cache/huggingface \
    WANDB_PROJECT=ARPO \
    WANDB_ENTITY=artysicistz-university-of-pennsylvania \
    WANDB_RUN_NAME=mcts_sft_v2_subsample_45pct_4.190e-6 \
    WANDB_API_KEY="$WANDB_API_KEY" \
    /mnt/kevinzyz/arpo_main/.venv/bin/torchrun --nproc_per_node=8 \
    scripts/train_mcts_sft_v2.py \
    --config configs/mcts_sft_v2_subsample_45pct.yaml \
    --run_name mcts_sft_v2_subsample_45pct_4.190e-6 \
    --model_path /home/kevinzyz/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/683d002dd99d8f95104d31e70391a39348857f4e \
    2>&1 | tee "$LOG_DIR/sft_45pct_train.log"
fi

# ---- Phase 3: eval 45% n=8 t=1 + n=1 t=0 ----
RES45_N8=/mnt/kevinzyz/arpo_main/checkpoints/clean_eval/v2_subsample_45pct_n8_t1/eval_results_at_0.json
if [ -f "$RES45_N8" ] && [ "$(jq 'length' "$RES45_N8" 2>/dev/null)" = "300" ]; then
  echo "[46 $(stamp)] SKIP eval 45% n8t1"
else
  echo "[46 $(stamp)] LAUNCH eval 45% n=8 t=1"
  sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main \
    config=configs/eval_clean_300tasks_v2_subsample_45pct_n8_t1.yaml \
    2>&1 | tee "$LOG_DIR/eval_45pct_n8_t1.log"
fi

RES45_N1=/mnt/kevinzyz/arpo_main/checkpoints/clean_eval/v2_subsample_45pct_n1_t0/eval_results_at_0.json
if [ -f "$RES45_N1" ] && [ "$(jq 'length' "$RES45_N1" 2>/dev/null)" = "300" ]; then
  echo "[46 $(stamp)] SKIP eval 45% n1t0"
else
  echo "[46 $(stamp)] LAUNCH eval 45% n=1 t=0"
  sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main \
    config=configs/eval_clean_300tasks_v2_subsample_45pct_n1_t0.yaml \
    2>&1 | tee "$LOG_DIR/eval_45pct_n1_t0.log"
fi

# ---- Phase 4: greedy backfills (4.6-resident only) ----
BACKFILLS=(
  "v3_noisy_combined|eval_clean_300tasks_v3_combined_n1_t0.yaml|checkpoints/clean_eval/v3_combined_n1_t0/eval_results_at_0.json"
)

for entry in "${BACKFILLS[@]}"; do
  IFS='|' read -r name cfg results <<< "$entry"
  full_results="/mnt/kevinzyz/arpo_main/$results"
  if [ -f "$full_results" ] && [ "$(jq 'length' "$full_results" 2>/dev/null)" = "300" ]; then
    echo "[46 $(stamp)] SKIP greedy $name"
    continue
  fi
  echo "[46 $(stamp)] LAUNCH greedy $name"
  sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main \
    config=configs/$cfg \
    2>&1 | tee "$LOG_DIR/greedy_${name}.log"
done

echo "[46 $(stamp)] chain done"
