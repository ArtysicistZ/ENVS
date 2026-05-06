#!/usr/bin/env bash
# Overnight orchestrator (2026-04-26). Fills 6 missing 300-task eval cells
# and auto-launches an ARPO follow-on training run.
#
# 4.6's verl is older and runs validation with zero noise (regardless of
# enable_noise=true) — confirmed via verl/trainer/ray_trainer.py:
#   `if is_val: ...  Run validation with zero noise`.
# So all NOISY evals must run on 4.4 (newer verl with noise_validate_with_noise).
# Only CLEAN evals run on 4.6.
#
# Pool partition (strict, never violated):
#   While 4.6 chain runs:  4.6 chain owns {4.6, 4.7} = 96 envs (its 2 CLEAN evals);
#                          4.4 idle (training).
#   After 4.6 chain done:  4.6 idle.
#   After training done:   4.4 chain owns {4.4, 4.7} = 96 envs (4 evals).
#                          4.4 chain WAITS for both training-done AND 4.6-chain-done
#                          before running so 4.7 is never contested.
#   ARPO:                  uses pools 4.4 + 4.6 + 4.7 (96 envs total).
#
# Schedule (dovetailed):
#   t=0     4.6 starts CLEAN chain   |  4.4 trains
#   t≈2h    4.6 chain done           |  4.4 still training
#   t≈3h    sentinel fires           |  4.4 chain begins (96 envs, 4 evals)
#   t≈7h    4.4 chain done           |  ARPO launches on 4.4
#   t≈13h   ARPO done                |
#
# Test mode: ORCH_TEST_MODE=1 ORCH_TEST_DIR=/tmp/orch_test
#   - Replaces verl invocations with sleep-3 stubs that write fake length-300 results JSON.
#   - Replaces sentinel with $TEST_DIR/training_done; replaces 4.6-done flag with
#     $TEST_DIR/orch_46_chain_done.
#   - Replaces ARPO command with a print-only echo.

set -uo pipefail
cd /mnt/kevinzyz/arpo_local

LOG_DIR=/mnt/kevinzyz/arpo_local/logs
mkdir -p "$LOG_DIR"

TEST_MODE="${ORCH_TEST_MODE:-0}"
TEST_DIR="${ORCH_TEST_DIR:-/tmp/orch_test}"

if [ "$TEST_MODE" = "1" ]; then
  mkdir -p "$TEST_DIR"
  FLAG_46_DONE="$TEST_DIR/orch_46_chain_done"
  SENTINEL="$TEST_DIR/training_done"
else
  FLAG_46_DONE=/tmp/orch_46_chain_done_2026_04_26
  SENTINEL=/mnt/kevinzyz/arpo_local/checkpoints/mcts_sft_v2_clean_aligned/beta05_2e-6/epoch_1/model.safetensors.index.json
fi
# Clear stale flag from a prior run.
[ "$TEST_MODE" = "0" ] && rm -f "$FLAG_46_DONE"

ENV_VERL=(
  RAY_TMPDIR=/mnt/ray_tmp
  PYTHONUNBUFFERED=1
  HYDRA_FULL_ERROR=1
  MKL_SERVICE_FORCE_INTEL=1
  MKL_THREADING_LAYER=GNU
  RAY_memory_usage_threshold=0.95
  RAY_memory_monitor_refresh_ms=0
)

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

fake_results() {
  local path="$1"
  mkdir -p "$(dirname "$path")"
  python3 -c "import json,sys; json.dump([{'i':i,'success':0} for i in range(300)], open(sys.argv[1],'w'))" "$path"
}

# Idempotent eval runner — local 4.4
run_eval_44() {
  local cfg="$1" results="$2" label="$3" log="$4"
  if [ "$TEST_MODE" = "1" ]; then
    local tag; tag=$(echo "$label" | tr ' /' '__')
    results="$TEST_DIR/44_${tag}.results.json"
    log="$TEST_DIR/44_${tag}.log"
  fi
  if [ -f "$results" ] && [ "$(jq 'length' "$results" 2>/dev/null)" = "300" ]; then
    echo "[44 $(stamp)] SKIP $label (already 300)"
    return 0
  fi
  echo "[44 $(stamp)] LAUNCH $label"
  if [ "$TEST_MODE" = "1" ]; then
    sleep 3
    fake_results "$results"
    echo "[44 $(stamp)] TEST FINISHED $label"
    return 0
  fi
  sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main config="$cfg" 2>&1 | tee "$log"
}

# Idempotent eval runner — remote 4.6 over ssh
run_eval_46() {
  local cfg="$1" results="$2" label="$3" log="$4"
  if [ "$TEST_MODE" = "1" ]; then
    local tag; tag=$(echo "$label" | tr ' /' '__')
    local trp="$TEST_DIR/46_${tag}.results.json"
    if [ -f "$trp" ] && [ "$(jq 'length' "$trp" 2>/dev/null)" = "300" ]; then
      echo "[46 $(stamp)] SKIP $label (already 300)"
      return 0
    fi
    echo "[46 $(stamp)] LAUNCH $label (test)"
    sleep 3
    fake_results "$trp"
    echo "[46 $(stamp)] TEST FINISHED $label"
    return 0
  fi
  local skip
  skip=$(ssh 10.100.4.6 "[ -f /mnt/kevinzyz/arpo_main/$results ] && [ \"\$(jq 'length' /mnt/kevinzyz/arpo_main/$results 2>/dev/null)\" = '300' ] && echo SKIP || echo RUN")
  if [ "$skip" = "SKIP" ]; then
    echo "[46 $(stamp)] SKIP $label (already 300)"
    return 0
  fi
  echo "[46 $(stamp)] LAUNCH $label"
  ssh 10.100.4.6 "cd /mnt/kevinzyz/arpo_main && \
    mkdir -p logs && \
    sudo -E ${ENV_VERL[@]} .venv/bin/python -m verl.trainer.main config=$cfg 2>&1 | tee $log"
}

echo "[orch $(stamp)] starting (TEST_MODE=$TEST_MODE)"

#####################################################################
# 4.6 SUBSHELL — 2 CLEAN evals (v3_aligned, v3_noisy_combined)
# Uses pools 4.6 + 4.7 = 96 envs.
# At end, touches FLAG_46_DONE so the 4.4 chain can proceed.
#####################################################################
(
  echo "[46 $(stamp)] starting 4.6 CLEAN chain"
  run_eval_46 \
    "configs/eval_clean_300tasks_mcts_sft_v3_noisy_aligned_n8_t1.yaml" \
    "checkpoints/clean_eval/mcts_sft_v3_noisy_aligned_n8_t1/eval_results_at_0.json" \
    "v3_aligned CLEAN" \
    "logs/eval_clean_v3_aligned.log"

  run_eval_46 \
    "configs/eval_clean_300tasks_mcts_sft_v3_noisy_combined_n8_t1.yaml" \
    "checkpoints/clean_eval/mcts_sft_v3_noisy_combined_n8_t1/eval_results_at_0.json" \
    "v3_noisy_combined CLEAN" \
    "logs/eval_clean_v3_combined.log"

  echo "[46 $(stamp)] 4.6 chain done; touching $FLAG_46_DONE"
  touch "$FLAG_46_DONE"
) > "$LOG_DIR/chain_46.log" 2>&1 &
PID_46=$!

#####################################################################
# 4.4 SUBSHELL — 4 evals after BOTH training-done AND 4.6-chain-done
# Uses pools 4.4 + 4.7 = 96 envs (4.7 free because 4.6 chain finished).
#####################################################################
(
  echo "[44 $(stamp)] waiting for v2_clean_aligned training (sentinel: $SENTINEL)"
  while [ ! -f "$SENTINEL" ]; do sleep 5; done
  echo "[44 $(stamp)] training done; waiting for 4.6 chain to finish ($FLAG_46_DONE)"
  while [ ! -f "$FLAG_46_DONE" ]; do sleep 5; done
  echo "[44 $(stamp)] both prerequisites met; starting 4.4 chain"

  run_eval_44 \
    "configs/eval_clean_300tasks_mcts_sft_v2_clean_aligned_n8_t1.yaml" \
    "checkpoints/clean_eval/mcts_sft_v2_clean_aligned_n8_t1/eval_results_at_0.json" \
    "v2_clean_aligned CLEAN" \
    "$LOG_DIR/eval_clean_v2_clean_aligned.log"

  run_eval_44 \
    "configs/eval_noisy_300tasks_mcts_sft_v2_clean_aligned_n8_t1.yaml" \
    "checkpoints/noisy_eval/mcts_sft_v2_clean_aligned_n8_t1/eval_results_at_0.json" \
    "v2_clean_aligned NOISY" \
    "$LOG_DIR/eval_noisy_v2_clean_aligned.log"

  run_eval_44 \
    "configs/eval_noisy_300tasks_mcts_sft_v3_noisy_aligned_n8_t1.yaml" \
    "checkpoints/noisy_eval/mcts_sft_v3_noisy_aligned_n8_t1/eval_results_at_0.json" \
    "v3_aligned NOISY" \
    "$LOG_DIR/eval_noisy_v3_aligned.log"

  run_eval_44 \
    "configs/eval_noisy_300tasks_mcts_sft_v2.1_no_mask_n8_t1.yaml" \
    "checkpoints/noisy_eval/mcts_sft_v2.1_no_mask_n8_t1/eval_results_at_0.json" \
    "v2.1_no_mask NOISY" \
    "$LOG_DIR/eval_noisy_v21_no_mask.log"

  echo "[44 $(stamp)] 4.4 chain done"
) > "$LOG_DIR/chain_44.log" 2>&1 &
PID_44=$!

#####################################################################
# Wait for both eval chains; immediately launch ARPO follow-on
#####################################################################
wait "$PID_46" "$PID_44"
echo "[orch $(stamp)] both eval chains finished. Launching ARPO follow-on."

if [ "$TEST_MODE" = "1" ]; then
  echo "[orch $(stamp)] TEST MODE: would launch ARPO follow-on (skipped)"
else
  cd /mnt/kevinzyz/arpo_local
  sudo -E "${ENV_VERL[@]}" .venv/bin/python -m verl.trainer.main \
    config=configs/arpo_86tasks_clean_from_v2_clean_aligned.yaml \
    2>&1 | tee "$LOG_DIR/arpo_86tasks_clean_from_v2_clean_aligned.log"
fi

echo "[orch $(stamp)] orchestrator exiting"
