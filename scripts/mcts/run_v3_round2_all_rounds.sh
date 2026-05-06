#!/bin/bash
# Run 2 rounds of v3-noisy MCTS collection on the 32 low-success tasks.
# Each round: 96 VMs on 2 clusters, 1 fire per rollout (SR-forced to 0.5),
# MCTS-SFT v2.1 as rollout policy. Output: round2a and round2b.
# 32 tasks × 2 rounds × 96 VMs = 6144 rollouts total (192 per task).
#
# Usage: bash scripts/mcts/run_v3_round2_all_rounds.sh

set -e
cd /mnt/kevinzyz/arpo_local
source .venv/bin/activate

CONFIG="configs/mcts_collection_v3_round2.yaml"
SCRIPT="scripts/mcts/run_mcts_collection.py"

echo "=========================================="
echo "v3-noisy round-2 — 2 rounds x 32 tasks (1 fire each)"
echo "=========================================="

for ROUND in a b; do
    DIR="checkpoints/mcts_trajectories_v3_noisy_round2${ROUND}"
    echo ""
    echo "=========================================="
    echo "Round ${ROUND}: output -> ${DIR}"
    echo "Started at: $(date)"
    echo "=========================================="

    sed -i "s|^output_dir:.*|output_dir: \"${DIR}\"|" "$CONFIG"
    python "$SCRIPT" --config "$CONFIG"

    echo "Round ${ROUND} finished at: $(date)"
done

echo ""
echo "=========================================="
echo "All 2 rounds complete at: $(date)"
echo "=========================================="
