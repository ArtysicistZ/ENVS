#!/bin/bash
# Run 3 rounds of MCTS collection for 18 hard tasks (336 total runs per task)
# Each round uses 112 VMs. Output dirs: rerun3a, rerun3b, rerun3c
#
# Usage: bash scripts/mcts/run_rerun3_all_rounds.sh

set -e

CONFIG="configs/mcts_collection_v2_rerun3_low18.yaml"
SCRIPT="scripts/mcts/run_mcts_collection.py"

echo "=========================================="
echo "MCTS v2 Rerun 3 — 3 rounds x 18 tasks"
echo "=========================================="

for ROUND in a b c; do
    DIR="checkpoints/mcts_trajectories_v2/rerun3${ROUND}_low18"
    echo ""
    echo "=========================================="
    echo "Round ${ROUND}: output -> ${DIR}"
    echo "Started at: $(date)"
    echo "=========================================="

    # Update output_dir in config
    sed -i "s|^output_dir:.*|output_dir: \"${DIR}\"|" "$CONFIG"

    sudo -E .venv/bin/python "$SCRIPT" --config "$CONFIG"

    echo "Round ${ROUND} finished at: $(date)"
done

echo ""
echo "=========================================="
echo "All 3 rounds complete at: $(date)"
echo "=========================================="
