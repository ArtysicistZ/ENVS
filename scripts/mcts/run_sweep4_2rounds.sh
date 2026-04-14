#!/bin/bash
# MCTS v2 Sweep 4 — 2 rounds x 86 trainable tasks, 72 VMs
#
# Runs two fresh MCTS collection rounds on the 86 trainable tasks
# using the 2 reachable VM servers (10.100.4.4 + 10.100.4.8 = 72 VMs).
#
# Usage:
#   bash scripts/mcts/run_sweep4_2rounds.sh

set -euo pipefail

cd /mnt/kevinzyz/arpo_main

VENV=".venv/bin"
PYTHON="${VENV}/python"
CONFIG="configs/mcts_collection_v2_sweep4.yaml"
SCRIPT="scripts/mcts/run_mcts_collection.py"

echo "=========================================="
echo "MCTS v2 Sweep 4 — 2 rounds x 86 tasks"
echo "VMs: 72 (32 + 40 across 2 servers)"
echo "Started at: $(date)"
echo "=========================================="

for ROUND in a b; do
    DIR="checkpoints/mcts_trajectories_v2/sweep4${ROUND}"
    echo ""
    echo "=========================================="
    echo "Round ${ROUND}: output -> ${DIR}"
    echo "Started at: $(date)"
    echo "=========================================="

    # Update output_dir in config
    sed -i "s|^output_dir:.*|output_dir: \"${DIR}\"|" "$CONFIG"

    sudo -E ${PYTHON} ${SCRIPT} --config "${CONFIG}"

    echo "Round ${ROUND} finished at: $(date)"
done

echo ""
echo "=========================================="
echo "Both rounds complete at: $(date)"
echo "=========================================="

# Print summary
echo ""
echo "Summary:"
for ROUND in a b; do
    DIR="checkpoints/mcts_trajectories_v2/sweep4${ROUND}"
    RESULTS="${DIR}/collection_results.json"
    if [ -f "${RESULTS}" ]; then
        sudo -E ${PYTHON} -c "
import json
with open('${RESULTS}') as f:
    data = json.load(f)
results = data.get('results', [])
n_tasks = len(results)
n_success = sum(1 for r in results if r.get('successful_trajectories', 0) > 0)
total_trajs = sum(r.get('successful_trajectories', 0) for r in results)
print(f'  Round ${ROUND}: {n_success}/{n_tasks} tasks with success, {total_trajs} total successful trajectories')
"
    else
        echo "  Round ${ROUND}: no results file found"
    fi
done
