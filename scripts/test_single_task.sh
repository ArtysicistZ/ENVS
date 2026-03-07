#!/bin/bash
# Test OSWorld + UI-TARS-2B with SINGLE TASK ONLY

echo "=============================================="
echo "Single Task Test - UI-TARS-2B"
echo "=============================================="
echo ""

# Create single task JSON
cat > single_task_test.json << 'EOF'
{
  "chrome": ["bb5e4c0d-f964-439c-97b6-bdb9747de3f4"]
}
EOF

echo "Testing with SINGLE task only (not 369 tasks!)"
echo "Task: Chrome cookie deletion"
echo "Max steps: 3"
echo ""

cd OSWorld

python run_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 3 \
    --model ui-tars-2b \
    --temperature 0.7 \
    --max_tokens 128 \
    --max_trajectory_length 5 \
    --test_all_meta_path ../single_task_test.json \
    --result_dir ../results_single/ \
    2>&1 | tee ../logs/test_single_task.log

cd ..

echo ""
echo "Single task test complete!"
echo ""
echo "Check results_single/ for trajectory and results"
