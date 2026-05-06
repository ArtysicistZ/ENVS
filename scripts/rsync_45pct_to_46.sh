#!/usr/bin/env bash
# Wait for 45% prep on 4.4 to finish, then rsync the dataset to 4.6.
# Pre-creates the destination dir on 4.6 (sudo, owned by jiajun) since
# /mnt/kevinzyz/arpo_main/checkpoints/ is owned by kevinzyz.
set -uo pipefail

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

PREP=/mnt/kevinzyz/arpo_local/checkpoints/mcts_trajectories_v2_subsample_45pct
echo "[rsync $(stamp)] waiting for $PREP/task_index.json"
until [ -f "$PREP/task_index.json" ]; do sleep 30; done
# Also wait for trees dir to be 'fully written' — task_index.json exists from
# early in the script, so verify trees count too
echo "[rsync $(stamp)] task_index ready; verifying tree write completion"
expected=$(jq 'to_entries | map(.value | length) | add' "$PREP/task_index.json")
echo "[rsync $(stamp)] expecting $expected tree files"
until [ "$(ls -1 "$PREP/trees/" 2>/dev/null | wc -l)" -ge "$expected" ]; do
  sleep 30
  written=$(ls -1 "$PREP/trees/" 2>/dev/null | wc -l)
  echo "[rsync $(stamp)] $written / $expected trees written"
done
echo "[rsync $(stamp)] all trees written"

# Pre-create destination on 4.6 owned by jiajun
ssh 10.100.4.6 "sudo mkdir -p /mnt/kevinzyz/arpo_main/checkpoints/mcts_trajectories_v2_subsample_45pct && sudo chown -R jiajun:jiajun /mnt/kevinzyz/arpo_main/checkpoints/mcts_trajectories_v2_subsample_45pct"

echo "[rsync $(stamp)] starting rsync to 4.6"
rsync -aP /mnt/kevinzyz/arpo_local/checkpoints/mcts_trajectories_v2_subsample_45pct/ \
          10.100.4.6:/mnt/kevinzyz/arpo_main/checkpoints/mcts_trajectories_v2_subsample_45pct/ \
          2>&1 | tail -3

echo "[rsync $(stamp)] done"
ssh 10.100.4.6 "ls /mnt/kevinzyz/arpo_main/checkpoints/mcts_trajectories_v2_subsample_45pct/ | head; echo 'tree count:'; ls /mnt/kevinzyz/arpo_main/checkpoints/mcts_trajectories_v2_subsample_45pct/trees/ | wc -l"
