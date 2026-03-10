#!/usr/bin/env bash
# Remove previous Ray session logs so the next run uses a fresh session and
# session_latest points to the new run. Run this before starting training if you
# want to avoid mixing old and new logs (e.g. to debug "Owner's node has crashed").
set -e
RAY_TMP="${RAY_TMP:-/tmp/ray}"
if [[ -d "$RAY_TMP" ]]; then
  for d in "$RAY_TMP"/session_*; do
    if [[ -d "$d" ]]; then
      echo "Removing Ray session: $d"
      rm -rf "$d"
    fi
  done
  echo "Previous Ray session logs removed. Next ray.init() will create a new session."
else
  echo "No Ray tmp dir found at $RAY_TMP (nothing to clear)."
fi
