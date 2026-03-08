#!/bin/bash
set -euo pipefail

# Legacy compatibility shim. The clean-room architecture no longer snapshots a
# mutable home at runtime; it validates the image-baked baseline metadata.

python3 /opt/osworld/reset/reset_runtime.py prepare-baseline
