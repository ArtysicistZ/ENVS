#!/bin/bash
set -euo pipefail

# Legacy compatibility shim. The clean-room architecture performs reset and
# verification through the root-level reset runtime.

python3 /opt/osworld/reset/reset_runtime.py reset
python3 /opt/osworld/reset/reset_runtime.py verify
