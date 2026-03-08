#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
CURRENT_PWD="$(pwd -P)"
DESKTOP_HOME="${OSWORLD_RESET_HOME:-${HOME}}"

if [[ "${CURRENT_PWD}" == "${DESKTOP_HOME}" || "${CURRENT_PWD}" == "${DESKTOP_HOME}/"* ]]; then
  cat >&2 <<EOF
Do not start the reset stack from inside ${DESKTOP_HOME}.
Current working directory: ${CURRENT_PWD}

The overlay service mounts on top of ${DESKTOP_HOME}, so running this command
from inside that tree can make osworld-home-overlay.service fail with EBUSY.

Use:
  cd /tmp
  bash ${REPO_ROOT}/OSWorld/desktop_env/providers/aws/scripts/start_local_reset_stack.sh
EOF
  exit 1
fi

cd "${REPO_ROOT}"
echo "Starting OSWorld reset stack from ${REPO_ROOT}"
sudo bash "${REPO_ROOT}/OSWorld/desktop_env/providers/aws/scripts/install_resetd.sh"

echo
echo "Reset daemon health:"
curl http://127.0.0.1:5001/health
echo
echo
echo "OSWorld server health:"
curl http://127.0.0.1:5000/health
echo
echo
echo "Services:"
sudo systemctl --no-pager --full status osworld-home-overlay.service osworld-resetd.service osworld-server.service
