#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

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
