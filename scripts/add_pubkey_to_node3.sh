#!/usr/bin/env bash
# Add your public key to ~/.ssh/authorized_keys on node 3 so SSH from node 2 works.
# You must run the printed command ON node 3 (e.g. get a shell via cluster console, job, or password login).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KEY="$REPO_ROOT/ssh_keys_for_users/ssh_keys_for_users_172.174.34.71/ssh_keys_for_users/kevinzyz_id_ed25519"

if [[ ! -f "$KEY" ]]; then
  echo "Key not found: $KEY"
  exit 1
fi

PUB="$(ssh-keygen -y -f "$KEY")"
echo "Public key (for node 3):"
echo "$PUB"
echo ""
echo "--- Run this ON node 3 (deepx-a100-40g-3) ---"
echo "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$PUB' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo Done."
echo ""
echo "Or copy the line above and on node 3 run: echo '<paste>' >> ~/.ssh/authorized_keys"
