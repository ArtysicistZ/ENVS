#!/usr/bin/env bash
# Start the remote env server with AWS provider.
# On EC2: auto-detects AWS_SUBNET_ID and AWS_SECURITY_GROUP_ID from this instance if not set.
# Usage: from repo root, run:  ./scripts/start_remote_env_aws.sh
# Or:    bash scripts/start_remote_env_aws.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RESTART=0

# Activate the arpo_env virtualenv (contains uvicorn, fastapi, boto3, etc.)
ARPO_ENV="${REPO_ROOT}/arpo_env/bin/activate"
if [ ! -f "$ARPO_ENV" ]; then
  # Also check parent-level arpo_env
  ARPO_ENV="/home/ubuntu/arpo_remote_env/arpo_env/bin/activate"
fi
if [ -f "$ARPO_ENV" ]; then
  # shellcheck disable=SC1090
  source "$ARPO_ENV"
fi

if [ "${1:-}" = "--restart" ]; then
  RESTART=1
elif [ -n "${1:-}" ]; then
  echo "Usage: $0 [--restart]"
  exit 2
fi

# Load .env if present (don't overwrite existing env)
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export AWS_REGION="${AWS_REGION:-us-east-1}"
# OSWORLD_SNAPSHOT_AMI: leave unset to use IMAGE_ID_MAP default (ami-092bc7644b0debfcd).
# When unset, remote_env_server defaults snapshot_name to "init_state" which makes
# _full_relaunch use the current instance's ImageId (same AMI as initial allocation).
export REMOTE_REPEAT_ACTION_THRESHOLD="${REMOTE_REPEAT_ACTION_THRESHOLD:-3}"
export REMOTE_REPEAT_ACTION_PENALTY="${REMOTE_REPEAT_ACTION_PENALTY:-0.5}"
export REMOTE_ACTION_PAUSE_SEC="${REMOTE_ACTION_PAUSE_SEC:-0.8}"

# Avoid noisy bind errors when the server is already running.
if pgrep -af "uvicorn scripts.remote_env_server:app" >/dev/null 2>&1; then
  if [ "$RESTART" -eq 1 ]; then
    echo "Remote env server already running; restarting..."
    pkill -9 -f "uvicorn scripts.remote_env_server:app" || true
    sleep 1
  else
    echo "Remote env server is already running; not starting a duplicate."
    pgrep -af "uvicorn scripts.remote_env_server:app" || true
    if command -v curl >/dev/null 2>&1; then
      echo "Health:"
      curl -sS --max-time 5 http://127.0.0.1:15001/health || true
      echo
    fi
    echo "Use '$0 --restart' to restart in foreground and stream logs here."
    exit 0
  fi
fi

# On EC2, auto-detect subnet and security group from instance metadata if not set
if [ -z "$AWS_SUBNET_ID" ] || [ -z "$AWS_SECURITY_GROUP_ID" ]; then
  METADATA_URL="http://169.254.169.254/latest/meta-data"
  # IMDSv2 requires a token (many accounts use this by default)
  TOKEN=$(curl -s -f -m 2 -X PUT "$METADATA_URL/../api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
  if [ -n "$TOKEN" ]; then
    META() { curl -s -f -m 2 -H "X-aws-ec2-metadata-token: $TOKEN" "$METADATA_URL/$1"; }
  else
    META() { curl -s -f -m 2 "$METADATA_URL/$1"; }
  fi
  if META "instance-id" >/dev/null 2>&1; then
    MAC=$(META "network/interfaces/macs/" | head -1 | tr -d '\n\r' | sed 's|/$||')
    if [ -n "$MAC" ]; then
      [ -z "$AWS_SUBNET_ID" ] && AWS_SUBNET_ID=$(META "network/interfaces/macs/${MAC}/subnet-id")
      if [ -z "$AWS_SECURITY_GROUP_ID" ]; then
        AWS_SECURITY_GROUP_ID=$(META "network/interfaces/macs/${MAC}/security-group-ids" | head -1 | tr -d '\n\r')
      fi
    fi
    export AWS_SUBNET_ID
    export AWS_SECURITY_GROUP_ID
    if [ -n "$AWS_SUBNET_ID" ] && [ -n "$AWS_SECURITY_GROUP_ID" ]; then
      echo "Auto-detected on EC2: AWS_SUBNET_ID=$AWS_SUBNET_ID AWS_SECURITY_GROUP_ID=$AWS_SECURITY_GROUP_ID"
    fi
  fi
fi

if [ -z "$AWS_SUBNET_ID" ] || [ -z "$AWS_SECURITY_GROUP_ID" ]; then
  echo "Error: Set AWS_SUBNET_ID and AWS_SECURITY_GROUP_ID (in .env or env), or run this script on an EC2 instance in the target VPC."
  exit 1
fi

echo "Starting remote env server (AWS provider) on 0.0.0.0:15001 ..."
echo "  AMI for new VMs: $OSWORLD_SNAPSHOT_AMI"
echo "  Python: $(which python)"
exec env PROVIDER=aws python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 15001
