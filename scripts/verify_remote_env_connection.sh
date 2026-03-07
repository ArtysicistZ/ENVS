#!/bin/bash
# Quick verification script: test cluster -> remote env server connectivity
# Usage: ./scripts/verify_remote_env_connection.sh [SERVER_URL]
# Default: http://35.175.248.181:15001

SERVER_URL="${1:-http://35.175.248.181:15001}"
SERVER_IP=$(echo "$SERVER_URL" | sed -E 's|https?://([^:/]+).*|\1|')
SERVER_PORT=$(echo "$SERVER_URL" | sed -E 's|https?://[^:]+:([0-9]+).*|\1|' || echo "15001")

echo "=== Verifying connection to remote env server ==="
echo "Server: $SERVER_URL"
echo ""

echo "1. Testing basic connectivity (TCP port $SERVER_PORT)..."
if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$SERVER_IP/$SERVER_PORT" 2>/dev/null; then
    echo "   ✓ Port $SERVER_PORT is reachable"
else
    echo "   ✗ Port $SERVER_PORT is NOT reachable (check security group/firewall)"
    exit 1
fi

echo ""
echo "2. Testing POST /env/reset (minimal task_config)..."
RESET_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/env/reset" \
    -H "Content-Type: application/json" \
    -d '{"task_config":{"id":"verify-test","instruction":"test"}}' \
    --connect-timeout 5 --max-time 20 2>&1)
RESET_HTTP_CODE=$(echo "$RESET_RESPONSE" | tail -n1)
RESET_BODY=$(echo "$RESET_RESPONSE" | head -n-1)

if [ "$RESET_HTTP_CODE" = "200" ]; then
    echo "   ✓ Reset succeeded (HTTP 200); server and env are ready."
    echo "   Response (truncated): $(echo "$RESET_BODY" | head -c 200)..."
elif [ "$RESET_HTTP_CODE" = "503" ]; then
    echo "   ✗ Reset returned 503 Service Unavailable. Server error detail:"
    echo "$RESET_BODY" | head -5
    if echo "$RESET_BODY" | grep -q "boto3\|AWS provider\|credentials"; then
        echo "   Fix on the EC2 host where the server runs: pip install boto3 && aws configure (or attach IAM role with EC2 permissions)."
    fi
    exit 1
else
    echo "   ✗ Reset failed (HTTP $RESET_HTTP_CODE)"
    echo "   Response: $RESET_BODY"
    exit 1
fi

echo ""
echo "3. Testing POST /env/evaluate..."
EVAL_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/env/evaluate" \
    -H "Content-Type: application/json" \
    -d '{}' \
    --max-time 10 2>&1)
EVAL_HTTP_CODE=$(echo "$EVAL_RESPONSE" | tail -n1)
EVAL_BODY=$(echo "$EVAL_RESPONSE" | head -n-1)

if [ "$EVAL_HTTP_CODE" = "200" ]; then
    echo "   ✓ Evaluate responded (HTTP 200): $EVAL_BODY"
elif [ "$EVAL_HTTP_CODE" = "503" ]; then
    echo "   Note: Evaluate 503 (env not ready) - optional; reset above passed."
else
    echo "   Evaluate HTTP $EVAL_HTTP_CODE: $EVAL_BODY"
fi

echo ""
echo "=== Connection test passed. Remote env server is ready. ==="
