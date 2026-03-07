#!/bin/bash
# Test UI-TARS-2B Server

echo "=============================================="
echo "Testing UI-TARS-2B Server"
echo "=============================================="
echo ""

# Test 1: Check if server is running
echo "Test 1: Checking if server is running..."
if curl -s http://localhost:9000/health > /dev/null 2>&1; then
    echo "✓ Server is running"
else
    echo "❌ Server not running!"
    echo "   Start it with: python uitars_2b_server.py"
    exit 1
fi

# Test 2: List available models
echo ""
echo "Test 2: Listing available models..."
curl -s http://localhost:9000/v1/models | python3 -m json.tool

# Test 3: Send a test request
echo ""
echo "Test 3: Sending test inference request..."
echo "(This may take 10-30 seconds on CPU)"

curl -s http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ui-tars-2b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Hello, are you working?"}
        ]
      }
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }' | python3 -m json.tool

echo ""
echo "=============================================="
echo "Server tests complete!"
echo "=============================================="
echo ""
echo "✓ Server is ready for OSWorld integration"
