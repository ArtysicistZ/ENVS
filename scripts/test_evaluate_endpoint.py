#!/usr/bin/env python3
"""
Test POST /env/evaluate from the cluster (same payload as RemoteEnvWorker).
Usage: python scripts/test_evaluate_endpoint.py [BASE_URL]
Default BASE_URL: http://35.175.248.181:15001
"""
import json
import sys

import requests

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://35.175.248.181:15001").rstrip("/")
EVALUATE_URL = f"{BASE_URL}/env/evaluate"
TIMEOUT = 300  # same as REMOTE_EVALUATE_TIMEOUT

def main():
    print(f"POST {EVALUATE_URL}")
    print(f"Body: {{}}  (empty JSON object)")
    print(f"Timeout: {TIMEOUT}s")
    print()
    try:
        r = requests.post(EVALUATE_URL, json={}, timeout=TIMEOUT)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text[:500]}")
        if r.headers.get("content-type", "").startswith("application/json"):
            try:
                data = r.json()
                score = float(data) if isinstance(data, (int, float)) else data
                print(f"Score (parsed): {score}")
            except (TypeError, ValueError):
                print(f"JSON body: {data}")
        r.raise_for_status()
        print("OK: 200 and sensible response.")
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out.")
        sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        print(f"ERROR: Connection failed: {e}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
