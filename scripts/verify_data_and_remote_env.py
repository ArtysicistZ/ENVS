#!/usr/bin/env python3
"""
Verify that (1) task data loads and has the right shape, and (2) the remote env
server returns a screenshot on reset (Chrome / API working).

Usage:
  python scripts/verify_data_and_remote_env.py [CONFIG_YAML] [--skip-reset]
  Default CONFIG: configs/smoke_remote_env.yaml
  --skip-reset: only verify data loading; do not call /env/reset.

Data flow verified:
  - train_files JSON (e.g. test_smoke_4.json) -> task IDs per domain
  - examples/<domain>/<id>.json -> full task_config (instruction, config for Chrome, evaluator)
  - task_config sent in POST /env/reset -> server should return obs_messages with screenshot (b64 image)
"""
import argparse
import json
import os
import sys

# Add project root so we can import verl and load config
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Verify data and remote env (screenshot, Chrome, API).")
    parser.add_argument("config", nargs="?", default="configs/smoke_remote_env.yaml", help="Config YAML path")
    parser.add_argument("--skip-reset", action="store_true", help="Only verify data; do not call /env/reset")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)
    if not os.path.isfile(config_path):
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    # Load config for data path and remote URL
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        train_files = OmegaConf.select(cfg, "data.train_files") or "OSWorld/evaluation_examples/test_smoke_4.json"
        remote_url = OmegaConf.select(cfg, "env.remote_server_url")
    except Exception as e:
        print(f"WARNING: Could not load config: {e}. Using defaults.")
        train_files = "OSWorld/evaluation_examples/test_smoke_4.json"
        remote_url = "http://67.202.26.249:15001"

    if not os.path.isabs(train_files):
        train_files = os.path.join(_PROJECT_ROOT, train_files)
    if not os.path.isfile(train_files):
        print(f"ERROR: train_files not found: {train_files}")
        sys.exit(1)

    print("=== 1. Data: task list and examples ===")
    with open(train_files, "r") as f:
        task_list = json.load(f)
    print(f"  train_files: {train_files}")
    print(f"  domains: {list(task_list.keys())}")
    base_path = os.path.dirname(train_files)
    for domain, ids in task_list.items():
        print(f"  {domain}: {len(ids)} task(s)")
        for tid in ids[:2]:
            ex_path = os.path.join(base_path, "examples", domain, tid + ".json")
            if os.path.isfile(ex_path):
                with open(ex_path, "r") as f:
                    tc = json.load(f)
                print(f"    example: {ex_path} -> instruction (len={len(tc.get('instruction',''))}), config (len={len(tc.get('config',[]))})")
            else:
                print(f"    MISSING: {ex_path}")
    print(f"  Total tasks: {sum(len(v) for v in task_list.values())}")
    print("  ✓ Data files present\n")

    print("=== 2. Task config shape (one full example) ===")
    from verl.utils.osworld import OSWorldTaskConfigDataset
    dataset = OSWorldTaskConfigDataset(train_files)
    if len(dataset) == 0:
        print("  ERROR: Dataset is empty")
        sys.exit(1)
    task_config = dataset[0]
    required = ["id", "instruction", "domain", "task_id"]
    for k in required:
        if k not in task_config:
            print(f"  ERROR: task_config missing key: {k}")
            sys.exit(1)
    print(f"  id: {task_config['id']}")
    print(f"  instruction: {task_config['instruction'][:80]}...")
    print(f"  domain: {task_config['domain']}")
    print(f"  config (Chrome/launch): {len(task_config.get('config', []))} item(s)")
    print("  ✓ Task config has instruction, id, domain, config (for Chrome)\n")

    if args.skip_reset:
        print("=== Skip reset (--skip-reset). Done. ===")
        return

    if not remote_url:
        print("=== No remote_server_url in config. Skip /env/reset. ===")
        return

    print("=== 3. Remote env: health and reset (screenshot) ===")
    import requests
    remote_url = remote_url.rstrip("/")
    try:
        r = requests.get(f"{remote_url}/health", timeout=5)
        if r.status_code != 200:
            print(f"  ERROR: /health returned {r.status_code}")
            sys.exit(1)
        print(f"  /health: 200 OK")
    except Exception as e:
        print(f"  ERROR: /health failed: {e}")
        sys.exit(1)

    print(f"  POST /env/reset with task_config (id={task_config['id']})...")
    try:
        r = requests.post(
            f"{remote_url}/env/reset",
            json={"task_config": task_config},
            timeout=120,
        )
        if r.status_code != 200:
            print(f"  ERROR: /env/reset returned {r.status_code}: {r.text[:300]}")
            sys.exit(1)
        data = r.json()
        obs = data.get("obs_messages")
        if not obs:
            print(f"  ERROR: obs_messages missing or empty. Response keys: {list(data.keys())}")
            sys.exit(1)
        # Check for at least one image (screenshot) in the first message content
        has_image = False
        for msg in obs:
            for c in msg.get("content", []):
                if c.get("type") == "image" and (c.get("b64") or c.get("image")):
                    has_image = True
                    break
            if has_image:
                break
        if not has_image:
            print(f"  ERROR: obs_messages has no image (screenshot). Content structure: {[list(m.keys()) for m in obs]}")
            sys.exit(1)
        print("  ✓ Reset returned obs_messages with screenshot (Chrome/screenshot/API working)\n")
    except requests.exceptions.Timeout:
        print("  ERROR: /env/reset timed out (VM/Chrome may be slow to start)")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: /env/reset failed: {e}")
        sys.exit(1)

    print("=== All checks passed: data in use, screenshot and API working. ===")


if __name__ == "__main__":
    main()
