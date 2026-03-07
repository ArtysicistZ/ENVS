#!/usr/bin/env python3
"""Quick check: GPUs visible to nvidia-smi vs GPUs visible to Ray. Run on the same machine where you start training."""
import os
import subprocess
import sys

def main():
    print("=== 1. nvidia-smi (physical GPUs) ===")
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        print(out)
        n_smi = out.count("GPU ")
        print(f"-> {n_smi} GPU(s) reported by nvidia-smi\n")
    except Exception as e:
        print(f"nvidia-smi failed: {e}\n")
        n_smi = 0

    print("=== 2. Ray cluster (what training uses) ===")
    try:
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        node_res = ray.state.available_resources_per_node()
        node_gpus = {n: info.get("GPU", 0) for n, info in node_res.items()}
        total = sum(node_gpus.values())
        print(f"RAY_ADDRESS: {os.environ.get('RAY_ADDRESS', '(not set = new cluster)')}")
        for node, g in node_gpus.items():
            print(f"  {node}: {g} GPU(s)")
        print(f"-> Total GPUs visible to Ray: {total}\n")
        if total == 0 and n_smi > 0:
            print("Ray sees 0 GPUs but nvidia-smi sees", n_smi, "-> Ray was likely started without --num-gpus.")
            print("Fix: unset RAY_ADDRESS and run training (it will start a new Ray that auto-detects GPUs),")
            print("  or restart the cluster with: ray start --head --num-gpus=" + str(n_smi))
        elif total > 0:
            print("Ray sees GPUs. Training should be able to use them.")
    except Exception as e:
        print(f"Ray check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
