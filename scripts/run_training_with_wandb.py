#!/usr/bin/env python3
"""
ARPO Training Script with W&B Logging

Wraps VERL training with comprehensive Weights & Biases logging.
"""

import wandb
import argparse
import subprocess
import json
import os
from pathlib import Path
import time

def setup_wandb(config_path="configs/wandb_config.yaml"):
    """Initialize wandb with configuration"""
    import yaml
    
    # Load wandb config
    with open(config_path) as f:
        wb_config = yaml.safe_load(f)
    
    if not wb_config.get('wandb', {}).get('enabled', False):
        print("⚠️  wandb logging disabled in config")
        return None
    
    wb_settings = wb_config['wandb']
    
    # Check if logged in
    if not wandb.api.api_key:
        print("❌ wandb not logged in!")
        print("   Run: wandb login")
        return None
    
    # Initialize wandb
    run = wandb.init(
        entity=wb_settings.get('entity'),
        project=wb_settings.get('project', 'arpo-uitars'),
        name=wb_settings.get('name', f'train-{int(time.time())}'),
        tags=wb_settings.get('tags', []),
        notes=wb_settings.get('notes', ''),
        config={
            "model": "UI-TARS-2B",
            "tasks": 128,
            "epochs": 10,
            "num_envs": 4,
            "inference_server": "colab_gpu",
        }
    )
    
    print(f"✅ wandb initialized: {wandb.run.url}")
    return run

def log_epoch_metrics(epoch, results_dir):
    """Log metrics for completed epoch"""
    # Parse results
    results = []
    for result_file in Path(results_dir).rglob("result.txt"):
        try:
            score = float(result_file.read_text().strip())
            results.append(score)
        except:
            pass
    
    if results:
        metrics = {
            "epoch": epoch,
            "average_reward": sum(results) / len(results),
            "success_rate": sum(1 for r in results if r >= 0.9) / len(results),
            "tasks_completed": len(results),
        }
        
        wandb.log(metrics)
        print(f"📊 Logged epoch {epoch} metrics: {metrics}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-config", default="configs/wandb_config.yaml")
    parser.add_argument("--training-config", default="configs/config_uitars_2b_mac.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--result-dir", default="results_training_128")
    args = parser.parse_args()
    
    # Setup wandb
    run = setup_wandb(args.wandb_config)
    
    if not run:
        print("Continuing without wandb logging...")
    
    try:
        # Run training (placeholder - adapt for your VERL training command)
        print("="*70)
        print("🚀 Starting ARPO Training with wandb logging")
        print("="*70)
        print()
        print("⚠️  This is a template. Adapt the training command below:")
        print()
        print("python -m verl.trainer.main \\")
        print(f"    config={args.training_config} \\")
        print("    data.train_files=test_data/osworld_examples/train_all_128.json \\")
        print("    trainer.total_episodes={} \\".format(args.epochs))
        print("    algorithm.enable_replay=true")
        print()
        print("💡 Integrate wandb.log() calls in VERL trainer for real-time logging")
        print()
        
        # For each epoch (example - adapt to actual training loop)
        for epoch in range(args.epochs):
            print(f"\n📊 Epoch {epoch + 1}/{args.epochs}")
            
            # Training happens here (VERL framework)
            # After epoch completes, log metrics
            
            if run:
                log_epoch_metrics(epoch + 1, args.result_dir)
            
            time.sleep(1)  # Placeholder
            
    finally:
        if run:
            wandb.finish()
            print("\n✅ wandb run finished")

if __name__ == "__main__":
    main()
