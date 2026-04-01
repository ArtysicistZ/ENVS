"""Precompute reference model log-probs for KTO training.

Runs the frozen reference model on all KTO examples (positives + negatives)
and saves per-example sum-of-log-probs for the action tokens.

This allows KTO training to run with only the policy model on GPU.

Usage:
    python scripts/precompute_ref_logps.py \
        --config configs/mcts_kto.yaml \
        --output checkpoints/mcts_kto/ref_logps.pt
"""

import argparse
import copy
import json
import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the KTO dataset and collator
import importlib.util
_kto_spec = importlib.util.spec_from_file_location(
    "train_mcts_kto",
    os.path.join(os.path.dirname(__file__), "train_mcts_kto.py"),
)
_kto_mod = importlib.util.module_from_spec(_kto_spec)
_kto_spec.loader.exec_module(_kto_mod)

MCTSKTODataset = _kto_mod.MCTSKTODataset
kto_collate_fn = _kto_mod.kto_collate_fn


def compute_logps(logits, labels):
    """Compute sum of log-probs for action tokens (labels != -100)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = (shift_labels != -100).float()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    per_sample = (gathered * mask).sum(dim=1)  # [B]
    return per_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mcts_kto.yaml")
    parser.add_argument("--output", default="checkpoints/mcts_kto/ref_logps.pt")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ref_path = cfg.get("ref_model_path", cfg["model_path"])
    print(f"Loading reference model from {ref_path}...")

    tokenizer = AutoTokenizer.from_pretrained(ref_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        ref_path, trust_remote_code=True, max_pixels=2116800, min_pixels=256
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForVision2Seq.from_pretrained(
        ref_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()

    # Load dataset
    print("Loading dataset...")
    dataset = MCTSKTODataset(
        tree_dir=cfg["tree_dir"],
        task_index_path=cfg["task_index_path"],
        mask_path=cfg["mask_path"],
        sr_path=cfg["sr_path"],
        negatives_path=cfg.get("negatives_path", "checkpoints/mcts_kto/kto_negatives_all.jsonl"),
        tokenizer=tokenizer,
        processor=processor,
        max_length=cfg.get("max_length", 12000),
        limit_images=cfg.get("limit_images", 3),
        beta=cfg.get("difficulty_beta", 0.5),
        max_step_ratio=cfg.get("max_step_ratio", 2.0),
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=kto_collate_fn, num_workers=0,
    )

    print(f"Computing ref log-probs for {len(dataset)} examples...")
    all_ref_logps = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Precomputing"):
            # Remove KTO-specific fields
            batch.pop("is_desirable", None)
            batch.pop("sample_weight", None)

            # Move to GPU
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            if batch.get("pixel_values") is not None:
                model_inputs["pixel_values"] = batch["pixel_values"].cuda()
            if batch.get("image_grid_thw") is not None:
                model_inputs["image_grid_thw"] = batch["image_grid_thw"].cuda()

            outputs = model(**model_inputs)
            ref_logps = compute_logps(outputs.logits, labels)
            all_ref_logps.append(ref_logps.cpu())

    all_ref_logps = torch.cat(all_ref_logps, dim=0)
    print(f"Computed {len(all_ref_logps)} ref log-probs")
    print(f"  Mean: {all_ref_logps.mean():.2f}, Std: {all_ref_logps.std():.2f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(all_ref_logps, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
