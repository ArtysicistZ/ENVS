"""
MCTS SFT v2 Training Script — Per-Step Deduplicated, Power-Scaled Weighting.

Trains on successful MCTS tree trajectories with:
  1. Per-step deduplication: shared prefix steps trained exactly once
  2. Step-level KEEP/REMOVE masks (agent-audited)
  3. Power-scaled difficulty weighting: per_step_weight = (1-SR_t)^beta / T_t
     All KEEP steps within a task get equal weight.

Data inputs:
  - Tree JSON files in combined_all/trees/
  - task_index.json: task_id → [tree filenames]
  - mcts_success.jsonl: per-task success rates
  - step_masks_v2.json: per-node KEEP/REMOVE masks

Usage:
    torchrun --nproc_per_node=8 scripts/train_mcts_sft_v2.py \
        --config configs/mcts_sft_v2.yaml
"""

import argparse
import copy
import datetime
import json
import math
import os
import sys
import time
import yaml
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from verl.utils.trajectory_sft import expand_episode
from verl.mcts.tree_io import load_mcts_tree, reconstruct_steps, _get_ancestor_chain

# Qwen2-VL specific
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    from qwen_vl_utils.vision_process import process_vision_info

DEFAULT_IMAGE_TOKEN = "<|image_pad|>"


# ---------------------------------------------------------------------------
# Helpers: map reconstructed steps back to source nodes
# ---------------------------------------------------------------------------

def map_steps_to_nodes(tree_data, leaf_id):
    """For each step in reconstruct_steps(tree, leaf_id), return (node_id, step_idx_within_node).

    This lets us look up the correct mask entry for each step.
    """
    chain = _get_ancestor_chain(tree_data, leaf_id)
    nodes = tree_data["nodes"]
    sources = []
    for i, nid in enumerate(chain):
        node = nodes[nid]
        own_steps = node.get("own_steps", [])
        if i < len(chain) - 1:
            child_nid = chain[i + 1]
            cut = nodes[child_nid].get("parent_steps_at_branch", len(own_steps))
            for j in range(cut):
                sources.append((nid, j))
        else:
            for j in range(len(own_steps)):
                sources.append((nid, j))
    return sources


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MCTSSFTDatasetV2(Dataset):
    """Per-step deduplicated SFT dataset from MCTS trees.

    Each unique KEEP step across all successful trajectories appears exactly once.
    All KEEP steps within a task get equal weight = (1-SR_t)^beta / T_t.
    """

    def __init__(
        self,
        tree_dir: str,
        task_index_path: str,
        mask_path: str,
        sr_path: str,
        tokenizer,
        processor,
        max_length: int = 12000,
        limit_images: int = 3,
        beta: float = 0.5,
        max_step_ratio: float = 10.0,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.limit_images = limit_images

        # Load per-task success rates
        task_sr = {}
        with open(sr_path) as f:
            for line in f:
                r = json.loads(line)
                task_sr[r["task_id"]] = r["sr"]

        # Load task index
        with open(task_index_path) as f:
            task_index = json.load(f)

        # Load step masks
        with open(mask_path) as f:
            masks = json.load(f)

        # Extract unique KEEP steps from all successful trajectories
        print("Loading trees and extracting unique KEEP steps...")
        seen_steps = set()  # (tree_filename, node_id, step_idx_in_node)
        self.examples = []  # list of dicts with messages, label, task_id, weight
        task_keep_counts = Counter()
        n_leaves = 0
        n_total_steps = 0
        n_removed = 0
        n_deduped = 0

        for task_id, tree_files in task_index.items():
            for tree_filename in tree_files:
                tree_path = os.path.join(tree_dir, tree_filename)
                real_path = os.path.realpath(tree_path)
                if not os.path.exists(real_path):
                    continue
                tree_data = load_mcts_tree(real_path)
                round_label = tree_filename.replace(task_id + "_", "").replace(".json", "")
                nodes = tree_data["nodes"]
                instruction = tree_data.get("instruction", "")

                # Find successful leaves
                for leaf_id, leaf in nodes.items():
                    if not (leaf.get("eval_score") and leaf["eval_score"] > 0):
                        continue
                    n_leaves += 1

                    # Reconstruct full trajectory
                    traj_steps = reconstruct_steps(tree_data, leaf_id)
                    step_sources = map_steps_to_nodes(tree_data, leaf_id)
                    n_total_steps += len(traj_steps)

                    # Build episode dict for expand_episode
                    episode = {
                        "task_id": task_id,
                        "instruction": instruction,
                        "eval_result": 1.0,
                        "limit_images": limit_images,
                        "steps": traj_steps,
                    }

                    # Expand into per-step examples
                    all_step_examples = expand_episode(episode, train_all_steps=True)

                    for k, (node_id, node_step_idx) in enumerate(step_sources):
                        step_id = (tree_filename, node_id, node_step_idx)

                        # Dedup: skip if already seen
                        if step_id in seen_steps:
                            n_deduped += 1
                            continue
                        seen_steps.add(step_id)

                        # Check mask
                        mask_key = f"{task_id}:{round_label}:{node_id}"
                        node_mask = masks.get(mask_key)
                        if node_mask is not None and node_step_idx < len(node_mask):
                            if node_mask[node_step_idx] == 0:
                                n_removed += 1
                                continue
                        # If no mask entry, default to KEEP

                        # Get the SFT example for step k
                        if k < len(all_step_examples):
                            example = all_step_examples[k]
                            example["task_id"] = task_id
                            self.examples.append(example)
                            task_keep_counts[task_id] += 1

        self.num_tasks = len(task_keep_counts)
        self.num_leaves = n_leaves

        print(f"  Successful leaves: {n_leaves}")
        print(f"  Total steps before dedup: {n_total_steps}")
        print(f"  Deduplicated: {n_deduped}")
        print(f"  Removed (mask=0): {n_removed}")
        print(f"  Unique KEEP steps (training examples): {len(self.examples)}")
        print(f"  Tasks with data: {self.num_tasks}")

        # Compute per-step weights
        self.weights = []
        raw_weights = []
        for ex in self.examples:
            tid = ex["task_id"]
            sr = task_sr.get(tid, 0.0)
            T_t = task_keep_counts[tid]
            w = ((1 - sr) ** beta) / T_t
            raw_weights.append(w)

        # Normalize to mean=1
        mean_w = sum(raw_weights) / len(raw_weights) if raw_weights else 1.0
        self.weights = [w / mean_w for w in raw_weights]

        # Cap at max_step_ratio
        n_capped = sum(1 for w in self.weights if w > max_step_ratio)
        if n_capped > 0:
            self.weights = [min(w, max_step_ratio) for w in self.weights]
            # Re-normalize after capping
            mean_w2 = sum(self.weights) / len(self.weights)
            self.weights = [w / mean_w2 for w in self.weights]
            print(f"  Capped {n_capped} steps at {max_step_ratio}x (re-normalized)")

        # Print weight distribution
        self._print_weight_summary(task_sr, task_keep_counts, beta)

    def _print_weight_summary(self, task_sr, task_keep_counts, beta):
        """Print gradient distribution by difficulty tier."""
        tiers = {"VeryHard (≤5)": [], "Hard (6-15)": [], "Medium (16-40)": [],
                 "Easy (41-70)": [], "VeryEasy (>70)": []}

        # Need n_success per task — approximate from sr and some reference
        # Use the examples to compute tier budgets
        task_budgets = {}
        for tid, T_t in task_keep_counts.items():
            sr = task_sr.get(tid, 0.0)
            budget = ((1 - sr) ** beta)
            task_budgets[tid] = budget

        total_budget = sum(task_budgets.values())
        if total_budget == 0:
            return

        # Group by approximate success count
        for tid, budget in task_budgets.items():
            sr = task_sr.get(tid, 0.0)
            T_t = task_keep_counts[tid]
            # Estimate n_success from sr (rough)
            n_approx = max(1, int(sr * 700))  # rough average runs
            if n_approx <= 5:
                tiers["VeryHard (≤5)"].append(budget)
            elif n_approx <= 15:
                tiers["Hard (6-15)"].append(budget)
            elif n_approx <= 40:
                tiers["Medium (16-40)"].append(budget)
            elif n_approx <= 70:
                tiers["Easy (41-70)"].append(budget)
            else:
                tiers["VeryEasy (>70)"].append(budget)

        print(f"\n  Gradient distribution (beta={beta}):")
        for tier, budgets in tiers.items():
            pct = sum(budgets) / total_budget * 100 if total_budget > 0 else 0
            print(f"    {tier:20s}: {len(budgets):2d} tasks, {pct:5.1f}%")

        # Weight stats
        if self.weights:
            ws = sorted(self.weights, reverse=True)
            print(f"\n  Weight stats (normalized, mean=1):")
            print(f"    Max: {ws[0]:.2f}, P99: {ws[int(len(ws)*0.01)]:.2f}, "
                  f"P50: {ws[len(ws)//2]:.2f}, Min: {ws[-1]:.2f}")

    def __len__(self):
        return len(self.examples)

    def _load_content(self, content):
        """Convert message content to text (for tokenization)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(self._load_content(c) for c in content)
        if isinstance(content, dict):
            if "text" in content:
                return content["text"]
            elif "image" in content:
                return "<|vision_start|><|image_pad|><|vision_end|>"
        raise ValueError(f"Unknown content type: {content}")

    def __getitem__(self, index):
        example = self.examples[index]
        message = copy.deepcopy(example["messages"])

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            message, return_video_kwargs=True
        )

        if image_inputs is not None and len(image_inputs) >= 1:
            input_ids = []
            labels = []
            attention_mask = []
            pixel_values = []
            image_grid_thw = []
            image_count = 0

            for msg in message:
                role = msg["role"]
                content = self._load_content(msg["content"])
                prompt = f"<|im_start|>{role}\n" + content + "<|im_end|>\n"

                cur_image_num = prompt.count("<|image_pad|>")
                if cur_image_num > 0:
                    result = self.processor(
                        image_inputs[image_count : image_count + cur_image_num],
                        [prompt],
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                    image_count += cur_image_num
                else:
                    result = self.processor(
                        None,
                        [prompt],
                        add_special_tokens=False,
                        return_tensors="pt",
                    )

                cur_input_ids = result.pop("input_ids")[0]
                cur_attention_mask = result.pop("attention_mask")[0]
                if "pixel_values" in result:
                    pixel_values.append(result["pixel_values"])
                if "image_grid_thw" in result:
                    image_grid_thw.append(result["image_grid_thw"])

                input_ids.append(cur_input_ids)
                attention_mask.append(cur_attention_mask)

                if role in ("system", "user"):
                    labels.append(torch.full_like(cur_input_ids, -100))
                else:
                    labels.append(cur_input_ids.clone())

            input_ids = torch.cat(input_ids, dim=0)
            labels = torch.cat(labels, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)

            pixel_values = (
                torch.cat(pixel_values, dim=0) if pixel_values else None
            )
            image_grid_thw = (
                torch.cat(image_grid_thw, dim=0) if image_grid_thw else None
            )
        else:
            input_ids = torch.zeros((0,), dtype=torch.int64)
            labels = torch.full((0,), -100, dtype=torch.int64)
            attention_mask = torch.zeros((0,), dtype=torch.int64)
            pixel_values = None
            image_grid_thw = None

        # Truncate to max_length
        if input_ids.size(0) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

            if image_grid_thw is not None and pixel_values is not None:
                image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
                merge_size = self.processor.image_processor.merge_size

                tokens_per_image = []
                patches_per_image = []
                for i in range(image_grid_thw.shape[0]):
                    t, h, w = image_grid_thw[i].tolist()
                    tokens_per_image.append(int(t * (h // merge_size) * (w // merge_size)))
                    patches_per_image.append(int(t * h * w))

                total_expected = sum(tokens_per_image)
                total_actual = (input_ids == image_token_id).sum().item()

                if total_actual != total_expected:
                    surviving = 0
                    cum = 0
                    for i in range(len(tokens_per_image)):
                        if cum + tokens_per_image[i] <= total_actual:
                            cum += tokens_per_image[i]
                            surviving += 1
                        else:
                            break
                    partial = total_actual - cum
                    keep_patches = sum(patches_per_image[:surviving])

                    if surviving > 0:
                        pixel_values = pixel_values[:keep_patches]
                        image_grid_thw = image_grid_thw[:surviving]
                    else:
                        pixel_values = None
                        image_grid_thw = None

                    if partial > 0:
                        pad_token_id = self.tokenizer.pad_token_id or 0
                        positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
                        idx = positions[-partial:]
                        input_ids = input_ids.clone()
                        attention_mask = attention_mask.clone()
                        labels = labels.clone()
                        input_ids[idx] = pad_token_id
                        attention_mask[idx] = 0
                        labels[idx] = -100

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "sample_weight": torch.tensor(self.weights[index], dtype=torch.float32),
        }
        return result


# ---------------------------------------------------------------------------
# Import shared components from v1
# ---------------------------------------------------------------------------
import importlib.util
_v1_spec = importlib.util.spec_from_file_location(
    "train_mcts_sft",
    os.path.join(os.path.dirname(__file__), "train_mcts_sft.py"),
)
_v1_mod = importlib.util.module_from_spec(_v1_spec)
_v1_spec.loader.exec_module(_v1_mod)

mcts_collate_fn = _v1_mod.mcts_collate_fn
MCTSTrainer = _v1_mod.MCTSTrainer
EpochSaveCallback = _v1_mod.EpochSaveCallback
StopAfterEpochCallback = _v1_mod.StopAfterEpochCallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="MCTS SFT v2 — Per-Step Deduplicated, Power-Scaled Weighting")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tree_dir", type=str, default=None)
    parser.add_argument("--task_index_path", type=str, default=None)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--sr_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--limit_images", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None,
                        help="Power exponent for difficulty weighting. 0.5=sqrt (default)")
    parser.add_argument("--max_step_ratio", type=float, default=None,
                        help="Cap per-step weight at this multiple of mean (default 10)")
    parser.add_argument("--freeze_vision_tower", action="store_true", default=None)
    parser.add_argument("--no_freeze_vision_tower", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--stop_after_epoch", type=int, default=None)
    parser.add_argument("--fsdp_offload", action="store_true", default=None)
    parser.add_argument("--no_save_model", action="store_true", default=None)
    parser.add_argument("--epoch_offset", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    cli_args = parser.parse_args()

    defaults = {
        "model_path": "ByteDance-Seed/UI-TARS-1.5-7B",
        "tree_dir": "checkpoints/mcts_trajectories_v2/combined_all/trees",
        "task_index_path": "checkpoints/mcts_trajectories_v2/combined_all/task_index.json",
        "mask_path": "checkpoints/mcts_trajectories_v2/combined_all/step_masks_v2.json",
        "sr_path": "checkpoints/mcts_trajectories_v2/combined_all/mcts_success.jsonl",
        "output_dir": "checkpoints/mcts_sft_v2",
        "num_epochs": 1,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-6,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "max_length": 12000,
        "limit_images": 3,
        "beta": 0.5,
        "max_step_ratio": 10.0,
        "freeze_vision_tower": True,
        "bf16": True,
        "logging_steps": 1,
        "fsdp_offload": False,
        "no_save_model": False,
        "epoch_offset": 0,
    }

    yaml_config = {}
    if cli_args.config:
        print(f"Loading config from {cli_args.config}...")
        with open(cli_args.config) as f:
            yaml_config = yaml.safe_load(f) or {}

    final = argparse.Namespace()
    for key, default_val in defaults.items():
        cli_val = getattr(cli_args, key, None)
        yaml_val = yaml_config.get(key, None)
        if cli_val is not None:
            setattr(final, key, cli_val)
        elif yaml_val is not None:
            setattr(final, key, type(default_val)(yaml_val) if default_val is not None else yaml_val)
        else:
            setattr(final, key, default_val)

    final.local_rank = cli_args.local_rank
    final.resume_from_checkpoint = cli_args.resume_from_checkpoint
    final.stop_after_epoch = cli_args.stop_after_epoch
    final.config = cli_args.config
    final.run_name = cli_args.run_name or yaml_config.get("run_name", None)

    if cli_args.no_freeze_vision_tower:
        final.freeze_vision_tower = False

    return final


def main():
    args = _parse_args()

    print(f"=== MCTS SFT v2 Training ===")
    print(f"  beta={args.beta}, max_step_ratio={args.max_step_ratio}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True,
        max_pixels=2116800, min_pixels=256,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
        device_map="cpu", low_cpu_mem_usage=True,
    )

    if args.freeze_vision_tower:
        print("Freezing vision tower...")
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}, Trainable: {trainable_params:,}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    dataset = MCTSSFTDatasetV2(
        tree_dir=args.tree_dir,
        task_index_path=args.task_index_path,
        mask_path=args.mask_path,
        sr_path=args.sr_path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=args.max_length,
        limit_images=args.limit_images,
        beta=args.beta,
        max_step_ratio=args.max_step_ratio,
    )

    n_gpus = max(1, torch.cuda.device_count())
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * n_gpus
    steps_per_epoch = len(dataset) // effective_batch

    print(f"  Effective batch: {effective_batch}")
    print(f"  Steps/epoch: {steps_per_epoch}")

    args_dict = {k: getattr(args, k) for k in vars(args) if not k.startswith("_")}
    args_dict["effective_batch_size"] = effective_batch
    args_dict["steps_per_epoch"] = steps_per_epoch
    args_dict["n_gpus"] = n_gpus

    dataset_stats = {
        "num_leaves": dataset.num_leaves,
        "num_unique_keep_steps": len(dataset),
        "num_tasks": dataset.num_tasks,
        "beta": args.beta,
        "max_step_ratio": args.max_step_ratio,
    }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        run_name=getattr(args, "run_name", None),
        fsdp="full_shard auto_wrap offload" if args.fsdp_offload else "full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer"],
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
        },
    )

    epoch_callback = EpochSaveCallback(
        output_dir=args.output_dir,
        processor=processor,
        args_dict=args_dict,
        dataset_stats=dataset_stats,
        no_save_model=args.no_save_model,
        epoch_offset=args.epoch_offset,
    )

    callbacks = [epoch_callback]
    if args.stop_after_epoch is not None:
        callbacks.append(StopAfterEpochCallback(args.stop_after_epoch))

    trainer = MCTSTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=mcts_collate_fn,
        processing_class=tokenizer,
        callbacks=callbacks,
        use_loss_weights=True,  # Always use weighted loss in v2
    )
    epoch_callback.trainer = trainer

    print("Starting MCTS SFT v2 training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("Done!")


if __name__ == "__main__":
    main()
