"""
MCTS KTO Training Script — Step-Level KTO with Branch-Point Negatives.

Trains on:
  - Positive: KEEP steps from successful MCTS trajectories (same as SFT v2.1)
  - Negative: Wrong steps from contrastive branch points (Phase 1)

KTO loss:
  - Desirable: push up log π(good_action | state) relative to reference
  - Undesirable: push down log π(bad_action | state) relative to reference
  - No pairing required — each step independently labeled

Reference model: frozen SFT v2.1 checkpoint.
Policy model: trainable copy of SFT v2.1 checkpoint.

Usage:
    torchrun --nproc_per_node=8 scripts/train_mcts_kto.py \
        --config configs/mcts_kto.yaml
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

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    from qwen_vl_utils.vision_process import process_vision_info

DEFAULT_IMAGE_TOKEN = "<|image_pad|>"

# Import shared components from v1
import importlib.util
_v1_spec = importlib.util.spec_from_file_location(
    "train_mcts_sft",
    os.path.join(os.path.dirname(__file__), "train_mcts_sft.py"),
)
_v1_mod = importlib.util.module_from_spec(_v1_spec)
_v1_spec.loader.exec_module(_v1_mod)

mcts_collate_fn = _v1_mod.mcts_collate_fn
EpochSaveCallback = _v1_mod.EpochSaveCallback
StopAfterEpochCallback = _v1_mod.StopAfterEpochCallback

# Also import map_steps_to_nodes from v2
_v2_spec = importlib.util.spec_from_file_location(
    "train_mcts_sft_v2",
    os.path.join(os.path.dirname(__file__), "train_mcts_sft_v2.py"),
)
_v2_mod = importlib.util.module_from_spec(_v2_spec)
_v2_spec.loader.exec_module(_v2_mod)

map_steps_to_nodes = _v2_mod.map_steps_to_nodes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MCTSKTODataset(Dataset):
    """Step-level KTO dataset: positives from SFT + negatives from branch points.

    Each example has:
        - messages: conversation up to the action
        - is_desirable: 1.0 (positive KEEP step) or 0.0 (negative wrong step)
        - sample_weight: difficulty weighting (same as SFT v2.1 for positives)
    """

    def __init__(
        self,
        tree_dir: str,
        task_index_path: str,
        mask_path: str,
        sr_path: str,
        negatives_path: str,
        tokenizer,
        processor,
        max_length: int = 12000,
        limit_images: int = 3,
        beta: float = 0.5,
        max_step_ratio: float = 2.0,
        auto_balance: bool = True,
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

        with open(task_index_path) as f:
            task_index = json.load(f)
        with open(mask_path) as f:
            masks = json.load(f)

        # ---- Load positives (same as MCTSSFTDatasetV2) ----
        print("Loading positive examples (KEEP steps from successful trajectories)...")
        seen_steps = set()
        self.examples = []
        self.is_desirable = []
        task_keep_counts = Counter()
        n_pos = 0

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

                for leaf_id, leaf in nodes.items():
                    if not (leaf.get("eval_score") and leaf["eval_score"] > 0):
                        continue

                    traj_steps = reconstruct_steps(tree_data, leaf_id)
                    step_sources = map_steps_to_nodes(tree_data, leaf_id)

                    episode = {
                        "task_id": task_id,
                        "instruction": instruction,
                        "eval_result": 1.0,
                        "limit_images": limit_images,
                        "steps": traj_steps,
                    }
                    all_step_examples = expand_episode(episode, train_all_steps=True)

                    for k, (node_id, node_step_idx) in enumerate(step_sources):
                        step_id = (tree_filename, node_id, node_step_idx)
                        if step_id in seen_steps:
                            continue
                        seen_steps.add(step_id)

                        mask_key = f"{task_id}:{round_label}:{node_id}"
                        node_mask = masks.get(mask_key)
                        if node_mask is not None and node_step_idx < len(node_mask):
                            if node_mask[node_step_idx] == 0:
                                continue

                        if k < len(all_step_examples):
                            example = all_step_examples[k]
                            example["task_id"] = task_id
                            self.examples.append(example)
                            self.is_desirable.append(1.0)
                            task_keep_counts[task_id] += 1
                            n_pos += 1

        print(f"  Positive examples: {n_pos}")

        # ---- Load negatives (Phase 1 branch-point negatives) ----
        print("Loading negative examples (branch-point wrong actions)...")
        n_neg = 0
        neg_task_counts = Counter()

        with open(negatives_path) as f:
            for line in f:
                neg = json.loads(line.strip())
                if not neg.get("steps"):
                    continue

                # Build episode and expand to get the last step (the wrong action)
                episode = {
                    "task_id": neg["task_id"],
                    "instruction": neg["instruction"],
                    "eval_result": 0.0,
                    "limit_images": neg.get("limit_images", limit_images),
                    "steps": neg["steps"],
                }
                all_step_examples = expand_episode(episode, train_all_steps=False)
                # train_all_steps=False gives only the LAST step (the wrong action)

                if all_step_examples:
                    example = all_step_examples[0]
                    example["task_id"] = neg["task_id"]
                    self.examples.append(example)
                    self.is_desirable.append(0.0)
                    neg_task_counts[neg["task_id"]] += 1
                    n_neg += 1

        self.n_pos = n_pos
        self.n_neg = n_neg
        self.num_tasks = len(task_keep_counts)
        print(f"  Negative examples: {n_neg}")
        print(f"  Total examples: {len(self.examples)} (pos={n_pos}, neg={n_neg})")
        print(f"  Pos/Neg ratio: {n_pos / max(n_neg, 1):.1f}:1")
        print(f"  Tasks with positives: {len(task_keep_counts)}")
        print(f"  Tasks with negatives: {len(neg_task_counts)}")

        # ---- Compute weights: ALL steps (pos+neg) within a task share the same weight ----
        # Weight = (1-SR_t)^beta / T_t_total, where T_t_total = pos + neg steps for task t
        # Same design as SFT v2.1 but T_t now includes negatives
        task_total_counts = Counter()
        for ex in self.examples:
            task_total_counts[ex["task_id"]] += 1

        raw_weights = []
        for ex in self.examples:
            tid = ex["task_id"]
            sr = task_sr.get(tid, 0.0)
            T_t = task_total_counts[tid]
            w = ((1 - sr) ** beta) / T_t
            raw_weights.append(w)

        # Normalize to mean=1
        mean_w = sum(raw_weights) / len(raw_weights)
        raw_weights = [w / mean_w for w in raw_weights]

        # Cap at max_step_ratio
        n_capped = sum(1 for w in raw_weights if w > max_step_ratio)
        if n_capped > 0:
            raw_weights = [min(w, max_step_ratio) for w in raw_weights]
            mean_w2 = sum(raw_weights) / len(raw_weights)
            raw_weights = [w / mean_w2 for w in raw_weights]
            print(f"  Capped {n_capped} steps at {max_step_ratio}x (re-normalized)")

        self.weights = raw_weights

        print(f"  Tasks with steps: {len(task_total_counts)}")
        ws = sorted(self.weights, reverse=True)
        print(f"  Weight stats: max={ws[0]:.2f}, p50={ws[len(ws)//2]:.2f}, min={ws[-1]:.2f}")

        # Auto-balance: scale for the pos/neg ratio imbalance
        # KTO paper recommends: lambda_u proportional to n_pos/n_neg
        self.lambda_d = 1.0
        self.lambda_u = float(n_pos) / max(float(n_neg), 1.0) if auto_balance else 1.0
        print(f"  Auto-balance: lambda_d={self.lambda_d:.1f}, lambda_u={self.lambda_u:.1f}")

        # Precomputed ref log-probs (loaded later via load_ref_logps)
        self.ref_logps = None

    def load_ref_logps(self, path):
        """Load precomputed reference model log-probs."""
        self.ref_logps = torch.load(path, map_location="cpu")
        assert len(self.ref_logps) == len(self.examples), \
            f"ref_logps length {len(self.ref_logps)} != examples {len(self.examples)}"
        print(f"  Loaded precomputed ref log-probs from {path}")

    def __len__(self):
        return len(self.examples)

    def _load_content(self, content):
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
                        [prompt], add_special_tokens=False, return_tensors="pt",
                    )
                    image_count += cur_image_num
                else:
                    result = self.processor(
                        None, [prompt], add_special_tokens=False, return_tensors="pt",
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
            pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else None
            image_grid_thw = torch.cat(image_grid_thw, dim=0) if image_grid_thw else None
        else:
            input_ids = torch.zeros((0,), dtype=torch.int64)
            labels = torch.full((0,), -100, dtype=torch.int64)
            attention_mask = torch.zeros((0,), dtype=torch.int64)
            pixel_values = None
            image_grid_thw = None

        # Truncate
        if input_ids.size(0) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

            if image_grid_thw is not None and pixel_values is not None:
                image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
                merge_size = self.processor.image_processor.merge_size
                tokens_per_image = []
                patches_per_image = []
                for i in range(image_grid_thw.shape[0]):
                    t, h, w = image_grid_thw[i].tolist()
                    tokens_per_image.append(int(t * (h // merge_size) * (w // merge_size)))
                    patches_per_image.append(int(t * h * w))
                total_actual = (input_ids == image_token_id).sum().item()
                total_expected = sum(tokens_per_image)
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

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "sample_weight": torch.tensor(self.weights[index], dtype=torch.float32),
            "is_desirable": torch.tensor(self.is_desirable[index], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# KTO Collate (extends mcts_collate_fn)
# ---------------------------------------------------------------------------

def kto_collate_fn(features):
    """Collate that handles is_desirable in addition to standard fields."""
    has_desirable = "is_desirable" in features[0]
    desirable_vals = []
    if has_desirable:
        for f in features:
            desirable_vals.append(f.pop("is_desirable"))

    batch = mcts_collate_fn(features)

    if has_desirable:
        batch["is_desirable"] = torch.stack(desirable_vals)

    return batch


# ---------------------------------------------------------------------------
# KTO Trainer
# ---------------------------------------------------------------------------

def _gather_log_probs(logits, labels):
    """Gather log probabilities of the target tokens."""
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return gathered


class KTOTrainer(Trainer):
    """KTO Trainer with ref model on GPU. Sequential forward: ref (no_grad) then policy."""

    def __init__(self, *args, ref_model=None, kto_beta=0.1,
                 lambda_d=1.0, lambda_u=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.kto_beta = kto_beta
        self.lambda_d = lambda_d
        self.lambda_u = lambda_u

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        is_desirable = inputs.pop("is_desirable")
        sample_weights = inputs.pop("sample_weight")

        labels = inputs["labels"]

        # ---- Reference forward (no grad) ----
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits
            shift_ref = ref_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = (shift_labels != -100).float()
            # SUM of log-probs (matches TRL)
            ref_logps = (_gather_log_probs(shift_ref, shift_labels) * mask).sum(dim=1)
            del ref_outputs, ref_logits, shift_ref

        # ---- Policy forward ----
        outputs = model(**inputs)
        policy_logits = outputs.logits
        shift_policy = policy_logits[..., :-1, :].contiguous()
        policy_logps = (_gather_log_probs(shift_policy, shift_labels) * mask).sum(dim=1)

        # Log ratio = sum of per-token log(pi/ref)
        log_ratio = policy_logps - ref_logps  # [B]

        # ---- APO-zero-unpaired loss (TRL variant, no KL baseline needed) ----
        # Works correctly with batch_size=1. No feedback loop risk.
        # Desirable: push log_ratio UP (increase policy prob vs ref)
        # Undesirable: push log_ratio DOWN (decrease policy prob vs ref)
        desirable_mask = is_desirable.to(policy_logps.device) > 0.5
        undesirable_mask = ~desirable_mask
        sample_weights = sample_weights.to(policy_logps.device)

        all_losses = []

        if desirable_mask.any():
            chosen_logratios = log_ratio[desirable_mask]
            w_d = sample_weights[desirable_mask]
            # 1 - sigmoid(beta * log_ratio) → loss decreases as policy improves on good actions
            chosen_losses = self.lambda_d * w_d * (1 - F.sigmoid(self.kto_beta * chosen_logratios))
            all_losses.append(chosen_losses)

        if undesirable_mask.any():
            rejected_logratios = log_ratio[undesirable_mask]
            w_u = sample_weights[undesirable_mask]
            # sigmoid(beta * log_ratio) → loss decreases as policy reduces prob on bad actions
            rejected_losses = self.lambda_u * w_u * F.sigmoid(self.kto_beta * rejected_logratios)
            all_losses.append(rejected_losses)

        if all_losses:
            # TRL-style: concatenate all losses, then nanmean over the combined tensor
            loss = torch.cat(all_losses, dim=0).nanmean()
        else:
            loss = torch.tensor(0.0, device=policy_logps.device, requires_grad=True)

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="MCTS KTO Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--tree_dir", type=str, default=None)
    parser.add_argument("--task_index_path", type=str, default=None)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--sr_path", type=str, default=None)
    parser.add_argument("--negatives_path", type=str, default=None)
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
    parser.add_argument("--kto_beta", type=float, default=None)
    parser.add_argument("--difficulty_beta", type=float, default=None)
    parser.add_argument("--max_step_ratio", type=float, default=None)
    parser.add_argument("--freeze_vision_tower", action="store_true", default=None)
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--auto_balance", action="store_true", default=None)
    parser.add_argument("--ref_logps_path", type=str, default=None)
    cli_args = parser.parse_args()

    defaults = {
        "model_path": "checkpoints/mcts_sft_v2.1/beta05_2e-6/epoch_1",
        "ref_model_path": "checkpoints/mcts_sft_v2.1/beta05_2e-6/epoch_1",
        "tree_dir": "checkpoints/mcts_trajectories_v2/combined_all/trees",
        "task_index_path": "checkpoints/mcts_trajectories_v2/combined_all/task_index.json",
        "mask_path": "checkpoints/mcts_trajectories_v2/combined_all/step_masks_v2.json",
        "sr_path": "checkpoints/mcts_trajectories_v2/combined_all/mcts_success.jsonl",
        "negatives_path": "checkpoints/mcts_kto/kto_negatives_phase1.jsonl",
        "output_dir": "checkpoints/mcts_kto/beta01_1e-6",
        "num_epochs": 1,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-6,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "max_length": 12000,
        "limit_images": 3,
        "kto_beta": 0.1,
        "difficulty_beta": 0.5,
        "max_step_ratio": 2.0,
        "freeze_vision_tower": True,
        "bf16": True,
        "logging_steps": 1,
        "auto_balance": True,
    }

    yaml_config = {}
    if cli_args.config:
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
    final.run_name = cli_args.run_name or yaml_config.get("run_name", None)
    final.ref_logps_path = cli_args.ref_logps_path or yaml_config.get("ref_logps_path", None)
    return final


def main():
    args = _parse_args()

    print(f"=== MCTS KTO Training ===")
    print(f"  kto_beta={args.kto_beta}, lr={args.learning_rate}")
    print(f"  policy: {args.model_path}")
    print(f"  ref:    {args.ref_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True,
        max_pixels=2116800, min_pixels=256,
    )

    print("Loading policy model...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
        device_map="cpu", low_cpu_mem_usage=True,
    )

    print("Loading reference model...")
    ref_model = AutoModelForVision2Seq.from_pretrained(
        args.ref_model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
        device_map="cpu", low_cpu_mem_usage=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    if args.freeze_vision_tower:
        print("Freezing vision tower (policy)...")
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Policy: Total={total_params:,}, Trainable={trainable_params:,}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    dataset = MCTSKTODataset(
        tree_dir=args.tree_dir,
        task_index_path=args.task_index_path,
        mask_path=args.mask_path,
        sr_path=args.sr_path,
        negatives_path=args.negatives_path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=args.max_length,
        limit_images=args.limit_images,
        beta=args.difficulty_beta,
        max_step_ratio=args.max_step_ratio,
        auto_balance=args.auto_balance,
    )

    n_gpus = max(1, torch.cuda.device_count())
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * n_gpus
    steps_per_epoch = len(dataset) // effective_batch
    print(f"  Dataset: {len(dataset)} examples (pos={dataset.n_pos}, neg={dataset.n_neg})")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Steps/epoch: {steps_per_epoch}")

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
        run_name=args.run_name,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer"],
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
        },
    )

    args_dict = {k: getattr(args, k) for k in vars(args) if not k.startswith("_")}
    dataset_stats = {
        "n_positives": dataset.n_pos,
        "n_negatives": dataset.n_neg,
        "n_tasks": dataset.num_tasks,
        "kto_beta": args.kto_beta,
        "lambda_d": dataset.lambda_d,
        "lambda_u": dataset.lambda_u,
    }

    epoch_callback = EpochSaveCallback(
        output_dir=args.output_dir,
        processor=processor,
        args_dict=args_dict,
        dataset_stats=dataset_stats,
    )

    trainer = KTOTrainer(
        model=model,
        ref_model=ref_model,
        kto_beta=args.kto_beta,
        lambda_d=dataset.lambda_d,
        lambda_u=dataset.lambda_u,
        args=training_args,
        train_dataset=dataset,
        data_collator=kto_collate_fn,
        processing_class=tokenizer,
        callbacks=[epoch_callback],
    )
    epoch_callback.trainer = trainer

    # Wrap ref_model in FSDP manually to shard across GPUs (same as policy)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import torch.distributed as dist
    import functools

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # Find the decoder layer class for wrapping
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
        auto_wrap = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2_5_VLDecoderLayer},
        )
        ref_model = FSDP(
            ref_model.to(f"cuda:{local_rank}"),
            auto_wrap_policy=auto_wrap,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            device_id=local_rank,
        )
    else:
        ref_model = ref_model.cuda()
    trainer.ref_model = ref_model

    print("Starting MCTS KTO training...")
    trainer.train()
    print("Done!")


if __name__ == "__main__":
    main()
