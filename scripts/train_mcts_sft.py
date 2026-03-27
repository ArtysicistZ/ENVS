"""
MCTS SFT Training Script — Step-Masked, Per-Task Balanced.

Trains a Qwen2.5-VL model on MCTS-collected GUI agent trajectories with:
  1. Step-level loss masking: REMOVE steps (mask=0) are skipped as training
     examples but remain visible as conversation context in later steps.
  2. Per-task balancing: Gradient mass per task ∝ n_t^α (default α=0.5,
     square root sampling) with per-trajectory length normalization.

Two balancing strategies (--balance_method):
  - "resample" (default, recommended): Task-balanced resampling at the
    dataloader level with UniMax-style repetition capping.
    Ref: An et al. (ICLR 2021) — resampling > reweighting under SGD.
  - "loss_weight": Per-sample loss weighting w = n_t^(α−1) / L_tj.
    Equivalent in expectation but less robust (Byrd & Lipton, ICML 2019).

Data inputs:
  - Trajectory JSONL: one episode per line with task_id, node_id, steps[], etc.
  - Step masks JSON: {"{task_id}:{node_id}": [0|1, ...]} per trajectory.

Usage:
    torchrun --nproc_per_node=8 scripts/train_mcts_sft.py \
        --config configs/mcts_sft.yaml

    Override any config param via CLI:
    torchrun --nproc_per_node=8 scripts/train_mcts_sft.py \
        --config configs/mcts_sft.yaml --alpha 0.3 --num_epochs 3
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
from typing import Any, Dict, List

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

# Qwen2-VL specific
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    from qwen_vl_utils.vision_process import process_vision_info

DEFAULT_IMAGE_TOKEN = "<|image_pad|>"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MCTSSFTDataset(Dataset):
    """Dataset that loads MCTS trajectories with step masks.

    - REMOVE steps (mask=0) are skipped entirely — no SFT example generated.
      They still appear as conversation context in later KEEP steps.
    - In resample mode: builds a resampled index mapping so __len__ reflects
      the resampled epoch size. The Trainer's own DistributedSampler then
      handles sharding correctly.
    - In loss_weight mode: computes per-sample loss weights.
    """

    def __init__(
        self,
        data_path: str,
        mask_path: str,
        tokenizer,
        processor,
        max_length: int = 12000,
        limit_images: int = 3,
        alpha: float = 0.5,
        compute_weights: bool = False,
        resample: bool = False,
        max_traj_repeat: int = 3,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.alpha = alpha

        # Load trajectories
        print(f"Loading trajectories from {data_path}...")
        episodes = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                ep["limit_images"] = limit_images
                episodes.append(ep)

        # Load step masks
        print(f"Loading step masks from {mask_path}...")
        with open(mask_path) as f:
            masks = json.load(f)

        # Count N_t (trajectories per task)
        self.task_traj_counts = Counter()
        for ep in episodes:
            self.task_traj_counts[ep["task_id"]] += 1

        # Expand episodes, filter REMOVE steps, store metadata
        self.examples = []
        self.example_task_ids = []   # task_id per example
        self.example_traj_keys = []  # "task_id:node_id" per example
        self.traj_keep_counts = Counter()  # L_tj (KEEP steps) per trajectory
        n_removed = 0
        n_kept = 0
        n_missing_mask = 0

        for ep in episodes:
            task_id = ep["task_id"]
            node_id = ep.get("node_id", "")
            mask_key = f"{task_id}:{node_id}"

            if mask_key not in masks:
                n_missing_mask += 1
                step_mask = [1] * len(ep["steps"])
            else:
                step_mask = masks[mask_key]

            if len(step_mask) != len(ep["steps"]):
                print(f"  WARNING: mask length mismatch for {mask_key}: "
                      f"mask={len(step_mask)}, steps={len(ep['steps'])}. Skipping.")
                continue

            # Expand all steps
            expanded = expand_episode(ep, train_all_steps=True)

            for ex in expanded:
                step_idx = ex["step_idx"]
                if step_mask[step_idx] == 1:  # KEEP
                    self.examples.append(ex)
                    self.example_task_ids.append(task_id)
                    self.example_traj_keys.append(mask_key)
                    self.traj_keep_counts[mask_key] += 1
                    n_kept += 1
                else:  # REMOVE
                    n_removed += 1

        # Compute per-sample loss weights (for loss_weight mode)
        self.weights = None
        if compute_weights:
            self._compute_weights()

        # Stats
        self.num_trajectories = len(episodes)
        task_ids = set(self.example_task_ids)
        self.num_tasks = len(task_ids)

        print(f"  {len(episodes)} trajectories, {self.num_tasks} tasks")
        print(f"  {n_kept} KEEP steps -> SFT examples")
        print(f"  {n_removed} REMOVE steps (skipped, visible as context)")
        if n_missing_mask > 0:
            print(f"  {n_missing_mask} trajectories missing mask (used all-KEEP)")

        # Print gradient mass distribution
        self._print_gradient_summary()

        # Build resampled index mapping (resample mode)
        self._index_map = None
        if resample:
            self._build_resample(max_traj_repeat)

    def _compute_weights(self):
        """Compute per-sample loss weights: w = n_t^(α−1) / L_tj, normalized to mean=1."""
        raw_weights = []
        for i in range(len(self.examples)):
            n_t = self.task_traj_counts[self.example_task_ids[i]]
            L_tj = self.traj_keep_counts[self.example_traj_keys[i]]
            raw_weights.append(n_t ** (self.alpha - 1) / L_tj)

        mean_w = sum(raw_weights) / len(raw_weights)
        self.weights = [w / mean_w for w in raw_weights]
        print(f"  Loss weights: alpha={self.alpha}, "
              f"range=[{min(self.weights):.4f}, {max(self.weights):.4f}], mean=1.0")

    def _print_gradient_summary(self):
        """Print per-task gradient mass distribution under the chosen alpha."""
        # Total gradient per task ∝ n_t^α (regardless of balance method)
        task_grad = {}
        for tid in set(self.example_task_ids):
            n_t = self.task_traj_counts[tid]
            task_grad[tid] = n_t ** self.alpha
        total_g = sum(task_grad.values())
        sorted_tasks = sorted(task_grad.items(), key=lambda x: x[1], reverse=True)
        top10 = sum(g for _, g in sorted_tasks[:10])
        bot10 = sum(g for _, g in sorted_tasks[-10:])
        n_rare = sum(1 for tid, n in self.task_traj_counts.items() if n <= 3)
        rare_g = sum(g for tid, g in task_grad.items() if self.task_traj_counts[tid] <= 3)
        print(f"  Gradient mass (alpha={self.alpha}): "
              f"top10={100*top10/total_g:.1f}%, bot10={100*bot10/total_g:.1f}%, "
              f"rare(n<=3, {n_rare} tasks)={100*rare_g/total_g:.1f}%")

    def _build_resample(self, max_traj_repeat, seed=42):
        """Build resampled index mapping with UniMax capping.

        Creates self._index_map so that __len__ returns the resampled size
        and __getitem__ maps through it. The Trainer's own DistributedSampler
        then correctly computes training steps.
        """
        g = torch.Generator()
        g.manual_seed(seed)
        n = len(self.examples)

        # Per-example sampling weight: w = n_t^(α−1) / L_tj
        weights = []
        for i in range(n):
            n_t = self.task_traj_counts[self.example_task_ids[i]]
            L_tj = self.traj_keep_counts[self.example_traj_keys[i]]
            weights.append(n_t ** (self.alpha - 1) / L_tj)
        total = sum(weights)
        w_tensor = torch.tensor([w / total for w in weights], dtype=torch.float64)

        # Sample n indices with replacement
        indices = torch.multinomial(w_tensor, n, replacement=True, generator=g)

        # UniMax capping: each trajectory appears at most K * L_tj times
        traj_L = dict(self.traj_keep_counts)
        traj_counts = Counter()
        capped = []
        for idx in indices.tolist():
            tk = self.example_traj_keys[idx]
            cap = max_traj_repeat * traj_L[tk]
            if traj_counts[tk] < cap:
                capped.append(idx)
                traj_counts[tk] += 1

        # Fill deficit by re-sampling from uncapped trajectories
        if len(capped) < n:
            deficit = n - len(capped)
            mask = torch.ones_like(w_tensor)
            capped_trajs = {tk for tk, c in traj_counts.items()
                           if c >= max_traj_repeat * traj_L[tk]}
            for i in range(n):
                if self.example_traj_keys[i] in capped_trajs:
                    mask[i] = 0.0
            masked = w_tensor * mask
            s = masked.sum()
            if s > 0:
                masked = masked / s
                extra = torch.multinomial(masked, deficit, replacement=True, generator=g)
                capped.extend(extra.tolist())
            else:
                capped.extend(capped[:deficit])
        capped = capped[:n]

        self._index_map = capped
        n_capped_trajs = sum(1 for tk, c in traj_counts.items()
                            if c >= max_traj_repeat * traj_L[tk])
        print(f"  Resampled: {len(self._index_map)} examples, "
              f"{n_capped_trajs} trajectories capped (K={max_traj_repeat})")

    def __len__(self):
        return len(self._index_map) if self._index_map is not None else len(self.examples)

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
        # Map through resampled indices if in resample mode
        if self._index_map is not None:
            index = self._index_map[index]
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
        }

        # Include sample_weight only in loss_weight mode
        if self.weights is not None:
            result["sample_weight"] = torch.tensor(self.weights[index], dtype=torch.float32)

        return result


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def mcts_collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for multimodal SFT batching. Handles optional sample_weight."""
    max_len = max(f["input_ids"].size(0) for f in features)

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    batch_pixel_values = []
    batch_image_grid_thw = []
    batch_weights = []
    has_weights = "sample_weight" in features[0]

    for f in features:
        seq_len = f["input_ids"].size(0)
        pad_len = max_len - seq_len

        if pad_len > 0:
            batch_input_ids.append(
                torch.cat([f["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
            )
            batch_labels.append(
                torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )
            batch_attention_mask.append(
                torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
        else:
            batch_input_ids.append(f["input_ids"])
            batch_labels.append(f["labels"])
            batch_attention_mask.append(f["attention_mask"])

        if f["pixel_values"] is not None:
            batch_pixel_values.append(f["pixel_values"])
        if f["image_grid_thw"] is not None:
            batch_image_grid_thw.append(f["image_grid_thw"])
        if has_weights:
            batch_weights.append(f["sample_weight"])

    result = {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
        "attention_mask": torch.stack(batch_attention_mask),
    }

    if batch_pixel_values:
        result["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
    if batch_image_grid_thw:
        result["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)
    if has_weights:
        result["sample_weight"] = torch.stack(batch_weights)

    return result


# ---------------------------------------------------------------------------
# Trainer variants
# ---------------------------------------------------------------------------

class MCTSTrainer(Trainer):
    """Trainer with optional per-sample loss weighting for loss_weight mode."""

    def __init__(self, *args, use_loss_weights=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_loss_weights = use_loss_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self._use_loss_weights and "sample_weight" in inputs:
            return self._weighted_loss(model, inputs, return_outputs)
        # Standard loss (for resampling mode — all samples equally weighted)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def _weighted_loss(self, model, inputs, return_outputs=False):
        """Per-sample weighted cross-entropy for loss_weight mode."""
        sample_weights = inputs.pop("sample_weight")

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        batch_size, seq_len, vocab_size = shift_logits.shape
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(batch_size, seq_len)

        mask = (shift_labels != -100).float()
        tokens_per_sample = mask.sum(dim=1).clamp(min=1)
        per_sample_loss = (per_token_loss * mask).sum(dim=1) / tokens_per_sample

        sample_weights = sample_weights.to(per_sample_loss.device)
        # .mean() so weights actually scale gradients (weights are pre-normalized to mean=1).
        # The old formula (.sum() / weights.sum()) cancels weights when batch_size=1.
        loss = (per_sample_loss * sample_weights).mean()

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class StopAfterEpochCallback(TrainerCallback):
    def __init__(self, stop_epoch: int):
        self.stop_epoch = stop_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if int(state.epoch) >= self.stop_epoch:
            control.should_training_stop = True


class EpochSaveCallback(TrainerCallback):
    """Saves model checkpoint and records metrics at the end of each epoch."""

    def __init__(self, output_dir, processor, args_dict, dataset_stats,
                 no_save_model=False, epoch_offset=0):
        self.output_dir = output_dir
        self.processor = processor
        self.trainer = None
        self.no_save_model = no_save_model
        self.epoch_offset = epoch_offset
        self._step_losses = []
        self.epoch_start_time = None

        self.log_path = os.path.join(output_dir, "training_log.json")
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path) as f:
                    self.training_log = json.load(f)
                print(f"[EpochSaveCallback] Loaded existing log with "
                      f"{len(self.training_log.get('epochs', []))} epochs")
            except (json.JSONDecodeError, IOError):
                self.training_log = None

        if not hasattr(self, 'training_log') or self.training_log is None:
            self.training_log = {
                "experiment": os.path.basename(self.output_dir),
                "start_time": datetime.datetime.now().isoformat(),
                "config": args_dict,
                "dataset": dataset_stats,
                "epochs": [],
            }

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        self._step_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._step_losses.append(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) + self.epoch_offset
        epoch_wall_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        if self.no_save_model:
            if state.is_world_process_zero:
                print(f"\n[EpochSaveCallback] Skipping model save (--no_save_model)")
        else:
            if self.trainer is not None:
                self.trainer.save_model(epoch_dir)
            if state.is_world_process_zero:
                self.processor.save_pretrained(epoch_dir)
                print(f"\n[EpochSaveCallback] Saved model to {epoch_dir}")

        if state.is_world_process_zero:
            avg_loss = sum(self._step_losses) / len(self._step_losses) if self._step_losses else None
            epoch_data = {
                "epoch": epoch,
                "checkpoint_path": os.path.abspath(epoch_dir),
                "training_metrics": {
                    "avg_loss": avg_loss,
                    "loss_values": self._step_losses[:],
                    "learning_rate": state.log_history[-1].get("learning_rate") if state.log_history else None,
                    "grad_norm": state.log_history[-1].get("grad_norm") if state.log_history else None,
                    "global_step": state.global_step,
                    "num_steps_this_epoch": len(self._step_losses),
                    "wall_time_seconds": round(epoch_wall_time, 1),
                },
                "eval_results": None,
                "eval_summary": None,
            }
            existing_idx = next(
                (i for i, e in enumerate(self.training_log["epochs"]) if e["epoch"] == epoch),
                None,
            )
            if existing_idx is not None:
                self.training_log["epochs"][existing_idx] = epoch_data
            else:
                self.training_log["epochs"].append(epoch_data)

            tmp_path = self.log_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(self.training_log, f, indent=2)
            os.replace(tmp_path, self.log_path)
            loss_str = f"avg_loss={avg_loss:.4f}" if avg_loss is not None else "avg_loss=N/A"
            print(f"[EpochSaveCallback] Epoch {epoch} — {loss_str}, "
                  f"wall_time={epoch_wall_time:.0f}s")

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_log["end_time"] = datetime.datetime.now().isoformat()
            total_time = None
            try:
                start = datetime.datetime.fromisoformat(self.training_log["start_time"])
                end = datetime.datetime.fromisoformat(self.training_log["end_time"])
                total_time = (end - start).total_seconds()
            except Exception:
                pass
            self.training_log["total_training_time_seconds"] = total_time

            tmp_path = self.log_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(self.training_log, f, indent=2)
            os.replace(tmp_path, self.log_path)
            print(f"[EpochSaveCallback] Training complete. Log saved to {self.log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse YAML config + CLI overrides. CLI args take precedence over YAML."""
    parser = argparse.ArgumentParser(
        description="MCTS SFT Training — Step-Masked, Per-Task Balanced")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (e.g. configs/mcts_sft.yaml)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--mask_path", type=str, default=None)
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
    parser.add_argument("--alpha", type=float, default=None,
                        help="Sampling exponent for per-task balancing. "
                             "Gradient mass per task ∝ n_t^α. "
                             "0.5=√n (default), 0.3=XLM-R, 1.0=natural, 0.0=task-uniform")
    parser.add_argument("--balance_method", type=str, default=None,
                        choices=["resample", "loss_weight"],
                        help="'resample' (default, recommended): task-balanced resampling "
                             "with UniMax capping. 'loss_weight': per-sample loss weighting.")
    parser.add_argument("--max_traj_repeat", type=int, default=None,
                        help="UniMax cap: max trajectory appearances per epoch (resample mode)")
    parser.add_argument("--freeze_vision_tower", action="store_true", default=None)
    parser.add_argument("--no_freeze_vision_tower", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--stop_after_epoch", type=int, default=None,
                        help="Stop after this epoch (for per-epoch eval)")
    parser.add_argument("--fsdp_offload", action="store_true", default=None)
    parser.add_argument("--no_save_model", action="store_true", default=None)
    parser.add_argument("--epoch_offset", type=int, default=None)
    cli_args = parser.parse_args()

    # Defaults (used if neither YAML nor CLI provides a value)
    defaults = {
        "model_path": "ByteDance-Seed/UI-TARS-1.5-7B",
        "data_path": "checkpoints/mcts_trajectories/combined/mcts_success.jsonl",
        "mask_path": "checkpoints/mcts_trajectories/combined/step_masks_train.json",
        "output_dir": "checkpoints/mcts_sft",
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
        "alpha": 0.5,
        "balance_method": "resample",
        "max_traj_repeat": 3,
        "freeze_vision_tower": True,
        "bf16": True,
        "logging_steps": 1,
        "fsdp_offload": False,
        "no_save_model": False,
        "epoch_offset": 0,
    }

    # Load YAML config if provided
    yaml_config = {}
    if cli_args.config:
        print(f"Loading config from {cli_args.config}...")
        with open(cli_args.config) as f:
            yaml_config = yaml.safe_load(f) or {}

    # Merge: CLI > YAML > defaults
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

    # Pass-through args that have no YAML equivalent
    final.local_rank = cli_args.local_rank
    final.resume_from_checkpoint = cli_args.resume_from_checkpoint
    final.stop_after_epoch = cli_args.stop_after_epoch
    final.config = cli_args.config

    # Handle --no_freeze_vision_tower override
    if cli_args.no_freeze_vision_tower:
        final.freeze_vision_tower = False

    return final


def main():
    args = _parse_args()

    if args.no_save_model and args.epoch_offset > 0:
        raise ValueError("--no_save_model is incompatible with multi-epoch training (epoch_offset > 0)")

    use_loss_weights = (args.balance_method == "loss_weight")
    print(f"Balance method: {args.balance_method} (alpha={args.alpha})")
    if args.balance_method == "resample":
        print(f"  UniMax cap: K={args.max_traj_repeat} per trajectory per epoch")

    # Load model, tokenizer, processor
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        max_pixels=2116800,
        min_pixels=256,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    if args.freeze_vision_tower:
        print("Freezing vision tower...")
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    use_resample = (args.balance_method == "resample")
    dataset = MCTSSFTDataset(
        data_path=args.data_path,
        mask_path=args.mask_path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=args.max_length,
        limit_images=args.limit_images,
        alpha=args.alpha,
        compute_weights=use_loss_weights,
        resample=use_resample,
        max_traj_repeat=args.max_traj_repeat,
    )

    n_gpus = max(1, torch.cuda.device_count())
    effective_batch_size = args.per_device_batch_size * args.gradient_accumulation_steps * n_gpus
    steps_per_epoch = len(dataset) // effective_batch_size
    total_steps = steps_per_epoch * args.num_epochs

    print(f"  Effective batch size: {effective_batch_size} "
          f"({args.per_device_batch_size} x {args.gradient_accumulation_steps} x {n_gpus} GPUs)")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")

    args_dict = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "mask_path": args.mask_path,
        "output_dir": args.output_dir,
        "num_epochs": args.num_epochs,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "max_length": args.max_length,
        "limit_images": args.limit_images,
        "alpha": args.alpha,
        "balance_method": args.balance_method,
        "max_traj_repeat": args.max_traj_repeat,
        "freeze_vision_tower": args.freeze_vision_tower,
        "bf16": args.bf16,
        "n_gpus": n_gpus,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "fsdp": "full_shard auto_wrap offload" if args.fsdp_offload else "full_shard auto_wrap",
    }
    dataset_stats = {
        "num_trajectories": dataset.num_trajectories,
        "num_sft_examples": len(dataset),
        "num_unique_tasks": dataset.num_tasks,
        "alpha": args.alpha,
        "balance_method": args.balance_method,
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
        print(f"  Will stop after epoch {args.stop_after_epoch}")

    trainer = MCTSTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=mcts_collate_fn,
        processing_class=tokenizer,
        callbacks=callbacks,
        use_loss_weights=use_loss_weights,
    )
    epoch_callback.trainer = trainer

    print("Starting MCTS SFT training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("Done!")


if __name__ == "__main__":
    main()
