"""
SFT Training Script for GUI Agent Trajectories.

Trains a Qwen2.5-VL model on successful trajectory demonstrations
using standard supervised fine-tuning (cross-entropy on assistant tokens).

Saves a clean model checkpoint after each epoch at {output_dir}/epoch_{N}/
and writes per-epoch training metrics to {output_dir}/training_log.json.

Usage:
    torchrun --nproc_per_node=8 scripts/train_sft.py \
        --model_path ByteDance-Seed/UI-TARS-1.5-7B \
        --data_path checkpoints/arpo-inference/sft_86tasks_trajectories.jsonl \
        --output_dir checkpoints/sft_86tasks \
        --num_epochs 5 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-5
"""

import argparse
import copy
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch
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


class SFTDataset(Dataset):
    """Dataset that loads trajectory JSONL, expands to per-step SFT examples,
    and tokenizes each example with the VLM processor."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        processor,
        max_length: int = 12000,
        limit_images: int = 3,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

        # Load and expand all trajectories
        print(f"Loading trajectories from {data_path}...")
        # Override limit_images in each episode before expansion
        episodes = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                ep["limit_images"] = limit_images
                episodes.append(ep)

        self.examples = []
        for ep in episodes:
            self.examples.extend(expand_episode(ep, train_all_steps=True))

        print(f"  {len(episodes)} trajectories -> {len(self.examples)} SFT examples")

        # Stats
        self.num_trajectories = len(episodes)
        task_ids = set(ex["task_id"] for ex in self.examples)
        self.num_tasks = len(task_ids)
        steps = [ex["total_steps"] for ex in self.examples]
        self.avg_steps = sum(steps) / len(steps) if steps else 0
        print(f"  {self.num_tasks} unique tasks")
        print(f"  avg steps/trajectory: {self.avg_steps:.1f}")

    def __len__(self):
        return len(self.examples)

    def _load_content(self, content):
        """Convert message content to text, replacing images with placeholder tokens."""
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

        # Extract images from message
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

                # SFT label masking: only train on assistant responses
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
            # No images (shouldn't happen for GUI agent, but handle gracefully)
            input_ids = torch.zeros((0,), dtype=torch.int64)
            labels = torch.full((0,), -100, dtype=torch.int64)
            attention_mask = torch.zeros((0,), dtype=torch.int64)
            pixel_values = None
            image_grid_thw = None

        # Truncate to max_length (right truncation — keep the beginning)
        if input_ids.size(0) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

            # After truncation, trim orphaned pixel_values/image_grid_thw
            # that lost their tokens (matching _trim_multimodal_after_truncation
            # in verl/utils/osworld.py).
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
                    # Right truncation: first images survive, last get cut
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

                    # Replace partial image tokens with pad
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
        }


def sft_collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for multimodal SFT batching.

    - Pads input_ids/labels/attention_mask to max length in batch (right-pad)
    - Concatenates pixel_values and image_grid_thw across samples
    """
    # Find max sequence length and pad token id
    max_len = max(f["input_ids"].size(0) for f in features)

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    batch_pixel_values = []
    batch_image_grid_thw = []

    for f in features:
        seq_len = f["input_ids"].size(0)
        pad_len = max_len - seq_len

        if pad_len > 0:
            # Right-pad for training (standard for causal LM; avoids MROPE position shifts)
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

    result = {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
        "attention_mask": torch.stack(batch_attention_mask),
    }

    if batch_pixel_values:
        result["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
    if batch_image_grid_thw:
        result["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)

    return result


class StopAfterEpochCallback(TrainerCallback):
    """Stops training after a target epoch.

    Used by the wrapper script to train one epoch at a time for evaluation,
    while keeping --num_epochs set to the TOTAL epochs so the LR schedule
    (cosine decay) is computed correctly over the full training run.
    """

    def __init__(self, stop_epoch: int):
        self.stop_epoch = stop_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if int(state.epoch) >= self.stop_epoch:
            control.should_training_stop = True


class EpochSaveCallback(TrainerCallback):
    """Saves a clean model checkpoint (model + tokenizer + processor) at the
    end of each epoch to a predictable path: {output_dir}/epoch_{N}/

    Also records per-epoch training metrics to {output_dir}/training_log.json.
    """

    def __init__(self, output_dir, processor, args_dict, dataset_stats, no_save_model=False, epoch_offset=0):
        self.output_dir = output_dir
        self.processor = processor
        self.trainer = None  # Set after Trainer construction
        self.no_save_model = no_save_model
        self.epoch_offset = epoch_offset
        self.epoch_metrics = []
        self.epoch_start_time = None
        # Accumulated loss between logging steps
        self._step_losses = []

        # Load existing training log if resuming, otherwise create new
        self.log_path = os.path.join(output_dir, "training_log.json")
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path) as f:
                    self.training_log = json.load(f)
                print(f"[EpochSaveCallback] Loaded existing log with {len(self.training_log.get('epochs', []))} epochs")
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

        # Save clean model copy using Trainer's save_model() which handles
        # FSDP/DeepSpeed state dict gathering across all ranks correctly.
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

        # Record epoch metrics (rank 0 only)
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
                "eval_results": None,  # Filled by wrapper script after evaluation
                "eval_summary": None,
            }
            # Replace existing epoch entry if resuming, otherwise append
            existing_idx = next(
                (i for i, e in enumerate(self.training_log["epochs"]) if e["epoch"] == epoch),
                None,
            )
            if existing_idx is not None:
                self.training_log["epochs"][existing_idx] = epoch_data
            else:
                self.training_log["epochs"].append(epoch_data)

            # Write log (atomic: write to temp then rename)
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


def main():
    parser = argparse.ArgumentParser(description="SFT Training for GUI Agent")
    parser.add_argument(
        "--model_path",
        type=str,
        default="ByteDance-Seed/UI-TARS-1.5-7B",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="checkpoints/arpo-inference/sft_86tasks_trajectories.jsonl",
        help="Path to trajectory JSONL",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/sft_86tasks",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=12000)
    parser.add_argument("--limit_images", type=int, default=3)
    parser.add_argument("--freeze_vision_tower", action="store_true", default=True)
    parser.add_argument("--no_freeze_vision_tower", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=1)
    # FSDP is always enabled (hardcoded in TrainingArguments) matching RL pipeline
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--stop_after_epoch", type=int, default=None,
                        help="Stop training after this epoch (for per-epoch eval). "
                             "Use with --num_epochs=TOTAL to keep LR schedule consistent.")
    parser.add_argument("--fsdp_offload", action="store_true", default=False,
                        help="Enable FSDP CPU offload (slower but uses less GPU memory)")
    parser.add_argument("--no_save_model", action="store_true", default=False,
                        help="Skip saving model checkpoints (still logs metrics)")
    parser.add_argument("--epoch_offset", type=int, default=0,
                        help="Offset added to epoch number for naming (e.g. 1 means epoch_1 becomes epoch_2)")
    args = parser.parse_args()

    if args.no_freeze_vision_tower:
        args.freeze_vision_tower = False

    if args.no_save_model and args.epoch_offset > 0:
        raise ValueError("--no_save_model is incompatible with multi-epoch training (epoch_offset > 0). "
                         "Each epoch needs the previous epoch's saved model.")

    # Load model, tokenizer, processor
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        max_pixels=2116800,
        min_pixels=256,
    )

    # Load model to CPU first, then let FSDP shard to GPUs (matches RL pipeline)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    # Freeze vision tower if requested (matches ARPO training config)
    if args.freeze_vision_tower:
        print("Freezing vision tower...")
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    # Ensure pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    dataset = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=args.max_length,
        limit_images=args.limit_images,
    )

    # Effective batch size
    n_gpus = max(1, torch.cuda.device_count())
    effective_batch_size = args.per_device_batch_size * args.gradient_accumulation_steps * n_gpus
    steps_per_epoch = len(dataset) // effective_batch_size
    total_steps = steps_per_epoch * args.num_epochs

    print(f"  Effective batch size: {effective_batch_size} "
          f"({args.per_device_batch_size} x {args.gradient_accumulation_steps} x {n_gpus} GPUs)")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")

    # Config dict for JSON log
    args_dict = {
        "model_path": args.model_path,
        "data_path": args.data_path,
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
        "avg_steps_per_trajectory": round(dataset.avg_steps, 2),
    }

    # Training arguments — FSDP full_shard matching RL pipeline VRAM optimization:
    #   - FSDP full_shard: shards params + gradients + optimizer across GPUs
    #   - bf16 mixed precision: params in bf16, reductions in fp32
    #   - Gradient checkpointing: use_reentrant=False (matches RL pipeline)
    #   - Flash Attention 2: O(n) memory for attention
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
        # FSDP config (full_shard + auto_wrap, optional CPU offload)
        fsdp="full_shard auto_wrap offload" if args.fsdp_offload else "full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer"],
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
        },
    )

    # Epoch callback for clean saves and JSON logging
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
        print(f"  Will stop after epoch {args.stop_after_epoch} (LR schedule computed for {args.num_epochs} total)")

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=sft_collate_fn,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    epoch_callback.trainer = trainer

    # Train (with optional resume)
    print("Starting SFT training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("Done!")


if __name__ == "__main__":
    main()
