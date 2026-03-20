#!/usr/bin/env python3
"""
Smoke test: measure action probability distribution and entropy across real OSWorld tasks.

Uses vLLM directly (single GPU) + remote env VMs to run a few tasks for a few steps,
generating K action candidates per step. Analyzes:
  - Per-step action entropy (categorical over action_type + target_bucket)
  - Distribution of action types
  - Frequency of "consensus" (all K samples agree) vs "split" (multiple strategies)
  - Token-level entropy from logprobs

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/research/smoke_action_entropy.py \
        --n-tasks 8 --n-steps 5 --n-samples 16 \
        --server-url http://10.100.4.7:15001
"""
import argparse
import base64
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

# -- Add project root to path --
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "OSWorld"))

from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

# ---- Constants ----
MODEL_PATH = "ByteDance-Seed/UI-TARS-1.5-7B"
MAX_PIXELS = 2116800
MIN_PIXELS = 256
IMAGE_FACTOR = 28
SCREEN_W, SCREEN_H = 1920, 1080

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
fail() # Use when you think the task is not feasible or cannot be completed.

## Note
- Use English in `Thought` and `Action` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}"""


# ============================================================
# Utility: parse action text into (action_type, target_bucket)
# ============================================================
COORD_BUCKET_PX = 48  # quantization grid size


def parse_action_category(text: str) -> Tuple[str, str]:
    """Parse model output into (action_type, target_bucket) for entropy computation."""
    text = text.strip()
    if "Action:" not in text:
        return ("UNPARSEABLE", "")

    action_str = text.split("Action:")[-1].strip().split("\n")[0].strip()

    # Extract function name
    m = re.match(r"(\w+)\(", action_str)
    if not m:
        return ("UNPARSEABLE", "")
    action_type = m.group(1)

    # For spatial actions, extract and bucket coordinates
    coord_match = re.search(r"start_box='.*?\((\d+),(\d+)\).*?'", action_str)
    if coord_match:
        x, y = int(coord_match.group(1)), int(coord_match.group(2))
        bx, by = x // COORD_BUCKET_PX, y // COORD_BUCKET_PX
        return (action_type, f"{bx},{by}")

    # For type/hotkey, bucket by content
    content_match = re.search(r"content='(.*?)'", action_str)
    if content_match:
        content = content_match.group(1)[:30]  # truncate for bucketing
        return (action_type, f"c:{content}")

    key_match = re.search(r"key='(.*?)'", action_str)
    if key_match:
        return (action_type, f"k:{key_match.group(1)}")

    direction_match = re.search(r"direction='(.*?)'", action_str)
    if direction_match:
        return (action_type, f"d:{direction_match.group(1)}")

    return (action_type, "")


def parse_action_with_pixel_coords(text: str) -> Tuple[str, Optional[int], Optional[int], str]:
    """Parse model output into (action_type, x_pixel, y_pixel, content_key).

    Returns pixel coordinates (from Qwen2.5-VL absolute coords) for spatial
    actions, or content/key for non-spatial actions. Used for spatial clustering.
    """
    text = text.strip()
    if "Action:" not in text:
        return ("UNPARSEABLE", None, None, "")
    action_str = text.split("Action:")[-1].strip().split("\n")[0].strip()
    m = re.match(r"(\w+)\(", action_str)
    if not m:
        return ("UNPARSEABLE", None, None, "")
    action_type = m.group(1)

    coord_match = re.search(r"start_box='.*?\((\d+),(\d+)\).*?'", action_str)
    if coord_match:
        x, y = int(coord_match.group(1)), int(coord_match.group(2))
        return (action_type, x, y, "")

    content_match = re.search(r"content='(.*?)'", action_str)
    if content_match:
        return (action_type, None, None, f"c:{content_match.group(1)[:30]}")
    key_match = re.search(r"key='(.*?)'", action_str)
    if key_match:
        return (action_type, None, None, f"k:{key_match.group(1)}")
    direction_match = re.search(r"direction='(.*?)'", action_str)
    if direction_match:
        return (action_type, None, None, f"d:{direction_match.group(1)}")
    return (action_type, None, None, "")


def compute_spatial_clusters(parsed_actions, dist_threshold=100):
    """Cluster parsed actions by type + spatial distance.

    Returns list of cluster labels (one per action).
    Two same-type actions with coords within dist_threshold pixels = same cluster.
    Different types = always different clusters.
    Non-spatial actions cluster by content key.
    """
    n = len(parsed_actions)
    labels = [-1] * n
    cluster_id = 0

    # Group indices by (action_type, content_key_for_non_spatial)
    groups = defaultdict(list)
    for i, (atype, x, y, ckey) in enumerate(parsed_actions):
        if x is not None and y is not None:
            groups[(atype, "SPATIAL")].append(i)
        else:
            groups[(atype, ckey)].append(i)

    for (atype, key), indices in groups.items():
        if key != "SPATIAL":
            # Non-spatial: all in one cluster
            for i in indices:
                labels[i] = cluster_id
            cluster_id += 1
        else:
            # Spatial: single-linkage clustering by Euclidean distance
            assigned = [False] * len(indices)
            for start_pos in range(len(indices)):
                if assigned[start_pos]:
                    continue
                assigned[start_pos] = True
                labels[indices[start_pos]] = cluster_id
                queue = [start_pos]
                while queue:
                    cur = queue.pop()
                    ci = indices[cur]
                    cx, cy = parsed_actions[ci][1], parsed_actions[ci][2]
                    for other_pos in range(len(indices)):
                        if assigned[other_pos]:
                            continue
                        oi = indices[other_pos]
                        ox, oy = parsed_actions[oi][1], parsed_actions[oi][2]
                        dist_sq = (cx - ox) ** 2 + (cy - oy) ** 2
                        if dist_sq <= dist_threshold ** 2:
                            assigned[other_pos] = True
                            labels[oi] = cluster_id
                            queue.append(other_pos)
                cluster_id += 1

    return labels


def compute_entropy(categories: List[Tuple[str, str]]) -> float:
    """Compute categorical entropy in bits."""
    counter = Counter(categories)
    total = len(categories)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_type_entropy(categories: List[Tuple[str, str]]) -> float:
    """Compute entropy over just action types (ignoring target)."""
    types = [c[0] for c in categories]
    counter = Counter(types)
    total = len(types)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_token_entropy_from_logprobs(logprobs_list) -> float:
    """Compute average token-level entropy from vLLM logprobs.

    Each element in logprobs_list is a dict mapping token_id -> Logprob.
    We use the selected token's logprob as an approximation.
    """
    if not logprobs_list:
        return 0.0
    total = 0.0
    count = 0
    for pos_logprobs in logprobs_list:
        if pos_logprobs is None:
            continue
        # The selected token's logprob gives us -log(p)
        for token_id, lp in pos_logprobs.items():
            logp = lp.logprob if hasattr(lp, 'logprob') else float(lp)
            # entropy contribution ≈ -p * log2(p) = -exp(logp) * logp / ln(2)
            p = math.exp(logp)
            if p > 0:
                total += -p * logp / math.log(2)
            count += 1
            break  # only the selected token
    return total / max(count, 1)


# ============================================================
# Remote Env interaction (direct HTTP, no Ray)
# ============================================================
class SimpleRemoteEnv:
    """Minimal remote env client for the smoke test."""

    def __init__(self, server_url: str, slot_id: int):
        self.server_url = server_url.rstrip("/")
        self.slot_id = slot_id
        self.history_messages = []
        self.is_done = False
        self.instruction = ""

    def _post(self, path, body, timeout=300):
        r = requests.post(f"{self.server_url}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def reset(self, task_config):
        self.is_done = False
        resp = self._post("/env/reset", {"task_config": task_config, "slot_id": self.slot_id}, timeout=600)
        self.is_done = resp.get("is_done", True)
        obs_wire = resp.get("obs_messages")
        if obs_wire:
            self.history_messages = self._wire_to_messages(obs_wire)
        else:
            self.history_messages = []
            self.is_done = True
        self.instruction = task_config.get("instruction", "")
        return not self.is_done

    def step(self, prediction):
        resp = self._post("/env/step", {"prediction": prediction, "slot_id": self.slot_id}, timeout=300)
        self.is_done = resp.get("is_done", True)
        obs_wire = resp.get("obs_messages")
        if obs_wire:
            self.history_messages = self._wire_to_messages(obs_wire)
        return not self.is_done

    @staticmethod
    def _wire_to_messages(wire):
        out = []
        for m in wire:
            content = []
            for c in m.get("content", []):
                if c.get("type") == "image" and "b64" in c:
                    content.append({
                        "type": "image",
                        "image": "data:image/jpeg;base64," + c["b64"],
                        "min_pixels": c.get("min_pixels", MIN_PIXELS),
                        "max_pixels": c.get("max_pixels", MAX_PIXELS),
                    })
                else:
                    content.append(c)
            out.append({"role": m["role"], "content": content})
        return out


# ============================================================
# vLLM-based multi-sample action generation
# ============================================================
LIMIT_IMAGES = 3  # must match vLLM limit_mm_per_prompt


def _truncate_to_last_n_images(messages, max_images):
    """Keep only the last max_images images in messages (drop older ones)."""
    import copy
    messages = copy.deepcopy(messages)
    image_positions = []
    for mi, msg in enumerate(messages):
        content = msg.get("content") or []
        if not isinstance(content, list):
            content = [content]
        for ci, c in enumerate(content):
            if isinstance(c, dict) and "image" in c:
                image_positions.append((mi, ci))
    if len(image_positions) <= max_images:
        return messages
    drop_set = set(image_positions[: len(image_positions) - max_images])
    out = []
    for mi, msg in enumerate(messages):
        content = msg.get("content") or []
        if not isinstance(content, list):
            content = [content]
        new_content = [
            c for ci, c in enumerate(content)
            if not (isinstance(c, dict) and "image" in c and (mi, ci) in drop_set)
        ]
        if new_content:
            out.append({**msg, "content": new_content})
    return out


def prepare_vllm_input(messages, processor, tokenizer):
    """Prepare a single vLLM input dict from conversation messages.

    Follows the exact same path as OSWorldDataset.__getitem__ + vllm_rollout_spmd.

    Key issue: the HF processor uses max_pixels from the message (2116800) to
    determine image grid → image_pad token count. But vLLM uses the model's
    default max_pixels (12845056). To fix this, we call the HF processor to get
    the actual input_ids (with correct image_pad count) AND to process images
    at the correct resolution. We then pass the processor's pixel_values to
    vLLM as pre-processed multi_modal_data so vLLM doesn't re-process them.

    Since vLLM's multi_modal_data expects PIL images, we instead strip the
    per-image min/max_pixels overrides from messages so both HF and vLLM use
    the model's default settings → matching token counts.
    """
    import copy
    messages = copy.deepcopy(messages)
    # Step 0: Truncate images to stay within vLLM's limit
    messages = _truncate_to_last_n_images(messages, LIMIT_IMAGES)
    # Step 1: Remove per-image min_pixels/max_pixels overrides so both HF
    # processor and vLLM use the model's default settings (12845056 max_pixels).
    # This ensures the image token count from apply_chat_template matches
    # what vLLM's image processor will produce.
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "image":
                    c.pop("min_pixels", None)
                    c.pop("max_pixels", None)
    # Step 2: processor.apply_chat_template → text with expanded image_pad tokens
    # (using model default pixel settings)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # Step 3: Extract PIL images (same call as OSWorldDataset)
    image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    # Step 4: Tokenize the prompt text (same as OSWorldDataset line 423)
    raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    vllm_input = {"prompt_token_ids": raw_prompt_ids}
    if image_inputs:
        vllm_input["multi_modal_data"] = {"image": image_inputs}
    return vllm_input


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Smoke test: action entropy distribution")
    parser.add_argument("--n-tasks", type=int, default=8, help="Number of tasks to run")
    parser.add_argument("--n-steps", type=int, default=5, help="Max steps per task")
    parser.add_argument("--n-samples", "-K", type=int, default=16, help="Action candidates per step")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--server-url", type=str, default="http://10.100.4.7:15001")
    parser.add_argument("--task-file", type=str,
                        default="OSWorld/evaluation_examples/test_all_300tasks_noproxy_softreset_clean.json")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: auto)")
    args = parser.parse_args()

    K = args.n_samples

    # ---- Load tasks ----
    task_file = os.path.join(PROJ_ROOT, args.task_file)
    with open(task_file) as f:
        raw = json.load(f)
    all_tasks = []
    base_path = os.path.dirname(task_file)
    for domain, task_ids in raw.items():
        for tid in task_ids:
            cfg_path = os.path.join(base_path, "examples", domain, tid + ".json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f2:
                    tc = json.load(f2)
                tc["domain"] = domain
                tc["id"] = tid
                tc["task_id"] = tid
                all_tasks.append(tc)
    print(f"Loaded {len(all_tasks)} tasks from {task_file}")

    # Pick a diverse subset
    import random
    random.seed(42)
    # Try to get tasks from different domains
    by_domain = defaultdict(list)
    for t in all_tasks:
        by_domain[t["domain"]].append(t)
    selected = []
    domains = sorted(by_domain.keys())
    while len(selected) < args.n_tasks and domains:
        for d in domains:
            if by_domain[d] and len(selected) < args.n_tasks:
                selected.append(by_domain[d].pop(random.randint(0, len(by_domain[d]) - 1)))
        domains = [d for d in domains if by_domain[d]]
    print(f"Selected {len(selected)} tasks from {len(set(t['domain'] for t in selected))} domains")
    for t in selected:
        print(f"  [{t['domain']}] {t['id'][:12]}  {t.get('instruction','')[:60]}")

    # ---- Load model ----
    print(f"\nLoading vLLM model: {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=12000,
        max_num_batched_tokens=14000,
        max_num_seqs=K,
        limit_mm_per_prompt={"image": LIMIT_IMAGES},
        enable_chunked_prefill=True,
        tensor_parallel_size=1,
    )
    # vLLM 0.7.3 has a bug with n>1 + Qwen2.5-VL multimodal (image token
    # count mismatch when expanding n outputs). Use n=1 and loop K times.
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=args.temperature,
        n=1,
        logprobs=1,
    )
    print("Model loaded.\n")

    # ---- Run episodes ----
    all_step_records = []  # List of per-step analysis dicts

    for task_idx, task_config in enumerate(selected):
        domain = task_config["domain"]
        task_id = task_config["id"][:12]
        instruction = task_config.get("instruction", "")
        print(f"\n{'='*70}")
        print(f"Task {task_idx+1}/{len(selected)}: [{domain}] {task_id}")
        print(f"  Instruction: {instruction[:100]}")

        env = SimpleRemoteEnv(args.server_url, slot_id=task_idx % 32)
        ok = env.reset(task_config)
        if not ok:
            print(f"  RESET FAILED, skipping.")
            continue
        print(f"  Reset OK, {len(env.history_messages)} messages in history")

        for step_idx in range(args.n_steps):
            if env.is_done:
                break

            # Build vLLM input from current history
            vllm_input = prepare_vllm_input(env.history_messages, processor, tokenizer)

            # Debug: check image_pad token count in prompt
            pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            n_pads = sum(1 for x in vllm_input["prompt_token_ids"] if x == pad_id)
            n_images = len(vllm_input.get("multi_modal_data", {}).get("image", []))
            print(f"    [debug] prompt_len={len(vllm_input['prompt_token_ids'])}, "
                  f"image_pad_tokens={n_pads}, n_pil_images={n_images}")

            # Generate K candidates (n=1 loop; vLLM 0.7.3 n>1 broken for multimodal)
            t0 = time.time()
            candidates_text = []
            all_logprobs = []
            for _k in range(K):
                _out = llm.generate([vllm_input], sampling_params=sampling_params, use_tqdm=False)
                candidates_text.append(_out[0].outputs[0].text)
                all_logprobs.append(_out[0].outputs[0].logprobs)
            gen_time = time.time() - t0

            # Parse each candidate to (action_type, target_bucket)
            categories = [parse_action_category(t) for t in candidates_text]
            action_entropy = compute_entropy(categories)
            type_entropy = compute_type_entropy(categories)

            # Spatial clustering analysis (100px threshold)
            parsed_with_coords = [parse_action_with_pixel_coords(t) for t in candidates_text]
            cluster_labels = compute_spatial_clusters(parsed_with_coords, dist_threshold=100)
            n_spatial_clusters = len(set(cluster_labels))
            # Cluster distribution
            cluster_counter = Counter(cluster_labels)
            top_cluster_frac = max(cluster_counter.values()) / K if cluster_counter else 0

            # Compute token-level entropy from logprobs
            avg_token_entropy = 0.0
            n_tokens_total = 0
            for lp in all_logprobs:
                if lp:
                    avg_token_entropy += compute_token_entropy_from_logprobs(lp)
                    n_tokens_total += 1
            if n_tokens_total > 0:
                avg_token_entropy /= n_tokens_total

            # Analyze distribution
            cat_counter = Counter(categories)
            type_counter = Counter(c[0] for c in categories)
            n_unique_cats = len(cat_counter)
            n_unique_types = len(type_counter)
            top_category = cat_counter.most_common(1)[0] if cat_counter else (("?", "?"), 0)
            top_frac = top_category[1] / K

            record = {
                "task_idx": task_idx,
                "task_id": task_config["id"],
                "domain": domain,
                "step": step_idx,
                "action_entropy_bits": round(action_entropy, 3),
                "type_entropy_bits": round(type_entropy, 3),
                "avg_token_entropy_bits": round(avg_token_entropy, 3),
                "n_unique_categories": n_unique_cats,
                "n_unique_action_types": n_unique_types,
                "top_category": f"{top_category[0][0]}({top_category[0][1]})",
                "top_fraction": round(top_frac, 2),
                "type_distribution": dict(type_counter),
                "category_distribution": {f"{k[0]}({k[1]})": v for k, v in cat_counter.most_common(10)},
                "gen_time_s": round(gen_time, 2),
                "n_samples": K,
                "n_spatial_clusters": n_spatial_clusters,
                "top_cluster_fraction": round(top_cluster_frac, 2),
                "raw_coords": [(p[0], p[1], p[2], p[3]) for p in parsed_with_coords],
            }
            all_step_records.append(record)

            # Print summary
            sc_label = f"clusters={n_spatial_clusters}" if n_spatial_clusters != n_unique_cats else f"clusters={n_spatial_clusters}(=cats)"
            print(f"  Step {step_idx}: H_act={action_entropy:.2f} H_type={type_entropy:.2f} "
                  f"cats={n_unique_cats} {sc_label} | "
                  f"top={top_category[0][0]}({top_category[0][1]}) {top_frac:.0%} | "
                  f"types={dict(type_counter)} | {gen_time:.1f}s")

            # Execute the most common action (plurality vote)
            best_cat = top_category[0]
            # Find a candidate matching the top category
            best_text = None
            for t, c in zip(candidates_text, categories):
                if c == best_cat:
                    best_text = t
                    break
            if best_text is None:
                best_text = candidates_text[0]

            # Check if it's a terminal action
            _, best_type_bucket = parse_action_category(best_text)
            best_type = parse_action_category(best_text)[0]
            if best_type in ("finished", "fail"):
                print(f"  -> Terminal action: {best_type}")
                env.is_done = True
                break

            # Step the env with the chosen action
            ok = env.step(best_text)
            if not ok:
                print(f"  -> Env step failed or done")
                break

    # ---- Analysis ----
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {len(all_step_records)} total steps across {len(selected)} tasks")
    print(f"{'='*70}\n")

    if not all_step_records:
        print("No step records collected. Check if the VMs are accessible.")
        return

    entropies = [r["action_entropy_bits"] for r in all_step_records]
    type_entropies = [r["type_entropy_bits"] for r in all_step_records]
    token_entropies = [r["avg_token_entropy_bits"] for r in all_step_records]
    unique_cats = [r["n_unique_categories"] for r in all_step_records]
    top_fracs = [r["top_fraction"] for r in all_step_records]

    def percentiles(vals, ps=[0, 10, 25, 50, 75, 90, 100]):
        s = sorted(vals)
        n = len(s)
        return {f"p{p}": s[min(int(p/100*n), n-1)] for p in ps}

    print("Action Entropy (bits) - full (action_type + target_bucket):")
    print(f"  mean={sum(entropies)/len(entropies):.3f}  {percentiles(entropies)}")

    print(f"\nAction Type Entropy (bits) - action_type only:")
    print(f"  mean={sum(type_entropies)/len(type_entropies):.3f}  {percentiles(type_entropies)}")

    print(f"\nToken-Level Entropy (bits) - avg per token:")
    print(f"  mean={sum(token_entropies)/len(token_entropies):.3f}  {percentiles(token_entropies)}")

    print(f"\nUnique Action Categories per Step (out of K={K}):")
    print(f"  mean={sum(unique_cats)/len(unique_cats):.1f}  {percentiles(unique_cats)}")

    print(f"\nTop Category Fraction (consensus strength):")
    print(f"  mean={sum(top_fracs)/len(top_fracs):.2f}  {percentiles(top_fracs)}")

    total = len(all_step_records)

    # Spatial clustering comparison
    spatial_clusters = [r["n_spatial_clusters"] for r in all_step_records]
    print(f"\nSpatial Clusters per Step (100px threshold, out of K={K}):")
    print(f"  mean={sum(spatial_clusters)/len(spatial_clusters):.1f}  {percentiles(spatial_clusters)}")

    # Compare bucket categories vs spatial clusters
    bucket_consensus = sum(1 for r in all_step_records if r["n_unique_categories"] == 1)
    spatial_consensus = sum(1 for r in all_step_records if r["n_spatial_clusters"] == 1)
    print(f"\nConsensus Comparison:")
    print(f"  48px grid bucketing:    {bucket_consensus}/{total} = {bucket_consensus/total:.1%} consensus")
    print(f"  100px spatial clusters: {spatial_consensus}/{total} = {spatial_consensus/total:.1%} consensus")
    print(f"  Noise eliminated:       {spatial_consensus - bucket_consensus} steps reclassified as consensus")

    # Steps where clustering REDUCES the branch count (coordinate noise removed)
    noise_steps = sum(1 for r in all_step_records if r["n_spatial_clusters"] < r["n_unique_categories"])
    same_steps = sum(1 for r in all_step_records if r["n_spatial_clusters"] == r["n_unique_categories"])
    print(f"  Steps where clustering reduces branches: {noise_steps}/{total} ({noise_steps/total:.1%})")
    print(f"  Steps where they agree:                  {same_steps}/{total} ({same_steps/total:.1%})")

    # Consensus vs split breakdown
    consensus_steps = sum(1 for r in all_step_records if r["n_unique_categories"] == 1)
    split_2 = sum(1 for r in all_step_records if r["n_unique_categories"] == 2)
    split_3plus = sum(1 for r in all_step_records if r["n_unique_categories"] >= 3)
    total = len(all_step_records)
    print(f"\nBranching Profile:")
    print(f"  Consensus (1 category):  {consensus_steps}/{total} = {consensus_steps/total:.1%}")
    print(f"  2-way split:             {split_2}/{total} = {split_2/total:.1%}")
    print(f"  3+ way split:            {split_3plus}/{total} = {split_3plus/total:.1%}")

    # Entropy histogram
    print(f"\nEntropy Histogram (action-level):")
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 4.0)]
    for lo, hi in bins:
        count = sum(1 for e in entropies if lo <= e < hi)
        bar = "#" * int(count / total * 50)
        print(f"  [{lo:.1f}, {hi:.1f}): {count:3d} ({count/total:5.1%}) {bar}")

    # Per-domain breakdown
    print(f"\nPer-Domain Mean Entropy:")
    domain_records = defaultdict(list)
    for r in all_step_records:
        domain_records[r["domain"]].append(r["action_entropy_bits"])
    for d in sorted(domain_records.keys()):
        vals = domain_records[d]
        print(f"  {d:25s}: H_mean={sum(vals)/len(vals):.3f}  n_steps={len(vals)}")

    # Per-step-index breakdown (does entropy change over the episode?)
    print(f"\nEntropy by Step Index (does uncertainty change over episode?):")
    step_records = defaultdict(list)
    for r in all_step_records:
        step_records[r["step"]].append(r["action_entropy_bits"])
    for s in sorted(step_records.keys()):
        vals = step_records[s]
        print(f"  Step {s}: H_mean={sum(vals)/len(vals):.3f}  n={len(vals)}")

    # Spatial clusters by step index (does branching opportunity change over episode?)
    print(f"\nSpatial Clusters by Step Index:")
    step_cluster_records = defaultdict(list)
    for r in all_step_records:
        step_cluster_records[r["step"]].append(r["n_spatial_clusters"])
    for s in sorted(step_cluster_records.keys()):
        vals = step_cluster_records[s]
        consensus_at_step = sum(1 for v in vals if v == 1)
        branch_worthy = sum(1 for v in vals if v >= 2)
        print(f"  Step {s:2d}: mean_clusters={sum(vals)/len(vals):.1f}  "
              f"consensus={consensus_at_step}/{len(vals)}  "
              f"branch_worthy={branch_worthy}/{len(vals)}  n={len(vals)}")

    # Overall action type distribution
    print(f"\nOverall Action Type Distribution (across all K samples, all steps):")
    type_total = Counter()
    for r in all_step_records:
        for t, c in r["type_distribution"].items():
            type_total[t] += c
    grand_total = sum(type_total.values())
    for t, c in type_total.most_common():
        print(f"  {t:20s}: {c:5d} ({c/grand_total:5.1%})")

    # Save results
    output_path = args.output or os.path.join(
        PROJ_ROOT, "docs", "research",
        f"action_entropy_smoke_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "n_tasks": args.n_tasks,
                "n_steps": args.n_steps,
                "n_samples": K,
                "temperature": args.temperature,
                "model": args.model_path,
                "task_file": args.task_file,
            },
            "summary": {
                "total_steps": len(all_step_records),
                "mean_action_entropy": round(sum(entropies)/len(entropies), 3),
                "mean_type_entropy": round(sum(type_entropies)/len(type_entropies), 3),
                "mean_token_entropy": round(sum(token_entropies)/len(token_entropies), 3),
                "spatial_consensus_fraction": round(spatial_consensus/total, 3),
                "bucket_consensus_fraction": round(bucket_consensus/total, 3),
                "consensus_fraction": round(consensus_steps/total, 3),
                "split_2_fraction": round(split_2/total, 3),
                "split_3plus_fraction": round(split_3plus/total, 3),
            },
            "step_records": all_step_records,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
