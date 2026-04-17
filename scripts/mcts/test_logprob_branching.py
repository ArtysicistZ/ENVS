#!/usr/bin/env python3
"""
Test script: validate logprob-based branching idea.

Does the model's logprob distribution at the action-type token reveal
meaningful alternatives? If so, we can branch without high temperature.

Usage:
    python scripts/mcts/test_logprob_branching.py \
        --server-url http://10.100.4.7:15001 \
        --task-ids 4783cc41-c03c-4e1b-89b4-50658f642bd5

Tests:
1. Generate 1 response with logprobs=20 at temp=1.0
2. Find the "Action:" position in generated tokens
3. Extract top-20 logprobs at the action-type token
4. Show probability distribution over action types
5. If alternatives exist, force-generate with alternative prefix
"""

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
import time

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "OSWorld"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("test_logprob")

# Known action types for UI-TARS
KNOWN_ACTION_TYPES = ["click", "type", "hotkey", "scroll", "wait", "finished", "fail",
                       "drag", "left_double", "right_single"]

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
type(content='xxx')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait()
finished(content='xxx')
fail()

## Note
- Use English in `Thought` and `Action` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}"""


def load_task_config(task_id: str) -> dict:
    task_file = os.path.join(PROJ_ROOT, "OSWorld/evaluation_examples/test_all_300tasks_noproxy_softreset_clean.json")
    with open(task_file) as f:
        raw = json.load(f)
    for domain, tids in raw.items():
        if task_id in tids:
            cfg_path = os.path.join(os.path.dirname(task_file), "examples", domain, task_id + ".json")
            with open(cfg_path) as f2:
                tc = json.load(f2)
            tc["domain"] = domain
            tc["id"] = task_id
            tc["task_id"] = task_id
            return tc
    raise ValueError(f"Task {task_id} not found")


def get_screenshot(server_url: str, slot_id: int, task_config: dict) -> str:
    """Reset VM and get initial screenshot."""
    import requests
    # Reset
    r = requests.post(f"{server_url}/env/reset",
                      json={"task_config": task_config, "slot_id": slot_id}, timeout=300)
    r.raise_for_status()
    # Get screenshot
    r2 = requests.post(f"{server_url}/env/history_messages",
                       json={"slot_id": slot_id}, timeout=30)
    r2.raise_for_status()
    messages = r2.json().get("messages") or r2.json().get("history_messages", [])
    for msg in reversed(messages):
        content = msg.get("content", [])
        if isinstance(content, list):
            for c in reversed(content):
                if isinstance(c, dict) and c.get("type") == "image":
                    b64 = c.get("b64", "")
                    if b64:
                        return b64
                    img_str = c.get("image", "")
                    if img_str.startswith("data:image"):
                        return img_str.split(",", 1)[1]
    return ""


def build_messages(instruction: str, screenshot_b64: str) -> list:
    return [
        {"role": "system", "content": [{"type": "text", "text": "Your are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "text", "text": SYSTEM_PROMPT.format(instruction=instruction)},
        ]},
        {"role": "user", "content": [
            {"type": "image", "image": f"data:image/jpeg;base64,{screenshot_b64}"},
        ]},
    ]


def find_action_colon_position(token_ids: list, tokenizer) -> int:
    """Find the position of the token right after 'Action:' in generated tokens.

    This is where the action type token appears.
    Returns the index into token_ids, or -1 if not found.
    """
    # Encode "Action:" to find its token pattern
    action_tokens = tokenizer.encode("Action:", add_special_tokens=False)
    # Also try with newline prefix
    action_tokens_nl = tokenizer.encode("\nAction:", add_special_tokens=False)

    # Search for pattern in generated tokens
    for pattern in [action_tokens, action_tokens_nl]:
        plen = len(pattern)
        for i in range(len(token_ids) - plen):
            if token_ids[i:i + plen] == pattern:
                # The action type token is right after this pattern
                pos = i + plen
                if pos < len(token_ids):
                    return pos
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://10.100.4.7:15001")
    parser.add_argument("--model-path", default="ByteDance-Seed/UI-TARS-1.5-7B")
    parser.add_argument("--task-ids", nargs="+", default=[
        "4783cc41-c03c-4e1b-89b4-50658f642bd5",   # os task, 5% rate
        "3ef2b351-8a84-4ff2-8724-d86eae9b842e",   # writer, 16% rate
        "ec71221e-ac43-46f9-89b8-ee7d80f7e1c5",   # vs_code, 22% rate
    ])
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--n-logprobs", type=int, default=20)
    args = parser.parse_args()

    # Load tokenizer
    from transformers import AutoProcessor, AutoTokenizer
    logger.info("Loading tokenizer...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Print token IDs for known action types
    logger.info("\n=== Action type token IDs ===")
    for atype in KNOWN_ACTION_TYPES:
        tids = tokenizer.encode(f" {atype}(", add_special_tokens=False)
        decoded = [tokenizer.decode([t]) for t in tids]
        logger.info("  %-15s tokens=%s  decoded=%s", atype, tids, decoded)

    # Encode "Action:" pattern
    action_pattern = tokenizer.encode("Action:", add_special_tokens=False)
    logger.info("  'Action:' pattern: %s  decoded=%s", action_pattern, [tokenizer.decode([t]) for t in action_pattern])

    # Load vLLM with logprobs support
    logger.info("\nLoading vLLM on 1 GPU...")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_path, tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem, max_model_len=32768,
        limit_mm_per_prompt={"image": 3}, trust_remote_code=True, dtype="bfloat16",
    )

    from qwen_vl_utils import process_vision_info

    # Test each task
    for task_id in args.task_ids:
        logger.info("\n" + "=" * 70)
        tc = load_task_config(task_id)
        logger.info("Task: [%s] %s", tc["domain"], tc.get("instruction", "")[:80])
        logger.info("=" * 70)

        # Get screenshot
        logger.info("Resetting VM and getting screenshot...")
        screenshot = get_screenshot(args.server_url, 0, tc)
        if not screenshot:
            logger.error("No screenshot!")
            continue

        # Build messages and prepare vLLM input
        messages = build_messages(tc["instruction"], screenshot)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)
        raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        vllm_input = {"prompt_token_ids": raw_prompt_ids}
        if image_inputs:
            vllm_input["multi_modal_data"] = {"image": image_inputs}

        # Phase 1: Generate with logprobs
        logger.info("\n--- Phase 1: Generate with logprobs=%d ---", args.n_logprobs)
        sp = SamplingParams(n=1, temperature=1.0, max_tokens=512, logprobs=args.n_logprobs)
        outputs = llm.generate([vllm_input], sampling_params=sp)
        out = outputs[0].outputs[0]

        generated_text = out.text
        generated_token_ids = list(out.token_ids)
        generated_logprobs = out.logprobs  # List[Dict[int, Logprob]]

        # Show generated text
        action_line = ""
        if "Action:" in generated_text:
            action_line = generated_text.split("Action:")[-1].strip().split("\n")[0]
        logger.info("Generated: ...Action: %s", action_line[:100])
        logger.info("Total tokens: %d", len(generated_token_ids))

        # Phase 2: Find action type position
        action_pos = find_action_colon_position(generated_token_ids, tokenizer)
        if action_pos < 0:
            logger.warning("Could not find 'Action:' in generated tokens!")
            # Try by scanning decoded text
            cumulative = ""
            for i, tid in enumerate(generated_token_ids):
                cumulative += tokenizer.decode([tid])
                if "Action:" in cumulative and i + 1 < len(generated_token_ids):
                    action_pos = i + 1
                    break

        if action_pos < 0:
            logger.error("Could not find action type position!")
            continue

        action_type_token = generated_token_ids[action_pos]
        action_type_decoded = tokenizer.decode([action_type_token])
        logger.info("\nAction type token at position %d: id=%d decoded='%s'",
                    action_pos, action_type_token, action_type_decoded)

        # Phase 3: Extract logprobs at action type position
        if generated_logprobs and action_pos < len(generated_logprobs):
            logprob_dict = generated_logprobs[action_pos]
            logger.info("\n--- Top %d logprobs at action-type position ---", args.n_logprobs)

            # Sort by probability
            entries = []
            for token_id, logprob_obj in logprob_dict.items():
                prob = math.exp(logprob_obj.logprob)
                decoded = tokenizer.decode([token_id]).strip()
                entries.append((prob, token_id, decoded, logprob_obj.logprob))
            entries.sort(reverse=True)

            for prob, tid, decoded, lp in entries:
                # Check if this is a known action type
                is_action = any(decoded.strip().startswith(at) for at in KNOWN_ACTION_TYPES)
                marker = " <-- ACTION TYPE" if is_action else ""
                logger.info("  p=%.4f (logp=%.2f) token=%d '%s'%s", prob, lp, tid, decoded, marker)

            # Compute distribution over action types specifically
            logger.info("\n--- Action type probability distribution ---")
            action_probs = {}
            for atype in KNOWN_ACTION_TYPES:
                # Check if any of the top logprobs match this action type's first token
                atype_tokens = tokenizer.encode(f" {atype}", add_special_tokens=False)
                first_token = atype_tokens[0]
                for token_id, logprob_obj in logprob_dict.items():
                    if token_id == first_token:
                        action_probs[atype] = math.exp(logprob_obj.logprob)
                        break

            if action_probs:
                total = sum(action_probs.values())
                for atype, prob in sorted(action_probs.items(), key=lambda x: -x[1]):
                    logger.info("  %-15s p=%.4f (%.1f%%)", atype, prob, prob * 100)
                logger.info("  Sum of known action types: %.4f", total)

                # Branch decision
                top_type = max(action_probs, key=action_probs.get)
                top_prob = action_probs[top_type]
                alternatives = [(at, p) for at, p in action_probs.items()
                                if at != top_type and p >= 0.03]
                logger.info("\n  Top: '%s' at %.1f%%", top_type, top_prob * 100)
                if alternatives:
                    logger.info("  BRANCH CANDIDATES:")
                    for at, p in sorted(alternatives, key=lambda x: -x[1]):
                        logger.info("    '%s' at %.1f%%", at, p * 100)
                else:
                    logger.info("  No alternatives above 3%% threshold")
            else:
                logger.info("  No known action types found in top logprobs!")

        # Phase 4: If alternatives exist, test forced prefix generation
        if action_probs and alternatives:
            logger.info("\n--- Phase 4: Forced prefix generation ---")
            for alt_type, alt_prob in sorted(alternatives, key=lambda x: -x[1])[:2]:
                # Build forced prefix: take generated tokens up to action_pos, then replace with alt
                prefix_tokens = list(generated_token_ids[:action_pos])
                alt_first_tokens = tokenizer.encode(f" {alt_type}(", add_special_tokens=False)

                # Append to prompt
                forced_prompt_ids = list(raw_prompt_ids) + prefix_tokens + alt_first_tokens
                forced_input = {"prompt_token_ids": forced_prompt_ids}
                if image_inputs:
                    forced_input["multi_modal_data"] = {"image": image_inputs}

                sp_forced = SamplingParams(n=1, temperature=1.0, max_tokens=128)
                forced_outputs = llm.generate([forced_input], sampling_params=sp_forced)
                forced_text = forced_outputs[0].outputs[0].text

                # Reconstruct: original thought + forced action
                thought_part = generated_text.split("Action:")[0] if "Action:" in generated_text else ""
                reconstructed = f"{thought_part}Action: {alt_type}({forced_text}"
                forced_action = f"{alt_type}({forced_text.split(chr(10))[0]}"
                logger.info("  Forced '%s' (p=%.1f%%): %s", alt_type, alt_prob * 100, forced_action[:120])

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
