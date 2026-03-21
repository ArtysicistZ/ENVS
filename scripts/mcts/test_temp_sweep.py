#!/usr/bin/env python3
"""
Quick test: what temperature range produces valid actions?

Generates K=16 at each temp from 1.0 to 1.5 in 0.05 steps.
Reports: parseable%, unique fingerprints, action type distribution.
"""

import argparse, copy, json, logging, os, re, sys, time
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "OSWorld"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("temp_sweep")

from verl.mcts.clustering import parse_action_with_coords, action_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://10.100.4.7:15001")
    parser.add_argument("--model-path", default="ByteDance-Seed/UI-TARS-1.5-7B")
    parser.add_argument("--task-id", default="3ef2b351-8a84-4ff2-8724-d86eae9b842e")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()

    # Load task and get screenshot
    task_file = os.path.join(PROJ_ROOT, "OSWorld/evaluation_examples/test_all_300tasks_noproxy_softreset_clean.json")
    with open(task_file) as f:
        raw = json.load(f)
    tc = None
    for domain, tids in raw.items():
        if args.task_id in tids:
            cfg_path = os.path.join(os.path.dirname(task_file), "examples", domain, args.task_id + ".json")
            with open(cfg_path) as f2:
                tc = json.load(f2)
            tc["domain"] = domain
            tc["id"] = args.task_id
            break
    logger.info("Task: [%s] %s", tc["domain"], tc.get("instruction", "")[:80])

    import requests
    r = requests.post(f"{args.server_url}/env/reset",
                      json={"task_config": tc, "slot_id": 0}, timeout=300)
    r.raise_for_status()
    r2 = requests.post(f"{args.server_url}/env/history_messages",
                       json={"slot_id": 0}, timeout=30)
    r2.raise_for_status()
    msgs = r2.json().get("messages") or r2.json().get("history_messages", [])
    screenshot = ""
    for msg in reversed(msgs):
        for c in reversed(msg.get("content", [])):
            if isinstance(c, dict) and c.get("type") == "image":
                screenshot = c.get("b64", "") or c.get("image", "").split(",")[-1]
                if screenshot:
                    break
        if screenshot:
            break
    logger.info("Screenshot: %d chars", len(screenshot))

    # Build prompt
    from transformers import AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

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

## User Instruction
{instruction}"""

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Your are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "text", "text": SYSTEM_PROMPT.format(instruction=tc["instruction"])},
        ]},
        {"role": "user", "content": [
            {"type": "image", "image": f"data:image/jpeg;base64,{screenshot}"},
        ]},
    ]
    from qwen_vl_utils import process_vision_info
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    vllm_input = {"prompt_token_ids": raw_ids}
    if image_inputs:
        vllm_input["multi_modal_data"] = {"image": image_inputs}

    # Load vLLM
    from vllm import LLM, SamplingParams
    logger.info("Loading vLLM...")
    llm = LLM(model=args.model_path, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem, max_model_len=32768,
              limit_mm_per_prompt={"image": 3}, trust_remote_code=True, dtype="bfloat16")

    # Sweep temperatures
    temps = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    K = args.k

    logger.info("\n=== Temperature sweep: K=%d per temp ===\n", K)
    logger.info("%-6s  %8s  %8s  %8s  %s", "temp", "parseable", "unique_fp", "n_types", "type_distribution")

    for temp in temps:
        batch = [copy.deepcopy(vllm_input) for _ in range(K)]
        sp = SamplingParams(n=1, temperature=temp, max_tokens=512)
        outputs = llm.generate(batch, sampling_params=sp)
        candidates = [out.outputs[0].text for out in outputs]

        # Analyze
        n_parseable = 0
        fps = []
        types = []
        for c in candidates:
            atype, x, y, ckey = parse_action_with_coords(c)
            if atype != "UNPARSEABLE":
                n_parseable += 1
            types.append(atype)
            fps.append(action_fingerprint(c, grid_size=50))

        from collections import Counter
        type_counts = Counter(types)
        unique_fps = len(set(fps))

        logger.info("%-6.2f  %5d/%d   %8d  %8d  %s",
                    temp, n_parseable, K, unique_fps, len(type_counts),
                    dict(type_counts))

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
