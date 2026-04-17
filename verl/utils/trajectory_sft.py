"""
Expand compact trajectories (JSONL) into multi-turn SFT training examples.

Each trajectory step K produces one training example where:
  - Input: system prompt + history of (screenshot, action) pairs + current screenshot
  - Label: action_K

The `limit_images` field in each episode controls how many screenshots are
kept in context, matching the truncation used during inference.

Usage:
    from verl.utils.trajectory_sft import load_sft_examples

    examples = load_sft_examples("trajectories_at_0.jsonl")
    for ex in examples:
        # ex["messages"] is the multi-turn conversation (system + user/assistant turns)
        # ex["label"] is the action text to predict
        # ex["task_id"], ex["step_idx"], ex["total_steps"] for metadata
        train(ex)
"""

import base64
import json
import gzip
from typing import List, Dict, Any, Iterator

# Must match the prompt used during inference (gui_agent.py)
SYSTEM_MESSAGE = {"role": "system", "content": [{"type": "text", "text": "Your are a helpful assistant."}]}

UITARS_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

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
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
fail() # Use when you think the task is not feasible or cannot be completed.

## Note
- Use English in `Thought` and `Action` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

IMAGE_MAX_PIXELS = 2116800
IMAGE_MIN_PIXELS = 256


def _make_image_message(screenshot_b64: str) -> dict:
    """Build a user message containing a single screenshot."""
    return {
        "role": "user",
        "content": [{
            "type": "image",
            "image": f"data:image/jpeg;base64,{screenshot_b64}",
            "min_pixels": IMAGE_MIN_PIXELS,
            "max_pixels": IMAGE_MAX_PIXELS,
        }],
    }


def _make_assistant_message(action_text: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": action_text}],
    }


def expand_episode(episode: Dict[str, Any], train_all_steps: bool = True) -> List[Dict[str, Any]]:
    """Expand one compact episode into SFT training examples.

    Args:
        episode: One line from the trajectories JSONL, with keys:
            task_id, instruction, eval_result, limit_images, steps[]
        train_all_steps: If True, generate one example per step (step 0..N-1).
            If False, only generate for the last step (the finishing action).

    Returns:
        List of training examples, each with:
            messages: list of message dicts (full conversation including the
                      assistant response to train on as the final message)
            label: the action text the model should produce (also last assistant msg)
            task_id, step_idx, total_steps: metadata
    """
    steps = episode["steps"]
    instruction = episode.get("instruction", "")
    limit_images = episode.get("limit_images", 8)
    task_id = episode.get("task_id", "")
    total_steps = len(steps)

    if not steps:
        return []

    step_range = range(total_steps) if train_all_steps else [total_steps - 1]
    examples = []

    for k in step_range:
        # Build conversation: system + instruction + history + current screenshot + label
        messages = [
            SYSTEM_MESSAGE,
            {"role": "user", "content": [{"type": "text", "text": UITARS_SYSTEM_PROMPT.format(instruction=instruction)}]},
        ]

        # Apply limit_images: only include the last `limit_images` screenshots
        # from steps 0..k (matching inference truncation).
        # All action text history is kept; only screenshots are truncated.
        start_img = max(0, k + 1 - limit_images)

        for i in range(k + 1):
            # Add screenshot only if within the visible window
            if i >= start_img:
                messages.append(_make_image_message(steps[i]["screenshot_b64"]))
            # Add previous actions (not for the current step — that's the label)
            if i < k:
                messages.append(_make_assistant_message(steps[i]["action"]))

        # Append the label as the final assistant message so tokenizers can
        # determine supervised targets by role (assistant → train, others → mask).
        messages.append(_make_assistant_message(steps[k]["action"]))

        examples.append({
            "messages": messages,
            "label": steps[k]["action"],
            "task_id": task_id,
            "step_idx": k,
            "total_steps": total_steps,
        })

    return examples


def load_sft_examples(
    path: str,
    train_all_steps: bool = True,
    min_steps: int = 1,
) -> List[Dict[str, Any]]:
    """Load all SFT examples from a trajectories JSONL file.

    Args:
        path: Path to trajectories JSONL (optionally .gz).
        train_all_steps: If True, one example per step. If False, only last step.
        min_steps: Skip episodes with fewer steps than this.

    Returns:
        List of SFT training examples.
    """
    open_fn = gzip.open if path.endswith(".gz") else open
    examples = []
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            if len(episode.get("steps", [])) < min_steps:
                continue
            examples.extend(expand_episode(episode, train_all_steps=train_all_steps))
    return examples


def iter_sft_examples(
    path: str,
    train_all_steps: bool = True,
    min_steps: int = 1,
) -> Iterator[Dict[str, Any]]:
    """Streaming version of load_sft_examples (for large files)."""
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            if len(episode.get("steps", [])) < min_steps:
                continue
            yield from expand_episode(episode, train_all_steps=train_all_steps)
