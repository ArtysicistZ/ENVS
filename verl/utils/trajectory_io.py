"""Efficient trajectory serialization for SFT/ENVS training.

Each episode is stored as a compact JSON object (one line in a JSONL file):
  {
    "task_id": str,
    "instruction": str,
    "eval_result": float,
    "limit_images": int,   # inference setting — must match SFT prompt reconstruction
    "steps": [
      {"screenshot_b64": "<base64 JPEG>", "action": "<model output>"},
      ...
    ]
  }

Each screenshot is stored ONCE (not once per subsequent step), eliminating the
massive redundancy from the multi-turn conversation format used during inference.

At SFT training time, reconstruct step K's prompt by taking the last `limit_images`
screenshots from steps[0..K] — matching the exact context window seen during inference.
"""

import gzip
import json
import os
from typing import Any, Dict, Iterator, List


class TrajectoryWriter:
    """Incrementally writes episodes to a JSONL file (one episode per line)."""

    def __init__(self, path: str, compress: bool = False):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._compress = compress
        self._f = gzip.open(path, "at", encoding="utf-8") if compress else open(path, "a", encoding="utf-8")

    def write(self, episode: Dict[str, Any]) -> None:
        self._f.write(json.dumps(episode, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_trajectories(path: str) -> List[Dict[str, Any]]:
    """Load all episodes from a JSONL (or .jsonl.gz) file."""
    open_fn = gzip.open if path.endswith(".gz") else open
    episodes = []
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def iter_trajectories(path: str) -> Iterator[Dict[str, Any]]:
    """Lazily iterate episodes from a JSONL file without loading all into memory."""
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def make_sft_conversations(episode: Dict[str, Any], system_prompt: str) -> List[Dict[str, Any]]:
    """Expand a compact episode into one SFT conversation per step.

    For step K, the prompt uses only the last `limit_images` screenshots
    from steps[0..K], matching the inference context window exactly.

    Returns a list of dicts:
      {
        "messages": [...],    # conversation up to step K (input)
        "label": str,         # action at step K (supervised target)
        "task_id": str,
        "step": int,
        "eval_result": float,
      }
    """
    limit = episode.get("limit_images", 8)
    steps = episode["steps"]
    instruction = episode["instruction"]
    task_id = episode["task_id"]
    eval_result = episode.get("eval_result", 0.0)

    sft_examples = []
    for k, step in enumerate(steps):
        # Build message list: system + instruction text + interleaved (screenshot, action) pairs
        # Apply limit_images: only include the last `limit` screenshots from steps[0..k]
        first_visible = max(0, k + 1 - limit)  # index of oldest visible screenshot

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Your are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": system_prompt.format(instruction=instruction)}]},
        ]

        for i in range(first_visible, k + 1):
            s = steps[i]
            # User turn: screenshot
            messages.append({
                "role": "user",
                "content": [{
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{s['screenshot_b64']}",
                }],
            })
            # Assistant turn: action (only for past steps, not the current one we're predicting)
            if i < k:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": s["action"]}],
                })

        sft_examples.append({
            "messages": messages,
            "label": step["action"],
            "task_id": task_id,
            "step": k,
            "eval_result": eval_result,
        })

    return sft_examples
