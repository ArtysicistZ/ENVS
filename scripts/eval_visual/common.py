"""Shared utilities for visual VLM benchmarks (ScreenSpot-V2, ScreenSpot-Pro, OSWorld-G).

Design goals:
- Reuse the existing UI-TARS coordinate parser from verl.trainer.gui_agent unchanged.
- Single-process vLLM offline inference; no Ray, no FastAPI, no remote servers.
- Per-sample try/except so one failure never crashes the loop.
- Append-resume JSONL: rerunning the same script picks up where it left off.
- Coordinate-space round-trip is verified before any benchmark runs.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

# Make verl importable when this script runs from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Lifted unchanged. Don't reimplement — these are battle-tested by every OSWorld rollout.
from verl.trainer.gui_agent import (  # noqa: E402
    IMAGE_FACTOR,
    MAX_PIXELS,
    MIN_PIXELS,
    parse_action_to_structure_output,
    smart_resize,
)


# ============================================================================
# Prompts
# ============================================================================

# The published UI-TARS-1.5 grounding prompt. Mirrors EnvWorker.ground_prompt
# in verl/trainer/gui_agent.py:521 — kept verbatim so eval matches what the
# model expects from training.
GROUNDING_PROMPT_TEMPLATE = (
    "Output only the coordinate of one point in your response. "
    "What element matches the following task: {instruction}"
)

# Full UI-TARS agent prompt — used for OSWorld-G refusal cases where we want
# the model to be able to emit fail() if the instruction is infeasible.
AGENT_PROMPT_HEADER = (
    "You are a GUI agent. You are given a task and a screenshot. "
    "Output an action in the UI-TARS action grammar.\n\n"
    "## Output Format\n"
    "```\n"
    "Thought: ...\n"
    "Action: ...\n"
    "```\n\n"
    "## Action Space\n"
    "click(start_box='<|box_start|>(x1,y1)<|box_end|>')\n"
    "fail() # Use when the task is not feasible.\n\n"
    "## User Instruction\n"
    "{instruction}"
)


# ============================================================================
# Image preprocessing
# ============================================================================


def smart_resize_image(
    image: Image.Image,
    max_pixels: int = MAX_PIXELS,
    min_pixels: int = MIN_PIXELS,
    factor: int = IMAGE_FACTOR,
) -> tuple[Image.Image, int, int]:
    """Resize image to UI-TARS smart_resize dimensions.

    Returns (resized_image, smart_h, smart_w). Coordinates the model emits are
    in the (smart_h, smart_w) frame; downstream we inverse-map them back to the
    original image's pixel space for bbox-hit scoring.
    """
    orig_w, orig_h = image.size
    smart_h, smart_w = smart_resize(orig_h, orig_w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    if (smart_h, smart_w) != (orig_h, orig_w):
        resized = image.resize((smart_w, smart_h), Image.Resampling.LANCZOS)
    else:
        resized = image
    return resized, smart_h, smart_w


def image_sha256(image: Image.Image) -> str:
    """Hash the raw RGB bytes of an image. Cheap drift detector across dataset versions."""
    rgb = image.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ============================================================================
# Coordinate parsing
# ============================================================================


@dataclass
class GroundingPrediction:
    """Result of parsing a UI-TARS grounding response.

    `point_orig` is the predicted (x, y) in original image pixel space, ready
    for bbox-hit scoring. `is_refusal` is True if the model declined (fail() or
    explicit 'infeasible' text). `parse_method` records which fallback fired —
    useful for debugging when the headline number looks off.
    """

    point_orig: tuple[float, float] | None
    is_refusal: bool
    raw_response: str
    parse_method: str
    parse_error: str | None = None


# Regex patterns tried in priority order. The first match wins.
_PAT_BOXED = re.compile(
    r"<\|box_start\|>\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*<\|box_end\|>"
)
_PAT_POINT_TAG = re.compile(
    r"<point>\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*</point>", re.IGNORECASE
)
_PAT_PAREN = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
_PAT_BRACKET = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
_PAT_TWO_NUMBERS = re.compile(r"(-?\d+(?:\.\d+)?)\s*[,\s]+\s*(-?\d+(?:\.\d+)?)")
_REFUSAL_TOKENS = ("fail()", "infeasible", "cannot be done", "not feasible", "impossible")


def _norm_to_orig(
    x_smart: float,
    y_smart: float,
    smart_h: int,
    smart_w: int,
    orig_h: int,
    orig_w: int,
) -> tuple[float, float]:
    """Inverse-map a coord from smart_resized space to original pixel space."""
    x_norm = x_smart / smart_w
    y_norm = y_smart / smart_h
    return (x_norm * orig_w, y_norm * orig_h)


def parse_grounding_response(
    response: str,
    smart_h: int,
    smart_w: int,
    orig_h: int,
    orig_w: int,
    allow_refusal: bool = False,
) -> GroundingPrediction:
    """Parse a UI-TARS grounding response into a point in original image space.

    Tries, in order:
      1. UI-TARS native `Action: click(start_box='...')` via the official parser.
      2. `<|box_start|>(x,y)<|box_end|>` token format.
      3. `<point>x y</point>` tag.
      4. First `(x, y)` tuple.
      5. First `[x, y]` bracket pair.
      6. First two numbers anywhere in the text.

    For each fallback, coordinates are interpreted as **absolute pixels in the
    smart_resized image** (Qwen2.5-VL output convention) and inverse-mapped to
    original pixel space.
    """
    response = response.strip()
    if not response:
        return GroundingPrediction(None, False, response, "empty_response", "empty response")

    if allow_refusal:
        low = response.lower()
        for token in _REFUSAL_TOKENS:
            if token in low:
                return GroundingPrediction(None, True, response, "refusal", None)

    # 1) UI-TARS official action grammar — only attempt if the response looks like one.
    if "Action:" in response and "start_box" in response:
        try:
            actions = parse_action_to_structure_output(
                response,
                factor=1000,  # ignored for qwen25vl path, parser uses smart_resize
                origin_resized_height=orig_h,
                origin_resized_width=orig_w,
                model_type="qwen25vl",
            )
            if actions:
                inputs = actions[0].get("action_inputs", {})
                box_str = inputs.get("start_box")
                if box_str:
                    # parse_action_to_structure_output returns box as string of normalized [0,1] floats
                    coords = json.loads(box_str.replace("'", '"')) if box_str.startswith("[") else None
                    if coords and len(coords) >= 2:
                        cx = (coords[0] + coords[2]) / 2 if len(coords) == 4 else coords[0]
                        cy = (coords[1] + coords[3]) / 2 if len(coords) == 4 else coords[1]
                        return GroundingPrediction(
                            (cx * orig_w, cy * orig_h),
                            False,
                            response,
                            "uitars_action",
                            None,
                        )
        except Exception as e:  # noqa: BLE001
            # Fall through to coord-only fallbacks
            _ = e

    for name, pat in (
        ("box_start_end", _PAT_BOXED),
        ("point_tag", _PAT_POINT_TAG),
        ("paren", _PAT_PAREN),
        ("bracket", _PAT_BRACKET),
    ):
        m = pat.search(response)
        if m:
            try:
                x_smart = float(m.group(1))
                y_smart = float(m.group(2))
                return GroundingPrediction(
                    _norm_to_orig(x_smart, y_smart, smart_h, smart_w, orig_h, orig_w),
                    False,
                    response,
                    name,
                    None,
                )
            except Exception as e:  # noqa: BLE001
                return GroundingPrediction(None, False, response, name, f"coerce_failed: {e}")

    m = _PAT_TWO_NUMBERS.search(response)
    if m:
        try:
            x_smart = float(m.group(1))
            y_smart = float(m.group(2))
            return GroundingPrediction(
                _norm_to_orig(x_smart, y_smart, smart_h, smart_w, orig_h, orig_w),
                False,
                response,
                "two_numbers_fallback",
                None,
            )
        except Exception as e:  # noqa: BLE001
            return GroundingPrediction(None, False, response, "two_numbers_fallback", f"coerce_failed: {e}")

    return GroundingPrediction(None, False, response, "no_match", "no parseable coordinate")


# ============================================================================
# Scoring
# ============================================================================


def point_in_bbox(point: tuple[float, float] | None, bbox: tuple[float, float, float, float]) -> bool:
    """bbox is (x1, y1, x2, y2) in original image pixel space, half-open is fine."""
    if point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def normalize_bbox(bbox: Iterable[float], image_w: int, image_h: int) -> tuple[float, float, float, float]:
    """Some datasets give [0,1]-normalized bboxes; some give pixel. Detect and convert to pixel.

    Heuristic: if every coordinate is <= 1.0001, treat as normalized.
    """
    b = list(bbox)
    if len(b) != 4:
        raise ValueError(f"bbox must have 4 entries, got {b}")
    if all(v <= 1.0001 for v in b):
        return (b[0] * image_w, b[1] * image_h, b[2] * image_w, b[3] * image_h)
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


# ============================================================================
# Append-resume JSONL writer
# ============================================================================


@dataclass
class ResultsLog:
    """Append-only JSONL with sample_id-keyed dedup. Resume-safe."""

    path: Path
    seen_ids: set[str] = field(default_factory=set)

    @classmethod
    def open(cls, path: str | Path) -> "ResultsLog":
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        if p.exists():
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        sid = rec.get("sample_id")
                        if sid is not None:
                            seen.add(str(sid))
                    except json.JSONDecodeError:
                        # Tolerate a partial last-line write from an interrupted run
                        continue
        return cls(path=p, seen_ids=seen)

    def has(self, sample_id: str) -> bool:
        return str(sample_id) in self.seen_ids

    def append(self, record: dict[str, Any]) -> None:
        sid = record.get("sample_id")
        if sid is None:
            raise ValueError("record must include 'sample_id'")
        line = json.dumps(record, ensure_ascii=False)
        with self.path.open("a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.seen_ids.add(str(sid))


# ============================================================================
# vLLM client
# ============================================================================


@dataclass
class GenerationConfig:
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0


class UITARSClient:
    """Thin vLLM wrapper that produces grounding responses for batched samples.

    Uses bfloat16 + greedy decode by default. Single tensor-parallel rank;
    multi-GPU is opt-in via tensor_parallel_size to avoid surprising the user
    when other GPUs are doing other work.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 12000,
        gpu_memory_utilization: float = 0.5,
        # Match ARPO training config (configs/arpo_8gpu.yaml): max_pixels=2.1M,
        # min_pixels=256. Using the same caps at eval ensures images are at the
        # same resolution UI-TARS-1.5-7B was trained with — no distribution
        # shift from the default 12.8M cap in verl/trainer/gui_agent.py.
        max_pixels: int = 2116800,
        min_pixels: int = 256,
        # Match training: max_num_seqs=8, max_num_batched_tokens=13024.
        # Setting max_num_seqs explicitly also prevents vLLM from clamping it
        # to 1 (it does that when max_model_len < its internal expected image
        # seq length of 32768 for generic VLMs).
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 13024,
    ):
        # Defer the heavy import so `--help` and unit tests don't pay for it.
        from vllm import LLM, SamplingParams  # noqa: F401  (SamplingParams used in generate)

        self.model_path = model_path
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
            disable_log_stats=True,
            enforce_eager=False,
        )

    def generate(
        self,
        prompts: list[dict[str, Any]],
        cfg: GenerationConfig | None = None,
    ) -> list[str]:
        """Batched generation. `prompts` is a list of {prompt, multi_modal_data}."""
        from vllm import SamplingParams

        cfg = cfg or GenerationConfig()
        sp = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        # vLLM may reorder by request_id; align by index since we passed them in order.
        outs = sorted(outs, key=lambda o: int(o.request_id) if str(o.request_id).isdigit() else 0)
        return [o.outputs[0].text for o in outs]


def build_grounding_prompt(image: Image.Image, instruction: str, system_prompt: str | None = None) -> dict[str, Any]:
    """Build a vLLM-ready prompt dict for a single grounding sample.

    Uses the Qwen2.5-VL chat template directly so we don't depend on transformers
    chat-template behavior diverging across versions.
    """
    sys_msg = system_prompt or "You are a helpful assistant."
    user_text = GROUNDING_PROMPT_TEMPLATE.format(instruction=instruction)
    # Qwen2.5-VL uses <|vision_start|><|image_pad|><|vision_end|> as the image slot.
    chat = (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {
        "prompt": chat,
        "multi_modal_data": {"image": image},
    }


def build_agent_prompt(image: Image.Image, instruction: str) -> dict[str, Any]:
    """Build a UI-TARS-grammar prompt that supports refusal via fail()."""
    user_text = AGENT_PROMPT_HEADER.format(instruction=instruction)
    chat = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"prompt": chat, "multi_modal_data": {"image": image}}


# ============================================================================
# Round-trip self-test (called by every runner before inference)
# ============================================================================


def selftest_coord_roundtrip() -> None:
    """Asserts the smart_resize coordinate inverse-map is exact on a known case.

    If this fails, the parser would silently mislabel hits as misses (or vice
    versa) and every benchmark number would be junk. Run it before any inference.
    """
    # Simulate a 4K screenshot.
    orig_h, orig_w = 2160, 3840
    smart_h, smart_w = smart_resize(orig_h, orig_w, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    # Pick a point at known fraction.
    fx, fy = 0.42, 0.71
    x_smart = fx * smart_w
    y_smart = fy * smart_h
    x_orig, y_orig = _norm_to_orig(x_smart, y_smart, smart_h, smart_w, orig_h, orig_w)
    err = abs(x_orig - fx * orig_w) + abs(y_orig - fy * orig_h)
    assert err < 1e-6, f"coord round-trip drift {err:.6f} (smart={smart_h}x{smart_w}, orig={orig_h}x{orig_w})"

    # Round-trip a normalized bbox dataset format too.
    bbox_norm = (0.1, 0.2, 0.3, 0.4)
    bbox_pix = normalize_bbox(bbox_norm, orig_w, orig_h)
    expected = (0.1 * orig_w, 0.2 * orig_h, 0.3 * orig_w, 0.4 * orig_h)
    assert all(abs(a - b) < 1e-6 for a, b in zip(bbox_pix, expected)), bbox_pix


# ============================================================================
# Stat helpers
# ============================================================================


@dataclass
class RunStats:
    n_total: int = 0
    n_scored: int = 0
    n_hit: int = 0
    n_parse_error: int = 0
    n_refusal_correct: int = 0
    n_refusal_wrong: int = 0
    by_class_hit: dict[str, int] = field(default_factory=dict)
    by_class_total: dict[str, int] = field(default_factory=dict)

    def record(self, sub_class: str, *, scored: bool, hit: bool, parse_error: bool) -> None:
        self.n_total += 1
        if parse_error:
            self.n_parse_error += 1
        if scored:
            self.n_scored += 1
            if hit:
                self.n_hit += 1
        self.by_class_total[sub_class] = self.by_class_total.get(sub_class, 0) + 1
        if hit:
            self.by_class_hit[sub_class] = self.by_class_hit.get(sub_class, 0) + 1

    def summary_dict(self) -> dict[str, Any]:
        accuracy = self.n_hit / self.n_total if self.n_total else 0.0
        per_class = {
            cls: {
                "hit": self.by_class_hit.get(cls, 0),
                "total": self.by_class_total[cls],
                "accuracy": self.by_class_hit.get(cls, 0) / self.by_class_total[cls],
            }
            for cls in sorted(self.by_class_total)
        }
        return {
            "n_total": self.n_total,
            "n_scored": self.n_scored,
            "n_hit": self.n_hit,
            "n_parse_error": self.n_parse_error,
            "accuracy": accuracy,
            "n_refusal_correct": self.n_refusal_correct,
            "n_refusal_wrong": self.n_refusal_wrong,
            "by_class": per_class,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def write_summary(out_path: Path, header: dict[str, Any], stats: RunStats) -> None:
    out = {"header": header, "stats": stats.summary_dict()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))


# ============================================================================
# Misc
# ============================================================================


def safe_str(v: Any) -> str:
    try:
        return str(v)
    except Exception:  # noqa: BLE001
        return "<unstringable>"


def ensure_pil_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img
