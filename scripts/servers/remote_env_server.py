#!/usr/bin/env python3
"""
Remote OSWorld env server (run on Mac or AWS CPU).
One env; exposes POST /env/reset, /env/step, /env/evaluate, /env/history_messages.
Cluster EnvWorkers call this over HTTP.

Aligns with ARPO_OSWorld_Evaluation / run_uitars.py:
- Same DesktopEnv: observation_type=screenshot, action_space=pyautogui.
- Reset returns obs_messages built from env screenshot (same as evaluation agent gets).
- Provider: On macOS (Darwin), defaults to VMware (no /dev/kvm; use VMware Fusion).
  On Linux, defaults to Docker. Override with env PROVIDER=aws (EC2), PROVIDER=vmware, or PROVIDER=docker.
  Use PROVIDER=aws on EC2 when boto3 and aws configure are set up (launches EC2 env instances; no local Docker VM).


Run on GPU cluster:

sg docker -c "export PROVIDER=docker && .venv/bin/python scripts/remote_env_server.py"

"""
REMOTE_ENV_STAMP = "b36ed69-lifespan"  # grep this on Mac to confirm you have latest
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
osworld_root = repo_root / "OSWorld"
for p in (repo_root, osworld_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_dotenv():
    """Load .env from repo root so OPENAI_API_KEY (and AWS_*, etc.) are available for tasks."""
    try:
        p = repo_root / ".env"
        if p.is_file():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


import base64
import logging
import os
import re
import threading
import traceback
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

# Load .env from repo root so OPENAI_API_KEY (and AWS_*, etc.) are available for OSWorld tasks
_load_dotenv()

# Unconditional print on import so we know this file is loaded (e.g. on Mac after git pull)
print("[remote_env_server] module loaded", file=sys.stderr, flush=True)

try:
    import docker
except ImportError:
    docker = None

from desktop_env.desktop_env import DesktopEnv
from verl.trainer.remote_env_protocol import messages_to_wire
from verl.trainer.gui_agent import (
    uitars_system_prompt,
    parse_action_to_structure_output,
    parsing_response_to_pyautogui_code,
    add_box_token,
    FINISH_WORD,
    WAIT_WORD,
    ENV_FAIL_WORD,
    CALL_USER,
)

def _default_provider() -> str:
    """Use VMware on macOS (no KVM); Docker on Linux."""
    if hasattr(os, "uname") and os.uname().sysname == "Darwin":
        return "vmware"
    return "docker"


# --- Global constants (shared across all slots) ---
_provider_name: str = (os.environ.get("PROVIDER") or _default_provider()).strip().lower()
max_steps = int(os.environ.get("REMOTE_MAX_STEPS", "32"))
OBSERVATION_TYPE = "screenshot"
IMAGE_MIN_PIXELS = int(os.environ.get("REMOTE_IMAGE_MIN_PIXELS", "3136"))
IMAGE_MAX_PIXELS = int(os.environ.get("REMOTE_IMAGE_MAX_PIXELS", "518400"))
ACTION_PAUSE_SEC = float(os.environ.get("REMOTE_ACTION_PAUSE_SEC", "1.0"))
REPEAT_ACTION_THRESHOLD = int(os.environ.get("REMOTE_REPEAT_ACTION_THRESHOLD", "3"))
REPEAT_ACTION_PENALTY = float(os.environ.get("REMOTE_REPEAT_ACTION_PENALTY", "0.5"))
FORMAT_PARSE_BASE_REWARD = float(os.environ.get("REMOTE_FORMAT_PARSE_BASE_REWARD", "0.03"))
FORMAT_EXEC_ACTION_BONUS = float(os.environ.get("REMOTE_FORMAT_EXEC_ACTION_BONUS", "0.02"))
FORMAT_FINISH_BONUS = float(os.environ.get("REMOTE_FORMAT_FINISH_BONUS", "0.05"))
FORMAT_STEP_SUCCESS_BONUS = float(os.environ.get("REMOTE_FORMAT_STEP_SUCCESS_BONUS", "0.04"))
SEMANTIC_REPEAT_PENALTY = float(os.environ.get("REMOTE_SEMANTIC_REPEAT_PENALTY", "0.12"))
SEMANTIC_REPEAT_THRESHOLD = int(os.environ.get("REMOTE_SEMANTIC_REPEAT_THRESHOLD", "2"))
COOKIE_BANNER_INTENT_PENALTY = float(os.environ.get("REMOTE_COOKIE_BANNER_INTENT_PENALTY", "0.12"))
COOKIE_BANNER_INTENT_THRESHOLD = int(os.environ.get("REMOTE_COOKIE_BANNER_INTENT_THRESHOLD", "3"))
INTENT_ZONE_MISMATCH_PENALTY = float(os.environ.get("REMOTE_INTENT_ZONE_MISMATCH_PENALTY", "0.10"))
EDGE_NOOP_CLICK_PENALTY = float(os.environ.get("REMOTE_EDGE_NOOP_CLICK_PENALTY", "0.08"))
BROWSER_MENU_NOOPEN_PENALTY = float(os.environ.get("REMOTE_BROWSER_MENU_NOOPEN_PENALTY", "0.12"))
BROWSER_MENU_OPEN_PROGRESS_BONUS = float(os.environ.get("REMOTE_BROWSER_MENU_OPEN_PROGRESS_BONUS", "0.04"))
WRONG_AFFORDANCE_PENALTY = float(os.environ.get("REMOTE_WRONG_AFFORDANCE_PENALTY", "0.18"))
SCREENSHOT_STALL_PENALTY = float(os.environ.get("REMOTE_SCREENSHOT_STALL_PENALTY", "0.08"))
SCREENSHOT_STALL_DIFF_THRESHOLD = float(os.environ.get("REMOTE_SCREENSHOT_STALL_DIFF_THRESHOLD", "0.2"))
SCREENSHOT_ZERO_DIFF_EPS = float(os.environ.get("REMOTE_SCREENSHOT_ZERO_DIFF_EPS", "0.02"))
CYCLE_REPEAT_PENALTY = float(os.environ.get("REMOTE_CYCLE_REPEAT_PENALTY", "0.35"))
WAIT_REPEAT_THRESHOLD = int(os.environ.get("REMOTE_WAIT_REPEAT_THRESHOLD", "5"))
WAIT_REPEAT_PENALTY = float(os.environ.get("REMOTE_WAIT_REPEAT_PENALTY", "0.25"))
ZERO_DIFF_REPEAT_BREAK_PENALTY = float(os.environ.get("REMOTE_ZERO_DIFF_REPEAT_BREAK_PENALTY", "0.35"))
PARSE_FAIL_REPEAT_THRESHOLD = int(os.environ.get("REMOTE_PARSE_FAIL_REPEAT_THRESHOLD", "2"))
PARSE_FAIL_REPEAT_PENALTY = float(os.environ.get("REMOTE_PARSE_FAIL_REPEAT_PENALTY", "0.20"))
INVALID_BOX_LITERAL_PENALTY = float(os.environ.get("REMOTE_INVALID_BOX_LITERAL_PENALTY", "0.25"))
ALLOW_ACTION_ONLY_RESPONSE = os.environ.get("REMOTE_ALLOW_ACTION_ONLY_RESPONSE", "1").strip() not in {"0", "false", "False"}
PARSE_REPAIR_RETRY = os.environ.get("REMOTE_PARSE_REPAIR_RETRY", "1").strip() not in {"0", "false", "False"}


# --- Per-slot state (one instance per parallel env slot) ---
@dataclass
class SlotState:
    env: DesktopEnv | None = None
    history_messages: list = field(default_factory=list)
    is_done: bool = False
    step_counter: int = 0
    instruction: str | None = None
    _last_action_signature: str | None = None
    _repeat_action_count: int = 0
    _last_semantic_action_key: str | None = None
    _semantic_repeat_count: int = 0
    _last_intent_key: str | None = None
    _intent_repeat_count: int = 0
    _last_screenshot_fingerprint: object = None
    _recent_action_signatures: deque = field(default_factory=lambda: deque(maxlen=4))
    _last_step_reward_components: dict = field(default_factory=dict)
    _eval_precondition_state: dict | None = None
    _parse_fail_streak: int = 0


# Slot pool: slot_id -> SlotState; protected by _slots_lock
_slots: dict[int, SlotState] = {}
_slots_lock = threading.RLock()
# Per-slot init locks: prevent two threads from double-initializing the same slot's DesktopEnv
_slot_init_locks: dict[int, threading.Lock] = {}
_slot_init_locks_meta = threading.Lock()
# Per-slot endpoint locks: serialize concurrent step/reset/evaluate for same slot
_slot_endpoint_locks: dict[int, threading.RLock] = {}
_slot_endpoint_locks_meta = threading.Lock()


def _get_slot_init_lock(slot_id: int) -> threading.Lock:
    with _slot_init_locks_meta:
        if slot_id not in _slot_init_locks:
            _slot_init_locks[slot_id] = threading.Lock()
        return _slot_init_locks[slot_id]


def _get_slot_endpoint_lock(slot_id: int) -> threading.RLock:
    with _slot_endpoint_locks_meta:
        if slot_id not in _slot_endpoint_locks:
            _slot_endpoint_locks[slot_id] = threading.RLock()
        return _slot_endpoint_locks[slot_id]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    provider = (os.environ.get("PROVIDER") or _default_provider()).strip().lower() or "docker"
    if provider not in ("docker", "vmware", "aws"):
        provider = "docker"
    logger.info("Startup: PROVIDER=%s", provider)
    print(f"[remote_env_server] Startup: PROVIDER={provider}", file=sys.stderr, flush=True)
    yield


app = FastAPI(title="OSWorld Remote Env", version="0.1.0", lifespan=_lifespan)


class ResetRequest(BaseModel):
    task_config: dict
    slot_id: int = 0


class StepRequest(BaseModel):
    prediction: str
    slot_id: int = 0


class SlotRequest(BaseModel):
    slot_id: int = 0


def _build_init_messages(screenshot_bytes: bytes, instruction_text: str) -> list:
    b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
    return [
        {"role": "system", "content": [{"type": "text", "text": "Your are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": uitars_system_prompt.format(instruction=instruction_text)}]},
        {
            "role": "user",
            "content": [{
                "type": "image",
                "image": f"data:image/jpeg;base64,{b64}",
                "min_pixels": IMAGE_MIN_PIXELS,
                "max_pixels": IMAGE_MAX_PIXELS,
            }],
        },
    ]


def _instruction_with_hints(task_config: dict) -> str:
    text = (task_config.get("instruction") or "").strip()
    domain = (task_config.get("domain") or "").strip().lower()
    lower = text.lower()
    if domain == "gimp":
        text += "\n\nImportant: Use GIMP (the image editor), not Ubuntu Settings/System Settings."
        if any(k in lower for k in ("vibrancy", "saturation", "brightness", "color", "contrast")):
            text += (
                "\n\nGIMP hint: Open GIMP, open/select the image, then use the top menu (Colors) for color changes. "
                "Do not repeatedly click the same sidebar area if no menu/dialog opens."
            )
    if "bing" in lower and "search" in lower:
        text += (
            "\n\nImportant: This is a browser setting in Chrome/Firefox (default search engine), not Ubuntu system settings. "
            "In Chrome, use the three-dot menu in the top-right (not a gear icon on the page)."
        )
    if ("privacy" in lower or "tracking" in lower or "cookies" in lower) and any(k in lower for k in ("chrome", "browser", "amazon", "internet")):
        text += (
            "\n\nBrowser privacy hint: Open the browser menu (three dots), then navigate to Settings/Privacy. "
            "Do not keep clicking the page content or the same menu icon repeatedly if the menu is already open."
        )
    if "shortcut" in lower and ("site" in lower or "page" in lower or "website" in lower):
        text += (
            "\n\nImportant: Create a browser/webpage shortcut on the Desktop (not just a terminal file/folder). "
            "Use the browser menu path for creating/saving a webpage shortcut. Do NOT use 'Save Target as...'."
        )
    if ("find discussions" in lower or "most replies" in lower or "community discussions" in lower) and "browser" not in lower:
        text += (
            "\n\nImportant: Use a search engine and click actual result links/pages. "
            "Do not type a synthetic URL made from the query text."
        )
    if any(k in lower for k in ("trash", "deleted", "recover")) and any(k in lower for k in ("ubuntu", "poster", "file", "night")):
        text += (
            "\n\nTrash recovery hint: Open Files, click Trash in the left sidebar, locate the target file, then Restore/Move out of Trash. "
            "Do not use browser search or web search for this task."
        )
    if "vlc" in lower and any(k in lower for k in ("video", "music video", "desktop", "play")):
        text += (
            "\n\nVLC hint: Open VLC directly, then use Media/Open File (or the VLC file picker) to open the video from Desktop. "
            "Do not repeatedly click the same launcher icon if VLC is already open."
        )
    text += (
        "\n\nOutput format requirement (strict): Always include an `Action:` line every step. "
        "Do not return only explanation text. Example: `Action: click(start_box='(x,y)')` or `Action: WAIT`."
    )
    if ALLOW_ACTION_ONLY_RESPONSE:
        text += (
            "\n\nFor this model, `Thought:` is optional when needed for reliability. "
            "You may return only `Action:` as long as it is parser-compatible."
        )
    return text


def _make_action_signature(parsed_responses, actions) -> str:
    try:
        def _bucket_num(v, size=24):
            try:
                x = float(v)
                return int(round(x / size) * size)
            except Exception:
                return v

        def _normalize_action_inputs(ai):
            if not isinstance(ai, dict):
                return ai
            out = {}
            for k, v in ai.items():
                if isinstance(v, (list, tuple)):
                    out[k] = [_bucket_num(x) for x in v]
                elif isinstance(v, dict):
                    out[k] = {kk: _bucket_num(vv) for kk, vv in v.items()}
                else:
                    out[k] = _bucket_num(v)
            return out

        if parsed_responses:
            compact = []
            for pr in parsed_responses:
                compact.append({
                    "action_type": pr.get("action_type"),
                    "action_inputs": _normalize_action_inputs(pr.get("action_inputs", {})),
                })
            return repr(compact)
        normalized_actions = []
        for a in actions:
            if isinstance(a, str):
                # Bucket click coords so tiny jitter does not evade loop detection.
                a = re.sub(
                    r"pyautogui\.click\(([-\d.]+),\s*([-\d.]+),",
                    lambda m: f"pyautogui.click({_bucket_num(m.group(1))}, {_bucket_num(m.group(2))},",
                    a,
                )
            normalized_actions.append(a)
        return repr(normalized_actions)
    except Exception:
        return "sig_error"


def _append_screenshot_message(messages: list, screenshot) -> bool:
    """Append a screenshot image message to history; returns True on success."""
    try:
        if screenshot is None:
            return False
        if not isinstance(screenshot, bytes):
            from PIL import Image
            buf = BytesIO()
            Image.open(BytesIO(screenshot) if isinstance(screenshot, bytes) else screenshot).save(buf, format="JPEG")
            screenshot = buf.getvalue()
        b64 = base64.b64encode(screenshot).decode("utf-8")
        messages.append({
            "role": "user",
            "content": [{
                "type": "image",
                "image": f"data:image/jpeg;base64,{b64}",
                "min_pixels": IMAGE_MIN_PIXELS,
                "max_pixels": IMAGE_MAX_PIXELS,
            }],
        })
        return True
    except Exception:
        print("Failed to append step screenshot to history_messages")
        print(traceback.format_exc())
        return False


def _screenshot_fingerprint(screenshot):
    try:
        from PIL import Image
        if isinstance(screenshot, bytes):
            img = Image.open(BytesIO(screenshot))
        else:
            img = Image.open(BytesIO(screenshot) if isinstance(screenshot, bytes) else screenshot)
        img = img.convert("L").resize((32, 18))
        return list(img.getdata())
    except Exception:
        return None


def _screenshot_diff_score(prev_fp, curr_fp) -> float:
    try:
        if not prev_fp or not curr_fp or len(prev_fp) != len(curr_fp):
            return 999.0
        return sum(abs(a - b) for a, b in zip(prev_fp, curr_fp)) / len(prev_fp)
    except Exception:
        return 999.0


def _extract_semantic_action_key(actions) -> str | None:
    try:
        if not actions or not isinstance(actions[0], str):
            return None
        a = actions[0]
        m = re.search(r"pyautogui\.write\('([^']*)'", a)
        if m:
            txt = m.group(1).strip()
            lower = txt.lower()
            # Repeated malformed URL assembly is a common browser failure mode.
            if (
                lower.count("http://") + lower.count("https://") > 1
                or (("http://" in lower or "https://" in lower or "www." in lower) and " " in lower)
                or (".com" in lower and " " in lower)
                or ("www." in lower and "https://" in lower)
            ):
                return "write_bad_url"
            for cmd in ("cd", "ls", "mkdir", "touch", "rm", "mv", "cp", "ln", "chmod", "chown", "echo", "cat", "pwd", "sudo"):
                if lower == cmd or lower.startswith(cmd + " "):
                    return f"write_cmd:{cmd}"
            return f"write:{txt[:80]}"
        m = re.search(r"pyautogui\.hotkey\(([^)]*)\)", a)
        if m:
            return f"hotkey:{m.group(1).replace(' ', '')}"
        m = re.search(r"pyautogui\.press\('([^']*)'\)", a)
        if m:
            return f"press:{m.group(1)}"
        return None
    except Exception:
        return None


def _extract_intent_key(prediction: str, actions) -> str | None:
    try:
        if not prediction:
            return None
        low = prediction.lower()
        if _is_click_like_action(actions):
            if any(s in low for s in ("cookie", "consent", "accept all", "confirm my choices", "privacy banner")):
                return "cookie_banner_intent"
            if "gear icon" in low and any(s in low for s in ("search engine", "bing", "browser settings", "chrome")):
                return "browser_gear_icon_intent"
            if "save target as" in low and "shortcut" in low:
                return "shortcut_save_target_as_intent"
        return None
    except Exception:
        return None


def _is_click_like_action(actions) -> bool:
    try:
        if not actions or not isinstance(actions[0], str):
            return False
        a = actions[0]
        return any(x in a for x in ("pyautogui.click(", "pyautogui.doubleClick(", "pyautogui.rightClick("))
    except Exception:
        return False


def _is_wait_action(actions) -> bool:
    try:
        return bool(actions) and actions[0] == "WAIT"
    except Exception:
        return False


def _extract_click_xy(actions) -> tuple[float, float] | None:
    try:
        if not actions or not isinstance(actions[0], str):
            return None
        a = actions[0]
        m = re.search(
            r"pyautogui\.(?:click|doubleClick|rightClick)\(([-\d.]+),\s*([-\d.]+)",
            a,
        )
        if not m:
            return None
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None


def _detect_invalid_box_literal(prediction: str) -> str | None:
    """Detect malformed start_box/end_box literals that commonly break parser (e.g. '', text labels)."""
    try:
        if not prediction:
            return None
        # Only inspect the action suffix to avoid matching prose examples in Thought.
        action_part = prediction.split("Action:")[-1] if "Action:" in prediction else prediction
        for _, raw in re.findall(r"(start_box|end_box)\s*=\s*'([^']*)'", action_part):
            s = (raw or "").strip()
            if not s:
                return "empty_box_literal"
            if "<|box_start|>" in s and "<|box_end|>" in s:
                continue
            # Common malformed labeled literal from small models, e.g. "()='Activities'".
            if any(ch.isalpha() for ch in s) and ("(" in s or ")" in s or "=" in s):
                return "invalid_box_label_literal"
            # Plain text/non-coordinate literal.
            if any(ch.isalpha() for ch in s):
                return "invalid_box_text_literal"
            # Accept "(x,y)" or "(x1,y1,x2,y2)" numeric tuples only.
            if re.fullmatch(
                r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?)?\s*\)",
                s,
            ):
                continue
            return "invalid_box_literal"
        return None
    except Exception:
        return None


def _repair_prediction_for_parser(prediction: str) -> tuple[str, str] | None:
    """Heuristic parser repair for common 2B formatting failures."""
    try:
        if not prediction:
            return None
        text = prediction.strip()
        if not text:
            return None

        # Case 1: Missing Action: line but contains a recognizable action call.
        if "Action:" not in text:
            m = re.search(
                r"\b(click|left_double|right_single|drag|hotkey|type|press|scroll|wait|finished|error_env)\s*\([^)]*\)?",
                text,
                re.DOTALL,
            )
            if m:
                action_str = m.group(0).strip()
                repaired = f"Action: {action_str}"
                return repaired, "inject_action_prefix"
            # If there is no action call, do not invent one.
            return None

        # Case 2: Truncated click-like action missing a closing tuple/paren; salvage visible coordinates.
        action_part = text.split("Action:")[-1]
        if any(fn in action_part for fn in ("click(", "left_double(", "right_single(")):
            fn_match = re.search(r"\b(click|left_double|right_single)\s*\(", action_part)
            coord_match = re.search(r"\((\d{1,4})\s*,\s*(\d{1,4})\)", action_part)
            if fn_match and coord_match and not re.search(r"start_box='\([^)]*\)'\)", action_part):
                fn = fn_match.group(1)
                x, y = coord_match.group(1), coord_match.group(2)
                repaired_action = f"{fn}(start_box='({x},{y})')"
                prefix = text.split("Action:")[0]
                repaired = f"{prefix}Action: {repaired_action}".strip()
                return repaired, "salvage_click_coords"

        return None
    except Exception:
        return None


def _task_family_from_instruction(instr: str | None) -> str:
    low = (instr or "").lower()
    if "trash" in low or ("recover" in low and "deleted" in low):
        return "trash"
    if "vlc" in low and ("video" in low or "music" in low or "play" in low):
        return "vlc"
    if "gimp" in low or any(k in low for k in ("vibrancy", "saturation", "brightness", "contrast")):
        return "gimp"
    if any(k in low for k in ("tracking", "cookies", "privacy")) and any(k in low for k in ("amazon", "browser", "chrome")):
        return "browser_privacy"
    if "bing" in low and "search" in low:
        return "browser_settings"
    if "shortcut" in low and any(k in low for k in ("site", "page", "website", "webpage")):
        return "browser_shortcut"
    if any(k in low for k in ("find discussions", "community discussions", "most replies")):
        return "browser_search"
    return "other"


def _is_edge_or_corner_click(x: float, y: float, w: float = 1920, h: float = 1080) -> bool:
    margin_x = 0.06 * w
    margin_y = 0.06 * h
    return x <= margin_x or x >= (w - margin_x) or y <= margin_y or y >= (h - margin_y)


def _click_in_expected_zone(family: str, x: float, y: float, w: float = 1920, h: float = 1080) -> bool:
    xn, yn = x / w, y / h
    # Broad, soft zones only. We use this only for extra penalty on non-progress clicks.
    if family == "trash":
        # Files left sidebar/list area or left dock (to open Files)
        return (xn < 0.35 and yn > 0.05) or (xn < 0.08)
    if family == "vlc":
        # left dock/app launcher, top menu bar, center content/file picker
        return (xn < 0.10) or (yn < 0.14) or (0.12 < xn < 0.92 and 0.12 < yn < 0.92)
    if family == "gimp":
        # left dock to open, top menu, main canvas/dialog area
        return (xn < 0.10) or (yn < 0.14) or (0.10 < xn < 0.96 and 0.10 < yn < 0.96)
    if family == "browser_privacy":
        # browser top-right menu/settings and settings content pane
        return (xn > 0.82 and yn < 0.18) or (0.08 < xn < 0.96 and 0.10 < yn < 0.96)
    if family == "browser_settings":
        return (xn > 0.82 and yn < 0.18) or (0.08 < xn < 0.96 and 0.10 < yn < 0.96)
    if family == "browser_shortcut":
        return (xn > 0.82 and yn < 0.18) or (0.08 < xn < 0.96 and 0.10 < yn < 0.96)
    if family == "browser_search":
        # address/search bar/top controls and main results/content area
        return (0.08 < xn < 0.95 and yn < 0.18) or (0.05 < xn < 0.96 and 0.14 < yn < 0.94)
    return True


def _looks_like_browser_menu_click(x: float, y: float, w: float = 1920, h: float = 1080) -> bool:
    xn, yn = x / w, y / h
    return xn > 0.88 and yn < 0.14


def _is_abab_cycle(sig_deque) -> bool:
    try:
        if len(sig_deque) < 4:
            return False
        a, b, c, d = list(sig_deque)[-4:]
        return a == c and b == d and a != b
    except Exception:
        return False


def _domain_matches_cookie(rule_domain: str, cookie_domain: str) -> bool:
    try:
        rd = (rule_domain or "").strip().lower()
        cd = (cookie_domain or "").strip().lower()
        if not rd or not cd:
            return False
        if rd == cd:
            return True
        rd_core = rd.lstrip(".")
        cd_core = cd.lstrip(".")
        return cd_core == rd_core or cd_core.endswith("." + rd_core)
    except Exception:
        return False


def _extract_cookie_deletion_rule(env) -> dict | None:
    try:
        evaluator = getattr(env, "evaluator", None) or {}
        if evaluator.get("func") != "is_cookie_deleted":
            return None
        expected = evaluator.get("expected") or {}
        rules = expected.get("rules") if isinstance(expected, dict) else None
        if not isinstance(rules, dict):
            return None
        if rules.get("type") != "domains":
            return None
        return rules
    except Exception:
        return None


def _cookie_deletion_evidence(env, when: str) -> dict | None:
    """Collect evidence for cookie-deletion evaluators (e.g., Amazon tracking task)."""
    try:
        rules = _extract_cookie_deletion_rule(env)
        if not rules:
            return None
        result_cfg = (getattr(env, "evaluator", {}) or {}).get("result")
        if not result_cfg or not getattr(env, "result_getter", None):
            return None
        cookie_data = env.result_getter(env, result_cfg)
        if not cookie_data:
            matched_domains = []
            total = 0
        else:
            total = len(cookie_data)
            cookie_domains = []
            for row in cookie_data:
                try:
                    cookie_domains.append(str(row[1]))
                except Exception:
                    continue
            matched_domains = sorted(
                {cd for cd in cookie_domains for rd in rules.get("domains", []) if _domain_matches_cookie(rd, cd)}
            )
        would_pass = 1.0 if len(matched_domains) == 0 else 0.0
        evidence = {
            "when": when,
            "rule_type": rules.get("type"),
            "rule_domains": list(rules.get("domains", [])),
            "total_cookies": total,
            "matched_cookie_domains": matched_domains[:20],
            "matched_count": len(matched_domains),
            "would_pass_is_cookie_deleted": would_pass,
        }
        print(
            "eval_cookie_evidence:"
            f" when={when} rules={evidence['rule_domains']!r}"
            f" matched_count={evidence['matched_count']}"
            f" matched_domains={evidence['matched_cookie_domains']!r}"
            f" total_cookies={evidence['total_cookies']}"
            f" would_pass={would_pass}"
        )
        return evidence
    except Exception as e:
        print(f"eval_cookie_evidence: error collecting {when}: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return None


def _history_deletion_evidence(env, when: str) -> dict | None:
    """Collect evidence for history-deletion evaluators."""
    try:
        evaluator = getattr(env, "evaluator", None) or {}
        if evaluator.get("func") != "check_history_deleted":
            return None
        expected = evaluator.get("expected") or {}
        rules = expected.get("rules") if isinstance(expected, dict) else None
        if not isinstance(rules, dict) or rules.get("type") != "rule":
            # OSWorld shape is expected={type: rule, rules:{type: keywords, keywords:[...]}}
            pass
        inner_rules = (expected or {}).get("rules", {}) if isinstance(expected, dict) else {}
        if not isinstance(inner_rules, dict) or inner_rules.get("type") != "keywords":
            return None
        result_cfg = evaluator.get("result")
        if not result_cfg or not getattr(env, "result_getter", None):
            return None
        history_data = env.result_getter(env, result_cfg) or []
        urls = []
        for row in history_data:
            try:
                urls.append(str(row[0]))
            except Exception:
                continue
        keywords = list(inner_rules.get("keywords", []))
        matches = sorted({u for u in urls for k in keywords if k in u})
        would_pass = 1.0 if not matches else 0.0
        evidence = {
            "kind": "history_deleted",
            "when": when,
            "keywords": keywords,
            "matched_count": len(matches),
            "matched_urls": matches[:20],
            "total_history_rows": len(history_data),
            "would_pass_check_history_deleted": would_pass,
        }
        print(
            "eval_history_evidence:"
            f" when={when} keywords={keywords!r} matched_count={len(matches)}"
            f" matched_urls={matches[:20]!r} total_rows={len(history_data)}"
            f" would_pass={would_pass}"
        )
        return evidence
    except Exception as e:
        print(f"eval_history_evidence: error collecting {when}: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return None


def _absence_metric_evidence(env, when: str) -> dict | None:
    return _cookie_deletion_evidence(env, when) or _history_deletion_evidence(env, when)


def _log_step_reward_final(slot: SlotState, format_reward: float, done_flag: bool) -> None:
    comps = slot._last_step_reward_components or {}
    parts = " ".join(f"{k}={v:+.2f}" for k, v in comps.items() if abs(v) > 1e-9)
    if parts:
        print(
            f"step_reward_final: format_reward={format_reward:.2f} is_done={done_flag} "
            f"step_counter={slot.step_counter} components[{parts}]"
        )
    else:
        print(f"step_reward_final: format_reward={format_reward:.2f} is_done={done_flag} step_counter={slot.step_counter}")


def _safe_env_pause(env) -> None:
    fn = getattr(env, "pause", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            print("env.pause() failed (ignored)")
            print(traceback.format_exc())


def _safe_env_unpause(env) -> None:
    fn = getattr(env, "unpause", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            print("env.unpause() failed (ignored)")
            print(traceback.format_exc())


def _get_slot(slot_id: int = 0) -> SlotState:
    """Return the SlotState for slot_id, creating a new DesktopEnv if this slot has no env yet.

    Thread safety: uses a per-slot init lock so only one thread ever initializes a given slot's
    DesktopEnv, even if multiple threads call _get_slot(same_id) concurrently.
    """
    with _slots_lock:
        if slot_id not in _slots:
            _slots[slot_id] = SlotState()
        slot = _slots[slot_id]

    # Guard env initialization with a per-slot lock to prevent TOCTOU double-init.
    # We check env outside the lock first for the fast path (already initialized).
    if slot.env is not None:
        return slot

    init_lock = _get_slot_init_lock(slot_id)
    with init_lock:
        # Re-check after acquiring init lock (double-checked locking pattern)
        if slot.env is not None:
            return slot
        # Default: VMware on macOS (no KVM), Docker on Linux. Override with PROVIDER=aws|vmware|docker.
        provider_name = (os.environ.get("PROVIDER") or _default_provider()).strip().lower()
        if provider_name not in ("docker", "vmware", "aws"):
            provider_name = "docker"
        if provider_name == "docker":
            if docker is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Docker provider requires the docker Python package. "
                        "Install it: pip install docker. Then ensure the Docker daemon is running (e.g. Docker Desktop or system docker)."
                    ),
                )
        if provider_name == "aws":
            print(f"[slot {slot_id}] ✓ Using provider: aws (EC2 instances; ensure boto3 and aws configure are set)")
        print(f"[slot {slot_id}] ✓ Using provider: {provider_name} (observation_type=screenshot, same as run_uitars)")
        global _provider_name
        _provider_name = provider_name
        region = os.environ.get("AWS_REGION", "us-east-1")
        snapshot_name = os.environ.get("OSWORLD_SNAPSHOT_AMI", "init_state")
        # For docker provider, pass path_to_vm=f"slot_{slot_id}" so provider can route to correct container
        path_to_vm_arg = f"slot_{slot_id}" if provider_name == "docker" else None
        try:
            slot.env = DesktopEnv(
                provider_name=provider_name,
                region=region,
                snapshot_name=snapshot_name,
                path_to_vm=path_to_vm_arg,
                action_space="pyautogui",
                screen_size=(1920, 1080),
                cache_dir=f"cache_dirs/cache_{slot_id}",
                headless=True,
                os_type="Ubuntu",
                require_a11y_tree=False,
            )
            print(f"[slot {slot_id}] DesktopEnv initialized successfully (provider={provider_name})")
        except HTTPException:
            raise
        except OSError as e:
            if e.errno == 28:  # No space left on device
                print(f"[slot {slot_id}] Env init failed: {e}", flush=True)
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "No space left on device on the env server. Free disk space (e.g. docker system prune -a, "
                        "remove cache_dirs/vms) or resize the EBS volume. Then restart the server."
                    ),
                ) from e
            raise
        except Exception as e:
            print(f"[slot {slot_id}] Env init failed: {type(e).__name__}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            is_docker_error = (
                docker is not None
                and isinstance(e, docker.errors.DockerException)
            ) or "docker" in str(e).lower() or "connection aborted" in str(e).lower()
            if is_docker_error:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Docker is not available. Install Docker Desktop and start it, or ensure the Docker "
                        "daemon is running and the socket is available (e.g. /var/run/docker.sock). "
                        f"Original error: {e}"
                    ),
                ) from e
            if isinstance(e, FileNotFoundError) and "SKIP_DOCKER_VM_DOWNLOAD" in str(e):
                raise HTTPException(status_code=503, detail=str(e)) from e
            raise

    return slot


@app.post("/env/reset")
def env_reset(body: ResetRequest):
    slot_id = body.slot_id
    endpoint_lock = _get_slot_endpoint_lock(slot_id)
    with endpoint_lock:
        return _env_reset_locked(slot_id, body)


def _env_reset_locked(slot_id: int, body: ResetRequest):
    """Inner reset logic, called with the per-slot endpoint lock held."""
    slot = _get_slot(slot_id)
    task_config = body.task_config
    env = slot.env

    try:
        obs = env.reset(task_config)
    except Exception as e:
        print(f"[slot {slot_id}] Env reset exception: {e}")
        print(traceback.format_exc())
        slot.is_done = True
        raise HTTPException(status_code=500, detail=f"env.reset() failed: {e}")

    # Reset SlotState ONLY after env.reset() succeeds to prevent stale state on failure
    slot.instruction = _instruction_with_hints(task_config)
    slot.step_counter = 0
    slot.is_done = False
    slot._last_action_signature = None
    slot._repeat_action_count = 0
    slot._last_semantic_action_key = None
    slot._semantic_repeat_count = 0
    slot._last_intent_key = None
    slot._intent_repeat_count = 0
    slot._last_screenshot_fingerprint = None
    slot._recent_action_signatures.clear()
    slot._eval_precondition_state = None
    slot._parse_fail_streak = 0
    slot.history_messages = []

    _safe_env_pause(env)
    screenshot = obs.get("screenshot")
    if screenshot is None:
        print(f"[slot {slot_id}] Reset: screenshot is None. Returning obs_messages=None.")
        slot.is_done = True
        return {"env_idx": slot_id, "obs_messages": None, "is_done": True, "format_reward": 0.0}
    if not isinstance(screenshot, bytes):
        from PIL import Image
        buf = BytesIO()
        Image.open(BytesIO(screenshot) if isinstance(screenshot, bytes) else screenshot).save(buf, format="JPEG")
        screenshot = buf.getvalue()

    slot.history_messages = _build_init_messages(screenshot, slot.instruction)
    slot._last_screenshot_fingerprint = _screenshot_fingerprint(screenshot)
    slot._eval_precondition_state = _absence_metric_evidence(env, when="reset_precondition")
    print(f"[slot {slot_id}] Reset OK: {len(screenshot)} bytes. Instruction: {slot.instruction[:60]}...")
    return {
        "env_idx": slot_id,
        "obs_messages": messages_to_wire(slot.history_messages),
        "is_done": False,
        "format_reward": 0.0,
    }


@app.post("/env/step")
def env_step(body: StepRequest):
    slot_id = body.slot_id
    endpoint_lock = _get_slot_endpoint_lock(slot_id)
    with endpoint_lock:
        return _env_step_locked(slot_id, body)


def _env_step_locked(slot_id: int, body: StepRequest):
    """Inner step logic, called with the per-slot endpoint lock held."""
    slot = _get_slot(slot_id)
    env = slot.env
    prediction = body.prediction
    action_parse_res_factor = 1000
    model_type = "qwen25vl"
    max_pixels = 16384 * 28 * 28
    min_pixels = 100 * 28 * 28
    obs_image_height, obs_image_width = 1080, 1920

    parsed_responses = []
    reward_components: dict[str, float] = {}
    parse_failed = False
    invalid_box_literal_key = None
    parse_input = prediction
    if PARSE_REPAIR_RETRY:
        repaired = _repair_prediction_for_parser(prediction)
        if repaired is not None and repaired[0] and repaired[0] != prediction:
            parse_input, repair_reason = repaired
            print(f"step_trace: parse_repair reason={repair_reason} before={prediction[:120]!r} after={parse_input[:120]!r}")
    try:
        if "Action:" not in (parse_input or ""):
            raise ValueError("Missing required 'Action:' line in model output")
        parsed_responses = parse_action_to_structure_output(
            parse_input, action_parse_res_factor, obs_image_height, obs_image_width, model_type, max_pixels, min_pixels
        )
        actions = []
        action_types = []
        for pr in parsed_responses:
            if "action_type" in pr:
                action_types.append(pr["action_type"])
                if pr["action_type"] == FINISH_WORD:
                    actions = ["DONE"]
                    break
                if pr["action_type"] in (WAIT_WORD, ENV_FAIL_WORD, CALL_USER):
                    actions = ["WAIT"] if pr["action_type"] == WAIT_WORD else ["FAIL"]
                    break
            code = parsing_response_to_pyautogui_code(pr, obs_image_height, obs_image_width, False)
            actions.append(code)

        # Validate that all action types are known; unknown types are a parse error
        _KNOWN_ACTION_TYPES = {
            "hotkey", "press", "keyup", "keydown", "type", "drag", "select",
            "scroll", "click", "left_single", "left_double", "right_single", "hover",
            FINISH_WORD, WAIT_WORD, ENV_FAIL_WORD, CALL_USER,
        }
        unknown_types = [at for at in action_types if at not in _KNOWN_ACTION_TYPES]
        if unknown_types:
            raise ValueError(f"Unknown action type(s): {unknown_types}")

        # Conservative shaping: reward parse quality lightly; task evaluator should dominate.
        format_reward = FORMAT_PARSE_BASE_REWARD
        reward_components["parse"] = FORMAT_PARSE_BASE_REWARD
        
        # Check if we have meaningful GUI actions (not just DONE/WAIT/FAIL)
        if actions and actions[0] not in ["DONE", "WAIT", "FAIL"]:
            format_reward += FORMAT_EXEC_ACTION_BONUS
            reward_components["exec"] = reward_components.get("exec", 0.0) + FORMAT_EXEC_ACTION_BONUS
        
        # Check for FINISH_WORD (task completion signal from LLM)
        if FINISH_WORD in action_types:
            format_reward += FORMAT_FINISH_BONUS
            reward_components["finish"] = reward_components.get("finish", 0.0) + FORMAT_FINISH_BONUS
        slot._parse_fail_streak = 0

    except Exception:
        print("Parse action error:", prediction)
        print(traceback.format_exc())
        parse_failed = True
        invalid_box_literal_key = _detect_invalid_box_literal(parse_input or prediction)
        slot._parse_fail_streak += 1
        # Keep a strong parse penalty, but avoid instantly collapsing every bad sample.
        format_reward = -0.4
        reward_components["parse_error"] = -0.4
        # Don't terminate immediately on malformed action text; give the policy another chance next step.
        actions = ["WAIT"]

    # Log parse outcome so we can see if training is sending meaningful actions (on-policy)
    pred_preview = (prediction or "")[:120].replace("\n", " ")
    action_preview = actions[0] if actions else "none"
    parse_status = "fail" if format_reward < 0 else "ok"
    print(f"step_parse: {parse_status} actions=[{action_preview}] format_reward={format_reward:.2f} pred_preview={pred_preview!r}")
    click_xy = _extract_click_xy(actions)
    task_family = _task_family_from_instruction(slot.instruction)

    action_signature = _make_action_signature(parsed_responses, actions)
    slot._recent_action_signatures.append(action_signature)
    if action_signature == slot._last_action_signature:
        slot._repeat_action_count += 1
    else:
        slot._last_action_signature = action_signature
        slot._repeat_action_count = 1
    print(f"[slot {slot_id}] step_trace: repeat_action_count={slot._repeat_action_count} threshold={REPEAT_ACTION_THRESHOLD}")

    if parse_failed and invalid_box_literal_key:
        format_reward = max(format_reward - INVALID_BOX_LITERAL_PENALTY, -1.0)
        reward_components[invalid_box_literal_key] = reward_components.get(invalid_box_literal_key, 0.0) - INVALID_BOX_LITERAL_PENALTY
        print(f"step_trace: {invalid_box_literal_key}; penalty={INVALID_BOX_LITERAL_PENALTY:.2f}")

    if parse_failed and slot._parse_fail_streak >= PARSE_FAIL_REPEAT_THRESHOLD:
        format_reward = max(format_reward - PARSE_FAIL_REPEAT_PENALTY, -1.0)
        reward_components["parse_fail_repeat"] = reward_components.get("parse_fail_repeat", 0.0) - PARSE_FAIL_REPEAT_PENALTY
        slot.is_done = True
        print(
            f"loop_breaker: parse_fail_streak x{slot._parse_fail_streak}; "
            f"terminating episode with penalty {PARSE_FAIL_REPEAT_PENALTY:.2f}"
        )
        slot._last_step_reward_components = reward_components
        _log_step_reward_final(slot, format_reward, slot.is_done)
        final_obs = messages_to_wire(slot.history_messages) if slot.history_messages else None
        return {"env_idx": slot_id, "obs_messages": final_obs, "is_done": True, "format_reward": format_reward}

    semantic_key = _extract_semantic_action_key(actions)
    if semantic_key and semantic_key == slot._last_semantic_action_key:
        slot._semantic_repeat_count += 1
    else:
        slot._last_semantic_action_key = semantic_key
        slot._semantic_repeat_count = 1 if semantic_key else 0
    if semantic_key and slot._semantic_repeat_count >= SEMANTIC_REPEAT_THRESHOLD:
        format_reward = max(format_reward - SEMANTIC_REPEAT_PENALTY, -1.0)
        reward_components["semantic_repeat"] = reward_components.get("semantic_repeat", 0.0) - SEMANTIC_REPEAT_PENALTY
        print(
            f"step_trace: semantic_repeat key={semantic_key!r} count={slot._semantic_repeat_count} "
            f"penalty={SEMANTIC_REPEAT_PENALTY:.2f}"
        )

    intent_key = _extract_intent_key(prediction, actions)
    if intent_key and intent_key == slot._last_intent_key:
        slot._intent_repeat_count += 1
    else:
        slot._last_intent_key = intent_key
        slot._intent_repeat_count = 1 if intent_key else 0
    if intent_key and slot._intent_repeat_count >= COOKIE_BANNER_INTENT_THRESHOLD:
        if intent_key == "cookie_banner_intent":
            format_reward = max(format_reward - COOKIE_BANNER_INTENT_PENALTY, -1.0)
            reward_components["cookie_banner_intent"] = (
                reward_components.get("cookie_banner_intent", 0.0) - COOKIE_BANNER_INTENT_PENALTY
            )
            print(
                f"step_trace: intent_repeat key={intent_key!r} count={slot._intent_repeat_count} "
                f"penalty={COOKIE_BANNER_INTENT_PENALTY:.2f}"
            )
    if intent_key in {"browser_gear_icon_intent", "shortcut_save_target_as_intent"}:
        format_reward = max(format_reward - WRONG_AFFORDANCE_PENALTY, -1.0)
        reward_components["wrong_affordance"] = reward_components.get("wrong_affordance", 0.0) - WRONG_AFFORDANCE_PENALTY
        print(f"step_trace: wrong_affordance key={intent_key!r} penalty={WRONG_AFFORDANCE_PENALTY:.2f}")

    if _is_abab_cycle(slot._recent_action_signatures):
        format_reward = max(format_reward - CYCLE_REPEAT_PENALTY, -1.0)
        reward_components["abab_cycle"] = reward_components.get("abab_cycle", 0.0) - CYCLE_REPEAT_PENALTY
        slot.is_done = True
        print(
            f"loop_breaker: abab_cycle detected last4={list(slot._recent_action_signatures)!r}; "
            f"terminating episode with penalty {CYCLE_REPEAT_PENALTY:.2f}"
        )
        slot._last_step_reward_components = reward_components
        _log_step_reward_final(slot, format_reward, slot.is_done)
        final_obs = messages_to_wire(slot.history_messages) if slot.history_messages else None
        return {"env_idx": slot_id, "obs_messages": final_obs, "is_done": True, "format_reward": format_reward}

    # Keep trajectory order consistent: assistant action text, then resulting screenshot(s).
    slot.history_messages.append({"role": "assistant", "content": [{"type": "text", "text": add_box_token(prediction)}]})

    repeat_limit = WAIT_REPEAT_THRESHOLD if _is_wait_action(actions) else REPEAT_ACTION_THRESHOLD
    if slot._repeat_action_count > repeat_limit:
        repeat_penalty = WAIT_REPEAT_PENALTY if _is_wait_action(actions) else REPEAT_ACTION_PENALTY
        format_reward = max(format_reward - repeat_penalty, -1.0)
        reward_components["repeat_loop"] = reward_components.get("repeat_loop", 0.0) - repeat_penalty
        slot.is_done = True
        print(
            f"loop_breaker: repeated action signature x{slot._repeat_action_count}; "
            f"action={actions[0] if actions else None!r} threshold={repeat_limit} penalty={repeat_penalty:.2f}"
        )
        slot._last_step_reward_components = reward_components
        _log_step_reward_final(slot, format_reward, slot.is_done)
        final_obs = messages_to_wire(slot.history_messages) if slot.history_messages else None
        return {"env_idx": slot_id, "obs_messages": final_obs, "is_done": True, "format_reward": format_reward}

    _safe_env_unpause(env)
    obs = None
    step_successful = False
    step_stall_penalized = False
    appended_any_step_screenshot = False
    for action in actions:
        obs, reward, step_done, info = env.step(action, pause=ACTION_PAUSE_SEC)
        if step_done:
            slot.is_done = True
        slot.step_counter += 1
        has_screenshot = obs is not None and obs.get("screenshot") is not None
        if has_screenshot:
            step_successful = True
            curr_fp = _screenshot_fingerprint(obs.get("screenshot"))
            if _is_click_like_action(actions):
                diff_score = _screenshot_diff_score(slot._last_screenshot_fingerprint, curr_fp)
                if diff_score < SCREENSHOT_STALL_DIFF_THRESHOLD:
                    format_reward = max(format_reward - SCREENSHOT_STALL_PENALTY, -1.0)
                    reward_components["stall"] = reward_components.get("stall", 0.0) - SCREENSHOT_STALL_PENALTY
                    step_stall_penalized = True
                    print(
                        f"step_trace: screenshot_stall diff={diff_score:.3f} "
                        f"penalty={SCREENSHOT_STALL_PENALTY:.2f}"
                    )
                    if click_xy is not None:
                        cx, cy = click_xy
                        if not _click_in_expected_zone(task_family, cx, cy):
                            format_reward = max(format_reward - INTENT_ZONE_MISMATCH_PENALTY, -1.0)
                            reward_components["intent_zone_mismatch"] = (
                                reward_components.get("intent_zone_mismatch", 0.0) - INTENT_ZONE_MISMATCH_PENALTY
                            )
                            print(
                                "step_trace: intent_zone_mismatch "
                                f"family={task_family} click=({cx:.1f},{cy:.1f}) "
                                f"penalty={INTENT_ZONE_MISMATCH_PENALTY:.2f}"
                            )
                        if _is_edge_or_corner_click(cx, cy) and diff_score <= SCREENSHOT_ZERO_DIFF_EPS:
                            format_reward = max(format_reward - EDGE_NOOP_CLICK_PENALTY, -1.0)
                            reward_components["edge_noop_click"] = (
                                reward_components.get("edge_noop_click", 0.0) - EDGE_NOOP_CLICK_PENALTY
                            )
                            print(
                                "step_trace: edge_noop_click "
                                f"click=({cx:.1f},{cy:.1f}) diff={diff_score:.3f} "
                                f"penalty={EDGE_NOOP_CLICK_PENALTY:.2f}"
                            )
                        if (
                            task_family in {"browser_privacy", "browser_settings", "browser_shortcut"}
                            and _looks_like_browser_menu_click(cx, cy)
                            and slot._repeat_action_count >= 2
                            and diff_score < SCREENSHOT_STALL_DIFF_THRESHOLD
                        ):
                            format_reward = max(format_reward - BROWSER_MENU_NOOPEN_PENALTY, -1.0)
                            reward_components["browser_menu_noopen"] = (
                                reward_components.get("browser_menu_noopen", 0.0) - BROWSER_MENU_NOOPEN_PENALTY
                            )
                            print(
                                "step_trace: browser_menu_noopen "
                                f"repeat={slot._repeat_action_count} diff={diff_score:.3f} "
                                f"penalty={BROWSER_MENU_NOOPEN_PENALTY:.2f}"
                            )
                        if (
                            task_family in {"browser_privacy", "browser_settings", "browser_shortcut"}
                            and _looks_like_browser_menu_click(cx, cy)
                            and diff_score >= SCREENSHOT_STALL_DIFF_THRESHOLD
                        ):
                            format_reward += BROWSER_MENU_OPEN_PROGRESS_BONUS
                            reward_components["browser_menu_progress"] = (
                                reward_components.get("browser_menu_progress", 0.0) + BROWSER_MENU_OPEN_PROGRESS_BONUS
                            )
                            print(
                                "step_trace: browser_menu_progress "
                                f"diff={diff_score:.3f} bonus={BROWSER_MENU_OPEN_PROGRESS_BONUS:.2f}"
                            )
                if (
                    slot._repeat_action_count >= 2
                    and diff_score <= SCREENSHOT_ZERO_DIFF_EPS
                    and not _is_wait_action(actions)
                ):
                    format_reward = max(format_reward - ZERO_DIFF_REPEAT_BREAK_PENALTY, -1.0)
                    reward_components["zero_diff_repeat_break"] = (
                        reward_components.get("zero_diff_repeat_break", 0.0) - ZERO_DIFF_REPEAT_BREAK_PENALTY
                    )
                    slot.is_done = True
                    print(
                        "loop_breaker: zero_diff_repeat_click;"
                        f" repeat_count={slot._repeat_action_count} diff={diff_score:.3f} "
                        f"penalty={ZERO_DIFF_REPEAT_BREAK_PENALTY:.2f}"
                    )
            slot._last_screenshot_fingerprint = curr_fp or slot._last_screenshot_fingerprint
            if _append_screenshot_message(slot.history_messages, obs.get("screenshot")):
                appended_any_step_screenshot = True
        if slot.step_counter >= max_steps:
            slot.is_done = True
        if slot.is_done:
            break

    _safe_env_pause(env)

    if step_successful and not step_stall_penalized and not parse_failed and not _is_wait_action(actions):
        format_reward += FORMAT_STEP_SUCCESS_BONUS
        reward_components["step_success"] = reward_components.get("step_success", 0.0) + FORMAT_STEP_SUCCESS_BONUS

    if obs is None and not actions:
        slot.is_done = True
        format_reward = max(format_reward - 0.1, -1.0)
        reward_components["no_obs"] = reward_components.get("no_obs", 0.0) - 0.1

    if slot.is_done:
        slot._last_step_reward_components = reward_components
        _log_step_reward_final(slot, format_reward, slot.is_done)
        # Return history so the client can see the final observation even on episode end
        final_obs = messages_to_wire(slot.history_messages) if slot.history_messages else None
        return {"env_idx": slot_id, "obs_messages": final_obs, "is_done": True, "format_reward": format_reward}

    if obs is None or obs.get("screenshot") is None:
        slot.is_done = True
        format_reward = max(format_reward - 0.1, -1.0)
        reward_components["missing_screenshot"] = reward_components.get("missing_screenshot", 0.0) - 0.1
        slot._last_step_reward_components = reward_components
        _log_step_reward_final(slot, format_reward, slot.is_done)
        final_obs = messages_to_wire(slot.history_messages) if slot.history_messages else None
        return {"env_idx": slot_id, "obs_messages": final_obs, "is_done": True, "format_reward": format_reward}

    if not appended_any_step_screenshot:
        _append_screenshot_message(slot.history_messages, obs["screenshot"])

    slot._last_step_reward_components = reward_components
    _log_step_reward_final(slot, format_reward, False)
    return {
        "env_idx": slot_id,
        "obs_messages": messages_to_wire(slot.history_messages),
        "is_done": False,
        "format_reward": format_reward,
    }


def _env_ready_for_evaluate(env) -> bool:
    """DesktopEnv with Docker provider only gets setup_controller after reset() calls _start_emulator()."""
    try:
        return getattr(env, "setup_controller", None) is not None
    except Exception:
        return False


@app.post("/env/evaluate")
def env_evaluate(body: SlotRequest):
    slot_id = body.slot_id
    endpoint_lock = _get_slot_endpoint_lock(slot_id)
    with endpoint_lock:
        return _env_evaluate_locked(slot_id)


def _env_evaluate_locked(slot_id: int):
    """Inner evaluate logic, called with the per-slot endpoint lock held."""
    slot = _get_slot(slot_id)
    env = slot.env
    _instr = (slot.instruction or "N/A")[:80]
    print(f"[slot {slot_id}] POST /env/evaluate received (instruction={_instr!r}, step_counter={slot.step_counter})")
    try:
        if not _env_ready_for_evaluate(env):
            print(f"[slot {slot_id}] Evaluation skipped: env not fully started; reset likely failed. Returning 503.")
            raise HTTPException(
                status_code=503,
                detail="Env not ready for evaluation (no setup_controller; reset may have failed). Client should retry.",
            )
        _safe_env_unpause(env)
        score = env.evaluate()
        post_evidence = _absence_metric_evidence(env, when="evaluate_post")
        if post_evidence is not None and slot._eval_precondition_state is not None:
            pre = slot._eval_precondition_state
            pre_match_count = int(pre.get("matched_count", 0))
            post_match_count = int(post_evidence.get("matched_count", 0))
            pre_pass = pre.get("would_pass_is_cookie_deleted", pre.get("would_pass_check_history_deleted"))
            print(
                "eval_absence_summary:"
                f" kind={pre.get('kind', 'cookie_deleted')} pre_matched_count={pre_match_count}"
                f" post_matched_count={post_match_count} pre_would_pass={pre_pass} raw_score={score}"
            )
            if float(score) == 1.0 and pre_match_count == 0:
                print(
                    "eval_precondition_fail: absence-only evaluator had no matching artifacts at reset; "
                    "treating as invalid success (score forced to 0.0)"
                )
                score = 0.0
        print(f"[slot {slot_id}] Evaluation completed: score={score}, step_counter={slot.step_counter}")
        return {"score": float(score)}
    except HTTPException:
        raise
    except AttributeError as e:
        if "setup_controller" in str(e):
            print(f"[slot {slot_id}] Evaluation skipped: env has no setup_controller (reset failed). Returning 503.")
            raise HTTPException(
                status_code=503,
                detail="Env not ready for evaluation (reset failed). Client should retry.",
            ) from e
        raise
    except Exception as e:
        err_msg = str(e)
        if "setup_controller" in err_msg:
            print(f"[slot {slot_id}] Evaluation skipped: env has no setup_controller (wrapped error). Returning 503.")
            raise HTTPException(
                status_code=503,
                detail="Env not ready for evaluation (no setup_controller). Client should retry.",
            ) from e
        print(f"[slot {slot_id}] Evaluation error:", e)
        print(traceback.format_exc())
        return {"score": 0.0}


@app.post("/env/history_messages")
def env_history_messages(body: SlotRequest):
    slot_id = body.slot_id
    slot = _get_slot(slot_id)
    return {"history_messages": messages_to_wire(slot.history_messages) if slot.history_messages else []}


@app.get("/env/history_messages")
def env_history_messages_get(slot_id: int = 0):
    """GET variant for compatibility — defaults to slot 0."""
    slot = _get_slot(slot_id)
    msgs = messages_to_wire(slot.history_messages) if slot.history_messages else []
    # Return under multiple keys for client compatibility
    return {"history_messages": msgs, "messages": msgs, "history": msgs}


@app.get("/health")
def health():
    """Health check: reports provider, pool size, and env statuses."""
    kvm_available = os.path.exists("/dev/kvm")
    with _slots_lock:
        pool_size = len(_slots)
        slot_statuses = {str(k): ("initialized" if v.env is not None else "not_initialized") for k, v in _slots.items()}
    return {
        "status": "ok",
        "provider": _provider_name,
        "observation_type": OBSERVATION_TYPE,
        "screenshot_source": "DesktopEnv.controller.get_screenshot() (same as run_uitars)",
        "kvm_available": kvm_available,
        "kvm_device": "/dev/kvm" if kvm_available else None,
        "pool_size": pool_size,
        "slot_statuses": slot_statuses,
        "message": "KVM hardware acceleration enabled" if kvm_available else "KVM not available, using software emulation"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=15001)
