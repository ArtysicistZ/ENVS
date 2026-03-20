"""Lightweight HTTP client for the remote env server.

This replaces RemoteEnvWorker for MCTS — no tokenizer/processor loading,
no train tensor management. Just HTTP calls to reset/step/evaluate/replay.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class MCTSEnvClient:
    """Lightweight env client for MCTS. Talks directly to the remote env server.

    Unlike RemoteEnvWorker, this does NOT:
    - Load a tokenizer or processor (MCTS handles inference in the driver)
    - Manage train tensors (no FSDP/DataProto)
    - Call process_message (MCTS builds vLLM inputs itself)

    It DOES:
    - reset, step, evaluate, replay, get_obs_screenshot
    """

    RESET_TIMEOUT = 300
    STEP_TIMEOUT = 60
    EVAL_TIMEOUT = 350
    REPLAY_TIMEOUT_PER_STEP = 5  # seconds per replayed action
    MAX_RETRIES = 2

    def __init__(self, worker_idx: int, remote_server_url: str, slot_id: int):
        self.worker_idx = worker_idx
        self.slot_id = slot_id
        self.remote_server_url = remote_server_url.rstrip("/")
        self.step_counter = 0
        self.is_done = False

    def _post(self, path: str, json_body: dict, timeout: float = 60) -> dict:
        url = f"{self.remote_server_url}{path}"
        r = requests.post(url, json=json_body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def reset(self, task_config: dict) -> dict:
        """Reset the VM to baseline and run task setup."""
        self.step_counter = 0
        self.is_done = False
        last_err = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                resp = self._post(
                    "/env/reset",
                    {"task_config": task_config, "slot_id": self.slot_id},
                    timeout=self.RESET_TIMEOUT,
                )
                self.is_done = resp.get("is_done", False)
                return resp
            except Exception as e:
                last_err = e
                logger.warning("Reset attempt %d failed for slot %d: %s",
                               attempt + 1, self.slot_id, e)
                if attempt < self.MAX_RETRIES:
                    time.sleep(5 * (attempt + 1))
        logger.error("Reset failed for slot %d after %d attempts: %s",
                      self.slot_id, self.MAX_RETRIES + 1, last_err)
        self.is_done = True
        return {"env_idx": self.worker_idx, "obs_messages": None, "is_done": True}

    def step(self, prediction: str) -> dict:
        """Execute one action on the VM."""
        try:
            resp = self._post(
                "/env/step",
                {"prediction": prediction, "slot_id": self.slot_id},
                timeout=self.STEP_TIMEOUT,
            )
            self.is_done = resp.get("is_done", False)
            self.step_counter += 1
            return resp
        except Exception as e:
            logger.error("Step failed for slot %d: %s", self.slot_id, e)
            self.is_done = True
            return {"env_idx": self.worker_idx, "obs_messages": None, "is_done": True}

    def evaluate(self) -> dict:
        """Evaluate the current VM state."""
        try:
            resp = self._post(
                "/env/evaluate",
                {"slot_id": self.slot_id},
                timeout=self.EVAL_TIMEOUT,
            )
            return resp
        except Exception as e:
            logger.error("Evaluate failed for slot %d: %s", self.slot_id, e)
            return {"env_idx": self.worker_idx, "score": 0.0}

    def replay(self, predictions: List[str], replay_pause_sec: float = 1.0) -> dict:
        """Replay a sequence of actions on the VM.

        The VM must already be reset+setup for the task. This replays the
        given action sequence to reach a specific state for branching.

        Falls back to sequential step() calls if /env/replay is not available.
        """
        timeout = max(60, len(predictions) * self.REPLAY_TIMEOUT_PER_STEP + 30)
        try:
            resp = self._post(
                "/env/replay",
                {
                    "predictions": predictions,
                    "slot_id": self.slot_id,
                    "replay_pause_sec": replay_pause_sec,
                },
                timeout=timeout,
            )
            self.step_counter = resp.get("steps_completed", len(predictions))
            self.is_done = not resp.get("success", True)
            return resp
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # /env/replay not implemented yet — fall back to sequential steps
                logger.warning("Replay endpoint not available, falling back to sequential step()")
                return self._replay_fallback(predictions, replay_pause_sec)
            raise
        except Exception as e:
            logger.warning("Replay failed for slot %d: %s. Falling back to sequential step()",
                            self.slot_id, e)
            return self._replay_fallback(predictions, replay_pause_sec)

    def _replay_fallback(self, predictions: List[str], pause_sec: float) -> dict:
        """Fallback: replay via sequential step() calls.

        Uses raw _post() instead of self.step() to avoid is_done side effects
        during replay — we're just reconstructing state, not running a real episode.
        """
        steps_completed = 0
        for pred in predictions:
            try:
                self._post(
                    "/env/step",
                    {"prediction": pred, "slot_id": self.slot_id},
                    timeout=self.STEP_TIMEOUT,
                )
                steps_completed += 1
            except Exception as e:
                logger.warning("Replay fallback step %d failed: %s", steps_completed, e)
                break
            if pause_sec > 0 and steps_completed < len(predictions):
                time.sleep(pause_sec)
        self.step_counter = steps_completed
        self.is_done = False  # replay doesn't mean we're done
        return {
            "success": steps_completed == len(predictions),
            "steps_completed": steps_completed,
            "fallback": True,
        }

    def get_obs_screenshot(self) -> Optional[str]:
        """Get the current screenshot as base64 JPEG.

        Fetches the latest observation messages and extracts the screenshot.
        """
        try:
            resp = self._post(
                "/env/history_messages",
                {"slot_id": self.slot_id},
                timeout=30,
            )
            messages = resp.get("messages") or resp.get("history_messages", [])
            # Extract the last screenshot from the messages
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
            return None
        except Exception as e:
            logger.error("get_obs_screenshot failed for slot %d: %s", self.slot_id, e)
            return None
