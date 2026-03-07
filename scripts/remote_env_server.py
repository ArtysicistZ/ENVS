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
"""
REMOTE_ENV_STAMP = "b36ed69-lifespan"  # grep this on Mac to confirm you have latest
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
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
import socket
import traceback
from contextlib import asynccontextmanager
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

# --- Runtime patching ---
def _patch_docker_provider_ports() -> None:
    """
    macOS can raise psutil.AccessDenied for psutil.net_connections(), which OSWorld's DockerProvider
    may use to find free ports. Patch the provider at runtime to find ports by socket bind
    + Docker container port inspection (no psutil).
    """
    try:
        from desktop_env.providers.docker.provider import DockerProvider  # type: ignore
    except Exception as e:
        logger.warning("Docker provider patch SKIPPED (import failed): %s. Port allocation may fail on macOS (psutil.AccessDenied).", e)
        return

    if getattr(DockerProvider, "_ARPO_PORT_PATCHED", False):
        return
    logger.info("Patching Docker provider for macOS: using socket bind + Docker ports (no psutil)...")

    def _get_docker_used_ports(self) -> set[int]:
        docker_ports: set[int] = set()
        try:
            for container in self.client.containers.list():
                ports = (container.attrs.get("NetworkSettings", {}) or {}).get("Ports") or {}
                for port_mappings in ports.values():
                    if port_mappings:
                        docker_ports.update(int(p["HostPort"]) for p in port_mappings)
        except Exception:
            pass
        return docker_ports

    def _is_port_available(self, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return True
        except OSError:
            return False

    def _get_available_port(self, start_port: int) -> int:
        docker_ports = _get_docker_used_ports(self)
        port = start_port
        while port < 65534:
            if port in docker_ports:
                port += 1
                continue
            if _is_port_available(self, port):
                return port
            port += 1
        raise RuntimeError(f"No available ports found starting from {start_port}")

    DockerProvider._get_available_port = _get_available_port  # type: ignore[attr-defined]

    # macOS (and some hosts) have no /dev/kvm; container.start() fails if we pass devices=["/dev/kvm"].
    # Patch start_emulator to only add /dev/kvm when it exists (use software emulation otherwise).
    _orig_start = DockerProvider.start_emulator
    _filelock = __import__("filelock", fromlist=["FileLock"]).FileLock
    _prov_logger = __import__("desktop_env.providers.docker.provider", fromlist=["logger"]).logger

    def _start_emulator_no_kvm_if_missing(self, path_to_vm: str, headless: bool, os_type: str, name=None):
        lock = _filelock(str(self.lock_file))
        try:
            with lock:
                self.vnc_port = self._get_available_port(8006)
                self.server_port = self._get_available_port(5000)
                self.chromium_port = self._get_available_port(9222)
                self.vlc_port = self._get_available_port(8080)
                devices = ["/dev/kvm"] if os.path.exists("/dev/kvm") else []
                if not devices:
                    _prov_logger.warning("/dev/kvm not found (e.g. on macOS); running VM with software emulation (slower).")
                self.container = self.client.containers.run(
                    "happysixd/osworld-docker",
                    environment=self.environment,
                    cap_add=["NET_ADMIN"],
                    devices=devices,
                    volumes={
                        os.path.abspath(path_to_vm): {"bind": "/System.qcow2", "mode": "ro"}
                    },
                    ports={
                        8006: self.vnc_port,
                        5000: self.server_port,
                        9222: self.chromium_port,
                        8080: self.vlc_port
                    },
                    detach=True,
                )
            _prov_logger.info(
                "Started container with ports - VNC: %s, Server: %s, Chrome: %s, VLC: %s",
                self.vnc_port, self.server_port, self.chromium_port, self.vlc_port
            )
            self._wait_for_vm_ready()
        except Exception as e:
            import traceback
            print("Exception:", e)
            print("Traceback:", traceback.format_exc())
            if self.container:
                try:
                    self.container.stop()
                    self.container.remove()
                except Exception:
                    pass
                self.container = None
            raise e

    DockerProvider.start_emulator = _start_emulator_no_kvm_if_missing  # type: ignore[attr-defined]
    DockerProvider._ARPO_PORT_PATCHED = True  # type: ignore[attr-defined]
    logger.info("Docker provider patched successfully (no psutil; /dev/kvm optional for macOS).")

# --- Server state: one env ---
env: DesktopEnv | None = None
_provider_name: str = "docker"  # set on first _get_env(); same as run_uitars / ARPO_OSWorld_Evaluation
history_messages: list = []
is_done = False
step_counter = 0
max_steps = 16
instruction: str | None = None
OBSERVATION_TYPE = "screenshot"  # same as run_uitars --observation_type screenshot
IMAGE_MIN_PIXELS = int(os.environ.get("REMOTE_IMAGE_MIN_PIXELS", "3136"))
IMAGE_MAX_PIXELS = int(os.environ.get("REMOTE_IMAGE_MAX_PIXELS", "129024"))
ACTION_PAUSE_SEC = float(os.environ.get("REMOTE_ACTION_PAUSE_SEC", "1.0"))
REPEAT_ACTION_THRESHOLD = int(os.environ.get("REMOTE_REPEAT_ACTION_THRESHOLD", "3"))
REPEAT_ACTION_PENALTY = float(os.environ.get("REMOTE_REPEAT_ACTION_PENALTY", "0.3"))
_last_action_signature: str | None = None
_repeat_action_count = 0

def _default_provider() -> str:
    """Use VMware on macOS (no KVM); Docker on Linux."""
    if hasattr(os, "uname") and os.uname().sysname == "Darwin":
        return "vmware"
    return "docker"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Always log so we confirm lifespan runs; then run Docker patch when provider is docker
    provider = (os.environ.get("PROVIDER") or _default_provider()).strip().lower() or "docker"
    if provider not in ("docker", "vmware", "aws"):
        provider = "docker"
    logger.info("Startup: PROVIDER=%s, will patch Docker=%s", provider, provider == "docker")
    print(f"[remote_env_server] Startup: PROVIDER={provider}, patching Docker={provider == 'docker'}", file=sys.stderr, flush=True)
    if provider == "docker":
        _patch_docker_provider_ports()
    yield


app = FastAPI(title="OSWorld Remote Env", version="0.1.0", lifespan=_lifespan)


class ResetRequest(BaseModel):
    task_config: dict


class StepRequest(BaseModel):
    prediction: str


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
    if domain == "gimp":
        text += "\n\nImportant: Use GIMP (the image editor), not Ubuntu Settings/System Settings."
    return text


def _make_action_signature(parsed_responses, actions) -> str:
    try:
        if parsed_responses:
            compact = []
            for pr in parsed_responses:
                compact.append({
                    "action_type": pr.get("action_type"),
                    "action_inputs": pr.get("action_inputs", {}),
                })
            return repr(compact)
        return repr(actions)
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


def _get_env():
    global env
    if env is None:
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
            _patch_docker_provider_ports()
        # Check KVM availability for logging (Docker only; AWS/VMware use their own)
        kvm_available = os.path.exists("/dev/kvm")
        if provider_name == "docker":
            if kvm_available:
                print("✓ KVM detected: /dev/kvm exists - VM will use hardware acceleration")
            else:
                print("⚠ KVM not found: /dev/kvm does not exist - VM will use software emulation (slower)")
        if provider_name == "aws":
            print("✓ Using provider: aws (EC2 instances; ensure boto3 and aws configure are set)")
        print(f"✓ Using provider: {provider_name} (observation_type=screenshot, same as run_uitars)")
        # Docker: QEMU/KVM VM inside container. VMware: Fusion VM. AWS: EC2 instances (no local Docker).
        global _provider_name
        _provider_name = provider_name
        region = os.environ.get("AWS_REGION", "us-east-1")
        try:
            env = DesktopEnv(
                provider_name=provider_name,
                region=region,
                action_space="pyautogui",
                screen_size=(1920, 1080),
                cache_dir="cache_dirs/cache_0",
                headless=True,
                os_type="Ubuntu",
                require_a11y_tree=False,
            )
            print(f"DesktopEnv initialized successfully (provider={provider_name}, KVM={kvm_available})")
            print(f"  → VM is ready: screenshots will come from VM via controller.get_screenshot()")
        except HTTPException:
            raise
        except OSError as e:
            if e.errno == 28:  # No space left on device
                print(f"Env init failed: {e}", flush=True)
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "No space left on device on the env server. Free disk space (e.g. docker system prune -a, "
                        "remove cache_dirs/vms) or resize the EBS volume. Then restart the server."
                    ),
                ) from e
            raise
        except Exception as e:
            print(f"Env init failed: {type(e).__name__}: {e}", flush=True)
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
    return env


@app.post("/env/reset")
def env_reset(body: ResetRequest):
    global history_messages, is_done, step_counter, instruction, _last_action_signature, _repeat_action_count
    task_config = body.task_config
    instruction = _instruction_with_hints(task_config)
    step_counter = 0
    is_done = False
    _last_action_signature = None
    _repeat_action_count = 0
    history_messages = []
    env = _get_env()

    trial = 0
    while trial < 8:
        try:
            obs = env.reset(task_config)
            break
        except Exception as e:
            print(f"Env reset exception: {e}")
            print(traceback.format_exc())
            trial += 1

    if trial >= 8:
        is_done = True
        return {"env_idx": 0, "obs_messages": None, "is_done": True, "format_reward": 0.0}

    _safe_env_pause(env)
    screenshot = obs.get("screenshot")
    if screenshot is None:
        print("Reset: screenshot is None (VM/container not ready or get_screenshot failed). Returning obs_messages=None.")
        is_done = True
        return {"env_idx": 0, "obs_messages": None, "is_done": True, "format_reward": 0.0}
    if isinstance(screenshot, bytes):
        pass
    else:
        from PIL import Image
        buf = BytesIO()
        Image.open(BytesIO(screenshot) if isinstance(screenshot, bytes) else screenshot).save(buf, format="JPEG")
        screenshot = buf.getvalue()

    history_messages = _build_init_messages(screenshot, instruction)
    # Screenshot comes from VM inside Docker container (QEMU/KVM) via controller.get_screenshot()
    # → http://localhost:5000/screenshot → pyautogui.screenshot() inside the VM
    print(f"Reset OK: VM screenshot obtained ({len(screenshot)} bytes), returning obs_messages with image. Instruction: {instruction[:60]}...")
    return {
        "env_idx": 0,
        "obs_messages": messages_to_wire(history_messages),
        "is_done": False,
        "format_reward": 0.0,
    }


@app.post("/env/step")
def env_step(body: StepRequest):
    global history_messages, is_done, step_counter, _last_action_signature, _repeat_action_count
    env = _get_env()
    prediction = body.prediction
    action_parse_res_factor = 1000
    model_type = "qwen25vl"
    max_pixels = 16384 * 28 * 28
    min_pixels = 100 * 28 * 28
    obs_image_height, obs_image_width = 1080, 1920

    parsed_responses = []
    try:
        parsed_responses = parse_action_to_structure_output(
            prediction, action_parse_res_factor, obs_image_height, obs_image_width, model_type, max_pixels, min_pixels
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
        
        # ARPO-style format_reward heuristics:
        # - Base reward for successful parsing (0.1)
        # - Bonus for meaningful actions (clicks, types, etc.) vs DONE/WAIT (0.05)
        # - Penalty for parse errors (-1.0)
        format_reward = 0.1  # Base reward for successful LLM action parsing
        
        # Check if we have meaningful GUI actions (not just DONE/WAIT/FAIL)
        if actions and actions[0] not in ["DONE", "WAIT", "FAIL"]:
            format_reward += 0.05  # Bonus for executable GUI actions
        
        # Check for FINISH_WORD (task completion signal from LLM)
        if FINISH_WORD in action_types:
            format_reward += 0.1  # Bonus for LLM indicating task completion
        
    except Exception:
        print("Parse action error:", prediction)
        print(traceback.format_exc())
        format_reward = -1.0
        actions = ["DONE"]

    # Log parse outcome so we can see if training is sending meaningful actions (on-policy)
    pred_preview = (prediction or "")[:120].replace("\n", " ")
    action_preview = actions[0] if actions else "none"
    parse_status = "fail" if format_reward < 0 else "ok"
    pred_chars = len(prediction or "")
    pred_tokens_est = pred_chars // 4  # ~4 chars/token for Qwen2.5 (think+action combined)
    print(f"step_parse: {parse_status} actions=[{action_preview}] format_reward={format_reward:.2f} tokens~{pred_tokens_est}({pred_chars}chars) pred_preview={pred_preview!r}")

    action_signature = _make_action_signature(parsed_responses, actions)
    if action_signature == _last_action_signature:
        _repeat_action_count += 1
    else:
        _last_action_signature = action_signature
        _repeat_action_count = 1
    print(f"step_trace: repeat_action_count={_repeat_action_count} threshold={REPEAT_ACTION_THRESHOLD}")

    # Keep trajectory order consistent: assistant action text, then resulting screenshot(s).
    history_messages.append({"role": "assistant", "content": [{"type": "text", "text": add_box_token(prediction)}]})

    if _repeat_action_count >= REPEAT_ACTION_THRESHOLD:
        # Cut off obvious local loops (same parsed action repeated with no progress signal).
        format_reward = max(format_reward - REPEAT_ACTION_PENALTY, -1.0)
        is_done = True
        print(
            f"loop_breaker: repeated action signature x{_repeat_action_count}; "
            f"terminating episode with penalty {REPEAT_ACTION_PENALTY:.2f}"
        )
        return {"env_idx": 0, "obs_messages": None, "is_done": True, "format_reward": format_reward}

    _safe_env_unpause(env)
    obs = None
    step_successful = False
    appended_any_step_screenshot = False
    for action in actions:
        obs, reward, step_done, info = env.step(action, pause=ACTION_PAUSE_SEC)
        if step_done:
            is_done = True
        step_counter += 1
        has_screenshot = obs is not None and obs.get("screenshot") is not None
        if has_screenshot:
            step_successful = True
            # Capture intermediate GUI state after each executed action so the next decision
            # sees UI changes even when one model response contains multiple actions.
            if _append_screenshot_message(history_messages, obs.get("screenshot")):
                appended_any_step_screenshot = True
        if step_counter >= max_steps:
            is_done = True
        if is_done:
            break
    
    _safe_env_pause(env)
    
    # Enhance format_reward based on step execution success
    if step_successful:
        format_reward += 0.05  # Bonus for successful step execution (screenshot obtained)
    
    if obs is None and not actions:
        is_done = True
        format_reward = max(format_reward - 0.1, -1.0)  # Penalty for no observation

    if is_done:
        return {"env_idx": 0, "obs_messages": None, "is_done": True, "format_reward": format_reward}

    if obs is None or obs.get("screenshot") is None:
        is_done = True
        format_reward = max(format_reward - 0.1, -1.0)  # Penalty for missing screenshot
        return {"env_idx": 0, "obs_messages": None, "is_done": True, "format_reward": format_reward}

    # Backward-compatible path: if no screenshot was appended in the action loop
    # (e.g. unusual provider behavior), append the final screenshot now.
    if not appended_any_step_screenshot:
        _append_screenshot_message(history_messages, obs["screenshot"])

    return {
        "env_idx": 0,
        "obs_messages": messages_to_wire(history_messages),
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
def env_evaluate():
    # Log immediately so server logs show evaluate was called (even if we 503 or return 0)
    _instr = (instruction or "N/A")[:80]
    print(f"POST /env/evaluate received (instruction={_instr!r}, step_counter={step_counter})")
    env = _get_env()
    try:
        if not _env_ready_for_evaluate(env):
            print("Evaluation skipped: env not fully started (no setup_controller); reset likely failed. Returning 503 so client can retry.")
            raise HTTPException(
                status_code=503,
                detail="Env not ready for evaluation (no setup_controller; reset may have failed). Client should retry.",
            )
        _safe_env_unpause(env)
        score = env.evaluate()
        print(f"Evaluation completed: score={score}, instruction={instruction}, step_counter={step_counter}")
        return float(score)
    except HTTPException:
        raise
    except AttributeError as e:
        if "setup_controller" in str(e):
            print("Evaluation skipped: env has no setup_controller (reset failed). Returning 503 so client can retry.")
            raise HTTPException(
                status_code=503,
                detail="Env not ready for evaluation (reset failed). Client should retry.",
            ) from e
        raise
    except Exception as e:
        err_msg = str(e)
        if "setup_controller" in err_msg:
            print("Evaluation skipped: env has no setup_controller (wrapped error). Returning 503 so client can retry.")
            raise HTTPException(
                status_code=503,
                detail="Env not ready for evaluation (no setup_controller). Client should retry.",
            ) from e
        print("Evaluation error:", e)
        print(traceback.format_exc())
        return 0.0


@app.post("/env/history_messages")
def env_history_messages():
    return {"history_messages": messages_to_wire(history_messages) if history_messages else []}


@app.get("/health")
def health():
    """Health check: reports provider, observation_type (screenshot), and VM status. Same env as ARPO_OSWorld_Evaluation."""
    kvm_available = os.path.exists("/dev/kvm")
    env_status = "initialized" if env is not None else "not_initialized"
    return {
        "status": "ok",
        "provider": _provider_name,
        "observation_type": OBSERVATION_TYPE,
        "screenshot_source": "DesktopEnv.controller.get_screenshot() (same as run_uitars)",
        "kvm_available": kvm_available,
        "kvm_device": "/dev/kvm" if kvm_available else None,
        "env_status": env_status,
        "message": "KVM hardware acceleration enabled" if kvm_available else "KVM not available, using software emulation"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=15001)
