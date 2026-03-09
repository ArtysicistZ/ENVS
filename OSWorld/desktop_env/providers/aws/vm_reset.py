"""Host-side client for the AWS reset daemon.

The reset daemon runs inside the VM on a dedicated port and owns the clean-room
workspace rollback lifecycle. The host provider only orchestrates:

1. baseline preparation validation
2. reset request
3. verification request
4. fallback-to-relaunch when any phase is not provably clean
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

from .config import AWS_RESETD_PORT, AWS_RESETD_REQUEST_TIMEOUT

logger = logging.getLogger("desktopenv.providers.aws.vm_reset")


@dataclass(frozen=True)
class ResetClientConfig:
    port: int = AWS_RESETD_PORT
    timeout: int = AWS_RESETD_REQUEST_TIMEOUT


def _daemon_base_url(ip: str, port: int) -> str:
    return f"http://{ip}:{port}"


def _result(
    *,
    status: str,
    reason_code: str,
    instance_id: str,
    details: dict[str, Any] | None = None,
    baseline_version: str = "unknown",
    reset_generation: int = 0,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason_code,
        "details": details or {},
        "instance_id": instance_id,
        "baseline_version": baseline_version,
        "reset_generation": reset_generation,
    }


def _request_json(
    *,
    method: str,
    ip: str,
    endpoint: str,
    payload: dict[str, Any] | None,
    config: ResetClientConfig,
) -> dict[str, Any]:
    url = f"{_daemon_base_url(ip, config.port)}{endpoint}"
    response = requests.request(method, url, json=payload, timeout=config.timeout)
    response.raise_for_status()
    return response.json()


@contextlib.contextmanager
def _instance_lock(instance_id: str):
    lock_dir = "/tmp/osworld-reset-locks"
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{instance_id}.lock")
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _normalize_daemon_response(
    response: dict[str, Any], *, default_status: str, default_reason: str, instance_id: str
) -> dict[str, Any]:
    return _result(
        status=response.get("status", default_status),
        reason_code=response.get("reason_code", default_reason),
        instance_id=response.get("instance_id", instance_id),
        details=response.get("details", {}),
        baseline_version=response.get("baseline_version", "unknown"),
        reset_generation=int(response.get("reset_generation", 0) or 0),
    )


def get_state(ip: str, instance_id: str, config: ResetClientConfig | None = None) -> dict[str, Any]:
    config = config or ResetClientConfig()
    try:
        response = _request_json(method="GET", ip=ip, endpoint="/state", payload=None, config=config)
        return _normalize_daemon_response(response, default_status="ok", default_reason="state_available", instance_id=instance_id)
    except Exception as exc:
        logger.warning("Reset daemon state unavailable for %s at %s: %s", instance_id, ip, exc)
        return _result(
            status="error",
            reason_code="daemon_unreachable",
            instance_id=instance_id,
            details={"error": str(exc)},
        )


def prepare_baseline(ip: str, instance_id: str, config: ResetClientConfig | None = None) -> dict[str, Any]:
    config = config or ResetClientConfig()
    payload = {"instance_id": instance_id}
    try:
        response = _request_json(method="POST", ip=ip, endpoint="/prepare_baseline", payload=payload, config=config)
        result = _normalize_daemon_response(
            response,
            default_status="ok",
            default_reason="baseline_ready",
            instance_id=instance_id,
        )
        logger.info(
            "Reset baseline preparation for %s on %s: %s (%s)",
            instance_id,
            ip,
            result["status"],
            result["reason_code"],
        )
        return result
    except Exception as exc:
        logger.warning("Reset baseline preparation failed for %s on %s: %s", instance_id, ip, exc)
        return _result(
            status="error",
            reason_code="daemon_unreachable",
            instance_id=instance_id,
            details={"error": str(exc)},
        )


def snapshot_home(ip: str, instance_id: str, port: int = AWS_RESETD_PORT) -> dict[str, Any]:
    """Compatibility alias for older callers.

    The new architecture no longer snapshots runtime state. This validates that
    the image-baked baseline is present and that daemon metadata is ready.
    """
    return prepare_baseline(ip, instance_id, ResetClientConfig(port=port))


def verify_state(ip: str, instance_id: str, config: ResetClientConfig | None = None) -> dict[str, Any]:
    """Call the reset daemon's /verify endpoint and return a structured result.

    Useful for retrying verification separately from the reset step when the
    osworld-server takes longer than the built-in timeout to restart.
    """
    config = config or ResetClientConfig()
    payload = {"instance_id": instance_id}
    try:
        response = _request_json(method="POST", ip=ip, endpoint="/verify", payload=payload, config=config)
        return _normalize_daemon_response(
            response,
            default_status="ok",
            default_reason="verified_clean",
            instance_id=instance_id,
        )
    except Exception as exc:
        logger.warning("Reset daemon verify call failed for %s on %s: %s", instance_id, ip, exc)
        return _result(
            status="error",
            reason_code="verification_unreachable",
            instance_id=instance_id,
            details={"error": str(exc)},
        )


def restore_home(ip: str, instance_id: str, port: int = AWS_RESETD_PORT) -> dict[str, Any]:
    """Compatibility alias for explicit restore+verify callers."""
    return soft_reset(ip, instance_id, ResetClientConfig(port=port))


def soft_reset(ip: str, instance_id: str, config: ResetClientConfig | None = None) -> dict[str, Any]:
    """Attempt a verified soft reset.

    Returns a structured outcome dict. The provider should reuse the instance
    only when `status == "reused_clean"`. All other outcomes must fall back to
    full terminate+relaunch.
    """
    config = config or ResetClientConfig()
    payload = {"instance_id": instance_id}

    with _instance_lock(instance_id):
        try:
            reset_response = _request_json(method="POST", ip=ip, endpoint="/reset", payload=payload, config=config)
            reset_result = _normalize_daemon_response(
                reset_response,
                default_status="ok",
                default_reason="reset_completed",
                instance_id=instance_id,
            )
        except Exception as exc:
            logger.warning("Reset daemon reset call failed for %s on %s: %s", instance_id, ip, exc)
            return _result(
                status="must_relaunch",
                reason_code="daemon_unreachable",
                instance_id=instance_id,
                details={"error": str(exc)},
            )

        if reset_result["status"] != "ok":
            return _result(
                status="must_relaunch",
                reason_code=reset_result["reason_code"],
                instance_id=instance_id,
                details=reset_result["details"],
                baseline_version=reset_result["baseline_version"],
                reset_generation=reset_result["reset_generation"],
            )

        try:
            verify_response = _request_json(method="POST", ip=ip, endpoint="/verify", payload=payload, config=config)
            verify_result = _normalize_daemon_response(
                verify_response,
                default_status="ok",
                default_reason="verified_clean",
                instance_id=instance_id,
            )
        except Exception as exc:
            logger.warning("Reset daemon verify call failed for %s on %s: %s", instance_id, ip, exc)
            return _result(
                status="must_relaunch",
                reason_code="verification_unreachable",
                instance_id=instance_id,
                details={"error": str(exc)},
                baseline_version=reset_result["baseline_version"],
                reset_generation=reset_result["reset_generation"],
            )

        if verify_result["status"] == "ok":
            return _result(
                status="reused_clean",
                reason_code="verified_clean",
                instance_id=instance_id,
                details=verify_result["details"],
                baseline_version=verify_result["baseline_version"],
                reset_generation=verify_result["reset_generation"],
            )

        return _result(
            status="must_relaunch",
            reason_code=verify_result["reason_code"],
            instance_id=instance_id,
            details=verify_result["details"],
            baseline_version=verify_result["baseline_version"],
            reset_generation=verify_result["reset_generation"],
        )


def pretty_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)
