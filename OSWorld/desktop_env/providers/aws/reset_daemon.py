"""HTTP daemon wrapper around the AWS clean-room reset runtime."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from flask import Flask, jsonify, request

try:
    from .reset_runtime import ResetConfig, ResetRuntime
except ImportError:  # pragma: no cover - support running as a copied standalone script
    from reset_runtime import ResetConfig, ResetRuntime

try:  # pragma: no cover - optional production server
    from waitress import serve as waitress_serve
except ImportError:  # pragma: no cover - fallback for development environments
    waitress_serve = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("desktopenv.providers.aws.reset_daemon")

app = Flask(__name__)
runtime = ResetRuntime(ResetConfig())


def _jsonify_result(result: Any):
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    elif isinstance(result, dict):
        payload = result
    else:
        payload = json.loads(json.dumps(result))
    status_code = 200 if payload.get("status") in {"ok", "busy"} else 503
    return jsonify(payload), status_code


@app.get("/health")
def health():
    state = runtime.state()
    payload = {
        "status": "ok",
        "reason_code": "daemon_healthy",
        "details": {
            "pid": os.getpid(),
            "runtime_state": state,
        },
        "instance_id": state.get("instance_id", "unknown-instance"),
        "baseline_version": state.get("baseline_version", "unknown"),
        "reset_generation": state.get("reset_generation", 0),
    }
    return jsonify(payload)


@app.get("/state")
def state():
    return jsonify(runtime.state())


@app.post("/prepare_baseline")
def prepare_baseline():
    return _jsonify_result(runtime.prepare_baseline())


@app.post("/reset")
def reset():
    return _jsonify_result(runtime.reset())


@app.post("/verify")
def verify():
    return _jsonify_result(runtime.verify())



def main() -> None:
    host = os.getenv("OSWORLD_RESET_BIND", "0.0.0.0")
    port = int(os.getenv("AWS_RESETD_PORT", "5001"))
    logger.info("Starting OSWorld reset daemon on %s:%d", host, port)
    if waitress_serve is not None:
        waitress_serve(app, host=host, port=port, threads=8)
        return
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
