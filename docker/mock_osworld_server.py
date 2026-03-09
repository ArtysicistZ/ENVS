#!/usr/bin/env python3
"""
Minimal mock of the OSWorld in-container stack.
Runs two Flask servers:
  - Port 5000: mock osworld-server (health, screenshot, execute_action, evaluate)
  - Port 5001: mock osworld-resetd (health, state, prepare_baseline, reset)
Used for testing the ArpoDockerProvider without the full desktop stack.
"""
import io
import json
import os
import threading
import time

from flask import Flask, jsonify, request, send_file

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ── State ─────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_reset_gen = 0
_baseline_ready = False
_start_time = time.time()
_screenshot_counter = 0

# ── Port 5000 — osworld-server ─────────────────────────────────────────────
app5000 = Flask("osworld-server")


@app5000.route("/health")
def srv_health():
    return jsonify({"status": "ok", "uptime": round(time.time() - _start_time, 1)})


@app5000.route("/screenshot")
def srv_screenshot():
    global _screenshot_counter
    if PIL_OK:
        with _state_lock:
            gen = _reset_gen
            _screenshot_counter += 1
            counter = _screenshot_counter
        img = Image.new("RGB", (1920, 1080), color=(30, 30, 60))
        d = ImageDraw.Draw(img)
        # Large bar that cycles color/position per screenshot so fingerprint always changes
        bar_y = 200 + (counter * 7) % 600
        bar_color = (counter * 17 % 200 + 55, counter * 37 % 200 + 55, counter * 73 % 200 + 55)
        d.rectangle([0, bar_y, 1920, bar_y + 80], fill=bar_color)
        d.text((60, 60),  f"OSWorld Mock — reset_gen={gen} n={counter}", fill=(200, 200, 255))
        d.text((60, 110), f"time={time.time():.2f}", fill=(180, 180, 180))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    # Fallback: 1-pixel PNG
    PNG1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return PNG1, 200, {"Content-Type": "image/png"}


@app5000.route("/execute_action", methods=["POST"])
def srv_execute():
    return jsonify({"status": "ok", "result": None})


@app5000.route("/execute", methods=["POST"])
def srv_execute_python():
    """Mock Python command execution (pyautogui). Always succeeds instantly."""
    return jsonify({"status": "success", "output": "", "error": ""})


@app5000.route("/run_python", methods=["POST"])
def srv_run_python():
    """Mock Python script execution. Always succeeds instantly."""
    return jsonify({"status": "success", "output": "", "error": ""})


@app5000.route("/run_bash_script", methods=["POST"])
def srv_run_bash():
    """Mock bash script execution. Always succeeds instantly."""
    return jsonify({"status": "success", "output": "", "error": "", "returncode": 0})


@app5000.route("/launch", methods=["POST"])
def srv_launch():
    """Mock application launch. Always succeeds instantly."""
    return jsonify({"status": "success"})


@app5000.route("/get_a11y_tree")
def srv_a11y():
    return jsonify({"status": "ok", "result": None})


@app5000.route("/evaluate", methods=["POST", "GET"])
def srv_evaluate():
    return jsonify({"score": 0.0})


@app5000.route("/info")
def srv_info():
    return jsonify({"os": "Ubuntu", "version": "22.04"})


# ── Port 5001 — osworld-resetd ─────────────────────────────────────────────
app5001 = Flask("osworld-resetd")


@app5001.route("/health")
def rst_health():
    with _state_lock:
        gen = _reset_gen
        ready = _baseline_ready
    return jsonify({
        "status": "ok",
        "reset_generation": gen,
        "baseline_version": "test-baseline",
        "baseline_ready": ready,
    })


@app5001.route("/state")
def rst_state():
    with _state_lock:
        gen = _reset_gen
    return jsonify({
        "status": "ok",
        "reason_code": "baseline_ready",
        "reset_generation": gen,
    })


@app5001.route("/prepare_baseline", methods=["POST"])
def rst_prepare():
    global _baseline_ready
    with _state_lock:
        _baseline_ready = True
    return jsonify({"status": "ok", "message": "baseline prepared"})


@app5001.route("/reset", methods=["POST"])
def rst_reset():
    global _reset_gen
    with _state_lock:
        _reset_gen += 1
        gen = _reset_gen
    # Simulate brief reset work
    time.sleep(0.5)
    return jsonify({"status": "ok", "reset_generation": gen})


# ── Launch both servers in threads ─────────────────────────────────────────
def _run(app, port):
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    t5000 = threading.Thread(target=_run, args=(app5000, 5000), daemon=True)
    t5001 = threading.Thread(target=_run, args=(app5001, 5001), daemon=True)
    t5000.start()
    t5001.start()
    print(f"Mock OSWorld servers running: port 5000 (server) + port 5001 (resetd)", flush=True)
    while True:
        time.sleep(60)
