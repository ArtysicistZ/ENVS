#!/usr/bin/env python3
"""Poll a remote env slot and materialize the latest screenshot for live viewing.

This is read-only: it only calls /env/history_messages.
It writes:
- latest.jpg
- optional timestamped frames/
- index.html with auto-refresh for browser viewing
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from urllib.parse import urljoin

import requests


def extract_b64(messages):
    for msg in reversed(messages or []):
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in reversed(content):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                continue
            b64 = item.get("b64")
            if b64:
                return b64
            image = item.get("image", "")
            if image.startswith("data:image"):
                return image.split(",", 1)[1]
    return None


def write_index(out_dir: Path, latest_name: str):
    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <meta http-equiv=\"refresh\" content=\"2\">
  <title>Remote Slot Watcher</title>
  <style>
    body {{ font-family: sans-serif; background: #111; color: #eee; margin: 20px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #444; }}
    .meta {{ margin-bottom: 12px; color: #bbb; }}
  </style>
</head>
<body>
  <div class=\"meta\">Auto-refresh every 2s. Reload if the image appears stale.</div>
  <img src=\"{latest_name}?t={int(time.time())}\" />
</body>
</html>
"""
    (out_dir / "index.html").write_text(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", required=True)
    ap.add_argument("--slot-id", type=int, required=True)
    ap.add_argument("--output-dir", default="/tmp/slot_watch")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--save-frames", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    latest_path = out_dir / "latest.jpg"
    meta_path = out_dir / "latest.json"
    write_index(out_dir, latest_path.name)

    last_b64 = None
    print(f"Watching slot {args.slot_id} at {args.server_url}")
    print(f"Open file://{out_dir / 'index.html'} or serve the directory with python -m http.server")

    while True:
        url = urljoin(args.server_url.rstrip("/") + "/", f"env/history_messages?slot_id={args.slot_id}")
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            messages = payload.get("messages") or payload.get("history_messages") or payload.get("history") or []
            b64 = extract_b64(messages)
            if not b64:
                print("No screenshot yet; waiting...")
                time.sleep(args.interval)
                continue
            if b64 != last_b64:
                jpg = base64.b64decode(b64)
                latest_path.write_bytes(jpg)
                meta_path.write_text(json.dumps({
                    "slot_id": args.slot_id,
                    "server_url": args.server_url,
                    "updated_at": time.time(),
                    "message_count": len(messages),
                }, indent=2))
                if args.save_frames:
                    ts = int(time.time() * 1000)
                    (frames_dir / f"frame_{ts}.jpg").write_bytes(jpg)
                write_index(out_dir, latest_path.name)
                last_b64 = b64
                print(f"Updated screenshot at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
        except KeyboardInterrupt:
            print("Stopped.")
            return
        except Exception as e:
            print(f"watch error: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
