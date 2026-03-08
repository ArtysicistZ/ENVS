from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _candidate_displays() -> list[str]:
    candidates: list[str] = []
    env_display = os.environ.get("DISPLAY", "").strip()
    if env_display:
        candidates.append(env_display)

    x11_dir = Path("/tmp/.X11-unix")
    if x11_dir.exists():
        for path in sorted(x11_dir.glob("X*")):
            suffix = path.name[1:]
            if suffix.isdigit():
                candidates.append(f":{suffix}")

    seen = set()
    ordered: list[str] = []
    for display in candidates:
        if display not in seen:
            ordered.append(display)
            seen.add(display)
    if not ordered:
        ordered.append(":0")
    return ordered


def _connectable(display_name: str) -> bool:
    try:
        from Xlib.display import Display

        display = Display(display_name)
        display.close()
        return True
    except Exception:
        return False


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: start_osworld_server.py <python_bin> <server_main.py>", file=sys.stderr)
        return 2

    python_bin = sys.argv[1]
    server_main = sys.argv[2]
    timeout = float(os.environ.get("OSWORLD_SERVER_WAIT_SECONDS", "60"))
    deadline = time.time() + timeout
    last_candidates: list[str] = []

    while time.time() < deadline:
        last_candidates = _candidate_displays()
        for display_name in last_candidates:
            if _connectable(display_name):
                os.environ["DISPLAY"] = display_name
                print(f"Using X display {display_name}", flush=True)
                os.execv(python_bin, [python_bin, server_main])
        time.sleep(1.0)

    print(
        "Timed out waiting for a usable X display. "
        f"Tried: {', '.join(last_candidates)}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
