from __future__ import annotations

import os
import sys
import time


def main() -> int:
    display_name = os.environ.get("DISPLAY", ":0")
    timeout = float(os.environ.get("OSWORLD_X_DISPLAY_WAIT_SECONDS", "60"))
    deadline = time.time() + timeout
    last_error = "display_not_ready"

    while time.time() < deadline:
        try:
            from Xlib.display import Display

            display = Display(display_name)
            display.close()
            return 0
        except Exception as exc:  # pragma: no cover - runtime only
            last_error = str(exc)
            time.sleep(1.0)

    print(
        f"Timed out waiting for X display {display_name}: {last_error}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
