from __future__ import annotations

import importlib
import importlib.util
import json
import platform
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[4]
SERVER_DIR = ROOT_DIR / "desktop_env" / "server"

for path in (ROOT_DIR, SERVER_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


MODULES = {
    "shared": [
        ("flask", "flask"),
        ("lxml.etree", "lxml"),
        ("PIL.Image", "Pillow"),
        ("pyautogui", "pyautogui"),
        ("pygetwindow", "PyGetWindow"),
        ("requests", "requests"),
        ("Xlib", "python-xlib"),
        ("pyxcursor", "local:desktop_env/server/pyxcursor.py"),
    ],
    "Linux": [
        ("pyatspi", "python3-pyatspi"),
    ],
    "Windows": [
        ("pywinauto", "pywinauto"),
        ("win32gui", "pywin32"),
    ],
    "Darwin": [
        ("AppKit", "pyobjc"),
        ("Quartz", "pyobjc"),
    ],
}


def main() -> int:
    platform_name = platform.system()
    required = list(MODULES["shared"])
    required.extend(MODULES.get(platform_name, []))

    missing: list[dict[str, str]] = []
    for module_name, package_name in required:
        try:
            if importlib.util.find_spec(module_name) is None:
                raise ModuleNotFoundError(f"No module named '{module_name}'")
        except Exception as exc:
            missing.append(
                {
                    "module": module_name,
                    "suggested_package": package_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not missing:
        print(json.dumps({"status": "ok", "platform": platform_name}, sort_keys=True))
        return 0

    print(
        json.dumps(
            {
                "status": "error",
                "platform": platform_name,
                "missing": missing,
            },
            indent=2,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
