from __future__ import annotations

import importlib
import json
import platform
import sys


MODULES = {
    "shared": [
        ("flask", "flask"),
        ("lxml.etree", "lxml"),
        ("PIL.Image", "Pillow"),
        ("pyautogui", "pyautogui"),
        ("requests", "requests"),
        ("Xlib", "python-xlib"),
        ("pyxcursor", "pyxcursor"),
    ],
    "Linux": [
        ("pyatspi", "pyatspi2"),
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
            importlib.import_module(module_name)
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
