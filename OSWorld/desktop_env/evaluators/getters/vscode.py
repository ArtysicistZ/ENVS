import logging
from typing import Any, Dict
import time
from .file import get_vm_file
from .replay import get_replay

logger = logging.getLogger("desktopenv.getters.vscode")


def get_vscode_config(env, config: Dict[str, Any]) -> str:
    os_type = env.vm_platform
    if os_type not in ("Windows", "Darwin", "Linux"):
        os_type = "Linux"
    vscode_extension_command = config["vscode_extension_command"]

    if os_type == "Darwin":
        trajectory = [
            {"type": "hotkey", "param": ["command", "shift", "p"]},
            {"type": "typewrite", "param": vscode_extension_command},
            {"type": "press", "param": "enter"}
        ]
    else:
        trajectory = [
            {"type": "hotkey", "param": ["ctrl", "shift", "p"]},
            {"type": "typewrite", "param": vscode_extension_command},
            {"type": "press", "param": "enter"}
        ]

    get_replay(env, trajectory)
    time.sleep(1.0)

    return get_vm_file(env, {
        "path": config["path"],
        "dest": config["dest"]
    })
