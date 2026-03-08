from __future__ import annotations

import os
from pathlib import Path


SERVER_STATE_ROOT_ENV = "OSWORLD_SERVER_STATE_ROOT"
DEFAULT_SERVER_STATE_ROOT = "/tmp/osworld-server"


def user_home() -> Path:
    return Path(os.path.expanduser("~")).resolve()


def resolve_user_path(raw_path: str | os.PathLike[str], *, home: Path | None = None) -> Path:
    """Resolve task-provided paths relative to the desktop user's home.

    OSWorld tasks commonly use relative paths like `setup.sh`, `Downloads/foo`,
    or `settings.json`. Those must resolve inside the disposable workspace, not
    the control-plane working directory.
    """
    home = (home or user_home()).resolve()
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(raw_path)))
    path = Path(expanded)
    if not path.is_absolute():
        path = home / path
    return path.resolve(strict=False)


def server_state_root() -> Path:
    root = Path(os.getenv(SERVER_STATE_ROOT_ENV, DEFAULT_SERVER_STATE_ROOT))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve(strict=False)


def server_artifact_path(*parts: str) -> Path:
    path = server_state_root().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
