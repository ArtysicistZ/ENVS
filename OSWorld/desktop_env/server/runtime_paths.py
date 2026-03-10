from __future__ import annotations

import os
from pathlib import Path


SERVER_STATE_ROOT_ENV = "OSWORLD_SERVER_STATE_ROOT"


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
    configured_root = os.getenv(SERVER_STATE_ROOT_ENV)
    if configured_root:
        root = Path(configured_root)
    else:
        runtime_dir = os.getenv("XDG_RUNTIME_DIR", "").strip()
        if not runtime_dir:
            raise RuntimeError(
                "OSWORLD_SERVER_STATE_ROOT or XDG_RUNTIME_DIR must be set "
                "before starting the OSWorld server."
            )
        root = Path(runtime_dir) / "osworld-server"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve(strict=False)


def server_artifact_path(*parts: str) -> Path:
    path = server_state_root().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def rewrite_runtime_compat_paths(
    value: str,
    *,
    runtime_home: str | None = None,
    runtime_dir: str | None = None,
) -> str:
    """Rewrite legacy task paths to the actual runtime home/runtime dir.

    The OSWorld task corpus still contains a small number of hardcoded setup
    commands like `/run/user/1000/bus`. Those should follow the live runtime
    directory without requiring evaluator or task-json changes.
    """
    rewritten = value
    runtime_home = runtime_home or os.getenv("HOME", "")
    runtime_dir = runtime_dir or os.getenv("XDG_RUNTIME_DIR", "")
    if runtime_home and runtime_home != "/home/user":
        rewritten = rewritten.replace("/home/user", runtime_home)
    if runtime_dir and runtime_dir != "/run/user/1000":
        rewritten = rewritten.replace("/run/user/1000", runtime_dir)
    return rewritten


def normalize_command_for_runtime(command: str | list[str], *, shell: bool) -> str | list[str]:
    if shell:
        if isinstance(command, str):
            return rewrite_runtime_compat_paths(command)
        return rewrite_runtime_compat_paths(" ".join(command))
    if isinstance(command, str):
        return rewrite_runtime_compat_paths(command)
    return [rewrite_runtime_compat_paths(arg) for arg in command]
