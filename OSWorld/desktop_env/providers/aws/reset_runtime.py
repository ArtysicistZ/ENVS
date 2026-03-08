"""Root-level clean-room reset runtime for AWS-backed OSWorld VMs.

This module is designed to run inside the VM as a privileged service. It owns
the disposable workspace reset lifecycle and returns structured results that
allow the host-side provider to decide between safe reuse and full relaunch.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import logging
import os
import pwd
import shutil
import subprocess
import time
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import Any

import requests

logger = logging.getLogger("desktopenv.providers.aws.reset_runtime")

RUNTIME_VERSION = "2"
DEFAULT_IGNORED_RELATIVE_PATHS = (
    ".Xauthority",
    ".ICEauthority",
    ".cache",
    ".dbus",
    ".config",
    ".local",
    ".local/share/recently-used.xbel",
)
DISPLAY_MANAGER_CANDIDATES = (
    "display-manager.service",
    "gdm3.service",
    "gdm.service",
    "lightdm.service",
    "sddm.service",
)


def _normalize_ignored_paths() -> tuple[str, ...]:
    env_value = os.getenv("OSWORLD_RESET_IGNORE_PATHS", "").strip()
    if not env_value:
        return DEFAULT_IGNORED_RELATIVE_PATHS

    merged = list(DEFAULT_IGNORED_RELATIVE_PATHS)
    merged.extend(p.strip("/") for p in env_value.split(",") if p.strip())
    # Preserve order but drop duplicates.
    return tuple(dict.fromkeys(merged))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _path_matches_ignored(rel_path: str, ignored_relative_paths: tuple[str, ...]) -> bool:
    return any(rel_path == ignored or rel_path.startswith(f"{ignored}/") for ignored in ignored_relative_paths)


def _path_is_allowed_runtime_artifact(rel_path: str, ignored_relative_paths: tuple[str, ...]) -> bool:
    return any(
        rel_path == ignored
        or rel_path.startswith(f"{ignored}/")
        or ignored.startswith(f"{rel_path}/")
        for ignored in ignored_relative_paths
    )


@dataclasses.dataclass
class ResetConfig:
    desktop_user: str = os.getenv("OSWORLD_RESET_USER", "user")
    workspace_home: Path = Path(os.getenv("OSWORLD_RESET_HOME", "/home/user"))
    allow_unsafe_home: bool = _env_flag("OSWORLD_ALLOW_UNSAFE_HOME", "0")
    control_plane_root: Path = Path(os.getenv("OSWORLD_CONTROL_PLANE_ROOT", "/opt/osworld/app/OSWorld"))
    baseline_home: Path = Path(os.getenv("OSWORLD_RESET_BASELINE_HOME", "/opt/osworld/baseline/home-user"))
    dconf_snapshot: Path = Path(os.getenv("OSWORLD_RESET_DCONF_SNAPSHOT", "/opt/osworld/baseline/dconf/user.dconf"))
    session_root: Path = Path(os.getenv("OSWORLD_RESET_SESSION_ROOT", "/var/lib/osworld/session"))
    state_root: Path = Path(os.getenv("OSWORLD_RESET_STATE_ROOT", "/var/lib/osworld-reset"))
    metadata_path: Path = Path(os.getenv("OSWORLD_RESET_METADATA_PATH", "/var/lib/osworld-reset/metadata.json"))
    state_path: Path = Path(os.getenv("OSWORLD_RESET_STATE_PATH", "/var/lib/osworld-reset/state.json"))
    lock_path: Path = Path(os.getenv("OSWORLD_RESET_LOCK_PATH", "/var/lib/osworld-reset/reset.lock"))
    baseline_manifest_path: Path = Path(
        os.getenv("OSWORLD_RESET_BASELINE_MANIFEST_PATH", "/var/lib/osworld-reset/baseline_home_manifest.json")
    )
    control_plane_stamp_path: Path = Path(
        os.getenv("OSWORLD_CONTROL_PLANE_STAMP_PATH", "/var/lib/osworld-reset/control_plane_build_id")
    )
    taint_marker_path: Path = Path(
        os.getenv("OSWORLD_RESET_TAINT_MARKER_PATH", "/var/lib/osworld-reset/system_taint.json")
    )
    baseline_version: str = os.getenv("OSWORLD_RESET_BASELINE_VERSION", "")
    ami_build_version: str = os.getenv("OSWORLD_RESET_AMI_BUILD_VERSION", "unknown")
    verification_policy_version: str = os.getenv("OSWORLD_RESET_VERIFICATION_POLICY_VERSION", "1")
    reset_generation_path: Path = Path(
        os.getenv("OSWORLD_RESET_GENERATION_PATH", "/var/lib/osworld-reset/reset_generation")
    )
    baseline_mode: str = os.getenv("OSWORLD_RESET_BASELINE_MODE", "minimal")
    osworld_server_url: str = os.getenv("OSWORLD_SERVER_URL", "http://127.0.0.1:5000")
    osworld_server_service: str = os.getenv("OSWORLD_SERVER_SERVICE", "osworld-server.service")
    osworld_graphical_session_service: str = os.getenv(
        "OSWORLD_GRAPHICAL_SESSION_SERVICE", "osworld-graphical-session.service"
    )
    screenshot_endpoint: str = os.getenv("OSWORLD_SCREENSHOT_ENDPOINT", "/screenshot")
    health_endpoint: str = os.getenv("OSWORLD_HEALTH_ENDPOINT", "/health")
    display_manager_service: str = os.getenv("OSWORLD_DISPLAY_MANAGER_SERVICE", "display-manager")
    user_tmp_dirs: tuple[Path, ...] = dataclasses.field(
        default_factory=lambda: (Path("/tmp"), Path("/var/tmp"))
    )
    ignored_relative_paths: tuple[str, ...] = dataclasses.field(
        default_factory=_normalize_ignored_paths
    )

    @property
    def workspace_upper(self) -> Path:
        return self.session_root / "home-upper"

    @property
    def workspace_work(self) -> Path:
        return self.session_root / "home-work"

    @property
    def overlay_dirs(self) -> tuple[Path, Path]:
        return self.workspace_upper, self.workspace_work


@dataclasses.dataclass
class ResetResult:
    status: str
    reason_code: str
    details: dict[str, Any]
    instance_id: str
    baseline_version: str
    reset_generation: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest(root: Path, ignored_relative_paths: tuple[str, ...] = ()) -> dict[str, Any]:
    root = root.resolve()
    manifest: dict[str, Any] = {}

    for current_root, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current_root)
        rel_root = "." if current_path == root else current_path.relative_to(root).as_posix()
        if rel_root != "." and _path_matches_ignored(rel_root, ignored_relative_paths):
            dirnames[:] = []
            continue

        dir_entry = current_path.lstat()
        manifest[rel_root] = {
            "type": "dir",
            "mode": dir_entry.st_mode & 0o7777,
            "uid": dir_entry.st_uid,
            "gid": dir_entry.st_gid,
        }

        for name in sorted(dirnames + filenames):
            path = current_path / name
            rel_path = path.relative_to(root).as_posix()
            if _path_matches_ignored(rel_path, ignored_relative_paths):
                continue
            entry = path.lstat()
            record: dict[str, Any] = {
                "mode": entry.st_mode & 0o7777,
                "uid": entry.st_uid,
                "gid": entry.st_gid,
            }
            if S_ISDIR(entry.st_mode):
                record["type"] = "dir"
            elif S_ISLNK(entry.st_mode):
                record["type"] = "symlink"
                record["target"] = os.readlink(path)
            elif S_ISREG(entry.st_mode):
                record["type"] = "file"
                record["size"] = entry.st_size
                record["sha256"] = _sha256_file(path)
            else:
                record["type"] = "other"
            manifest[rel_path] = record
    return manifest


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return _sha256_bytes(_stable_json(manifest).encode("utf-8"))


class ResetRuntime:
    def __init__(self, config: ResetConfig | None = None):
        self.config = config or ResetConfig()

    def _result(
        self,
        *,
        status: str,
        reason_code: str,
        details: dict[str, Any] | None = None,
        reset_generation: int | None = None,
        baseline_version: str | None = None,
    ) -> ResetResult:
        metadata = self._load_metadata(optional=True)
        return ResetResult(
            status=status,
            reason_code=reason_code,
            details=details or {},
            instance_id=self._resolve_instance_id(),
            baseline_version=baseline_version or metadata.get("baseline_version", "unknown"),
            reset_generation=reset_generation if reset_generation is not None else self._load_reset_generation(),
        )

    def _load_metadata(self, *, optional: bool = False) -> dict[str, Any]:
        if not self.config.metadata_path.exists():
            if optional:
                return {}
            raise FileNotFoundError(self.config.metadata_path)
        return json.loads(self.config.metadata_path.read_text(encoding="utf-8"))

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        self.config.state_root.mkdir(parents=True, exist_ok=True)
        self.config.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def _write_state(self, state: dict[str, Any]) -> None:
        self.config.state_root.mkdir(parents=True, exist_ok=True)
        self.config.state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _load_state(self) -> dict[str, Any]:
        if not self.config.state_path.exists():
            return {
                "status": "unprepared",
                "reason_code": "baseline_missing",
                "details": {},
                "instance_id": self._resolve_instance_id(),
                "baseline_version": "unknown",
                "reset_generation": self._load_reset_generation(),
            }
        return json.loads(self.config.state_path.read_text(encoding="utf-8"))

    def _load_reset_generation(self) -> int:
        if not self.config.reset_generation_path.exists():
            return 0
        try:
            return int(self.config.reset_generation_path.read_text(encoding="utf-8").strip())
        except Exception:
            return 0

    def _load_taint_marker(self, *, optional: bool = True) -> dict[str, Any]:
        if not self.config.taint_marker_path.exists():
            if optional:
                return {}
            raise FileNotFoundError(self.config.taint_marker_path)
        return json.loads(self.config.taint_marker_path.read_text(encoding="utf-8"))

    def _write_taint_marker(self, payload: dict[str, Any]) -> None:
        self.config.state_root.mkdir(parents=True, exist_ok=True)
        self.config.taint_marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.taint_marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _clear_taint_marker(self) -> None:
        if self.config.taint_marker_path.exists():
            self.config.taint_marker_path.unlink()

    def _bump_reset_generation(self) -> int:
        generation = self._load_reset_generation() + 1
        self.config.state_root.mkdir(parents=True, exist_ok=True)
        self.config.reset_generation_path.write_text(f"{generation}\n", encoding="utf-8")
        return generation

    def _run(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, check=check, capture_output=True, text=True)

    def _run_shell(self, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, check=check, capture_output=True, text=True, shell=True)

    def _resolve_instance_id(self) -> str:
        instance_id = os.getenv("AWS_INSTANCE_ID", "").strip()
        if instance_id:
            return instance_id
        try:
            token_resp = requests.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                timeout=2,
            )
            token_resp.raise_for_status()
            token = token_resp.text
            resp = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-id",
                headers={"X-aws-ec2-metadata-token": token},
                timeout=2,
            )
            resp.raise_for_status()
            return resp.text.strip()
        except Exception:
            return "unknown-instance"

    def _desktop_uid(self) -> int:
        return pwd.getpwnam(self.config.desktop_user).pw_uid

    def _desktop_gid(self) -> int:
        return pwd.getpwnam(self.config.desktop_user).pw_gid

    def _ensure_overlay_storage_permissions(self) -> None:
        uid = self._desktop_uid()
        gid = self._desktop_gid()
        for path in self.config.overlay_dirs:
            path.mkdir(parents=True, exist_ok=True)
            os.chown(path, uid, gid)
            os.chmod(path, 0o700)

    def _ensure_runtime_layout(self) -> None:
        self.config.state_root.mkdir(parents=True, exist_ok=True)
        self.config.session_root.mkdir(parents=True, exist_ok=True)
        self._ensure_overlay_storage_permissions()

    def _validate_workspace_target(self) -> None:
        banned_prefix = Path("/home/ubuntu")
        try:
            if self.config.workspace_home == banned_prefix or banned_prefix in self.config.workspace_home.parents:
                raise RuntimeError("unsafe_workspace_target: /home/ubuntu is permanently forbidden as an OSWorld reset workspace")
        except RuntimeError:
            raise
        if self.config.allow_unsafe_home:
            return
        expected_home = Path(f"/home/{self.config.desktop_user}")
        if self.config.desktop_user != "user" or self.config.workspace_home != expected_home:
            raise RuntimeError(
                "unsafe_workspace_target: OSWorld reset must use the isolated 'user' runtime home by default"
            )

    def _overlay_mount_options(self) -> str:
        return ",".join(
            [
                f"lowerdir={self.config.baseline_home}",
                f"upperdir={self.config.workspace_upper}",
                f"workdir={self.config.workspace_work}",
            ]
        )

    def _home_overlay_status(self) -> tuple[bool, bool, str]:
        mountpoint = self._run(["mountpoint", "-q", self.config.workspace_home.as_posix()], check=False)
        if mountpoint.returncode != 0:
            return False, False, ""

        result = self._run(
            ["findmnt", "-n", "-o", "FSTYPE,OPTIONS", "--target", self.config.workspace_home.as_posix()],
            check=False,
        )
        if result.returncode != 0:
            return True, False, ""

        output = result.stdout.strip()
        if not output:
            return True, False, ""

        fstype, _, options = output.partition(" ")
        expected = self._overlay_mount_options()
        matches = fstype == "overlay" and all(opt in options for opt in expected.split(","))
        return True, matches, options

    def ensure_home_overlay_mounted(self) -> ResetResult:
        self._ensure_runtime_layout()
        try:
            self._validate_workspace_target()
        except RuntimeError as exc:
            return self._result(
                status="error",
                reason_code="unsafe_workspace_target",
                details={
                    "desktop_user": self.config.desktop_user,
                    "workspace_home": self.config.workspace_home.as_posix(),
                    "error": str(exc),
                },
            )
        if not self.config.baseline_home.exists():
            return self._result(status="error", reason_code="baseline_missing")

        self.config.workspace_home.mkdir(parents=True, exist_ok=True)
        is_mountpoint, matches, options = self._home_overlay_status()
        if is_mountpoint and matches:
            return self._result(status="ok", reason_code="home_overlay_ready")
        if is_mountpoint and not matches:
            return self._result(
                status="error",
                reason_code="overlay_mount_mismatch",
                details={"findmnt_options": options},
            )

        try:
            self._run(
                [
                    "mount",
                    "-t",
                    "overlay",
                    "overlay",
                    "-o",
                    self._overlay_mount_options(),
                    self.config.workspace_home.as_posix(),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            return self._result(
                status="error",
                reason_code="overlay_mount_failed",
                details={"command": exc.cmd, "stderr": exc.stderr, "stdout": exc.stdout},
            )

        _, matches, options = self._home_overlay_status()
        if not matches:
            return self._result(
                status="error",
                reason_code="overlay_mount_failed",
                details={"findmnt_options": options},
            )
        return self._result(status="ok", reason_code="home_overlay_ready")

    def _control_plane_build_id(self) -> str:
        if not self.config.control_plane_stamp_path.exists():
            return "missing"
        return self.config.control_plane_stamp_path.read_text(encoding="utf-8").strip() or "missing"

    def mark_tainted(
        self,
        *,
        source: str,
        scope: str,
        command: str = "",
        details: dict[str, Any] | None = None,
    ) -> ResetResult:
        payload = {
            "source": source,
            "scope": scope,
            "command": command,
            "details": details or {},
            "marked_at_epoch": time.time(),
            "instance_id": self._resolve_instance_id(),
        }
        self._write_taint_marker(payload)
        result = self._result(
            status="ok",
            reason_code="taint_marked",
            details=payload,
        )
        self._write_state(result.to_dict())
        return result

    def _capture_baseline_dconf_if_missing(self) -> None:
        if self.config.dconf_snapshot.exists():
            return
        self.config.dconf_snapshot.parent.mkdir(parents=True, exist_ok=True)
        result = self._run_shell(f"runuser -u {self.config.desktop_user} -- dconf dump /", check=False)
        if result.returncode == 0:
            self.config.dconf_snapshot.write_text(result.stdout, encoding="utf-8")

    def _overlay_upper_clean(self) -> bool:
        if not self.config.workspace_upper.exists():
            return True
        manifest = _build_manifest(self.config.workspace_upper)
        return all(
            rel_path == "."
                or _path_is_allowed_runtime_artifact(rel_path, self.config.ignored_relative_paths)
                for rel_path in manifest
        )

    def _overlay_upper_summary(self) -> dict[str, Any]:
        if not self.config.workspace_upper.exists():
            return {"entries": 0, "sample": []}
        manifest = _build_manifest(self.config.workspace_upper)
        entries = sorted(
            rel_path for rel_path in manifest if rel_path != "."
        )
        return {"entries": len(entries), "sample": entries[:50]}

    def _stop_control_plane_server(self) -> None:
        self._run(["systemctl", "stop", self.config.osworld_server_service], check=False)

    def _start_control_plane_server(self) -> None:
        self._run(["systemctl", "start", self.config.osworld_server_service], check=False)

    def _service_exists(self, unit: str) -> bool:
        result = self._run(["systemctl", "list-unit-files", unit], check=False)
        return result.returncode == 0

    def _stop_graphical_session(self) -> None:
        unit = self.config.osworld_graphical_session_service
        if self._service_exists(unit):
            self._run(["systemctl", "stop", unit], check=False)

    def _start_graphical_session(self) -> None:
        unit = self.config.osworld_graphical_session_service
        if self._service_exists(unit):
            self._run(["systemctl", "start", unit], check=False)

    def _restart_display_stack(self) -> None:
        self._run(["loginctl", "terminate-user", self.config.desktop_user], check=False)
        if self._service_exists(self.config.osworld_graphical_session_service):
            self._start_graphical_session()
        else:
            display_manager_unit = self._resolve_display_manager_service()
            if display_manager_unit is not None:
                active = self._run(["systemctl", "is-active", display_manager_unit], check=False)
                if active.returncode == 0:
                    self._run(["systemctl", "restart", display_manager_unit], check=False)
        self._start_control_plane_server()

    def _resolve_display_manager_service(self) -> str | None:
        configured = self.config.display_manager_service
        candidates = [configured, *DISPLAY_MANAGER_CANDIDATES]
        seen: set[str] = set()
        for unit in candidates:
            if not unit or unit in seen:
                continue
            seen.add(unit)
            result = self._run(["systemctl", "list-unit-files", unit], check=False)
            if result.returncode == 0:
                return unit
        return None

    def _kill_task_user_processes(self) -> None:
        self._run(["loginctl", "kill-user", self.config.desktop_user, "--signal=KILL"], check=False)
        self._run(["pkill", "-KILL", "-u", self.config.desktop_user], check=False)

    def _list_user_processes(self) -> list[str]:
        result = self._run(["pgrep", "-a", "-u", self.config.desktop_user], check=False)
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _wait_for_no_user_processes(self, timeout: float = 15.0, poll: float = 0.5) -> tuple[bool, list[str]]:
        deadline = time.time() + timeout
        last_seen = self._list_user_processes()
        while time.time() < deadline:
            if not last_seen:
                return True, []
            time.sleep(poll)
            last_seen = self._list_user_processes()
        return not last_seen, last_seen

    def _umount_home_overlay(self) -> None:
        is_mountpoint, matches, options = self._home_overlay_status()
        if not is_mountpoint:
            raise RuntimeError(f"{self.config.workspace_home} is not mounted as an overlay workspace")
        if not matches:
            raise RuntimeError(f"{self.config.workspace_home} mount does not match expected overlay options: {options}")
        self._run(["umount", self.config.workspace_home.as_posix()], check=True)

    def _mount_home_overlay(self) -> None:
        result = self.ensure_home_overlay_mounted()
        if result.status != "ok":
            raise RuntimeError(result.reason_code)

    def _reset_overlay_storage(self) -> None:
        for path in self.config.overlay_dirs:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        self._ensure_overlay_storage_permissions()

    def _restore_dconf(self) -> None:
        if not self.config.dconf_snapshot.exists():
            return
        self._run(["pkill", "-KILL", "dconf"], check=False)
        self._run_shell(f"runuser -u {self.config.desktop_user} -- dconf reset -f /", check=False)
        escaped = self.config.dconf_snapshot.as_posix()
        self._run_shell(
            f"runuser -u {self.config.desktop_user} -- sh -lc 'dconf load / < {escaped}'",
            check=False,
        )
        self._run(["pkill", "-KILL", "dconf"], check=False)

    def _clear_user_temp(self) -> None:
        uid = str(self._desktop_uid())
        for tmp_root in self.config.user_tmp_dirs:
            if not tmp_root.exists():
                continue
            self._run_shell(
                f"find {tmp_root} -xdev -mindepth 1 -user {uid} -exec rm -rf {{}} +",
                check=False,
            )

    def _clear_user_crontab(self) -> None:
        self._run(["crontab", "-u", self.config.desktop_user, "-r"], check=False)

    def _server_health_ok(self) -> bool:
        try:
            response = requests.get(f"{self.config.osworld_server_url}{self.config.health_endpoint}", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _screenshot_ok(self) -> bool:
        try:
            response = requests.get(
                f"{self.config.osworld_server_url}{self.config.screenshot_endpoint}",
                timeout=10,
            )
            return response.status_code == 200 and bool(response.content)
        except Exception:
            return False

    def _wait_for_server_health(self, timeout: float = 10.0, poll: float = 0.5) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._server_health_ok() and self._screenshot_ok():
                return True
            time.sleep(poll)
        return False

    def _detect_unsupported_system_drift(self, metadata: dict[str, Any]) -> dict[str, Any] | None:
        taint = self._load_taint_marker()
        if taint:
            return {
                "field": "system_taint",
                "expected": "clean",
                "actual": taint.get("scope", "tainted"),
                "taint": taint,
            }
        current = {
            "expected_session_user": self.config.desktop_user,
            "control_plane_build_id": self._control_plane_build_id(),
        }
        for field, value in current.items():
            expected = metadata.get(field)
            if expected is None:
                continue
            if value != expected:
                return {"field": field, "expected": expected, "actual": value}
        return None

    def prepare_baseline(self) -> ResetResult:
        self._ensure_runtime_layout()
        self._write_state(self._result(status="busy", reason_code="preparing_baseline").to_dict())
        try:
            self._validate_workspace_target()
        except RuntimeError as exc:
            result = self._result(
                status="error",
                reason_code="unsafe_workspace_target",
                details={
                    "desktop_user": self.config.desktop_user,
                    "workspace_home": self.config.workspace_home.as_posix(),
                    "error": str(exc),
                },
            )
            self._write_state(result.to_dict())
            return result
        if not self.config.baseline_home.exists():
            result = self._result(status="error", reason_code="baseline_missing")
            self._write_state(result.to_dict())
            return result

        logger.info("Preparing baseline: ensuring home overlay is mounted")
        overlay_result = self.ensure_home_overlay_mounted()
        if overlay_result.status != "ok":
            self._write_state(overlay_result.to_dict())
            return overlay_result

        logger.info("Preparing baseline: capturing dconf snapshot if missing")
        self._capture_baseline_dconf_if_missing()
        self._clear_taint_marker()
        baseline_version = self.config.baseline_version or f"baseline-{int(time.time())}"
        metadata = {
            "runtime_version": RUNTIME_VERSION,
            "ami_build_version": self.config.ami_build_version,
            "baseline_version": baseline_version,
            "baseline_mode": self.config.baseline_mode,
            "expected_session_user": self.config.desktop_user,
            "verification_policy_version": self.config.verification_policy_version,
            "control_plane_build_id": self._control_plane_build_id(),
            "prepared_at_epoch": time.time(),
        }
        self._write_metadata(metadata)
        result = self._result(
            status="ok",
            reason_code="baseline_ready",
            baseline_version=baseline_version,
        )
        self._write_state(result.to_dict())
        return result

    def reset(self) -> ResetResult:
        self._ensure_runtime_layout()
        with open(self.config.lock_path, "a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                metadata = self._load_metadata()
            except FileNotFoundError:
                result = self._result(status="error", reason_code="baseline_missing")
                self._write_state(result.to_dict())
                return result

            drift = self._detect_unsupported_system_drift(metadata)
            if drift is not None:
                result = self._result(
                    status="error",
                    reason_code="unsupported_system_drift",
                    details=drift,
                )
                self._write_state(result.to_dict())
                return result

            try:
                self._write_state(self._result(status="busy", reason_code="resetting").to_dict())
                self._stop_control_plane_server()
                self._stop_graphical_session()
                self._kill_task_user_processes()
                drained, survivors = self._wait_for_no_user_processes()
                if not drained:
                    result = self._result(status="error", reason_code="session_stop_failed")
                    result.details["surviving_processes"] = survivors
                    self._write_state(result.to_dict())
                    return result

                self._umount_home_overlay()
                self._reset_overlay_storage()
                self._mount_home_overlay()
                self._clear_user_temp()
                self._clear_user_crontab()
                self._restore_dconf()
                self._restart_display_stack()
                generation = self._bump_reset_generation()
                result = self._result(
                    status="ok",
                    reason_code="reset_completed",
                    reset_generation=generation,
                    baseline_version=metadata.get("baseline_version", "unknown"),
                )
                self._write_state(result.to_dict())
                return result
            except subprocess.CalledProcessError as exc:
                result = self._result(
                    status="error",
                    reason_code="workspace_restore_failed",
                    details={"command": exc.cmd, "stderr": exc.stderr, "stdout": exc.stdout},
                )
                self._write_state(result.to_dict())
                return result
            except RuntimeError as exc:
                result = self._result(
                    status="error",
                    reason_code="workspace_restore_failed",
                    details={"error": str(exc)},
                )
                self._write_state(result.to_dict())
                return result
            except Exception as exc:
                result = self._result(
                    status="error",
                    reason_code="verification_failed",
                    details={"error": str(exc)},
                )
                self._write_state(result.to_dict())
                return result
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def verify(self) -> ResetResult:
        try:
            metadata = self._load_metadata()
        except FileNotFoundError:
            result = self._result(status="error", reason_code="baseline_missing")
            self._write_state(result.to_dict())
            return result

        drift = self._detect_unsupported_system_drift(metadata)
        if drift is not None:
            result = self._result(
                status="error",
                reason_code="unsupported_system_drift",
                details=drift,
            )
            self._write_state(result.to_dict())
            return result

        is_mountpoint, matches, options = self._home_overlay_status()
        if not is_mountpoint or not matches:
            result = self._result(
                status="error",
                reason_code="overlay_mount_mismatch",
                details={"findmnt_options": options},
            )
            self._write_state(result.to_dict())
            return result

        if not self._wait_for_server_health():
            result = self._result(status="error", reason_code="server_health_failed")
            self._write_state(result.to_dict())
            return result

        if not self._overlay_upper_clean():
            result = self._result(
                status="error",
                reason_code="workspace_not_clean",
                details={
                    "overlay_upper": self.config.workspace_upper.as_posix(),
                    **self._overlay_upper_summary(),
                },
            )
            self._write_state(result.to_dict())
            return result

        result = self._result(
            status="ok",
            reason_code="verified_clean",
            baseline_version=metadata.get("baseline_version", "unknown"),
        )
        self._write_state(result.to_dict())
        return result

    def state(self) -> dict[str, Any]:
        return self._load_state()


def _build_runtime_from_env() -> ResetRuntime:
    logging.basicConfig(level=logging.INFO)
    return ResetRuntime(ResetConfig())


def main() -> int:
    parser = argparse.ArgumentParser(description="OSWorld AWS reset runtime CLI")
    parser.add_argument("command", choices=["prepare-baseline", "mount-home", "umount-home", "reset", "verify", "state"])
    args = parser.parse_args()

    runtime = _build_runtime_from_env()
    if args.command == "prepare-baseline":
        result = runtime.prepare_baseline()
    elif args.command == "mount-home":
        result = runtime.ensure_home_overlay_mounted()
    elif args.command == "umount-home":
        try:
            runtime._umount_home_overlay()
            result = runtime._result(status="ok", reason_code="home_overlay_unmounted")
        except Exception as exc:
            result = runtime._result(
                status="error",
                reason_code="overlay_unmount_failed",
                details={"error": str(exc)},
            )
    elif args.command == "reset":
        result = runtime.reset()
    elif args.command == "verify":
        result = runtime.verify()
    else:
        print(json.dumps(runtime.state(), indent=2, sort_keys=True))
        return 0

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
