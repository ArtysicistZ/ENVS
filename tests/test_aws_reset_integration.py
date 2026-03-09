import os
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import requests
from flask import Flask, jsonify, Response
from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "OSWorld"))

from desktop_env.providers.aws import reset_daemon, vm_reset
from desktop_env.providers.aws.reset_runtime import ResetConfig, ResetRuntime


def _free_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
    except PermissionError as exc:  # pragma: no cover - sandbox-dependent
        raise unittest.SkipTest(f"localhost socket bind unavailable in this environment: {exc}") from exc


class _ServerThread(threading.Thread):
    def __init__(self, app, host: str, port: int):
        super().__init__(daemon=True)
        self._server = make_server(host, port, app)
        self.host = host
        self.port = port

    def run(self):
        self._server.serve_forever()

    def shutdown(self):
        self._server.shutdown()


class FakeResetRuntime(ResetRuntime):
    def __init__(self, config: ResetConfig):
        super().__init__(config)
        self.control_plane_build_id = "control-build-id"
        self.server_started = False
        self.overlay_mounted = True

    def _resolve_instance_id(self) -> str:
        return "i-integration"

    def _control_plane_build_id(self) -> str:
        return self.control_plane_build_id

    def _capture_baseline_dconf_if_missing(self) -> None:
        self.config.dconf_snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not self.config.dconf_snapshot.exists():
            self.config.dconf_snapshot.write_text("", encoding="utf-8")

    def _ensure_overlay_storage_permissions(self) -> None:
        for path in self.config.overlay_dirs:
            path.mkdir(parents=True, exist_ok=True)

    def ensure_home_overlay_mounted(self):
        self.overlay_mounted = True
        return self._result(status="ok", reason_code="home_overlay_ready")

    def _home_overlay_status(self) -> tuple[bool, bool, str]:
        if self.overlay_mounted:
            return True, True, "overlay"
        return False, False, ""

    def _stop_control_plane_server(self) -> None:
        self.server_started = False

    def _start_control_plane_server(self) -> None:
        self.server_started = True

    def _restart_display_stack(self) -> None:
        self._start_control_plane_server()

    def _kill_task_user_processes(self) -> None:
        return None

    def _umount_home_overlay(self) -> None:
        self.overlay_mounted = False
        if self.config.workspace_home.exists():
            shutil.rmtree(self.config.workspace_home)

    def _mount_home_overlay(self) -> None:
        shutil.copytree(self.config.baseline_home, self.config.workspace_home, symlinks=True)
        self.config.workspace_upper.mkdir(parents=True, exist_ok=True)
        self.config.workspace_work.mkdir(parents=True, exist_ok=True)
        self.overlay_mounted = True

    def _restore_dconf(self) -> None:
        return None

    def _clear_user_temp(self) -> None:
        return None

    def _clear_user_crontab(self) -> None:
        return None


class TestAWSResetIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.baseline_home = root / "baseline-home"
        self.workspace_home = root / "home-user"
        self.control_plane_root = root / "control"
        self.state_root = root / "state"
        self.session_root = root / "session"
        self.dconf_snapshot = root / "baseline-dconf" / "user.dconf"

        (self.baseline_home / "Desktop").mkdir(parents=True)
        (self.baseline_home / ".config").mkdir(parents=True)
        (self.baseline_home / "Desktop" / "baseline.txt").write_text("clean", encoding="utf-8")
        (self.baseline_home / ".config" / "prefs.json").write_text("{}", encoding="utf-8")
        (self.control_plane_root / "server").mkdir(parents=True)
        (self.control_plane_root / "server" / "main.py").write_text("print('ok')\n", encoding="utf-8")
        shutil.copytree(self.baseline_home, self.workspace_home)

        self.mock_server_port = _free_port()
        self.mock_server = Flask("mock_osworld")

        @self.mock_server.get("/health")
        def health():
            return jsonify({"status": "ok"})

        @self.mock_server.get("/screenshot")
        def screenshot():
            return Response(b"\x89PNG\r\n\x1a\nmock", mimetype="image/png")

        self.mock_server_thread = _ServerThread(self.mock_server, "127.0.0.1", self.mock_server_port)
        self.mock_server_thread.start()

        self.resetd_port = _free_port()
        self.runtime = FakeResetRuntime(
            ResetConfig(
                desktop_user="user",
                workspace_home=self.workspace_home,
                allow_unsafe_home=True,
                control_plane_root=self.control_plane_root,
                baseline_home=self.baseline_home,
                dconf_snapshot=self.dconf_snapshot,
                session_root=self.session_root,
                state_root=self.state_root,
                metadata_path=self.state_root / "metadata.json",
                state_path=self.state_root / "state.json",
                lock_path=self.state_root / "reset.lock",
                baseline_manifest_path=self.state_root / "baseline_home_manifest.json",
                control_plane_stamp_path=self.state_root / "control_plane_build_id",
                reset_generation_path=self.state_root / "generation.txt",
                osworld_server_url=f"http://127.0.0.1:{self.mock_server_port}",
                ignored_relative_paths=(".cache",),
            )
        )
        self.runtime.server_started = True
        self._old_runtime = reset_daemon.runtime
        reset_daemon.runtime = self.runtime
        self.resetd_thread = _ServerThread(reset_daemon.app, "127.0.0.1", self.resetd_port)
        self.resetd_thread.start()

    def tearDown(self):
        self.resetd_thread.shutdown()
        self.mock_server_thread.shutdown()
        reset_daemon.runtime = self._old_runtime
        self.tmpdir.cleanup()

    def test_host_client_can_drive_real_reset_daemon_and_mock_osworld_server(self):
        baseline = vm_reset.prepare_baseline(
            "127.0.0.1",
            "i-integration",
            vm_reset.ResetClientConfig(port=self.resetd_port, timeout=10),
        )
        self.assertEqual(baseline["status"], "ok")

        (self.workspace_home / "Desktop" / "baseline.txt").write_text("dirty", encoding="utf-8")
        (self.workspace_home / "Desktop" / "leftover.tmp").write_text("temp", encoding="utf-8")
        self.runtime.config.workspace_upper.mkdir(parents=True, exist_ok=True)
        (self.runtime.config.workspace_upper / "junk").write_text("junk", encoding="utf-8")

        result = vm_reset.soft_reset(
            "127.0.0.1",
            "i-integration",
            vm_reset.ResetClientConfig(port=self.resetd_port, timeout=10),
        )
        self.assertEqual(result["status"], "reused_clean")
        self.assertEqual(result["reason_code"], "verified_clean")
        self.assertEqual((self.workspace_home / "Desktop" / "baseline.txt").read_text(encoding="utf-8"), "clean")
        self.assertFalse((self.workspace_home / "Desktop" / "leftover.tmp").exists())

        state_resp = requests.get(f"http://127.0.0.1:{self.resetd_port}/state", timeout=5)
        self.assertEqual(state_resp.status_code, 200)
        self.assertEqual(state_resp.json()["reason_code"], "verified_clean")

if __name__ == "__main__":
    unittest.main()
