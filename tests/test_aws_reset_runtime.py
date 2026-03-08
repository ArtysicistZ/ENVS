import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "OSWorld"))

from desktop_env.providers.aws.reset_runtime import ResetConfig, ResetRuntime


class FakeResetRuntime(ResetRuntime):
    def __init__(self, config: ResetConfig):
        super().__init__(config)
        self.fingerprints = {
            "package_fingerprint": "pkg-fp",
            "user_fingerprint": "user-fp",
            "control_plane_manifest_sha256": "control-fp",
        }
        self.server_started = False
        self.user_processes_quiesced = True
        self.overlay_mounted = True

    def _resolve_instance_id(self) -> str:
        return "i-test"

    def _package_fingerprint(self) -> str:
        return self.fingerprints["package_fingerprint"]

    def _user_fingerprint(self) -> str:
        return self.fingerprints["user_fingerprint"]

    def _control_plane_manifest_hash(self) -> str:
        return self.fingerprints["control_plane_manifest_sha256"]

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

    def _assert_no_user_processes(self) -> bool:
        return self.user_processes_quiesced

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

    def _server_health_ok(self) -> bool:
        return self.server_started

    def _screenshot_ok(self) -> bool:
        return self.server_started


class TestAWSResetRuntime(unittest.TestCase):
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

        self.config = ResetConfig(
            desktop_user="user",
            workspace_home=self.workspace_home,
            control_plane_root=self.control_plane_root,
            baseline_home=self.baseline_home,
            dconf_snapshot=self.dconf_snapshot,
            session_root=self.session_root,
            state_root=self.state_root,
            metadata_path=self.state_root / "metadata.json",
            state_path=self.state_root / "state.json",
            lock_path=self.state_root / "reset.lock",
            baseline_manifest_path=self.state_root / "baseline_home_manifest.json",
            reset_generation_path=self.state_root / "generation.txt",
            osworld_server_url="http://127.0.0.1:5000",
            ignored_relative_paths=(".cache",),
        )
        self.runtime = FakeResetRuntime(self.config)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_prepare_reset_verify_restores_clean_workspace(self):
        prepared = self.runtime.prepare_baseline()
        self.assertEqual(prepared.status, "ok")
        self.assertTrue(self.config.metadata_path.exists())
        self.assertTrue(self.config.baseline_manifest_path.exists())

        (self.workspace_home / "Desktop" / "baseline.txt").write_text("dirty", encoding="utf-8")
        (self.workspace_home / "Desktop" / "leftover.tmp").write_text("temp", encoding="utf-8")
        self.config.workspace_upper.mkdir(parents=True, exist_ok=True)
        (self.config.workspace_upper / "junk").write_text("junk", encoding="utf-8")

        reset = self.runtime.reset()
        self.assertEqual(reset.status, "ok")
        verify = self.runtime.verify()
        self.assertEqual(verify.status, "ok")
        self.assertEqual((self.workspace_home / "Desktop" / "baseline.txt").read_text(encoding="utf-8"), "clean")
        self.assertFalse((self.workspace_home / "Desktop" / "leftover.tmp").exists())
        self.assertEqual(self.runtime._load_reset_generation(), 1)

    def test_verify_detects_unsupported_system_drift(self):
        self.runtime.prepare_baseline()
        self.runtime.server_started = True
        self.runtime.fingerprints["package_fingerprint"] = "changed"

        verify = self.runtime.verify()
        self.assertEqual(verify.status, "error")
        self.assertEqual(verify.reason_code, "unsupported_system_drift")
        self.assertEqual(verify.details["field"], "package_fingerprint")

    def test_reset_fails_when_user_processes_survive(self):
        self.runtime.prepare_baseline()
        self.runtime.user_processes_quiesced = False

        reset = self.runtime.reset()
        self.assertEqual(reset.status, "error")
        self.assertEqual(reset.reason_code, "session_stop_failed")

    def test_verify_allows_known_runtime_artifacts(self):
        self.runtime.prepare_baseline()
        self.runtime.server_started = True
        (self.workspace_home / ".cache").mkdir(parents=True, exist_ok=True)
        (self.workspace_home / ".cache" / "session").write_text("runtime", encoding="utf-8")
        self.config.workspace_upper.mkdir(parents=True, exist_ok=True)
        (self.config.workspace_upper / ".cache").mkdir(parents=True, exist_ok=True)
        (self.config.workspace_upper / ".cache" / "session").write_text("runtime", encoding="utf-8")

        verify = self.runtime.verify()
        self.assertEqual(verify.status, "ok")


if __name__ == "__main__":
    unittest.main()
