import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "OSWorld"))

from desktop_env.server.runtime_paths import (  # noqa: E402
    DEFAULT_SERVER_STATE_ROOT,
    SERVER_STATE_ROOT_ENV,
    normalize_command_for_runtime,
    resolve_user_path,
    rewrite_runtime_compat_paths,
    server_artifact_path,
)


class TestOSWorldServerRuntimePaths(unittest.TestCase):
    def test_relative_paths_resolve_inside_user_home(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir()
            self.assertEqual(resolve_user_path("setup.sh", home=home), home / "setup.sh")
            self.assertEqual(resolve_user_path("Downloads/file.txt", home=home), home / "Downloads" / "file.txt")

    def test_absolute_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            absolute = Path(tmpdir) / "outside" / "data.txt"
            self.assertEqual(resolve_user_path(absolute, home=home), absolute)

    def test_server_artifacts_live_outside_control_plane(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old = os.environ.get(SERVER_STATE_ROOT_ENV)
            os.environ[SERVER_STATE_ROOT_ENV] = tmpdir
            try:
                artifact = server_artifact_path("screenshots", "screenshot.png")
            finally:
                if old is None:
                    os.environ.pop(SERVER_STATE_ROOT_ENV, None)
                else:
                    os.environ[SERVER_STATE_ROOT_ENV] = old
            self.assertEqual(artifact, Path(tmpdir) / "screenshots" / "screenshot.png")
            self.assertNotIn("/opt/osworld", artifact.as_posix())

    def test_default_server_state_root_is_tmp(self):
        self.assertEqual(DEFAULT_SERVER_STATE_ROOT, "/tmp/osworld-server")

    def test_runtime_path_rewrite_updates_legacy_bus_path(self):
        command = "export DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus' && gnome-terminal --working-directory=/home/user"
        normalized = rewrite_runtime_compat_paths(
            command,
            runtime_home="/home/user",
            runtime_dir="/run/user/1001",
        )
        self.assertIn("/run/user/1001/bus", normalized)
        self.assertIn("/home/user", normalized)

    def test_runtime_command_normalization_rewrites_list_arguments(self):
        command = [
            "gnome-terminal",
            "--working-directory=/home/user",
            "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
        ]
        old_home = os.environ.get("HOME")
        old_runtime = os.environ.get("XDG_RUNTIME_DIR")
        os.environ["HOME"] = "/home/user"
        os.environ["XDG_RUNTIME_DIR"] = "/run/user/1001"
        try:
            normalized = normalize_command_for_runtime(command, shell=False)
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
            if old_runtime is None:
                os.environ.pop("XDG_RUNTIME_DIR", None)
            else:
                os.environ["XDG_RUNTIME_DIR"] = old_runtime
        self.assertEqual(normalized[1], "--working-directory=/home/user")
        self.assertEqual(normalized[2], "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus")


if __name__ == "__main__":
    unittest.main()
