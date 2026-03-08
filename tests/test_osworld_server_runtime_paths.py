import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "OSWorld"))

from desktop_env.server.runtime_paths import (  # noqa: E402
    DEFAULT_SERVER_STATE_ROOT,
    SERVER_STATE_ROOT_ENV,
    resolve_user_path,
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


if __name__ == "__main__":
    unittest.main()
