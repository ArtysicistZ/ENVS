import json
import shlex
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "test_data" / "osworld_examples" / "tasks"
TRAIN_MANIFEST = REPO_ROOT / "test_data" / "osworld_examples" / "train_all_128.json"
INSTALL_SCRIPT = REPO_ROOT / "OSWorld" / "desktop_env" / "providers" / "aws" / "scripts" / "install_resetd.sh"
PROVISION_SCRIPT = REPO_ROOT / "OSWorld" / "desktop_env" / "providers" / "aws" / "scripts" / "provision_osworld_desktop.sh"
LAUNCH_SCRIPT = (
    REPO_ROOT / "OSWorld" / "desktop_env" / "providers" / "aws" / "scripts" / "launch_osworld_graphical_session.sh"
)
RESET_RUNTIME = REPO_ROOT / "OSWorld" / "desktop_env" / "providers" / "aws" / "reset_runtime.py"
RUNTIME_PATHS = REPO_ROOT / "OSWorld" / "desktop_env" / "server" / "runtime_paths.py"


def _task_ids() -> set[str]:
    manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for values in manifest.values():
        ids.update(values)
    return ids


def _iter_tasks():
    ids = _task_ids()
    for path in TASK_ROOT.rglob("*.json"):
        if path.stem not in ids:
            continue
        yield path, json.loads(path.read_text(encoding="utf-8"))


class TestOSWorld128RuntimeContract(unittest.TestCase):
    def test_manifest_contains_expected_128_tasks(self):
        self.assertEqual(len(_task_ids()), 128)

    def test_corpus_runtime_path_assumptions_are_supported(self):
        runtime_paths_text = RUNTIME_PATHS.read_text(encoding="utf-8")
        home_user_refs = 0
        runtime_dir_refs = 0

        for _, task in _iter_tasks():
            text = json.dumps(task)
            if "/home/user" in text:
                home_user_refs += 1
            if "/run/user/1000" in text:
                runtime_dir_refs += 1

        self.assertGreater(home_user_refs, 0)
        self.assertGreater(runtime_dir_refs, 0)
        self.assertIn('rewritten.replace("/home/user", runtime_home)', runtime_paths_text)
        self.assertIn('rewritten.replace("/run/user/1000", runtime_dir)', runtime_paths_text)

    def test_required_app_and_tool_surface_is_provisioned(self):
        provision_text = PROVISION_SCRIPT.read_text(encoding="utf-8")
        first_tokens = Counter()
        corpus_text = []

        for _, task in _iter_tasks():
            corpus_text.append(json.dumps(task))
            for cfg in task.get("config", []):
                if cfg.get("type") not in {"launch", "execute", "command"}:
                    continue
                command = cfg.get("parameters", {}).get("command")
                if not command:
                    continue
                if isinstance(command, list):
                    token = command[0]
                else:
                    token = shlex.split(command)[0]
                first_tokens[token] += 1

        # These all occur in the real 128-task corpus and must be available on the guest VM.
        self.assertGreater(first_tokens["google-chrome"], 0)
        self.assertGreater(first_tokens["code"], 0)
        self.assertGreater(first_tokens["/usr/bin/thunderbird"], 0)
        self.assertGreater(first_tokens["vlc"], 0)
        self.assertGreater(first_tokens["gimp"], 0)
        self.assertGreater(first_tokens["socat"], 0)
        self.assertGreater(first_tokens["python"], 0)
        self.assertTrue(any("gnome-terminal" in text for text in corpus_text))

        for required in (
            "ensure_google_chrome",
            "ensure_vscode",
            "thunderbird",
            "vlc",
            "gimp",
            "libreoffice",
            "gnome-terminal",
            "jq",
            "expect",
            "sqlite3",
            "socat",
            "python-is-python3",
            "libglib2.0-bin",
            "psmisc",
        ):
            self.assertIn(required, provision_text)

    def test_critical_app_config_paths_are_seeded_and_verified(self):
        install_text = INSTALL_SCRIPT.read_text(encoding="utf-8")
        launch_text = LAUNCH_SCRIPT.read_text(encoding="utf-8")
        reset_text = RESET_RUNTIME.read_text(encoding="utf-8")

        # Paths referenced directly by the 128-task corpus or by evaluator getters.
        required_layout = (
            ".config/Code/User",
            ".config/Code/User/settings.json",
            ".config/google-chrome/Default",
            ".config/google-chrome/Default/Preferences",
            ".config/google-chrome/Local State",
            ".config/google-chrome/Default/Bookmarks",
            ".config/libreoffice/4/user",
            ".config/vlc",
            ".config/vlc/vlcrc",
            ".config/GIMP/2.10",
            ".thunderbird",
        )
        for rel_path in required_layout:
            self.assertTrue(
                rel_path in install_text or rel_path in launch_text,
                msg=f"{rel_path} must be created in the baseline or session home",
            )

        for verified_path in (
            ".config/Code/User/settings.json",
            ".config/google-chrome/Default/Preferences",
            ".config/google-chrome/Local State",
            ".config/google-chrome/Default/Bookmarks",
            ".config/vlc/vlcrc",
            ".thunderbird",
            ".config/GIMP/2.10",
            ".config/libreoffice/4/user",
        ):
            self.assertIn(verified_path, reset_text)


if __name__ == "__main__":
    unittest.main()
