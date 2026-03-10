import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "OSWorld"))

from desktop_env.providers.aws import vm_reset


def _mock_response(payload):
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = payload
    return response


class TestAWSVMResetClient(unittest.TestCase):
    @patch("desktop_env.providers.aws.vm_reset.requests.request")
    def test_soft_reset_returns_reused_clean_after_verify(self, mock_request):
        mock_request.side_effect = [
            _mock_response(
                {
                    "status": "ok",
                    "reason_code": "reset_completed",
                    "instance_id": "i-1",
                    "baseline_version": "b1",
                    "reset_generation": 1,
                }
            ),
            _mock_response(
                {
                    "status": "ok",
                    "reason_code": "verified_clean",
                    "instance_id": "i-1",
                    "baseline_version": "b1",
                    "reset_generation": 1,
                }
            ),
        ]

        result = vm_reset.soft_reset("10.0.0.1", "i-1")
        self.assertEqual(result["status"], "reused_clean")
        self.assertEqual(result["reason_code"], "verified_clean")

    @patch("desktop_env.providers.aws.vm_reset.requests.request")
    def test_soft_reset_requests_relaunch_when_verify_fails(self, mock_request):
        mock_request.side_effect = [
            _mock_response(
                {
                    "status": "ok",
                    "reason_code": "reset_completed",
                    "instance_id": "i-1",
                    "baseline_version": "b1",
                    "reset_generation": 2,
                }
            ),
            _mock_response(
                {
                    "status": "error",
                    "reason_code": "workspace_not_clean",
                    "instance_id": "i-1",
                    "baseline_version": "b1",
                    "reset_generation": 2,
                    "details": {"problem": "manifest mismatch"},
                }
            ),
        ]

        result = vm_reset.soft_reset("10.0.0.1", "i-1")
        self.assertEqual(result["status"], "must_relaunch")
        self.assertEqual(result["reason_code"], "workspace_not_clean")

    @patch("desktop_env.providers.aws.vm_reset.requests.request", side_effect=RuntimeError("no daemon"))
    def test_soft_reset_requests_relaunch_when_daemon_is_unreachable(self, _mock_request):
        result = vm_reset.soft_reset("10.0.0.1", "i-1")
        self.assertEqual(result["status"], "must_relaunch")
        self.assertEqual(result["reason_code"], "daemon_unreachable")


if __name__ == "__main__":
    unittest.main()
