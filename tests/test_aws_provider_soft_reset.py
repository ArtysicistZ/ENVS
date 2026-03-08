import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "OSWorld"))

from desktop_env.providers.aws.provider import AWSProvider


class TestAWSProviderSoftReset(unittest.TestCase):
    @patch.object(AWSProvider, "_full_relaunch", return_value="i-new")
    @patch("desktop_env.providers.aws.provider.vm_reset.soft_reset")
    @patch("desktop_env.providers.aws.provider.boto3.client")
    def test_revert_to_snapshot_reuses_instance_when_soft_reset_is_clean(
        self, mock_boto_client, mock_soft_reset, mock_full_relaunch
    ):
        mock_boto_client.return_value.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": "running"}, "PrivateIpAddress": "10.0.0.1"}]}]
        }
        mock_soft_reset.return_value = {
            "status": "reused_clean",
            "reason_code": "verified_clean",
            "details": {},
        }

        provider = AWSProvider(region="us-east-1")
        result = provider.revert_to_snapshot("i-123", "ami-xyz")
        self.assertEqual(result, "i-123")
        mock_full_relaunch.assert_not_called()

    @patch.object(AWSProvider, "_full_relaunch", return_value="i-new")
    @patch("desktop_env.providers.aws.provider.vm_reset.soft_reset")
    @patch("desktop_env.providers.aws.provider.boto3.client")
    def test_revert_to_snapshot_falls_back_when_soft_reset_requests_relaunch(
        self, mock_boto_client, mock_soft_reset, mock_full_relaunch
    ):
        mock_boto_client.return_value.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": "running"}, "PrivateIpAddress": "10.0.0.1"}]}]
        }
        mock_soft_reset.return_value = {
            "status": "must_relaunch",
            "reason_code": "workspace_not_clean",
            "details": {"problem": "manifest mismatch"},
        }

        provider = AWSProvider(region="us-east-1")
        result = provider.revert_to_snapshot("i-123", "ami-xyz")
        self.assertEqual(result, "i-new")
        mock_full_relaunch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
