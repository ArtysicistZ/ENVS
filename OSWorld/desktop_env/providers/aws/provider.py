import boto3
from botocore.exceptions import ClientError

import logging
import os
import time
import requests
from datetime import datetime, timedelta, timezone
from desktop_env.providers.base import Provider

# TTL configuration
from desktop_env.providers.aws.config import (
    AWS_RESETD_PORT,
    AWS_SCHEDULER_ROLE_ARN,
    DEFAULT_TTL_MINUTES,
    ENABLE_TTL,
)
from desktop_env.providers.aws.scheduler_utils import schedule_instance_termination
from desktop_env.providers.aws import vm_reset
from desktop_env.providers.aws.manager import _VM_USER_DATA

logger = logging.getLogger("desktopenv.providers.aws.AWSProvider")
logger.setLevel(logging.INFO)

WAIT_DELAY = 15
MAX_ATTEMPTS = 10
VM_READY_TIMEOUT = 120  # seconds to wait for OSWorld server inside the VM to be reachable
VM_READY_POLL = 5       # poll interval in seconds


class AWSProvider(Provider):

    def _check_port_5000(self, ip: str) -> bool:
        """Return True if port 5000 responds to /health or /screenshot."""
        for endpoint in ("/health", "/screenshot"):
            try:
                r = requests.get(f"http://{ip}:5000{endpoint}", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
        return False

    def _resetd_reset(self, ip: str, instance_id: str) -> dict:
        """Call /reset on the reset daemon directly (no verify). Returns response dict."""
        config = vm_reset.ResetClientConfig(port=AWS_RESETD_PORT, timeout=60)
        return vm_reset._request_json(
            method="POST", ip=ip, endpoint="/reset",
            payload={"instance_id": instance_id}, config=config,
        )

    def _fix_port_5000_via_resetd(self, ip: str, instance_id: str) -> bool:
        """Use the reset daemon (port 5001) to fix port 5000.

        Sends /reset to restart the correct service. Single attempt with
        up to 60s wait for port 5000 to come up.
        """
        logger.info(
            "Fix-port-5000: sending /reset to %s:%s ...", ip, AWS_RESETD_PORT,
        )
        try:
            result = self._resetd_reset(ip, instance_id)
            logger.info("Reset result: %s", result)
        except Exception as exc:
            logger.warning("Reset call failed: %s", exc)
            return False

        # Wait for the server wrapper to find X display and start Flask
        for _ in range(12):  # 12 × 5s = 60s
            time.sleep(5)
            if self._check_port_5000(ip):
                logger.info("Port 5000 is UP after fix via resetd.")
                return True

        logger.warning("Port 5000 still down after fix via resetd.")
        return False

    def _wait_for_vm_server(self, ip: str, instance_id: str = "", port: int = 5000):
        """Wait until the OSWorld HTTP server inside the VM is reachable."""
        health_url = f"http://{ip}:{port}/health"
        deadline = time.time() + VM_READY_TIMEOUT
        logger.info("Waiting for VM server at %s (timeout=%ss)...", health_url, VM_READY_TIMEOUT)

        # Phase 1: quick check — port 5000 might already be up
        for _ in range(6):  # 6 × 5s = 30s
            if self._check_port_5000(ip):
                logger.info(f"VM server at {ip}:5000 is ready.")
                return
            logger.info(f"VM not ready yet, retrying in {VM_READY_POLL}s...")
            time.sleep(VM_READY_POLL)

        # Phase 2: port 5000 is down — try fixing via reset daemon
        if instance_id and self._is_resetd_reachable(ip):
            if self._fix_port_5000_via_resetd(ip, instance_id):
                return

        # Phase 3: final check with remaining time
        while time.time() < deadline:
            if self._check_port_5000(ip):
                logger.info(f"VM server at {ip}:5000 is ready.")
                return
            time.sleep(VM_READY_POLL)

        raise TimeoutError(
            f"VM server at {health_url} did not become ready within {VM_READY_TIMEOUT}s"
        )

    def start_emulator(self, path_to_vm: str, headless: bool, *args, **kwargs):
        logger.info("Starting AWS VM...")
        ec2_client = boto3.client('ec2', region_name=self.region)

        try:
            # Check the current state of the instance (do not use MaxResults with InstanceIds - API forbids it)
            response = ec2_client.describe_instances(InstanceIds=[path_to_vm])
            state = response['Reservations'][0]['Instances'][0]['State']['Name']
            logger.info(f"Instance {path_to_vm} current state: {state}")

            if state == 'running':
                logger.info(f"Instance {path_to_vm} is already running. Skipping start.")
                return

            if state == 'stopped':
                ec2_client.start_instances(InstanceIds=[path_to_vm])
                logger.info(f"Instance {path_to_vm} is starting...")

                waiter = ec2_client.get_waiter('instance_running')
                waiter.wait(
                    InstanceIds=[path_to_vm],
                    WaiterConfig={'Delay': WAIT_DELAY, 'MaxAttempts': MAX_ATTEMPTS}
                )
                logger.info(f"Instance {path_to_vm} is now running.")
            else:
                logger.warning(f"Instance {path_to_vm} is in state '{state}' and cannot be started.")

        except ClientError as e:
            logger.error(f"Failed to start the AWS VM {path_to_vm}: {str(e)}")
            raise


    def get_ip_address(self, path_to_vm: str) -> str:
        logger.info("Getting AWS VM IP address...")
        ec2_client = boto3.client('ec2', region_name=self.region)

        try:
            response = ec2_client.describe_instances(InstanceIds=[path_to_vm])
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    private_ip_address = instance.get('PrivateIpAddress', '')
                    public_ip_address = instance.get('PublicIpAddress', '')

                    if public_ip_address:
                        vnc_url = f"http://{public_ip_address}:5910/vnc.html"
                        logger.info("="*80)
                        logger.info(f"VNC Web Access URL: {vnc_url}")
                        logger.info(f"Public IP: {public_ip_address}")
                        logger.info(f"Private IP: {private_ip_address}")
                        logger.info("="*80)
                    else:
                        logger.warning("No public IP address available for VNC access")

                    if private_ip_address:
                        self._wait_for_vm_server(private_ip_address, instance_id=path_to_vm)
                        try:
                            vm_reset.snapshot_home(private_ip_address, path_to_vm, port=AWS_RESETD_PORT)
                        except Exception as e:
                            logger.warning(f"Failed to prepare reset baseline on {private_ip_address}: {e}")

                    return private_ip_address
            return ''
        except ClientError as e:
            logger.error(f"Failed to retrieve IP address for the instance {path_to_vm}: {str(e)}")
            raise

    def save_state(self, path_to_vm: str, snapshot_name: str):
        logger.info("Saving AWS VM state...")
        ec2_client = boto3.client('ec2', region_name=self.region)

        try:
            image_response = ec2_client.create_image(InstanceId=path_to_vm, Name=snapshot_name)
            image_id = image_response['ImageId']
            logger.info(f"AMI {image_id} created successfully from instance {path_to_vm}.")
            return image_id
        except ClientError as e:
            logger.error(f"Failed to create AMI from the instance {path_to_vm}: {str(e)}")
            raise

    def _is_resetd_reachable(self, ip: str, timeout: int = 5) -> bool:
        """Quick health probe of the reset daemon. Returns True only if port 5001 responds."""
        try:
            r = requests.get(f"http://{ip}:{AWS_RESETD_PORT}/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        """Revert to clean state via soft reset. No silent fallbacks — raises on failure."""
        ec2_client = boto3.client('ec2', region_name=self.region)

        response = ec2_client.describe_instances(InstanceIds=[path_to_vm])
        instance = response['Reservations'][0]['Instances'][0]
        state = instance['State']['Name']
        private_ip = instance.get('PrivateIpAddress', '')

        if state != 'running' or not private_ip:
            raise RuntimeError(
                f"Instance {path_to_vm} is in state '{state}' (ip={private_ip!r}); "
                f"cannot soft-reset a non-running instance."
            )

        if not self._is_resetd_reachable(private_ip):
            raise RuntimeError(
                f"Reset daemon not reachable on {private_ip}:{AWS_RESETD_PORT} "
                f"(5s probe). Instance {path_to_vm} cannot be soft-reset."
            )

        logger.info(f"Soft-resetting {path_to_vm} at {private_ip} via reset daemon...")
        reset_result = vm_reset.soft_reset(
            private_ip, path_to_vm,
            vm_reset.ResetClientConfig(port=AWS_RESETD_PORT, timeout=30),
        )
        if reset_result.get("status") == "reused_clean":
            logger.info(
                "Soft reset complete for %s: %s (%s). Reusing instance.",
                path_to_vm,
                reset_result.get("status"),
                reset_result.get("reason_code"),
            )
            return path_to_vm

        raise RuntimeError(
            f"Soft reset failed for {path_to_vm}: "
            f"reason={reset_result.get('reason_code')} "
            f"details={reset_result.get('details')}"
        )

    def _full_relaunch(self, path_to_vm: str, snapshot_name: str):
        """Terminate the existing instance and launch a fresh one from the AMI. ~89s."""
        logger.info(f"Full relaunch: reverting AWS VM to snapshot AMI: {snapshot_name}...")
        ec2_client = boto3.client('ec2', region_name=self.region)

        try:
            # Step 1: Retrieve the original instance details (do not use MaxResults with InstanceIds)
            instance_details = ec2_client.describe_instances(InstanceIds=[path_to_vm])
            instance = instance_details['Reservations'][0]['Instances'][0]
            # Resolve security groups with fallbacks
            security_groups = [sg['GroupId'] for sg in instance.get('SecurityGroups', []) if 'GroupId' in sg]
            if not security_groups:
                env_sg = os.getenv('AWS_SECURITY_GROUP_ID')
                if env_sg:
                    security_groups = [env_sg]
                    logger.info("SecurityGroups missing on instance; using AWS_SECURITY_GROUP_ID from env")
                else:
                    raise ValueError("No security groups found on instance and AWS_SECURITY_GROUP_ID not set")

            # Resolve subnet with fallbacks
            subnet_id = instance.get('SubnetId')
            if not subnet_id:
                nis = instance.get('NetworkInterfaces', []) or []
                if nis and isinstance(nis, list):
                    for ni in nis:
                        if isinstance(ni, dict) and ni.get('SubnetId'):
                            subnet_id = ni.get('SubnetId')
                            break
                if not subnet_id:
                    env_subnet = os.getenv('AWS_SUBNET_ID')
                    if env_subnet:
                        subnet_id = env_subnet
                        logger.info("SubnetId missing on instance; using AWS_SUBNET_ID from env")
                    else:
                        raise ValueError("SubnetId not available on instance, NetworkInterfaces, or environment")

            # Resolve instance type with fallbacks
            instance_type = instance.get('InstanceType') or os.getenv('AWS_INSTANCE_TYPE') or 't3.large'
            if instance.get('InstanceType') is None:
                logger.info(f"InstanceType missing on instance; using '{instance_type}' from env/default")

            # Resolve AMI: snapshot_name is only an AMI ID when it starts with "ami-"; otherwise use current instance's image (e.g. "init_state" -> same AMI)
            image_id = snapshot_name if (snapshot_name and str(snapshot_name).startswith("ami-")) else (instance.get("ImageId") or "")
            if not image_id:
                raise ValueError("Cannot revert: snapshot_name is not an AMI ID and instance has no ImageId. Use ami-... or ensure instance has ImageId.")
            if image_id != snapshot_name:
                logger.info(f"Snapshot name {snapshot_name!r} is not an AMI ID; using current instance image {image_id} for revert.")

            # Step 2: Terminate the old instance (skip if already terminated/shutting-down)
            state = (instance.get('State') or {}).get('Name')
            if state in ['shutting-down', 'terminated']:
                logger.info(f"Old instance {path_to_vm} is already in state '{state}', skipping termination.")
            else:
                try:
                    ec2_client.terminate_instances(InstanceIds=[path_to_vm])
                    logger.info(f"Old instance {path_to_vm} has been terminated.")
                except ClientError as e:
                    error_code = getattr(getattr(e, 'response', {}), 'get', lambda *_: None)('Error', {}).get('Code') if hasattr(e, 'response') else None
                    if error_code in ['InvalidInstanceID.NotFound', 'IncorrectInstanceState']:
                        logger.info(f"Ignore termination error for {path_to_vm}: {error_code}")
                    else:
                        raise

            # Step 3: Launch a new instance from the snapshot(AMI) with performance optimization
            logger.info(f"Launching a new instance from AMI {image_id}...")

            # TTL configuration follows the same env flags as allocation (centralized)
            enable_ttl = ENABLE_TTL
            default_ttl_minutes = DEFAULT_TTL_MINUTES
            ttl_seconds = max(0, default_ttl_minutes * 60)

            run_instances_params = {
                "MaxCount": 1,
                "MinCount": 1,
                "ImageId": image_id,
                "InstanceType": instance_type,
                "EbsOptimized": True,
                "InstanceInitiatedShutdownBehavior": "terminate",
                "UserData": _VM_USER_DATA,
                "NetworkInterfaces": [
                    {
                        "SubnetId": subnet_id,
                        "AssociatePublicIpAddress": True,
                        "DeviceIndex": 0,
                        "Groups": security_groups
                    }
                ],
                "BlockDeviceMappings": [
                    {
                        "DeviceName": "/dev/sda1",
                        "Ebs": {
                            "VolumeSize": 30,
                            "VolumeType": "gp3",
                            "Throughput": 1000,
                            "Iops": 4000
                        }
                    }
                ]
            }

            new_instance = ec2_client.run_instances(**run_instances_params)
            new_instance_id = new_instance['Instances'][0]['InstanceId']
            logger.info(f"New instance {new_instance_id} launched from AMI {image_id}.")

            try:
                logger.info(f"Waiting for instance {new_instance_id} to be running...")
                ec2_client.get_waiter('instance_running').wait(InstanceIds=[new_instance_id])
            except Exception as wait_err:
                logger.error(f"Waiter failed for {new_instance_id}, terminating leaked instance: {wait_err}")
                try:
                    ec2_client.terminate_instances(InstanceIds=[new_instance_id])
                except Exception:
                    pass
                raise

            logger.info(f"Instance {new_instance_id} is ready.")
            # Schedule cloud-side termination via EventBridge Scheduler
            try:
                if enable_ttl:
                    schedule_instance_termination(self.region, new_instance_id, ttl_seconds, AWS_SCHEDULER_ROLE_ARN, logger)
            except Exception as e:
                logger.warning(f"Failed to create EventBridge Scheduler for {new_instance_id}: {e}")

            try:
                instance_details = ec2_client.describe_instances(InstanceIds=[new_instance_id])
                instance = instance_details['Reservations'][0]['Instances'][0]
                public_ip = instance.get('PublicIpAddress', '')
                if public_ip:
                    vnc_url = f"http://{public_ip}:5910/vnc.html"
                    logger.info("="*80)
                    logger.info(f"New Instance VNC Web Access URL: {vnc_url}")
                    logger.info(f"Public IP: {public_ip}")
                    logger.info(f"New Instance ID: {new_instance_id}")
                    logger.info("="*80)
            except Exception as e:
                logger.warning(f"Failed to get VNC address for new instance {new_instance_id}: {e}")

            return new_instance_id

        except Exception as e:
            logger.error(f"Failed to revert to snapshot {snapshot_name} for the instance {path_to_vm}: {e}")
            raise


    def stop_emulator(self, path_to_vm, region=None):
        logger.info(f"Stopping AWS VM {path_to_vm}...")
        ec2_client = boto3.client('ec2', region_name=self.region)

        try:
            ec2_client.terminate_instances(InstanceIds=[path_to_vm])
            logger.info(f"Instance {path_to_vm} has been terminated.")
        except ClientError as e:
            logger.error(f"Failed to stop the AWS VM {path_to_vm}: {str(e)}")
            raise
