import boto3
from botocore.exceptions import ClientError

import logging
import os
import time
import requests
from datetime import datetime, timedelta, timezone
from desktop_env.providers.base import Provider

# TTL configuration
from desktop_env.providers.aws.config import ENABLE_TTL, DEFAULT_TTL_MINUTES, AWS_SCHEDULER_ROLE_ARN
from desktop_env.providers.aws.scheduler_utils import schedule_instance_termination

logger = logging.getLogger("desktopenv.providers.aws.AWSProvider")
logger.setLevel(logging.INFO)

WAIT_DELAY = 15
MAX_ATTEMPTS = 10
VM_READY_TIMEOUT = 360  # seconds to wait for OSWorld server inside the VM to be reachable
VM_READY_POLL = 10      # poll interval in seconds


class AWSProvider(Provider):

    def _wait_for_vm_server(self, ip: str, port: int = 5000):
        """Wait until the OSWorld HTTP server inside the VM is reachable on port 5000."""
        url = f"http://{ip}:{port}/screenshot"
        deadline = time.time() + VM_READY_TIMEOUT
        logger.info(f"Waiting for VM server at {url} (timeout={VM_READY_TIMEOUT}s)...")
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    logger.info(f"VM server at {url} is ready.")
                    return
            except Exception:
                pass
            logger.info(f"VM not ready yet, retrying in {VM_READY_POLL}s...")
            time.sleep(VM_READY_POLL)
        raise TimeoutError(f"VM server at {url} did not become ready within {VM_READY_TIMEOUT}s")

    def start_emulator(self, path_to_vm: str, headless: bool, *args, **kwargs):
        logger.info("Starting AWS VM...")
        ec2_client = boto3.client('ec2', region_name=self.region)

        try:
            # Check the current state of the instance (do not use MaxResults with InstanceIds - API forbids it)
            response = ec2_client.describe_instances(InstanceIds=[path_to_vm])
            state = response['Reservations'][0]['Instances'][0]['State']['Name']
            logger.info(f"Instance {path_to_vm} current state: {state}")

            if state == 'running':
                # If the instance is already running, skip starting it
                logger.info(f"Instance {path_to_vm} is already running. Skipping start.")
                return

            if state == 'stopped':
                # Start the instance if it's currently stopped
                ec2_client.start_instances(InstanceIds=[path_to_vm])
                logger.info(f"Instance {path_to_vm} is starting...")

                # Wait until the instance reaches 'running' state
                waiter = ec2_client.get_waiter('instance_running')
                waiter.wait(
                    InstanceIds=[path_to_vm],
                    WaiterConfig={'Delay': WAIT_DELAY, 'MaxAttempts': MAX_ATTEMPTS}
                )
                logger.info(f"Instance {path_to_vm} is now running.")
            else:
                # For all other states (terminated, pending, etc.), log a warning
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
                        logger.info(f"🖥️  VNC Web Access URL: {vnc_url}")
                        logger.info(f"📡 Public IP: {public_ip_address}")
                        logger.info(f"🏠 Private IP: {private_ip_address}")
                        logger.info("="*80)
                        print(f"\n🌐 VNC Web Access URL: {vnc_url}")
                        print(f"📍 Please open the above address in the browser for remote desktop access\n")
                    else:
                        logger.warning("No public IP address available for VNC access")

                    if private_ip_address:
                        self._wait_for_vm_server(private_ip_address)

                    return private_ip_address
                    # return public_ip_address
            return ''  # Return an empty string if no IP address is found
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

    # Apps to kill during soft reset — desktop applications only; never kills python/server processes
    _SOFT_RESET_KILL_PATTERN = (
        "chromium|firefox|libreoffice|soffice|vlc|gedit|mousepad|"
        "thunar|nautilus|nemo|evince|eog|gimp|inkscape|code|kate|xed"
    )

    def _soft_reset_vm(self, ip: str, port: int = 5000):
        """Kill all desktop apps inside the running VM via its HTTP API.

        Does NOT kill python/server processes, so the VM's port-5000 server stays up.
        Raises RuntimeError if the HTTP call fails (caller should fall back to full relaunch).
        """
        cmd = f"pkill -9 -f '{self._SOFT_RESET_KILL_PATTERN}' || true; sleep 1"
        url = f"http://{ip}:{port}/setup/execute"
        resp = requests.post(url, json={"command": cmd, "shell": True}, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"Soft reset returned HTTP {resp.status_code}: {resp.text[:200]}")
        logger.info(f"Soft reset sent to {ip}:{port} — desktop apps killed.")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        """Revert to clean state. Tries a fast soft reset (~5s) first; falls back to
        full terminate+relaunch (~89s) if the instance is unreachable or unhealthy."""
        ec2_client = boto3.client('ec2', region_name=self.region)

        # --- Fast path: soft reset the running instance ---
        try:
            response = ec2_client.describe_instances(InstanceIds=[path_to_vm])
            instance = response['Reservations'][0]['Instances'][0]
            state = instance['State']['Name']
            private_ip = instance.get('PrivateIpAddress', '')

            if state == 'running' and private_ip:
                logger.info(f"Soft-resetting {path_to_vm} at {private_ip} (skipping terminate+relaunch)...")
                self._soft_reset_vm(private_ip)
                logger.info(f"Soft reset complete. Reusing instance {path_to_vm}.")
                return path_to_vm
            else:
                logger.info(f"Instance {path_to_vm} is in state '{state}'; falling back to full relaunch.")
        except Exception as e:
            logger.warning(f"Soft reset failed ({e}); falling back to full terminate+relaunch.")

        # --- Slow path: terminate old instance and launch fresh from AMI ---
        return self._full_relaunch(path_to_vm, snapshot_name)

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
                            # "VolumeInitializationRate": 300
                            "VolumeSize": 30,  # Size in GB
                            "VolumeType": "gp3",  # General Purpose SSD
                            "Throughput": 1000,
                            "Iops": 4000  # Adjust IOPS as needed
                        }
                    }
                ]
            }
            
            new_instance = ec2_client.run_instances(**run_instances_params)
            new_instance_id = new_instance['Instances'][0]['InstanceId']
            logger.info(f"New instance {new_instance_id} launched from AMI {image_id}.")
            logger.info(f"Waiting for instance {new_instance_id} to be running...")
            ec2_client.get_waiter('instance_running').wait(InstanceIds=[new_instance_id])

            logger.info(f"Instance {new_instance_id} is ready.")
            # Schedule cloud-side termination via EventBridge Scheduler (auto-resolve role ARN)
            try:
                if enable_ttl:
                    schedule_instance_termination(self.region, new_instance_id, ttl_seconds, AWS_SCHEDULER_ROLE_ARN, logger)
            except Exception as e:
                logger.warning(f"Failed to create EventBridge Scheduler for {new_instance_id}: {e}")

            # Schedule cloud-side termination via EventBridge Scheduler (same as allocation path)
            try:
                if enable_ttl and os.getenv('AWS_SCHEDULER_ROLE_ARN'):
                    scheduler_client = boto3.client('scheduler', region_name=self.region)
                    schedule_name = f"osworld-ttl-{new_instance_id}-{int(time.time())}"
                    eta_scheduler = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
                    schedule_expression = f"at({eta_scheduler.strftime('%Y-%m-%dT%H:%M:%S')})"
                    target_arn = "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances"
                    input_payload = '{"InstanceIds":["' + new_instance_id + '"]}'
                    scheduler_client.create_schedule(
                        Name=schedule_name,
                        ScheduleExpression=schedule_expression,
                        FlexibleTimeWindow={"Mode": "OFF"},
                        Target={
                            "Arn": target_arn,
                            "RoleArn": os.getenv('AWS_SCHEDULER_ROLE_ARN'),
                            "Input": input_payload
                        },
                        State='ENABLED',
                        Description=f"OSWorld TTL terminate for {new_instance_id}"
                    )
                    logger.info(f"Scheduled EC2 termination via EventBridge Scheduler for snapshot revert: name={schedule_name}, when={eta_scheduler.isoformat()} (UTC)")
                else:
                    logger.info("TTL enabled but AWS_SCHEDULER_ROLE_ARN not set; skipping scheduler for snapshot revert.")
            except Exception as e:
                logger.warning(f"Failed to create EventBridge Scheduler for {new_instance_id}: {e}")

            try:
                instance_details = ec2_client.describe_instances(InstanceIds=[new_instance_id])
                instance = instance_details['Reservations'][0]['Instances'][0]
                public_ip = instance.get('PublicIpAddress', '')
                if public_ip:
                    vnc_url = f"http://{public_ip}:5910/vnc.html"
                    logger.info("="*80)
                    logger.info(f"🖥️  New Instance VNC Web Access URL: {vnc_url}")
                    logger.info(f"📡 Public IP: {public_ip}")
                    logger.info(f"🆔 New Instance ID: {new_instance_id}")
                    logger.info("="*80)
                    print(f"\n🌐 New Instance VNC Web Access URL: {vnc_url}")
                    print(f"📍 Please open the above address in the browser for remote desktop access\n")
            except Exception as e:
                logger.warning(f"Failed to get VNC address for new instance {new_instance_id}: {e}")

            return new_instance_id

        except ClientError as e:
            logger.error(f"Failed to revert to snapshot {snapshot_name} for the instance {path_to_vm}: {str(e)}")
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
