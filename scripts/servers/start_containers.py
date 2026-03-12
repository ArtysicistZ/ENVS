#!/usr/bin/env python3
"""
Batch-start OSWorld containers and wait for health checks.

Usage:
    python scripts/servers/start_containers.py           # start 64 containers (slots 0-63)
    python scripts/servers/start_containers.py 16        # start 16 containers (slots 0-15)
    python scripts/servers/start_containers.py 0 7       # start 8 containers (slots 0-7)
    python scripts/servers/start_containers.py --restart # stop+remove existing, then start fresh

Environment overrides:
    OSWORLD_DOCKER_IMAGE       (default: osworld:latest)
    OSWORLD_DOCKER_NETWORK     (default: osworld-net)
    OSWORLD_CONTAINER_TMP_SIZE (default: 512m)
"""
import argparse
import os
import sys
import time
import threading
import requests
import docker

IMAGE    = os.environ.get("OSWORLD_DOCKER_IMAGE",       "osworld:latest")
NETWORK  = os.environ.get("OSWORLD_DOCKER_NETWORK",     "osworld-net")
TMP_SIZE = os.environ.get("OSWORLD_CONTAINER_TMP_SIZE", "512m")

SERVER_PORT  = 5000
RESETD_PORT  = 5001
HEALTH_TIMEOUT = 120  # seconds to wait for /health


def ensure_network(client: docker.DockerClient):
    try:
        client.networks.get(NETWORK)
    except docker.errors.NotFound:
        client.networks.create(NETWORK, driver="bridge")
        print(f"Created network: {NETWORK}")


def stop_and_remove(client: docker.DockerClient, name: str):
    try:
        c = client.containers.get(name)
        c.stop(timeout=5)
        c.remove(force=True)
        print(f"  Removed {name}")
    except docker.errors.NotFound:
        pass


def get_container_ip(client: docker.DockerClient, name: str) -> str | None:
    for _ in range(10):
        try:
            c = client.containers.get(name)
            c.reload()
            nets = c.attrs.get("NetworkSettings", {}).get("Networks", {})
            ip = nets.get(NETWORK, {}).get("IPAddress")
            if not ip:
                for info in nets.values():
                    ip = info.get("IPAddress")
                    if ip:
                        break
            if ip:
                return ip
        except Exception:
            pass
        time.sleep(1)
    return None


def wait_for_health(ip: str, port: int, name: str, timeout: int = HEALTH_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{ip}:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_container(client: docker.DockerClient, slot_id: int, restart: bool) -> bool:
    """Start container, return True if newly started (needs health check), False if skipped."""
    name = f"osworld-slot-{slot_id}"

    if restart:
        stop_and_remove(client, name)
    else:
        try:
            c = client.containers.get(name)
            c.reload()
            if c.status == "running":
                print(f"  {name}: already running, skipping")
                return False
            c.remove(force=True)
        except docker.errors.NotFound:
            pass

    client.containers.run(
        image=IMAGE,
        name=name,
        detach=True,
        privileged=True,
        cgroupns="host",
        network=NETWORK,
        tmpfs={
            "/run":       "rw,nosuid,nodev,size=64m",
            "/run/lock":  "rw,nosuid,nodev,size=16m",
            "/tmp":       f"rw,nosuid,nodev,size={TMP_SIZE}",
            "/var/log":   "rw,nosuid,nodev,size=64m",
            "/var/tmp":   "rw,nosuid,nodev,size=64m",
            "/var/cache": "rw,nosuid,nodev,size=128m",
        },
        volumes={"/sys/fs/cgroup": {"bind": "/sys/fs/cgroup", "mode": "rw"}},
        shm_size="256m",
        environment={"OSWORLD_SLOT_ID": str(slot_id)},
        restart_policy={"Name": "unless-stopped"},
    )
    print(f"  {name}: started")
    return True


def check_health_worker(client, slot_id, results, lock):
    name = f"osworld-slot-{slot_id}"
    ip = get_container_ip(client, name)
    if not ip:
        with lock:
            results[slot_id] = "NO_IP"
        print(f"  [{name}] FAIL: could not get IP")
        return

    # Wait for resetd (port 5001) first — it comes up before the server
    resetd_ok = wait_for_health(ip, RESETD_PORT, name)
    server_ok = wait_for_health(ip, SERVER_PORT, name)

    status = "OK" if (resetd_ok and server_ok) else "TIMEOUT"
    with lock:
        results[slot_id] = status
    tag = "healthy" if status == "OK" else "TIMEOUT"
    print(f"  [{name}] {tag}  (ip={ip}, resetd={resetd_ok}, server={server_ok})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", type=int, default=64,
                        help="Total count (if no end given) or first slot index")
    parser.add_argument("end",   nargs="?", type=int, default=None,
                        help="Last slot index (inclusive)")
    parser.add_argument("--restart", action="store_true",
                        help="Stop and remove existing containers before starting")
    parser.add_argument("--no-health", action="store_true",
                        help="Skip health checks after starting")
    args = parser.parse_args()

    if args.end is None:
        slot_start, slot_end = 0, args.start - 1
    else:
        slot_start, slot_end = args.start, args.end

    slots = list(range(slot_start, slot_end + 1))
    print(f"Starting {len(slots)} containers (slots {slot_start}–{slot_end}), "
          f"image={IMAGE}, network={NETWORK}, restart={args.restart}")

    client = docker.from_env()
    ensure_network(client)

    newly_started = []
    for slot_id in slots:
        if start_container(client, slot_id, args.restart):
            newly_started.append(slot_id)

    if not newly_started or args.no_health:
        print(f"\nDone. {len(slots) - len(newly_started)} already running, "
              f"{len(newly_started)} newly started.")
        return

    print(f"\nWaiting for health checks on {len(newly_started)} new containers "
          f"(timeout={HEALTH_TIMEOUT}s each)...")

    results = {}
    lock = threading.Lock()
    threads = [
        threading.Thread(target=check_health_worker, args=(client, sid, results, lock), daemon=True)
        for sid in newly_started
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok      = sum(1 for v in results.values() if v == "OK")
    failed  = [sid for sid, v in results.items() if v != "OK"]
    print(f"\nHealth check: {ok}/{len(newly_started)} healthy.")
    if failed:
        print(f"  FAILED slots: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
