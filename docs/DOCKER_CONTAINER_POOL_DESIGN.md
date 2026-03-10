# OSWorld Docker Container Pool — Design Document

**Goal:** Replace the current fleet of individual EC2 instances (one per environment) with a
pool of Docker containers running on a single large EC2 host, targeting ≥ 64 parallel
OSWorld environments with robust, fast reset and identical agent-facing behaviour.

---

## 1. Why Docker on One Host

### Current pain points
| Problem | Root cause |
|---------|-----------|
| AMI baking after every code change | Code is baked into AMI; no live sync |
| IAM/SSM/keypair friction | Each EC2 instance needs full AWS identity |
| ~90 s full-relaunch fallback | EC2 lifecycle (terminate + start new instance) |
| Per-instance TTL scheduler | EventBridge rule per instance |
| Hard to test locally | Real EC2 needed for even a single env |

### What Docker gives us
- **One image, 64 containers** — Docker's copy-on-write layers mean the 15–20 GB base image
  is stored once on disk. Each container adds only its writable layer (~500 MB–2 GB).
- **Code changes = image rebuild + `docker pull`** — no AMI bake cycle.
- **Reset = overlay wipe inside the container** — existing `reset_runtime.py` logic works
  unchanged inside a `--privileged` container.
- **Local testability** — spin up 1–4 containers on any developer laptop/VM with Docker.
- **No AWS account dependency for the VM layer** — only the host EC2 instance needs IAM.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  EC2 Host  m6i.16xlarge (64 vCPU / 256 GB RAM)                      │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  remote_env_server.py  :15001  (DockerProvider)                 │ │
│  └──────────────┬──────────────────────────────────────────────────┘ │
│                 │ allocates / resets container slots                  │
│  ┌──────┐ ┌──────┐ ┌──────┐        ┌──────┐                         │
│  │ C-0  │ │ C-1  │ │ C-2  │  ...   │ C-63 │   Docker containers     │
│  │:5000 │ │:5000 │ │:5000 │        │:5000 │   (osworld-server)      │
│  │:5001 │ │:5001 │ │:5001 │        │:5001 │   (osworld-resetd)      │
│  └──────┘ └──────┘ └──────┘        └──────┘                         │
│  172.20.0.2  .3     .4   ...        .65    ← Docker bridge network  │
└──────────────────────────────────────────────────────────────────────┘
         ↑ HTTP :15001
Ray workers (GPU cluster) — agent inference lives here, not on host
```

**Each container** is an isolated desktop environment:
- `Xvfb :0` — virtual 1920×1080 display
- `Openbox` (or XFCE) — lightweight window manager (GNOME apps still work)
- `osworld-server` on internal port 5000 — screenshot, `/execute`, `/run_python`
- `osworld-resetd` on internal port 5001 — overlay-FS reset daemon
- All four systemd services (home-overlay, graphical-session, resetd, server) — **unchanged**

The host's `DockerProvider` connects to each container by its **Docker-network private IP**
on the standard ports 5000/5001. No host-port remapping needed.

---

## 3. EC2 Host Sizing

### Resource budget per container
| Resource | Idle | Under load (task running) |
|----------|------|--------------------------|
| vCPU | 0.3 | 1.5 |
| RAM | 1.5 GB | 3 GB |
| Disk (writable layer) | 500 MB | 2 GB |
| Network | negligible | negligible |

Training is **I/O bound on the agent side**: the VM sits idle while the GPU generates the
next action, then executes one click/keystroke and returns a screenshot. Peak concurrency
across 64 VMs is low — not all execute simultaneously.

### Recommended instances
| Instance | vCPU | RAM | EBS / NVMe | On-Demand $/hr | Verdict |
|----------|------|-----|------------|---------------|---------|
| `m6i.16xlarge` | 64 | 256 GB | 25 Gbps EBS | ~$3.20 | **Primary recommendation** |
| `m6i.32xlarge` | 128 | 512 GB | 25 Gbps EBS | ~$6.40 | Comfortable headroom |
| `m6a.16xlarge` | 64 | 256 GB | 25 Gbps EBS | ~$2.90 | AMD, slightly cheaper |
| `c6a.32xlarge` | 128 | 256 GB | 25 Gbps EBS | ~$4.90 | CPU-heavy, RAM tight |

**Storage:** Attach a 2 TB `gp3` EBS volume (4000 IOPS, 1000 MB/s) for Docker storage
driver. NVMe instance store is faster but ephemeral.

```
/var/lib/docker  →  2 TB gp3 EBS
  Image layers (shared): ~15 GB
  64 container writable layers: 64 × 2 GB = ~128 GB
  Headroom: 1857 GB
```

---

## 4. Docker Image Design

### Base: `osworld-desktop:latest`

```dockerfile
FROM ubuntu:22.04
LABEL maintainer="osworld"
ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:0 \
    DBUS_SESSION_BUS_ADDRESS=autolaunch: \
    HOME=/home/user

# ── System packages ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Init system
    systemd systemd-sysv dbus dbus-user-session \
    # Desktop
    xvfb openbox xfwm4 xfce4-session x11-xserver-utils \
    x11-utils wmctrl xdotool xclip xdg-utils \
    # Screenshot
    gnome-screenshot scrot \
    # Apps (OSWorld task set)
    libreoffice gimp vlc thunderbird \
    ffmpeg socat sqlite3 jq git expect unzip zip curl wget \
    # Python
    python3.10 python3-pip python3-pyatspi python3-dbus \
    # Process tools
    procps psmisc util-linux \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Google Chrome ─────────────────────────────────────────────────────
RUN wget -qO /tmp/chrome.deb \
    https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y /tmp/chrome.deb && rm /tmp/chrome.deb

# ── VS Code ───────────────────────────────────────────────────────────
RUN wget -qO /tmp/vscode.deb \
    "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64" && \
    apt-get install -y /tmp/vscode.deb && rm /tmp/vscode.deb

# ── Python runtime deps ───────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    flask waitress requests pyautogui pyperclip \
    pillow python-xlib ewmh \
    boto3  # for any AWS calls from within container

# ── OSWorld app (control plane + reset runtime) ───────────────────────
# Copy repo at build time — rebuild image when code changes
COPY OSWorld /opt/osworld/app/OSWorld
RUN chmod -R 0444 /opt/osworld/app/OSWorld && \
    find /opt/osworld/app/OSWorld -type d -exec chmod 0555 {} +

# ── Desktop user ─────────────────────────────────────────────────────
RUN useradd -m -s /bin/bash user && \
    install -d -o user -g user -m 0755 /home/user

# ── Run install_resetd.sh to set up overlay dirs + baseline + systemd units ──
ENV PYTHON_BIN=/usr/bin/python3 \
    OSWORLD_PROVISION_DESKTOP=0   # skip desktop provisioning (already installed)
RUN bash /opt/osworld/app/OSWorld/desktop_env/providers/aws/scripts/install_resetd.sh

# ── Mask services not needed in container ────────────────────────────
RUN systemctl mask \
    getty@tty1.service \
    serial-getty@ttyS0.service \
    systemd-logind.service \
    apt-daily.timer apt-daily-upgrade.timer \
    motd-news.timer

# ── Container init ───────────────────────────────────────────────────
# systemd is PID 1; it starts all osworld-*.service units on boot
STOPSIGNAL SIGRTMIN+3
EXPOSE 5000 5001
ENTRYPOINT ["/sbin/init"]
```

### Build & push
```bash
cd /home/ubuntu/yincheng_arpo
docker build -t osworld-desktop:latest \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -f docker/Dockerfile .

# Tag with git SHA for versioning
docker tag osworld-desktop:latest \
  osworld-desktop:$(git rev-parse --short HEAD)
```

**Image size estimate:** ~16–20 GB uncompressed, ~7–10 GB compressed (ECR). All 64
containers share the read-only layers — total host disk for image data ≈ 16 GB.

---

## 5. Container Launch

### Run command per container
```bash
docker run -d \
  --name osworld-slot-${N} \
  --hostname osworld-${N} \
  --privileged \
  --tmpfs /run:rw,noexec,nosuid,size=128m \
  --tmpfs /run/lock:rw,noexec,nosuid,size=16m \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
  --cgroupns host \
  --shm-size 512m \
  --memory 4g \
  --cpus 2 \
  osworld-desktop:latest
```

**Why `--privileged`:**
The existing `reset_runtime.py` mounts/unmounts the OverlayFS on `/home/user`. This
requires `CAP_SYS_ADMIN` which `--privileged` provides. This is acceptable here because:
- The host is a single-tenant training machine
- Container isolation is for resource management, not multi-tenant security
- All 64 containers are running the same trusted image

**Alternative without `--privileged`:** Replace overlay FS reset with `rsync` restore (see
§6 Option B). Removes the privileged requirement entirely.

### Startup sequence (inside container, systemd-managed)
```
systemd (PID 1)
  ├─ osworld-home-overlay.service  → mounts /home/user overlay
  ├─ osworld-graphical-session.service  → Xvfb :0 + Openbox
  ├─ osworld-resetd.service  → port 5001 (root)
  └─ osworld-server.service  → port 5000 (user)
```

Cold-start time: **~15–20 seconds** (systemd + Xvfb + Python server ready).

### Pre-start all 64 containers at host boot
```bash
# /etc/rc.local or a host-level systemd service
for N in $(seq 0 63); do
  docker start osworld-slot-${N} 2>/dev/null || \
  docker run -d --name osworld-slot-${N} ... osworld-desktop:latest
done
```

---

## 6. Reset Mechanism

### Option A: Overlay FS (recommended — zero code change)

Existing `reset_runtime.py` works **unchanged** inside privileged containers:

```
Reset sequence (5–8 s):
  1. osworld-resetd receives POST /reset
  2. Stops osworld-server + X session
  3. Kills all `user` processes (loginctl + pkill -KILL)
  4. umount /home/user
  5. rm -rf home-upper/ home-work/ && mkdir them fresh  ← sub-1s
  6. mount overlay
  7. clear /tmp, /var/tmp, user crontab, /run/user/{uid}
  8. Restore dconf from baseline snapshot
  9. Restart X session → Xvfb :0 up
  10. Restart osworld-server → port 5000 up
  11. POST /verify confirms osworld-server healthy
```

### Option B: rsync restore (no `--privileged` required)

Replace the overlay mount/unmount section of `reset_runtime.py` with:

```python
def _reset_home_rsync(self) -> None:
    """Replace overlay FS wipe with rsync restore. No mount caps needed."""
    subprocess.run(
        ["rsync", "--delete", "-a",
         str(self.config.baseline_home) + "/",
         str(self.config.workspace_home) + "/"],
        check=True, timeout=30,
    )
    subprocess.run(
        ["chown", "-R", f"{self.config.desktop_user}:{self.config.desktop_user}",
         str(self.config.workspace_home)],
        check=True,
    )
```

**Trade-offs:**

| | Option A (overlay) | Option B (rsync) |
|--|--|--|
| Speed | 5–8 s | 8–20 s (depends on /home/user size) |
| `--privileged` | Required | Not required |
| Code changes | None | Modify `reset_runtime.py` |
| Reliability | Proven | Slightly less battle-tested |

**Recommendation:** Start with Option A (no code changes, proven fast). Move to Option B
only if `--privileged` becomes a blocker.

### Provider-side relaunch fallback

If `/reset` fails on a container, the `DockerProvider` falls back to:
```python
subprocess.run(["docker", "restart", container_name])
# Then wait for port 5001 (resetd) and port 5000 (server) to come up
# Typical restart time: 10–15 s
```

This is equivalent to the current "terminate + relaunch EC2" but takes 10 s instead of 90 s.

---

## 7. DockerProvider Implementation

Create `OSWorld/desktop_env/providers/docker/provider.py`:

```python
"""DockerProvider — manages a fixed pool of pre-started Docker containers.

path_to_vm convention: "slot-{N}" e.g. "slot-0" ... "slot-63"
"""
import subprocess, time, json, logging, requests
from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.docker")

CONTAINER_PREFIX = "osworld-slot-"
OSWORLD_SERVER_PORT = 5000
RESETD_PORT = 5001
READY_TIMEOUT = 120   # s to wait for container services after restart
READY_POLL   = 2

class DockerProvider(Provider):

    def __init__(self, n_slots: int = 64, image: str = "osworld-desktop:latest", **kwargs):
        self.n_slots = n_slots
        self.image = image
        self._slot_ips: dict[str, str] = {}   # slot_name → container IP

    # ── Slot allocation ──────────────────────────────────────────────

    def _container_name(self, path_to_vm: str) -> str:
        return f"{CONTAINER_PREFIX}{path_to_vm.replace('slot-', '')}"

    def _get_container_ip(self, container_name: str) -> str:
        out = subprocess.check_output(
            ["docker", "inspect", "-f",
             "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
             container_name],
            text=True,
        ).strip()
        if not out:
            raise RuntimeError(f"No IP for container {container_name}")
        return out

    def get_ip_address(self, path_to_vm: str) -> str:
        cname = self._container_name(path_to_vm)
        if path_to_vm not in self._slot_ips:
            self._slot_ips[path_to_vm] = self._get_container_ip(cname)
        return self._slot_ips[path_to_vm]

    # ── Lifecycle ────────────────────────────────────────────────────

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> str:
        ip = self.get_ip_address(path_to_vm)
        # 1. Try soft reset via resetd
        try:
            r = requests.post(
                f"http://{ip}:{RESETD_PORT}/reset",
                json={"instance_id": path_to_vm},
                timeout=90,
            )
            if r.status_code == 200 and r.json().get("status") == "ok":
                # Verify
                rv = requests.post(
                    f"http://{ip}:{RESETD_PORT}/verify",
                    json={"instance_id": path_to_vm},
                    timeout=30,
                )
                if rv.status_code == 200 and rv.json().get("status") == "ok":
                    logger.info("Soft reset OK for %s", path_to_vm)
                    return path_to_vm
        except Exception as exc:
            logger.warning("Soft reset failed for %s: %s — falling back to docker restart", path_to_vm, exc)

        # 2. Fallback: docker restart
        cname = self._container_name(path_to_vm)
        subprocess.run(["docker", "restart", cname], check=True, timeout=30)
        self._slot_ips.pop(path_to_vm, None)   # IP may change after restart
        self._wait_for_ready(path_to_vm)
        return path_to_vm

    def _wait_for_ready(self, path_to_vm: str, timeout: float = READY_TIMEOUT) -> None:
        ip = self.get_ip_address(path_to_vm)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"http://{ip}:{OSWORLD_SERVER_PORT}/health", timeout=3)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(READY_POLL)
        raise RuntimeError(f"Container {path_to_vm} not ready after {timeout}s")

    def start_emulator(self, path_to_vm: str, headless: bool = True) -> None:
        """Containers are pre-started; this is a no-op or ensures the container is running."""
        cname = self._container_name(path_to_vm)
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", cname],
            capture_output=True, text=True,
        )
        if result.stdout.strip() != "true":
            subprocess.run(["docker", "start", cname], check=True)
            self._wait_for_ready(path_to_vm)

    def get_vm_list(self) -> list[str]:
        return [f"slot-{i}" for i in range(self.n_slots)]
```

### Wiring into `remote_env_server.py`

Add `docker` as a recognised provider:

```python
# In _get_env() or DesktopEnv constructor
if os.environ.get("PROVIDER") == "docker":
    from desktop_env.providers.docker.provider import DockerProvider
    provider = DockerProvider(
        n_slots=int(os.environ.get("DOCKER_N_SLOTS", "64")),
        image=os.environ.get("DOCKER_IMAGE", "osworld-desktop:latest"),
    )
```

Environment variable to select provider: `PROVIDER=docker`.

---

## 8. Agent Interaction — Unchanged

Every agent-facing interface stays **bit-for-bit identical**:

| Interface | Status |
|-----------|--------|
| `POST /execute` — click, scroll, type | Unchanged (pyautogui inside container) |
| `GET /screenshot` — PNG 1920×1080 | Unchanged (scrot/Xvfb) |
| `POST /run_python` — arbitrary Python | Unchanged |
| Coordinate system (0–1920, 0–1080) | Unchanged |
| Action format `click(start_box=...)` | Unchanged |
| Reward shaping, loop detection | Unchanged |
| Training HTTP API (`/env/reset`, `/env/step`) | Unchanged |

The container's Xvfb runs at 1920×1080. Openbox (or XFCE) provides window management.
GNOME-specific apps (LibreOffice, Chrome, GIMP, Thunderbird, VLC) work fine under a
non-GNOME WM — they use GTK/Qt directly and don't require a GNOME session.

**Screenshot quality check:** Xvfb + Openbox renders all apps identically to Xvfb + GNOME
session for pyautogui-driven interaction. The only difference is the desktop wallpaper and
panel; OSWorld tasks don't depend on those.

---

## 9. Testability Without GPU

### Local single-container test
```bash
# Start one container
docker run -d --name osworld-test --privileged \
  --tmpfs /run --tmpfs /run/lock \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw --cgroupns host \
  osworld-desktop:latest

# Wait ~20s for services, then:
docker exec osworld-test curl -s http://localhost:5001/health
docker exec osworld-test curl -s http://localhost:5000/health

# Get a screenshot
CONTAINER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' osworld-test)
curl -s http://${CONTAINER_IP}:5000/screenshot > /tmp/screen.png
open /tmp/screen.png   # or display /tmp/screen.png

# Execute a click
curl -s -X POST http://${CONTAINER_IP}:5000/execute \
  -H 'Content-Type: application/json' \
  -d '{"command": ["python3", "-c", "import pyautogui; pyautogui.click(960,540)"], "shell": false}'

# Trigger a reset
curl -s -X POST http://${CONTAINER_IP}:5001/reset \
  -H 'Content-Type: application/json' \
  -d '{"instance_id": "slot-0"}'
```

### Remote env server against Docker pool
```bash
PROVIDER=docker \
DOCKER_N_SLOTS=4 \
python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 15001
```

### End-to-end test (no GPU — mock agent)
```bash
# scripts/test_e2e_vm_pipeline.py already has mock-action tests
# Point REMOTE_ENV_URL at the docker-backed server
REMOTE_ENV_URL=http://localhost:15001 \
python scripts/test_e2e_vm_pipeline.py --provider docker --n-slots 4
```

### CI/CD on any Linux host with Docker
```yaml
# .github/workflows/docker-test.yml (example)
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t osworld-desktop:ci -f docker/Dockerfile .
      - run: |
          docker run -d --name slot-0 --privileged ... osworld-desktop:ci
          sleep 20
          python scripts/test_e2e_vm_pipeline.py --provider docker --n-slots 1
```

**No GPU required at any point in the above.** The GPU is only needed by the training
cluster for agent inference, which connects to the already-running remote env server.

---

## 10. Code Changes Summary

| File | Change | Risk |
|------|--------|------|
| `docker/Dockerfile` | **New** | Low |
| `docker/compose.yml` | **New** (optional) | Low |
| `OSWorld/desktop_env/providers/docker/provider.py` | **New** | Low |
| `scripts/remote_env_server.py` | Add `PROVIDER=docker` branch in `_get_env()` | Very low |
| `OSWorld/desktop_env/providers/aws/reset_runtime.py` | **None** (Option A) | Zero |
| `OSWorld/desktop_env/controllers/python.py` | None | Zero |
| `OSWorld/desktop_env/desktop_env.py` | None | Zero |
| `verl/trainer/gui_agent.py` | None | Zero |
| `scripts/test_e2e_vm_pipeline.py` | Add `--provider docker` flag | Low |

**Total new code: ~300 lines.  Modified code: ~20 lines.**

---

## 11. Migration Path

### Phase 1: Build & validate the image (1–2 days)
1. Write `docker/Dockerfile`
2. `docker build` and verify all apps launch: Chrome, LibreOffice, GIMP, VLC, Thunderbird
3. Run `test_e2e_vm_pipeline.py` against a single container locally
4. Verify reset works: trigger task → change files → reset → files gone

### Phase 2: Implement DockerProvider (0.5 days)
1. Write `provider.py` as above
2. Wire into `remote_env_server.py` (PROVIDER=docker)
3. Run existing E2E tests with 4 containers

### Phase 3: Scale to 64 on target host (0.5 days)
1. Launch `m6i.16xlarge` with 2 TB EBS
2. Install Docker, pull image
3. Start 64 containers
4. Point training run at the server
5. Monitor RAM / CPU / reset timing

### Phase 4: Remove AWS VM dependency (optional)
- Retire `AWSProvider` or keep it for fallback
- Remove EventBridge scheduler logic
- Remove SSM, keypair, AMI bake pipeline

---

## 12. Open Questions / Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| GNOME-specific task behaviour breaks under Openbox | Medium | Run all 124 tasks in Openbox; audit failures; add XFCE if needed |
| OverlayFS-on-OverlayFS blocked by kernel | Low | Most kernels ≥ 5.11 support nested overlay; use `--privileged` |
| Chrome sandbox needs extra capabilities | Medium | Launch Chrome with `--no-sandbox` flag (already done in most OSWorld configs) |
| Container OOM under heavy app load | Low | Set `--memory 4g --memory-swap 6g`; monitor per-container RSS |
| Docker daemon becomes single point of failure | Low | Restart policy `always`; add watchdog to restart dead containers |
| `jq` missing (tasks 3299584d, 9656a811) | Confirmed | Pre-install `jq` in Dockerfile (already planned above) |
| dconf socket isolation between containers | Low | Each container has isolated `/run/user/1000/` — no cross-container leakage |

---

## 13. Key Numbers

| Metric | Current (EC2 per VM) | Docker pool |
|--------|---------------------|-------------|
| Parallel envs | 1 per EC2 (scalable but $$$) | 64 on 1 host |
| Reset time | 5–8 s (overlay) | 5–8 s (overlay, unchanged) |
| Full relaunch fallback | ~90 s (EC2 terminate+start) | ~15 s (docker restart) |
| Code update | New AMI bake (~15 min) | `docker build && docker pull` (~5 min) |
| Local testability | No (real EC2 required) | Yes (any Docker host) |
| Host cost (64 envs) | 64 × t3.large × ~$0.08/hr = $5.12/hr | 1 × m6i.16xlarge × $3.20/hr |
| Cost per env-hour | $0.08 | $0.05 |

---

## 14. Audit Findings & Required Fixes

Three independent audits were run against this design. All blockers must be resolved before
deployment. Findings are grouped by source.

---

### 14.1 Reset / Overlay FS Audit

#### BLOCKER R1 — install_resetd.sh fails during image build (no X socket)

`install_resetd.sh` calls `fail_no_desktop()` if `/tmp/.X11-unix/X0` doesn't exist, which is
always true during `docker build`. The Dockerfile's `RUN install_resetd.sh` will exit 1 and
the build fails.

**Fix:** Add an env var gate to the install script and set it in the Dockerfile:
```dockerfile
ENV OSWORLD_SKIP_X_SOCKET_CHECK=1
RUN bash /opt/osworld/app/OSWorld/desktop_env/providers/aws/scripts/install_resetd.sh
```
And in `install_resetd.sh`, wrap the `fail_no_desktop` call:
```bash
if [ "${OSWORLD_SKIP_X_SOCKET_CHECK:-0}" != "1" ] && ! has_x_socket "${DISPLAY_NUM}"; then
    fail_no_desktop
fi
```

#### BLOCKER R2 — Overlay-on-overlay requires Linux ≥ 5.15

Docker containers run on the host kernel. Overlay-on-overlay (overlayfs mounted inside an
overlayfs container root) was unreliable before kernel 5.15. The EC2 host must run
**Amazon Linux 2023 or Ubuntu 22.04 HVM** (both ship kernel 6.x).

**Fix:** Add a startup check in the container init or in `_ensure_runtime_layout()`:
```bash
KVER=$(uname -r | awk -F. '{print $1*100+$2}')
[ "$KVER" -lt 515 ] && echo "ERROR: overlay-on-overlay needs kernel ≥ 5.15" && exit 1
```

#### BLOCKER R3 — systemd-logind must NOT be masked

The design masked `systemd-logind` to save memory. However `loginctl kill-user` (called by
`_kill_task_user_processes()`) depends on logind. Without it, user process cleanup fails
silently and the overlay unmount raises `RuntimeError`, collapsing to a slow docker restart.

**Fix:** Remove `systemd-logind.service` from the masked list in the Dockerfile. Memory
overhead is negligible (~8 MB). Alternatively modify `_kill_task_user_processes()` to use
only `pkill -KILL -u user` and remove the `loginctl` call — this works regardless of logind.

#### HIGH R4 — dconf restore runs after D-Bus is stopped

`_restore_dconf()` is called after the graphical session (and its D-Bus daemon) are stopped.
`dconf load` fails silently because no session bus exists.

**Fix:** Spawn a temporary D-Bus daemon for the restore, or restore the dconf binary files
directly (`~/.config/dconf/user`) from the overlay baseline instead of using the CLI tool.
The overlay wipe already copies the baseline's dconf file — the `_restore_dconf()` call is
redundant when using Option A (overlay reset). Verify and document this.

---

### 14.2 Task Compatibility Audit

#### BLOCKER T1 — gnome-terminal-server hard-coded in main.py (affects ~25% of tasks)

`OSWorld/desktop_env/server/main.py` contains XPath queries and accessibility tree walks
that hard-code `"gnome-terminal-server"` as the process name. Under Openbox with
`xfce4-terminal` or `xterm`, these queries return empty — OS and multi-app tasks that
require reading terminal output will fail evaluation.

**Affected tasks:** ~7 OS tasks + ~7 multi-app tasks = ~14 tasks (11% of training set).

**Fix:** Replace the hard-coded name with a generic terminal detector:
```python
# Instead of: app.name == "gnome-terminal-server"
TERMINAL_APP_NAMES = {"gnome-terminal-server", "xfce4-terminal", "xterm", "konsole", "terminator"}
if app.getRoleName() == "application" and app.name in TERMINAL_APP_NAMES:
    ...
```
**OR** keep `gnome-terminal` installed (it works under Openbox — only the GNOME session is
not required, the binary still runs).

**Recommended:** Install `gnome-terminal` in the Dockerfile (it runs under any WM).
This is zero-code-change and maximally compatible.

#### Required Dockerfile additions (from task audit)

```dockerfile
# These are needed for task coverage and were missing from the initial design:
RUN apt-get install -y \
    at-spi2-core at-spi2-atk \   # Accessibility tree for terminal eval
    gnome-terminal \              # Terminal: keep gnome-terminal-server compat
    pulseaudio pulseaudio-utils \ # Audio volume tasks (pactl)
    jq \                          # Tasks 3299584d, 9656a811
    inotify-tools                 # VS Code file watcher stability
```

#### WARNING T2 — PulseAudio must be started per-container

Audio volume tasks use `pactl`. PulseAudio needs either system-mode daemon or per-session
daemon. In containers (no logind), start it in the graphical session script:
```bash
# In launch_osworld_graphical_session.sh:
pulseaudio --start --log-target=syslog 2>/dev/null || true
```

#### WARNING T3 — dconf baseline snapshot must be re-baked under Openbox

The baseline dconf snapshot was created under GNOME. Under Openbox, some GNOME-specific
keys may differ or be absent. Re-bake the snapshot:
1. Build the Docker image
2. Start a container, wait for graphical session
3. `docker exec container runuser -u user -- dconf dump / > /tmp/openbox_dconf.ini`
4. Compare with existing snapshot; use the new one as the baseline

---

### 14.3 DockerProvider & Training Pipeline Audit

#### BLOCKER P1 — remote_env_server.py has ONE global DesktopEnv (fatal for 64 slots)

**This is the most critical finding.** The current `remote_env_server.py` stores a single
global `env` object. All 64 training workers hitting `/env/reset` and `/env/step` concurrently
will corrupt each other's state.

**Fix required:** Refactor `remote_env_server.py` to a slot-indexed pool:

```python
# New global state (replace single env):
_envs: dict[int, DesktopEnv] = {}          # slot_id → env
_env_lock = threading.Lock()

# New endpoint: requires slot_id in request body
@app.post("/env/reset")
async def env_reset(request: Request):
    body = await request.json()
    slot_id = body.get("slot_id", 0)        # training worker sends its slot
    task_config = body["task_config"]
    with _env_lock:
        env = _get_or_create_env(slot_id)
    result = env.reset(task_config)
    ...

@app.post("/env/step")
async def env_step(request: Request):
    body = await request.json()
    slot_id = body.get("slot_id", 0)
    prediction = body["prediction"]
    env = _envs[slot_id]
    ...
```

The training worker (`RemoteEnvWorker` in `gui_agent.py`) must send its `worker_idx` as
`slot_id` in every request. This is a **2-file change** (`remote_env_server.py` +
`gui_agent.py`), but it is load-bearing.

#### BLOCKER P2 — Thread-unsafe IP cache in DockerProvider

`_slot_ips` is a plain dict accessed from multiple threads without locking. Under concurrent
resets (which invalidate and re-fetch IPs), this can return stale IPs or corrupt the dict.

**Fix:**
```python
import threading
self._slot_ips: dict[str, str] = {}
self._slot_ip_lock = threading.Lock()

def get_ip_address(self, path_to_vm: str) -> str:
    with self._slot_ip_lock:
        if path_to_vm not in self._slot_ips:
            self._slot_ips[path_to_vm] = self._get_container_ip(self._container_name(path_to_vm))
        return self._slot_ips[path_to_vm]
```

#### BLOCKER P3 — Reset timeout not propagated

The `revert_to_snapshot()` calls `requests.post(.../reset, timeout=90)` but the resetd's
own timeout is 120 s (`AWS_RESETD_REQUEST_TIMEOUT`). These must be consistent. Also, the
verify call has no timeout defined at all.

**Fix:** Mirror the timeout from `config.py`:
```python
from desktop_env.providers.aws.config import AWS_RESETD_REQUEST_TIMEOUT
RESET_TIMEOUT = AWS_RESETD_REQUEST_TIMEOUT  # 120 s
VERIFY_TIMEOUT = 30
```

#### WARNING P4 — Per-slot locking needed to prevent concurrent reset + step on same slot

If a reset and a step request arrive simultaneously for the same slot, the step runs on a
partially-reset environment. Add per-slot RLock:
```python
self._slot_locks: dict[str, threading.RLock] = {
    f"slot-{i}": threading.RLock() for i in range(n_slots)
}
```

Acquire the lock in both `revert_to_snapshot()` and any direct env access.

---

### 14.4 Updated Code Changes Summary

| File | Change | Severity |
|------|--------|----------|
| `docker/Dockerfile` | **New** — add missing packages, `OSWORLD_SKIP_X_SOCKET_CHECK=1` | Required |
| `OSWorld/desktop_env/providers/aws/scripts/install_resetd.sh` | Add `OSWORLD_SKIP_X_SOCKET_CHECK` gate | **BLOCKER R1** |
| `scripts/remote_env_server.py` | Slot-indexed env pool (single global → dict) | **BLOCKER P1** |
| `verl/trainer/gui_agent.py` | Send `slot_id` in each HTTP request | **BLOCKER P1** |
| `OSWorld/desktop_env/providers/docker/provider.py` | **New** — add RLock, timeout constants, retry | **BLOCKER P2/P3** |
| `OSWorld/desktop_env/providers/aws/scripts/launch_osworld_graphical_session.sh` | Start PulseAudio | **WARNING T2** |
| `OSWorld/desktop_env/providers/aws/reset_runtime.py` | Remove `loginctl` OR don't mask logind | **BLOCKER R3** |

---

### 14.5 Pre-Deployment Checklist

```
BLOCKERS (must fix before any scale test):
[ ] R1: install_resetd.sh OSWORLD_SKIP_X_SOCKET_CHECK flag + Dockerfile env var
[ ] R2: Verify EC2 host kernel ≥ 5.15 (check: uname -r)
[ ] R3: Unmask systemd-logind OR remove loginctl dependency in reset_runtime.py
[ ] T1: Install gnome-terminal in Dockerfile (zero code change)
[ ] P1: Refactor remote_env_server.py to slot-indexed env pool
[ ] P2: Add RLock to DockerProvider._slot_ips
[ ] P3: Align reset/verify timeouts with config.py constants

HIGH PRIORITY (fix before production training run):
[ ] R4: Validate dconf restore works in Docker (or skip — overlay already resets it)
[ ] T2: Start PulseAudio in graphical session script
[ ] T3: Re-bake baseline dconf snapshot under Openbox
[ ] P4: Add per-slot RLock in DockerProvider

VALIDATION (run before full 64-slot deploy):
[ ] Single container: all 4 services start, screenshot works, /execute works
[ ] Single container: trigger reset → verify /home/user is wiped → screenshot clean
[ ] 4-container test: concurrent resets don't corrupt each other
[ ] Run 1 Chrome task, 1 LibreOffice task, 1 GIMP task, 1 OS-shell task end-to-end
[ ] Measure reset time: target ≤ 10 s (overlay) or ≤ 20 s (rsync fallback)
[ ] 64-container boot: confirm all slots healthy within 3 minutes
```

---

*Document version: 2026-03-09 (rev 2 — post-audit). Author: Claude Code.*
