# Remote env with nested KVM (Fusion → Linux VM → Docker)

Use this path when you want **KVM acceleration** for the OSWorld Docker container: run a **Linux VM inside VMware Fusion** with nested virtualization enabled, then run **Docker + the env server inside that VM**. The container will see `/dev/kvm` inside the guest.

**Architecture:**
- **Mac** → VMware Fusion → **Ubuntu VM** (nested virt enabled) → Docker → OSWorld container (with KVM)
- **Env server** runs inside the Ubuntu VM.
- **Tunnel** runs from inside the Ubuntu VM to the cluster.
- **Resource (training):** cluster.

---

## 1. VMware Fusion: create Ubuntu VM and enable nested virtualization

1. **Create a new VM** in Fusion (e.g. Ubuntu 22.04 LTS). Allocate at least 4 CPU cores, 8 GB RAM, 50+ GB disk (the Docker provider will download an OSWorld qcow2 image inside the VM).
2. **Enable nested virtualization** so the guest can use KVM:
   - Select the VM → **Settings** (or Virtual Machine → Settings).
   - **Processors & Memory** → check **“Hypervisor applications”** (allow hypervisor apps in the guest).
   - Click **OK**.
3. Install Ubuntu in the VM, install VMware Tools / open-vm-tools, and ensure the VM has network (NAT or Bridged) so it can reach the internet and your cluster (for SSH).

---

## 2. Inside the Ubuntu VM: install Docker and repo

SSH into the VM (or use the Fusion console) and run:

```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io git python3.11 python3.11-venv
sudo usermod -aG docker $USER
# Log out and back in (or newgrp docker) so docker runs without sudo

# Clone repo (same as on cluster; has OSWorld submodule + verl)
git clone --recursive https://github.com/hanshengzhu0001/arpo_remote_env.git
cd arpo_remote_env
git submodule update --init --recursive   # if OSWorld is empty

# Python env (server needs: fastapi, desktop_env, verl.trainer)
python3.11 -m venv arpo_env
source arpo_env/bin/activate
pip install -U pip
pip install fastapi uvicorn "uvicorn[standard]" requests pillow pydantic
# OSWorld + desktop_env (from submodule)
pip install -e OSWorld
# Repo root may have requirements or installable packages (verl, etc.)
pip install -r requirements.txt 2>/dev/null || true
pip install -e . 2>/dev/null || true
# If verl is not a package, ensure PYTHONPATH includes repo root when running uvicorn (it does by default via scripts/remote_env_server.py)
```

If the server fails on import, install the missing package (e.g. `pip install transformers` or whatever the traceback shows). The server adds `repo_root` and `OSWorld` to `sys.path`, so running from the repo root is enough.

---

## 3. Inside the Ubuntu VM: run env server with Docker (KVM)

Use **Docker** provider so the OSWorld container runs inside the VM and sees `/dev/kvm`:

```bash
cd ~/arpo_remote_env
source arpo_env/bin/activate
export PROVIDER=docker
python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 18082
```

- `PROVIDER=docker`: use Docker (and the container will get KVM inside this Linux guest).
- `--host 0.0.0.0`: listen on all interfaces so the tunnel can bind to 127.0.0.1:18082 and forward to the cluster.
- First run may take a while: the Docker manager downloads the OSWorld Ubuntu qcow2 into `./docker_vm_data` inside the VM.

Leave this terminal running. You should see logs like “Docker provider patched…” and “Started container…” (and no “/dev/kvm not found” if KVM is available in the guest).

---

## 4. Inside the Ubuntu VM: open reverse tunnel to the cluster

In a **second** terminal (or SSH session) on the same Ubuntu VM:

```bash
ssh -R 15001:127.0.0.1:18082 kevinzyz@deepx-a100-40g-2 -N
```

Replace `kevinzyz@deepx-a100-40g-2` with your cluster login. Leave this session open. The cluster’s `localhost:15001` will forward to the VM’s `127.0.0.1:18082`.

---

## 5. On the cluster: run training

```bash
ssh kevinzyz@deepx-a100-40g-2
cd ~/hansenzuishuai
curl http://127.0.0.1:15001/health
python -m verl.trainer.main config=configs/smoke_remote_env_tunnel.yaml
```

You should see `"provider":"docker"` and `"kvm_available":true` in the health response (because the server runs in the Linux VM where KVM is available).

---

## Summary

| Where        | What |
|-------------|------|
| **Mac**     | VMware Fusion with Ubuntu VM (nested “Hypervisor applications” enabled). |
| **Ubuntu VM** | Docker, repo, `PROVIDER=docker` env server on port 18082, then `ssh -R 15001:127.0.0.1:18082 ...` to cluster. |
| **Cluster** | `config=configs/smoke_remote_env_tunnel.yaml` (uses `http://127.0.0.1:15001`). |

If the VM cannot reach the cluster (e.g. firewall), use a tunnel from the **Mac** instead: on the Mac, forward Mac’s 18082 to the VM’s 18082 (Fusion port forwarding or `ssh -L 18082:VM_IP:18082` from Mac to VM), then run `ssh -R 15001:127.0.0.1:18082 kevinzyz@deepx-a100-40g-2 -N` from the Mac.
