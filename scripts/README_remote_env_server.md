# Remote OSWorld env server (2-machine setup)

**Architecture:**
- **Resource (GPUs, model, rollout):** current cluster.
- **Executions (reset / step / evaluate):** remote env server (on Mac with VMware, or inside a Linux VM with Docker+KVM).

The cluster runs ARPO training and sends env requests (reset, step, evaluate) to the remote server over HTTP.

**Two options for where the env runs:**
- **Simple (Mac + VMware):** Run the server on the Mac; it uses VMware Fusion by default. See sections below.
- **Nested KVM (Fusion → Linux VM → Docker):** Run a Linux VM in Fusion with nested virtualization, then run the server + Docker inside that VM for KVM-accelerated containers. See **[README_remote_env_nested_kvm.md](README_remote_env_nested_kvm.md)**.

## 1. On Mac (env server – VMware)

Install VMware Fusion and ensure `vmrun` is on PATH. The server defaults to VMware on macOS.

```bash
cd ~/arpo_remote_env   # or your clone
source arpo_env/bin/activate
python -m uvicorn scripts.remote_env_server:app --host 127.0.0.1 --port 18082
```

Server listens on `127.0.0.1:18082`. Keep this terminal open. You should see `Startup: PROVIDER=vmware`.

## 2. On cluster (training – resource)

**Option A – Cluster can reach your Mac (same VPN/network)**  
- Set `env.remote_server_url` in `configs/smoke_remote_env.yaml` to your Mac URL, e.g. `http://YOUR_MAC_IP:18082`.  
- From cluster: `curl http://YOUR_MAC_IP:18082/health` then run:
  `python -m verl.trainer.main config=configs/smoke_remote_env.yaml`

**Option B – Cluster cannot reach your Mac (SSH reverse tunnel)**  
1. On Mac (env server already running on 18082): open a second terminal and run:
   `ssh -R 15001:127.0.0.1:18082 kevinzyz@<CLUSTER_HOST>`  
   Leave this SSH session open.  
2. On cluster: `curl http://127.0.0.1:15001/health` then:
   `python -m verl.trainer.main config=configs/smoke_remote_env_tunnel.yaml`

Tunnel maps cluster `localhost:15001` → Mac `localhost:18082`. Training config `smoke_remote_env_tunnel.yaml` already uses `http://127.0.0.1:15001`.

## 3. On Linux / AWS (env server – Docker + KVM)

To **fix 503** and **use KVM** when the env server runs on Linux (e.g. AWS):

1. **Install the Docker Python package** (required for the Docker provider):
   ```bash
   pip install docker
   ```
2. **Ensure the Docker daemon is running** (Docker Engine or Docker Desktop).
3. **Use KVM acceleration:** Run the server on a host that has `/dev/kvm` (e.g. most AWS EC2 instance types). The server checks for `/dev/kvm` at startup and passes it into the OSWorld container when present; you’ll see `✓ KVM detected` in the logs. If you see `⚠ KVM not found`, the VM will use software emulation (slower).

```bash
# On the Linux/AWS host (e.g. after SSH)
cd /path/to/repo
source .venv/bin/activate   # or your venv
pip install docker         # if not already installed
python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 15001
```

Point the cluster at this host (e.g. `env.remote_server_url: "http://34.227.191.7:15001"`) and ensure the security group allows the cluster IP on port 15001.

## API (for debugging)

- `POST /env/reset`  Body: `{"task_config": {...}}`  → `{env_idx, obs_messages, is_done, format_reward}`
- `POST /env/step`   Body: `{"prediction": "..."}`   → same shape
- `POST /env/evaluate` Body: `{}` → float
- `POST /env/history_messages` Body: `{}` → `{history_messages: [...]}`
- `GET /health` → `{status: "ok", provider: "vmware", ...}`

Images in JSON use a `b64` field (raw base64); the cluster adds the `data:image/jpeg;base64,` prefix when building messages.
