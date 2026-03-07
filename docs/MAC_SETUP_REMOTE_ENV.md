# Mac setup: clone → env → run remote env server

Use a **dedicated virtual env** (venv or conda). It keeps this project’s dependencies separate from `base` and avoids version clashes. `base` is fine for a quick try, but for anything ongoing, use an env.

---

## 1. Clone and submodules

```bash
cd ~
git clone https://github.com/hanshengzhu0001/arpo_remote_env.git
cd arpo_remote_env
git submodule update --init --recursive
```

---

## 2. Create and use a virtual env (recommended)

**Option A – venv**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Option B – conda**

```bash
conda create -n arpo_env python=3.11 -y
conda activate arpo_env
```

---

## 3. Install dependencies (Mac / CPU only)

- **FastAPI + server:** required for the remote env server.

```bash
pip install "fastapi>=0.100" "uvicorn[standard]" requests pydantic
```

- **Repo requirements** (skip GPU-only packages: `vllm`, `flash_attn`, `liger_kernel`):

```bash
pip install -r requirements.txt
```

If something in `requirements.txt` fails (e.g. GPU-only), install the rest and leave that one out.

- **OSWorld / desktop_env** (for the server’s `DesktopEnv`):

```bash
pip install -r OSWorld/requirements.txt
# If there is a desktop_env-specific requirements file:
# pip install -r OSWorld/desktop_env/requirements.txt
```

---

## 4. Docker + VM image (for the env server)

The server runs one OSWorld env using the **Docker** provider and an Ubuntu VM image.

- Install **Docker Desktop** (or Docker Engine) and ensure the Docker daemon is running.
- **Start Docker on Mac:** open **Docker Desktop** from Applications (or run `open -a Docker` in Terminal) and wait until the menu bar icon shows Docker is running (not "Starting…").
- Follow **OSWorld’s Docker setup** for the Ubuntu VM (e.g. build or download the image and put it where the Docker provider expects).
- See `OSWorld/desktop_env/providers/docker/` and the main OSWorld README for VM image and `docker_vm_data` (or equivalent) setup.

Without Docker and the VM, the server will fail when it tries to create `DesktopEnv(provider_name="docker", ...)`.

---

## 5. Run the remote env server

From the repo root, with your venv/conda env activated:

```bash
python scripts/remote_env_server.py
```

Server listens on `0.0.0.0:5000`. Check:

```bash
curl http://localhost:5000/health
```

---

## 6. Point the cluster at this Mac

- Get your Mac’s IP on the LAN (e.g. `System Settings → Network`, or `ifconfig`).
- On the cluster, set in your config (e.g. `configs/smoke_remote_env.yaml`):

  `env.remote_server_url: "http://YOUR_MAC_IP:5000"`

- Ensure the cluster can reach that URL (same network, VPN, or port forward).
- Use `num_envs: 1` (one server = one env) and run training as usual.

---

## Summary

| Step | What to do |
|------|------------|
| 1 | Clone repo, then `git submodule update --init --recursive` |
| 2 | Create and activate a venv or conda env (recommended over `base`) |
| 3 | Install FastAPI/uvicorn, then `requirements.txt`, then OSWorld deps |
| 4 | Install Docker and set up OSWorld’s Ubuntu VM image |
| 5 | Run `python scripts/remote_env_server.py` and test with `curl …/health` |
| 6 | Set `env.remote_server_url` on the cluster to `http://MAC_IP:5000` |
