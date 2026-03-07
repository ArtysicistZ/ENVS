# arpo_remote_env

ARPO (Agentic Replay Policy Optimization) with OSWorld and **remote env** support: run the environment on a Mac (or CPU server) and training on a GPU cluster over HTTP, with an optional SSH tunnel when the cluster cannot reach the Mac directly.

**Repo:** [github.com/hanshengzhu0001/arpo_remote_env](https://github.com/hanshengzhu0001/arpo_remote_env)

---

## Quick start

### 1. Clone and submodules

```bash
git clone https://github.com/hanshengzhu0001/arpo_remote_env.git
cd arpo_remote_env
git submodule update --init --recursive
```

### 2. Smoke test (two options)

**Option A – All on cluster (local envs)**  
Uses Docker + VMs on the cluster. No Mac needed.

```bash
cd arpo_remote_env
pip install -r requirements.txt   # and GPU deps (vllm, etc.) as needed
ray stop   # if an old Ray cluster is running
python -m verl.trainer.main config=configs/smoke_4gpu.yaml
```

**Option B – Remote env (Mac + cluster)**  
Env runs on your Mac; cluster runs training and talks to the Mac over HTTP (direct or SSH tunnel).

**On the Mac:**

```bash
cd arpo_remote_env
python3 -m venv arpo_env && source arpo_env/bin/activate
pip install -r requirements.txt   # skip GPU-only packages if needed
python scripts/remote_env_server.py
```

Leave that running (server on port **15001**).

**On the cluster:**

- If the cluster **can** reach your Mac (same VPN/network): set `env.remote_server_url` in `configs/smoke_remote_env.yaml` to `http://YOUR_MAC_IP:15001`, then:
  ```bash
  python -m verl.trainer.main config=configs/smoke_remote_env.yaml
  ```
- If the cluster **cannot** reach your Mac: use an SSH reverse tunnel. On the Mac (with the env server already running): `ssh -R 15001:localhost:15001 USER@CLUSTER_HOST`. On the cluster: `curl http://127.0.0.1:15001/health` then:
  ```bash
  python -m verl.trainer.main config=configs/smoke_remote_env_tunnel.yaml
  ```

Full details: [scripts/README_remote_env_server.md](scripts/README_remote_env_server.md).

---

## Configs

| Config | Use case |
|--------|----------|
| `configs/smoke.yaml` | 1 GPU, 2 envs, ~1 h |
| `configs/smoke_4gpu.yaml` | 4 GPUs, 4 envs, ~25–30 min |
| `configs/smoke_8gpu.yaml` | 8 GPUs, 8 envs |
| `configs/smoke_remote_env.yaml` | Remote env (Mac IP in config) |
| `configs/smoke_remote_env_tunnel.yaml` | Remote env via SSH tunnel (127.0.0.1:15001) |

Data paths use `OSWorld/evaluation_examples/test_smoke_4.json` (4 tasks). Set `env.remote_server_url` to your Mac’s URL when using remote env without tunnel.

---

## Layout

```
arpo_remote_env/
├── configs/           # Training configs (smoke, remote env, tunnel)
├── scripts/
│   ├── remote_env_server.py   # Run on Mac for Option B
│   └── README_remote_env_server.md
├── verl/              # VERL trainer + RemoteEnvWorker
├── OSWorld/           # Submodule (desktop_env, evaluators)
├── notebooks/         # Cluster_Smoke_Test, evaluation, etc.
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- GPU cluster: Ray, PyTorch, vLLM, transformers; see `requirements.txt`
- Remote env server: FastAPI, uvicorn, OSWorld/desktop_env deps
- For local envs on cluster: Docker (+ VM image per OSWorld docs)

---

## Links

- **Paper:** [ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay](https://arxiv.org/abs/2505.16282)
- **ARPO:** [JIA-Lab-research/ARPO](https://github.com/JIA-Lab-research/ARPO)
- **OSWorld:** [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)

## License

Apache-2.0
