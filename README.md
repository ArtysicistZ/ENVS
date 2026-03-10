# ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay

This repository provides an enhanced implementation of **ARPO (Agentic Replay Policy Optimization)** with remote environment support, enabling scalable GUI agent training across distributed hardware — run environments on a Mac or AWS while training on a GPU cluster over HTTP.

**Paper:** [ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay](https://arxiv.org/abs/2505.16282)
**Original Code:** [JIA-Lab-research/ARPO](https://github.com/JIA-Lab-research/ARPO)
**Benchmark:** [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)

---

## What is ARPO?

Training GUI agents with RL suffers from **sparse rewards** — most rollouts fail, producing zero gradients. ARPO solves this by combining **GRPO (Group Relative Policy Optimization)** with an **experience replay buffer**.

**The problem:**
```
All rollouts fail: rewards = [0, 0, 0, 0, 0, 0, 0, 0]
  -> Advantages all zero -> No gradient signal -> Training stalls
```

**ARPO's solution:** Inject a previously successful trajectory from the replay buffer:
```
After injection: rewards = [1, 0, 0, 0, 0, 0, 0, 0]
  -> Advantages = [+2.65, -0.38, -0.38, ...] -> Gradients flow!
```

### Key Results (from the paper)

| Model | 128 Training Tasks | OSWorld Overall (369) |
|-------|-------------------|-----------------------|
| UI-Tars-1.5 (Baseline) | 68.7% | 23.5% |
| UI-Tars-1.5 + GRPO | 72.9% | 26.0% |
| **UI-Tars-1.5 + ARPO** | **83.9%** | **29.9%** |

+11% improvement on training tasks, +3.9% generalization to unseen tasks.

---

## Architecture

```
                          GPU Cluster                          Environment Host (Mac/AWS/Docker)
                    ┌──────────────────────┐              ┌──────────────────────────┐
                    │  VERL Trainer (Ray)   │              │  Remote Env Server       │
                    │                      │   HTTP/SSH   │  (FastAPI on port 15001)  │
                    │  ┌────────────────┐  │◄────────────►│                          │
                    │  │ vLLM Rollout   │  │  screenshots │  ┌────────────────────┐  │
                    │  │ (UI-Tars-7B)   │  │  & actions   │  │ OSWorld Desktop VM │  │
                    │  └────────────────┘  │              │  │ (Ubuntu + apps)    │  │
                    │  ┌────────────────┐  │              │  └────────────────────┘  │
                    │  │ GRPO + Replay  │  │              │                          │
                    │  │ Buffer         │  │              │  Providers:              │
                    │  └────────────────┘  │              │  - Docker (Linux)        │
                    │  ┌────────────────┐  │              │  - VMware (macOS)        │
                    │  │ FSDP Actor     │  │              │  - AWS EC2 (cloud)       │
                    │  └────────────────┘  │              └──────────────────────────┘
                    └──────────────────────┘
```

**Two deployment modes:**
- **Local:** Environments run as Docker containers on the same GPU cluster
- **Remote:** Environments run on a separate host (Mac/AWS), communicating via HTTP (direct or SSH tunnel)

---

## Repository Structure

```
ARPO/
├── verl/                        # Core training framework
│   ├── trainer/
│   │   ├── main.py              # Entry point
│   │   ├── ray_trainer.py       # Main training loop (~2000 lines)
│   │   ├── core_algos.py        # GRPO advantage computation
│   │   ├── replay_buffer.py     # Experience replay buffer
│   │   ├── gui_agent.py         # EnvWorker & RemoteEnvWorker
│   │   ├── remote_env_protocol.py  # HTTP wire protocol (base64 images)
│   │   └── config.py            # Configuration dataclasses
│   ├── workers/
│   │   ├── rollout/             # vLLM inference (vllm_rollout_spmd.py)
│   │   ├── reward/              # Reward computation (custom.py)
│   │   └── actor/, ref/         # FSDP model sharding
│   └── utils/
│       └── osworld.py           # Dataset loading & tokenization
├── scripts/
│   ├── servers/                 # Model & env servers
│   │   └── remote_env_server.py # FastAPI server for remote environments
│   ├── training/                # Training launch scripts
│   ├── testing/                 # Test & validation scripts
│   └── utils/                   # Diagnostic & maintenance utilities
├── configs/                     # Training configurations
│   ├── smoke.yaml               # 1 GPU, quick test
│   ├── smoke_4gpu.yaml          # 4 GPUs, 4 envs
│   ├── smoke_8gpu.yaml          # 8 GPUs, 8 envs
│   ├── smoke_remote_env.yaml    # Remote env (direct HTTP)
│   ├── smoke_remote_env_tunnel.yaml      # Remote env via SSH tunnel
│   └── smoke_remote_env_8gpu_a100.yaml   # Paper-aligned full training
├── examples/
│   ├── osworld_full_arpo.sh     # Full ARPO: 128 tasks, 2 nodes, 8 GPUs
│   ├── osworld_subset32.sh      # Subset: 32 tasks, 1 node, 8 GPUs
│   └── config.yaml              # Base config template
├── OSWorld/                     # OSWorld benchmark (submodule)
│   ├── desktop_env/             # VM providers, evaluators, actions
│   ├── evaluation_examples/     # Task definitions (JSON)
│   └── mm_agents/               # Agent implementations
├── docs/                        # Documentation
│   ├── PAPER_SUMMARY.md         # Detailed algorithm explanation
│   ├── COMPREHENSIVE_GUIDE.md   # Full walkthrough & reproduction guide
│   ├── DATA_FLOW.md             # Architecture diagrams
│   └── TRAINING_GUIDE.md        # Step-by-step training instructions
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- GPU cluster: PyTorch, Ray, vLLM, transformers
- For remote envs: FastAPI, uvicorn
- For local envs: Docker (+ OSWorld VM image)

### Option A: Local Environments (all on GPU cluster)

```bash
git clone https://github.com/hanshengzhu0001/arpo_remote_env.git
cd arpo_remote_env
pip install -r requirements.txt
ray stop
python -m verl.trainer.main config=configs/smoke_4gpu.yaml
```

### Option B: Remote Environments (Mac/AWS + GPU cluster)

**On the environment host (Mac or AWS):**

```bash
pip install -r requirements.txt
python scripts/servers/remote_env_server.py  # Starts on port 15001
```

**On the GPU cluster:**

If the cluster can reach the env host directly:
```bash
# Set env.remote_server_url in the config to http://ENV_HOST_IP:15001
python -m verl.trainer.main config=configs/smoke_remote_env.yaml
```

If you need an SSH tunnel:
```bash
# On the env host: ssh -R 15001:localhost:15001 USER@CLUSTER_HOST
python -m verl.trainer.main config=configs/smoke_remote_env_tunnel.yaml
```

---

## Training Configurations

### Smoke Tests

| Config | GPUs | Envs | Tasks | Use Case |
|--------|------|------|-------|----------|
| `smoke.yaml` | 1 | 2 | 4 | Quick sanity check |
| `smoke_4gpu.yaml` | 4 | 4 | 4 | Multi-GPU test |
| `smoke_8gpu.yaml` | 8 | 8 | 4 | Full GPU test |
| `smoke_remote_env.yaml` | 1 | 1 | 4 | Remote env test |

### Paper-Aligned Training

**Full replication** (`examples/osworld_full_arpo.sh`):
- 128 tasks (filtered from 369), 256 Docker VMs, 8 rollouts/task
- 2 nodes x 8 GPUs, 15 epochs
- UI-Tars-1.5-7B, 64K context, 15 screenshots max

**Subset training** (`examples/osworld_subset32.sh`):
- 32 tasks, 16 envs, 8 rollouts/task
- 1 node x 8 GPUs, 15 epochs
- Experience replay enabled

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-6 | AdamW optimizer |
| Rollout temperature | 1.0 | High exploration |
| Eval temperature | 0.5 | More deterministic |
| Clip ratio | [0.2, 0.3] | Asymmetric (from DAPO) |
| KL penalty | Disabled | No reference model needed |
| Max steps/trajectory | 15 | Action budget per task |
| Experience replay | Enabled | Per-task buffer with FIFO eviction |

---

## How It Works

### 1. Distributed Rollout

The trainer spawns environment workers (local Docker or remote HTTP) and a vLLM inference engine. For each episode:
1. **Reset** — VM boots, task is configured, initial screenshot captured
2. **Step loop** — Model sees screenshot history, predicts action (click, type, scroll...), environment executes it
3. **Evaluate** — OSWorld's rule-based checker scores the task (0 or 1)

### 2. Experience Replay Buffer

```python
# Per-task buffer stores successful trajectories
if all rollout rewards == 0 and task_id in replay_buffer:
    # Inject cached success -> non-zero advantages -> gradients flow
    rollouts[0] = replay_buffer.sample(task_id)

# Store new successes for future use
for rollout in rollouts:
    if rollout.reward > 0:
        replay_buffer.store(task_id, rollout)
```

### 3. GRPO Policy Update

- Group-normalize rewards within each task: `advantage = (r - mean) / std`
- Clipped policy gradient (no value function, no KL penalty)
- Token-level optimization over the full action sequence

### 4. Reward Design

| Component | Value | Purpose |
|-----------|-------|---------|
| Task reward | +1.0 (success) / 0.0 (fail) | Binary task completion |
| Format penalty | -1.0 (invalid) / 0.0 (valid) | Encourage valid action syntax |

---

## Remote Environment Protocol

The remote env server exposes a REST API for environment interaction:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/env/reset` | POST | Initialize VM with task config |
| `/env/step` | POST | Execute action, return next screenshot |
| `/env/evaluate` | POST | Get task success score |
| `/env/history_messages` | POST | Retrieve conversation history |

Images are transmitted as base64-encoded JPEG in JSON payloads, preserving resolution constraints.

---

## Links

- **Paper:** [arxiv.org/abs/2505.16282](https://arxiv.org/abs/2505.16282)
- **Original ARPO:** [JIA-Lab-research/ARPO](https://github.com/JIA-Lab-research/ARPO)
- **OSWorld:** [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)
- **UI-Tars-1.5:** [huggingface.co/Zhenyu00/UITars-1.5](https://huggingface.co/Zhenyu00/UITars-1.5)
- **VERL Framework:** [volcengine/verl](https://github.com/volcengine/verl)

## Citation

```bibtex
@article{lu2024arpo,
  title={ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay},
  author={Lu, Fanbin and Zhong, Zhisheng and Liu, Shu and Fu, Chi-Wing and Jia, Jiaya},
  journal={arXiv preprint arXiv:2505.16282},
  year={2024}
}
```

## License

Apache-2.0
