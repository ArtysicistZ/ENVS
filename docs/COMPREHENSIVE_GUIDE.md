# ARPO Comprehensive Guide: Codebase Walkthrough & Reproduction Instructions

> **Audience**: A Windows user who just cloned this repo and wants to understand everything and reproduce ARPO results using a GPU cluster (for training/rollout) and VMs on AWS (for desktop environments with screenshots).

---

## Table of Contents

1. [What is ARPO?](#1-what-is-arpo)
2. [Repository Structure Overview](#2-repository-structure-overview)
3. [File-by-File Breakdown](#3-file-by-file-breakdown)
4. [Architecture & Data Flow](#4-architecture--data-flow)
5. [The ARPO Algorithm In Detail](#5-the-arpo-algorithm-in-detail)
6. [Hardware & Infrastructure Requirements](#6-hardware--infrastructure-requirements)
7. [Software Dependencies](#7-software-dependencies)
8. [Setup Instructions (Windows + GPU Cluster + AWS VMs)](#8-setup-instructions)
9. [Configuration Reference](#9-configuration-reference)
10. [Running Training](#10-running-training)
11. [Evaluation & Checkpoints](#11-evaluation--checkpoints)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What is ARPO?

**ARPO (Agentic Replay Policy Optimization)** is a reinforcement learning method for training vision-language GUI agents on desktop tasks.

**Paper**: *ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay*
**Authors**: Fanbin Lu, Zhisheng Zhong, Shu Liu, Chi-Wing Fu, Jiaya Jia (CUHK, SmartMore, HKUST)
**Link**: https://arxiv.org/abs/2505.16282

### The Problem

Training GUI agents with RL suffers from **sparse rewards**: most rollouts fail (reward=0). When all rollouts for a task fail:
- Mean reward = 0, standard deviation = 0
- Normalized advantage = (0-0)/0 = undefined -> 0
- **No gradient signal -> training stalls completely**

### The Solution

ARPO = **GRPO (Group Relative Policy Optimization)** + **Experience Replay Buffer**

When all rollouts fail for a task, ARPO injects a previously-successful trajectory from a per-task replay buffer:

```
Before injection: rewards = [0, 0, 0, 0, 0, 0, 0, 0]  -> advantages all 0 -> no learning
After injection:  rewards = [1, 0, 0, 0, 0, 0, 0, 0]  -> advantages [+2.65, -0.38, ...] -> learning!
```

### Key Results (from the paper)

| Model                 | 128 Training Tasks | OSWorld Overall (369) |
|-----------------------|--------------------|-----------------------|
| UI-TARS-1.5 (base)   | 68.7%              | 23.5%                 |
| UI-TARS-1.5 + GRPO   | 72.9%              | 26.0%                 |
| **UI-TARS-1.5 + ARPO** | **83.9%**        | **29.9%**             |

### Key Hyperparameters (paper)

| Parameter              | Value   | Notes                                     |
|------------------------|---------|--------------------------------------------|
| Base model             | UI-TARS-1.5 (Qwen2.5-VL 7B) | Vision-language model       |
| Learning rate          | 1e-6    | AdamW optimizer                            |
| Clip ratio (low/high)  | 0.2/0.3 | Asymmetric clipping from DAPO             |
| KL divergence          | Disabled | No reference model needed                 |
| Rollout temperature    | 1.0     | High exploration during training           |
| Eval temperature       | 0.6     | More deterministic during evaluation       |
| Rollouts per task      | 8       | For GRPO group normalization               |
| Max steps per trajectory | 15    | GUI interaction steps                      |
| Training epochs        | 15      | Total training epochs                      |
| Tasks                  | 128     | Filtered from 369 OSWorld tasks            |
| Parallel environments  | 256     | Docker containers                          |

---

## 2. Repository Structure Overview

```
arpo_remote_env/
|
|-- verl/                          # Core: VERL training framework (modified for ARPO)
|   |-- trainer/                   # Training orchestration
|   |   |-- main.py                # Entry point
|   |   |-- ray_trainer.py         # Main training loop (RayPPOTrainer)
|   |   |-- gui_agent.py           # GUI agent action parsing (UI-TARS format)
|   |   |-- core_algos.py          # GRPO/PPO loss & advantage computation
|   |   |-- config.py              # Configuration dataclasses
|   |   |-- metrics.py             # Training metrics computation
|   |   |-- remote_env_protocol.py # HTTP protocol for remote environments
|   |   |-- replay_buffer.py       # Experience replay buffer (ARPO key innovation)
|   |
|   |-- workers/                   # Distributed worker implementations
|   |   |-- fsdp_workers.py        # Main FSDP worker (actor + rollout + critic)
|   |   |-- actor/                 # Policy network (gradient updates)
|   |   |-- critic/                # Value function (optional, not used in GRPO)
|   |   |-- rollout/               # vLLM-based text generation
|   |   |-- reward/                # Reward computation
|   |   |-- sharding_manager/      # FSDP <-> vLLM weight synchronization
|   |   |-- config.py              # Worker configuration aggregation
|   |
|   |-- protocol.py                # DataProto: universal data exchange format
|   |-- utils/                     # Utilities (dataset, tokenizer, checkpointing, logging...)
|   |-- models/                    # Model patches (Qwen2-VL, Ulysses attention)
|   |-- single_controller/         # Ray-based distributed orchestration
|
|-- OSWorld/                       # Git submodule: OSWorld benchmark environment
|   |-- desktop_env/               # Desktop environment (VM management, actions, evaluators)
|   |-- evaluation_examples/       # Task definitions (JSON per domain)
|
|-- configs/                       # YAML training configurations
|   |-- smoke.yaml                 # 1 GPU smoke test
|   |-- smoke_4gpu.yaml            # 4 GPU smoke test
|   |-- smoke_8gpu.yaml            # 8 GPU smoke test
|   |-- smoke_remote_env.yaml      # Remote env (direct IP)
|   |-- smoke_remote_env_tunnel.yaml  # Remote env (SSH tunnel)
|   |-- config_uitars_2b_mac.yaml  # Mac CPU config (2B model)
|   |-- wandb_config.yaml          # W&B logging config
|
|-- scripts/                       # Helper scripts
|   |-- remote_env_server.py       # FastAPI server for remote env (run on VM/Mac)
|   |-- model_merger.py            # Merge FSDP sharded checkpoints
|   |-- uitars_2b_server.py        # UI-TARS-2B inference server (Flask, CPU)
|   |-- uitars_7b_server.py        # UI-TARS-7B inference server (Flask, CPU)
|   |-- run_training_with_wandb.py # Training wrapper with W&B
|
|-- examples/                      # Example training launch scripts
|   |-- osworld_full_arpo.sh       # Full ARPO: 2 nodes, 8 GPUs, 128 tasks, 15 epochs
|   |-- osworld_subset32.sh        # Subset: 1 node, 8 GPUs, 32 tasks, 15 epochs
|   |-- config.yaml                # Base config for examples
|   |-- baselines/                 # Non-OSWorld baselines (CLEVR, GeoQA)
|
|-- osworld_patches/               # Modifications for OSWorld compatibility
|   |-- run_uitars.py              # Single-env evaluation runner
|   |-- run_multienv_uitars.py     # Multi-env evaluation runner
|   |-- uitars_agent.py            # UITARSAgent implementation
|
|-- notebooks/                     # Jupyter notebooks
|-- papers/                        # ARPO paper PDF
|-- evaluation_examples/           # Symlink/copy of task JSONs
|-- docs/                          # Documentation
|-- requirements.txt               # Python dependencies
```

---

## 3. File-by-File Breakdown

### 3.1 Training Core (`verl/trainer/`)

#### `main.py` - Entry Point
- Creates a Ray remote `Runner` actor
- Calls `RayPPOTrainer.fit()` to start the training loop
- Usage: `python -m verl.trainer.main config=configs/smoke_4gpu.yaml`

#### `ray_trainer.py` - Training Orchestrator (~1500 lines)
The brain of the system. `RayPPOTrainer` manages:
- **Worker initialization**: Creates FSDP workers for actor/rollout/critic/ref roles
- **Resource allocation**: Maps GPU pools to worker groups via Ray placement groups
- **Training loop** (`fit()`):
  1. Sample a batch of tasks
  2. For each task: run environment rollouts (interact with OSWorld VMs)
  3. Collect trajectories (screenshots + actions + rewards)
  4. Apply experience replay (inject successes when all rollouts fail)
  5. Compute GRPO advantages
  6. Update policy with clipped PPO loss
  7. Log metrics, save checkpoints
- **Validation** (`validate()`): Run evaluation rollouts at lower temperature
- **Remote env support**: Makes HTTP calls to `remote_env_server.py` for environment interaction

#### `gui_agent.py` - Action Parser (~1200 lines)
Handles the UI-TARS action format:
- **System prompt**: Defines the agent's instruction format with action types
- **Action types**: `LEFT_CLICK`, `RIGHT_CLICK`, `TYPE_TEXT`, `PRESS_HOTKEY`, `SCROLL`, `WAIT`, `FINISH`, `FAIL`, `CALL_USER`
- **Parsing**: Converts LLM text output -> structured action -> pyautogui code
- **Image handling**: Resizes screenshots, adds bounding box tokens
- Critical for bridging the LLM output to actual desktop interactions

#### `core_algos.py` - RL Algorithms (~440 lines)
Contains all advantage estimation and loss functions:
- **`compute_grpo_outcome_advantage()`**: ARPO's advantage method - group-normalizes rewards within task groups. When std=0 (all same reward), advantages become 0 (this is where replay buffer saves the day)
- **`compute_policy_loss()`**: PPO clipped surrogate loss with asymmetric clipping (clip_low=0.2, clip_high=0.3)
- **`compute_gae_advantage_return()`**: Standard GAE (not used in ARPO, but available)
- **KL controllers**: Adaptive and fixed KL penalty (disabled in ARPO)

#### `replay_buffer.py` - Experience Replay (~52 lines)
The key ARPO innovation:
- **`ReplayBuffer`**: Maintains `pos_dataset[task_id] -> list of successful trajectories`
- **`update_replay_buffer(task_id, trajectory, reward)`**: Stores trajectories with reward > 0.1 (FIFO eviction)
- **`get_pos(task_id)`**: Randomly samples a cached success for injection
- Small file, huge impact - this is what prevents vanishing gradients

#### `config.py` - Configuration Dataclasses
Defines: `PPOConfig` (top-level), `DataConfig`, `AlgorithmConfig`, `TrainerConfig`, `EnvConfig`
- Supports deep initialization via `deep_post_init()`
- Key fields: `enable_replay`, `disable_kl`, `adv_estimator`

#### `remote_env_protocol.py` - HTTP Protocol (~68 lines)
Serializes/deserializes messages between training cluster and remote env server:
- Converts `data:image/jpeg;base64,...` to wire format (raw base64 in `b64` field)
- Used by both `ray_trainer.py` and `remote_env_server.py`

#### `metrics.py` - Metrics Computation
Computes and logs: score stats, reward stats, advantage stats, value stats, timing, throughput.

### 3.2 Workers (`verl/workers/`)

#### `fsdp_workers.py` - FSDP Worker
The main worker class that combines multiple roles:
- Roles: `actor`, `rollout`, `critic`, `ref`, or combinations like `actor_rollout`
- Initializes FSDP-wrapped model, optimizer, vLLM engine
- Manages model/optimizer CPU offloading for memory efficiency

#### `actor/dp_actor.py` - Policy Network
- **`compute_log_prob()`**: Forward pass to get action log probabilities (no grad)
- **`update_policy()`**: Compute PPO loss, backpropagate, update weights
- Supports padding-free attention, Ulysses sequence parallelism, multi-modal inputs

#### `rollout/vllm_rollout_spmd.py` - Text Generation
- Uses vLLM for efficient batched inference
- **`generate_sequences()`**: Generate action text from screenshot prompts
- Handles multi-modal data (images), response masking, n>1 rollouts
- Manages vLLM sleep/wake for memory sharing with training

#### `reward/custom.py` - Reward Functions
- `math_compute_score()`: Math task scoring (format 0.1 + accuracy 0.9)
- `r1v_compute_score()`: R1V task scoring (format 0.5 + accuracy 0.5)
- For OSWorld: rewards come from the environment evaluator (binary: 1.0 success / 0.0 failure)

#### `sharding_manager/fsdp_vllm.py` - Weight Sync
- Context manager that syncs weights between FSDP (training) and vLLM (inference)
- On enter: wake vLLM, copy FSDP weights to vLLM engine
- On exit: sleep vLLM, free GPU memory for training

### 3.3 Data Protocol (`verl/protocol.py`)

`DataProto` is the universal data carrier:
- `batch` (TensorDict): input_ids, attention_mask, position_ids, responses, log_probs, advantages
- `non_tensor_batch` (Dict): images, ground_truth text, raw prompt IDs
- `meta_info` (Dict): temperature, EOS token, step count
- Methods: `chunk()`, `concat()`, `select()`, `to()` for distributed data manipulation

### 3.4 Utilities (`verl/utils/`)

| File | Purpose |
|------|---------|
| `osworld.py` | OSWorld dataset loading, task config parsing, prompt formatting |
| `dataset.py` | General RLHF dataset (parquet/HuggingFace), tokenization, image processing |
| `fsdp_utils.py` | FSDP wrapping policies, CPU offloading for model and optimizer |
| `model_utils.py` | GPU memory reporting, model size inspection |
| `seqlen_balancing.py` | Load-balance sequences across GPUs (Karmarkar-Karp algorithm) |
| `tokenizer.py` | HuggingFace tokenizer/processor loading |
| `torch_functional.py` | Log probs, masked ops, padding, LR schedulers, AnyPrecisionAdamW |
| `checkpoint/` | Save/load FSDP sharded checkpoints, manage checkpoint rotation |
| `logger/` | Multi-backend logging (console, W&B, TensorBoard, MLflow) |
| `reward_score/` | Task-specific reward functions (math, R1V) |

### 3.5 Model Patches (`verl/models/`)

| File | Purpose |
|------|---------|
| `monkey_patch.py` | Patches flash_attention for Ulysses sequence parallelism |
| `transformers/qwen2_vl.py` | Qwen2-VL specific: RoPE position IDs, multi-modal forward pass |

### 3.6 Single Controller (`verl/single_controller/`)

Ray-based distributed orchestration:
- **`decorator.py`**: Dispatch patterns (ONE_TO_ALL, DP_COMPUTE, etc.) for distributing work
- **`worker.py`**: Base worker with rank/world_size info
- **`worker_group.py`**: Groups workers into resource pools
- **`ray/base.py`**: Ray placement groups, worker scheduling

### 3.7 Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `remote_env_server.py` | **Critical**: FastAPI server that wraps OSWorld. Run on any machine with Docker. Exposes `/env/reset`, `/env/step`, `/env/evaluate`, `/env/history_messages`, `/health` |
| `model_merger.py` | Merges FSDP rank-sharded checkpoints into a single HuggingFace model |
| `uitars_2b_server.py` | Flask inference server for UI-TARS-2B on CPU |
| `uitars_7b_server.py` | Flask inference server for UI-TARS-7B on CPU |

### 3.8 OSWorld Patches (`osworld_patches/`)

| File | Purpose |
|------|---------|
| `run_uitars.py` | End-to-end evaluation loop: loads tasks, runs agent, reports scores |
| `run_multienv_uitars.py` | Same but with multiple parallel environments |
| `uitars_agent.py` | `UITARSAgent` class: image resizing, action prediction, history management |

### 3.9 OSWorld Submodule (`OSWorld/`)

The OSWorld benchmark (https://github.com/xlang-ai/OSWorld):
- `desktop_env/desktop_env.py`: Main environment class (`DesktopEnv`) - manages VMs
- `desktop_env/providers/`: VM providers (Docker, AWS, VMware, VirtualBox, GCP, Azure)
- `desktop_env/evaluators/`: Task completion evaluators (rule-based per domain)
- `desktop_env/actions.py`: Action space definition
- `evaluation_examples/examples/`: 369 task configs organized by domain (chrome, libreoffice, gimp, etc.)

---

## 4. Architecture & Data Flow

### 4.1 Two-Machine Architecture

```
+------------------------------------------+     HTTP      +-----------------------------------+
|          GPU Cluster (Training)           |  <-------->   |     VM Host (Environments)        |
|                                           |               |                                   |
|  python -m verl.trainer.main              |  /env/reset   |  python scripts/remote_env_server |
|                                           |  /env/step    |                                   |
|  +-- RayPPOTrainer                        |  /env/evaluate|  +-- DesktopEnv (Docker/AWS)      |
|  |   +-- FSDPWorker (actor + rollout)     |  /health      |  |   +-- Ubuntu VM                |
|  |   |   +-- Qwen2-VL model (FSDP)       |               |  |   +-- Screenshots              |
|  |   |   +-- vLLM engine (generation)     |               |  |   +-- GUI actions              |
|  |   +-- ReplayBuffer                     |               |  |                                 |
|  |   +-- Metrics & Logging                |               |  +-- GuiAgent (action parser)     |
|  +---                                     |               +-----------------------------------+
+------------------------------------------+
```

### 4.2 Training Loop Data Flow (one epoch)

```
1. SAMPLE TASKS
   Dataset (JSON) -> sample batch of task configs

2. ENVIRONMENT ROLLOUT (per task, n rollouts each)
   For each task:
     a. POST /env/reset {task_config}           -> initial screenshot
     b. Build prompt: system + instruction + screenshot
     c. vLLM generates action text               (on GPU)
     d. POST /env/step {prediction: action_text} -> next screenshot + format_reward
     e. Repeat b-d for up to max_steps (15)
     f. POST /env/evaluate                       -> binary reward (0 or 1)

3. EXPERIENCE REPLAY (ARPO key step)
   For each task group:
     If all rewards == 0 AND replay buffer has a success for this task:
       Replace one failed trajectory with cached success
     If any reward > 0:
       Store successful trajectory in replay buffer

4. ADVANTAGE COMPUTATION (GRPO)
   For each task group:
     mu = mean(rewards), sigma = std(rewards)
     advantage_i = (reward_i - mu) / (sigma + eps)

5. POLICY UPDATE (PPO with clipping)
   ratio = pi_new(action|obs) / pi_old(action|obs)
   loss = -min(ratio * advantage, clip(ratio, 1-0.2, 1+0.3) * advantage)
   Backward pass -> gradient update

6. LOGGING & CHECKPOINTING
   Log to W&B / console
   Save checkpoint every N epochs
```

### 4.3 Weight Synchronization (FSDP <-> vLLM)

During training, the same model serves two roles:
1. **FSDP model**: For gradient computation (training)
2. **vLLM engine**: For fast batched generation (rollout)

The `FSDPVLLMShardingManager` handles this:
- Before rollout: wake vLLM, copy latest FSDP weights -> vLLM
- After rollout: sleep vLLM, free GPU memory for training
- This sharing is what makes single-GPU training possible

---

## 5. The ARPO Algorithm In Detail

### 5.1 GRPO (Base Algorithm)

GRPO computes advantages by normalizing rewards **within each task group**:

```
Given G tasks, each with n rollouts:
  For task g with rollouts {o_1, ..., o_n} and rewards {r_1, ..., r_n}:
    mu_g = mean(r_1, ..., r_n)
    sigma_g = std(r_1, ..., r_n)
    advantage_i = (r_i - mu_g) / sigma_g
```

Policy loss (PPO-style with asymmetric clipping):
```
ratio = pi_theta(a|s) / pi_old(a|s)
L = -min(ratio * A, clip(ratio, 1-eps_low, 1+eps_high) * A)

where eps_low = 0.2, eps_high = 0.3
```

### 5.2 Experience Replay (ARPO Addition)

```python
class ReplayBuffer:
    pos_dataset: Dict[task_id, List[trajectory]]  # per-task buffer

    def update(task_id, trajectory, reward):
        if reward > 0.1:  # success
            pos_dataset[task_id].append(trajectory)
            if len(pos_dataset[task_id]) > buffer_size:
                pos_dataset[task_id].pop(0)  # FIFO

    def inject(task_id, trajectories):
        rewards = [t.reward for t in trajectories]
        if std(rewards) == 0 and task_id in pos_dataset:
            # All same reward (likely all 0) -> inject success
            trajectories[0] = random.choice(pos_dataset[task_id])
        return trajectories
```

### 5.3 Reward Design

For OSWorld tasks:
- **Task reward**: 1.0 (success) or 0.0 (failure) - from OSWorld evaluator
- **Format reward**: 0.0 (valid action parse) or -1.0 (parse failure)
- **Total**: task_reward + format_reward

### 5.4 No KL Divergence

ARPO removes the KL penalty term entirely (`disable_kl: true`, `kl_coef: 0`). This means:
- No reference model needed (saves memory)
- Policy can diverge more freely from initialization
- Simpler training loop

---

## 6. Hardware & Infrastructure Requirements

### 6.1 For Full Paper Reproduction (recommended)

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **GPU Cluster** | 2 nodes x 8 GPUs (A100 80GB or H100) | 16 GPUs total |
| **Environment VMs** | 128-256 Docker containers | Ubuntu desktop VMs |
| **VM Host** | CPU server with Docker | AWS, GCP, or dedicated |
| **Network** | Low latency between cluster and VM host | Same region/VPC |
| **Storage** | ~100GB for model + checkpoints | SSD preferred |
| **RAM per GPU node** | 256GB+ system RAM | For FSDP offloading |

### 6.2 For Smaller Reproduction (subset)

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **GPU Cluster** | 1 node x 4-8 GPUs (A100 40GB+) | |
| **Environment VMs** | 4-16 Docker containers | |
| **VM Host** | AWS instance or local Docker | |
| **Storage** | ~50GB | |

### 6.3 For Smoke Testing

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **GPU** | 1-4 GPUs (any modern NVIDIA) | RTX 3090+ |
| **Environments** | 1-4 Docker containers | |
| **VM Host** | Same machine or any Docker host | |

### 6.4 Your Setup (GPU Cluster + AWS VMs)

This is a great match for the remote env architecture:
- **GPU cluster**: Runs `python -m verl.trainer.main` with appropriate config
- **AWS instances**: Run `python scripts/remote_env_server.py` (one per env)
- Connection: HTTP over network (direct IP or SSH tunnel)

---

## 7. Software Dependencies

### 7.1 GPU Cluster

```
# Core ML
torch==2.5.1
transformers==4.57.6
accelerate==1.4.0
qwen-vl-utils==0.0.11

# GPU-only (DO NOT install on CPU machines)
vllm                    # Fast inference engine
flash_attn              # Flash Attention 2
liger_kernel            # Fused CUDA kernels

# Distributed
ray>=2.9.0              # Distributed computing
wandb                   # Experiment tracking

# Communication with env server
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
requests==2.32.5
pydantic>=2.0.0
```

### 7.2 Environment Server (AWS VM)

```
# From requirements.txt (full list needed)
pip install -r requirements.txt

# OSWorld dependencies
pip install -r OSWorld/requirements.txt

# Docker must be installed and running
# The OSWorld Docker provider pulls Ubuntu VM images automatically
```

### 7.3 Windows Development Machine

You only need this for code editing and SSH access. No training runs on Windows.

```
# Git (with submodule support)
git clone https://github.com/hanshengzhu0001/arpo_remote_env.git
cd arpo_remote_env
git submodule update --init --recursive

# SSH client for connecting to cluster/AWS
# WSL2 or PowerShell SSH both work
```

---

## 8. Setup Instructions

### 8.1 Step 1: Clone the Repository (Windows)

```powershell
# In PowerShell or Git Bash
git clone https://github.com/hanshengzhu0001/arpo_remote_env.git
cd arpo_remote_env
git submodule update --init --recursive
```

### 8.2 Step 2: Set Up the GPU Cluster

SSH into your GPU cluster and:

```bash
# Clone repo on cluster
git clone https://github.com/hanshengzhu0001/arpo_remote_env.git
cd arpo_remote_env
git submodule update --init --recursive

# Create conda environment
conda create -n arpo python=3.10 -y
conda activate arpo

# Install dependencies
pip install -r requirements.txt

# Install GPU-only packages
pip install vllm
pip install flash-attn --no-build-isolation
pip install liger-kernel

# Verify
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
python -c "import vllm; print('vLLM OK')"
```

### 8.3 Step 3: Set Up AWS VM for Environments

Each AWS instance runs one `remote_env_server.py` (which manages one Docker-based OSWorld environment).

```bash
# On AWS instance (Ubuntu recommended)
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and back in for docker group

# Clone repo
git clone https://github.com/hanshengzhu0001/arpo_remote_env.git
cd arpo_remote_env
git submodule update --init --recursive

# Create environment
conda create -n arpo python=3.10 -y
conda activate arpo

# Install dependencies
pip install -r requirements.txt
pip install -r OSWorld/requirements.txt

# Start the environment server
python scripts/remote_env_server.py
# Server listens on 0.0.0.0:15001
```

**To verify**: From another machine, run:
```bash
curl http://AWS_INSTANCE_IP:15001/health
# Should return: {"status":"ok"}
```

**For multiple environments**: Launch multiple AWS instances, each running `remote_env_server.py` on port 15001. The training config's `env.remote_server_url` should point to one of them (current implementation supports one remote env per training run; for multi-env, you would need to either modify `ray_trainer.py` or run separate Docker envs on the cluster itself).

### 8.4 Step 4: Connect Cluster to AWS Env

**Option A: Direct network access** (recommended if in same VPC)

Set in your config YAML:
```yaml
env:
  remote_server_url: "http://AWS_INSTANCE_IP:15001"
```

Test connectivity from cluster:
```bash
curl http://AWS_INSTANCE_IP:15001/health
```

**Option B: SSH tunnel** (if cluster cannot reach AWS directly)

From the AWS instance:
```bash
ssh -R 15001:localhost:15001 YOUR_USER@CLUSTER_HOST
```

Then on cluster, use `http://127.0.0.1:15001` as the remote server URL.

### 8.5 Step 5: Download the Model

The model downloads automatically on first use via HuggingFace, but you can pre-download:

```bash
# On GPU cluster
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = 'ByteDance-Seed/UI-TARS-2B-SFT'  # or 'UI-TARS-1.5-7B' for paper reproduction
AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
print('Model downloaded successfully')
"
```

For the paper's full reproduction, you need `UI-TARS-1.5-7B` (the 7B model). The `UI-TARS-2B-SFT` is a smaller model for development/testing.

### 8.6 Step 6: Prepare Task Data

The smoke test data is already included: `evaluation_examples/test_smoke_4.json` (4 Chrome tasks).

For full training, you need the filtered task list. The paper filters OSWorld's 369 tasks down to 128 "trainable" tasks (those where the base model has at least 1/16 success). The example script references:
- `evaluation_examples/test_success_uitars1.5_wo_impossible.json` (128 tasks for full ARPO)
- `evaluation_examples/test_success_middle_difficult.json` (32 tasks for subset)

You may need to create these filtered lists by running baseline evaluation first.

---

## 9. Configuration Reference

### 9.1 Available Configs

| Config | GPUs | Tasks | Envs | Use Case |
|--------|------|-------|------|----------|
| `configs/smoke.yaml` | 1 | 4 | 2 | Minimal smoke test |
| `configs/smoke_4gpu.yaml` | 4 | 4 | 4 | Multi-GPU smoke test |
| `configs/smoke_8gpu.yaml` | 8 | 4 | 16 | 8-GPU smoke test |
| `configs/smoke_remote_env.yaml` | 4 | 4 | 1 (remote) | Remote env smoke test |
| `configs/smoke_remote_env_tunnel.yaml` | 4 | 4 | 1 (remote) | Remote env via SSH tunnel |
| `configs/config_uitars_2b_mac.yaml` | 0 (CPU) | 8 | 1 | Mac CPU development |

### 9.2 Key Configuration Sections

```yaml
# === DATA ===
data:
  train_files: path/to/tasks.json       # Task definition file
  val_files: path/to/tasks.json         # Validation tasks
  max_prompt_length: 64000              # Context window (tokens)
  max_response_length: 8192             # Max action output (tokens)
  rollout_batch_size: 16                # Tasks per rollout batch
  max_pixels: 2116800                   # Max image pixels
  min_pixels: 256                       # Min image pixels

# === ALGORITHM (ARPO-specific) ===
algorithm:
  adv_estimator: grpo                   # MUST be "grpo" for ARPO
  disable_kl: true                      # No KL divergence
  use_kl_loss: false
  kl_coef: 0
  enable_replay: true                   # Experience replay (THE key ARPO feature)

# === ACTOR (Policy Model) ===
worker:
  actor:
    global_batch_size: 8                # Total batch size across all GPUs
    micro_batch_size_per_device_for_update: 1
    ppo_epochs: 1                       # PPO update epochs per batch
    clip_ratio_low: 0.2                 # Asymmetric clipping lower bound
    clip_ratio_high: 0.3                # Asymmetric clipping upper bound
    max_grad_norm: 1.0                  # Gradient clipping
    padding_free: true                  # Memory-efficient attention (GPU only)
    model:
      model_path: UI-TARS-1.5-7B       # HuggingFace model ID
      enable_gradient_checkpointing: true
      trust_remote_code: true
      freeze_vision_tower: false        # Fine-tune vision encoder too
    optim:
      lr: 1.0e-6                        # Learning rate
      weight_decay: 1.0e-2
      strategy: adamw_bf16              # bf16 optimizer for GPU
      lr_warmup_ratio: 0.05
    fsdp:
      enable_full_shard: true           # FSDP sharding
      enable_cpu_offload: false         # Enable if memory constrained
      enable_rank0_init: true           # false reduces peak memory
    offload:
      offload_params: false             # CPU offload parameters
      offload_optimizer: false          # CPU offload optimizer states

# === ROLLOUT (vLLM Generation) ===
  rollout:
    temperature: 1.0                    # High exploration
    n: 8                                # Rollouts per task (GRPO needs n > 1)
    gpu_memory_utilization: 0.6         # vLLM GPU memory fraction
    tensor_parallel_size: 1             # TP parallelism
    limit_images: 15                    # Max screenshots per trajectory
    max_num_batched_tokens: 128000      # vLLM batch token limit
    val_override_config:
      temperature: 0.6                  # Lower temperature for eval
      n: 1

# === ENVIRONMENT ===
env:
  num_envs: 128                         # Number of parallel environments
  max_steps: 15                         # Max steps per trajectory
  remote_server_url: ""                 # Set for remote env mode

# === TRAINER ===
trainer:
  total_episodes: 15                    # Training epochs
  n_gpus_per_node: 8
  nnodes: 2                             # Number of nodes
  val_freq: 8                           # Validate every N episodes
  val_before_train: true
  save_freq: 8                          # Save checkpoint every N episodes
  save_limit: 3                         # Max checkpoints to keep
  logger: ["console", "wandb"]
  project_name: arpo-training
  experiment_name: my_experiment
```

### 9.3 Memory Optimization Options

If you hit OOM errors, progressively enable these:

```yaml
# Level 1: Reduce batch/token sizes
data:
  max_prompt_length: 2048               # Down from 64000
  max_response_length: 512              # Down from 8192
worker:
  actor:
    global_batch_size: 4                # Down from 8
  rollout:
    gpu_memory_utilization: 0.3         # Down from 0.6
    max_num_batched_tokens: 10240       # Down from 128000
    limit_images: 10                    # Down from 15

# Level 2: Enable CPU offloading
worker:
  actor:
    fsdp:
      enable_cpu_offload: true
      enable_rank0_init: false
    offload:
      offload_params: true
      offload_optimizer: true

# Level 3: Reduce rollouts
worker:
  rollout:
    n: 2                                # Down from 8 (minimum for GRPO)
```

---

## 10. Running Training

### 10.1 Smoke Test (verify everything works)

**With local Docker environments** (no remote env needed):
```bash
# On GPU cluster (4+ GPUs)
cd arpo_remote_env
python -m verl.trainer.main config=configs/smoke_4gpu.yaml
```

**With remote environment**:
```bash
# On AWS (Terminal 1):
python scripts/remote_env_server.py

# On GPU cluster (Terminal 2):
# First edit configs/smoke_remote_env.yaml: set env.remote_server_url
python -m verl.trainer.main config=configs/smoke_remote_env.yaml
```

### 10.2 Subset Training (32 tasks, 1 node)

```bash
# Uses examples/osworld_subset32.sh as reference
python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=evaluation_examples/test_success_middle_difficult.json \
    data.val_files=evaluation_examples/test_success_middle_difficult.json \
    data.max_prompt_length=64000 \
    data.max_response_length=8192 \
    data.rollout_batch_size=2 \
    worker.actor.global_batch_size=1 \
    worker.actor.model.model_path=UI-TARS-1.5-7B \
    worker.rollout.temperature=1.0 \
    worker.rollout.n=8 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.max_num_batched_tokens=128000 \
    algorithm.disable_kl=True \
    algorithm.kl_coef=0 \
    algorithm.enable_replay=True \
    env.num_envs=16 \
    env.max_steps=15 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_episodes=15
```

### 10.3 Full ARPO (128 tasks, 2 nodes) - Paper Reproduction

```bash
# Uses examples/osworld_full_arpo.sh as reference
python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=evaluation_examples/test_success_uitars1.5_wo_impossible.json \
    data.val_files=evaluation_examples/test_success_uitars1.5_wo_impossible.json \
    data.max_prompt_length=64000 \
    data.max_response_length=8192 \
    data.rollout_batch_size=16 \
    worker.actor.global_batch_size=8 \
    worker.actor.model.model_path=UI-TARS-1.5-7B \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.padding_free=true \
    worker.rollout.temperature=1.0 \
    worker.rollout.n=8 \
    worker.rollout.limit_images=15 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.max_num_batched_tokens=128000 \
    algorithm.disable_kl=True \
    algorithm.kl_coef=0 \
    env.num_envs=128 \
    env.max_steps=15 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.total_episodes=15 \
    trainer.save_freq=8 \
    trainer.val_freq=8
```

Note: The full ARPO script (`osworld_full_arpo.sh`) does NOT have `algorithm.enable_replay=True` explicitly, which suggests replay may be handled differently in the full setup or was added later. The subset script does include it.

### 10.4 Monitoring Training

**Console output**: Shows per-epoch metrics (reward, loss, success rate)

**Weights & Biases**: If configured (`trainer.logger: ["console", "wandb"]`), view at https://wandb.ai

**Key metrics to watch**:
- `train/avg_reward`: Should increase over epochs
- `train/success_rate`: Should improve
- `train/replay_buffer_size`: Number of cached successes
- `train/policy_loss`: Should decrease
- `val/avg_reward`: Validation performance

---

## 11. Evaluation & Checkpoints

### 11.1 Checkpoints

Checkpoints are saved as FSDP-sharded state dicts:
```
checkpoints/
  global_step_100/
    rank_0/
      model_state_dict.pt
      optimizer_state_dict.pt
    rank_1/
      ...
    hf_config/
      config.json
```

### 11.2 Merging Checkpoints

To get a single HuggingFace model from sharded checkpoints:

```bash
python scripts/model_merger.py \
    --checkpoint_dir checkpoints/global_step_100 \
    --output_dir merged_model/
```

### 11.3 Evaluation

Use the OSWorld evaluation scripts:

```bash
# Single-env evaluation
python osworld_patches/run_uitars.py \
    --model_path merged_model/ \
    --task_file evaluation_examples/test_all.json

# Multi-env evaluation
python osworld_patches/run_multienv_uitars.py \
    --model_path merged_model/ \
    --task_file evaluation_examples/test_all.json \
    --num_envs 8
```

---

## 12. Troubleshooting

### Common Issues

**"Docker is not available"** on env server
- Install Docker and ensure daemon is running: `sudo systemctl start docker`
- Add your user to docker group: `sudo usermod -aG docker $USER`

**CUDA OOM on GPU cluster**
- Reduce `max_num_batched_tokens`, `gpu_memory_utilization`, batch sizes
- Enable CPU offloading (see Section 9.3)
- Reduce `limit_images` and context lengths

**Ray connection errors**
- Start Ray: `ray start --head --num-gpus=N`
- Check: `ray status`
- Stop old instance: `ray stop`

**"Connection refused" to remote env**
- Verify env server is running: `curl http://IP:15001/health`
- Check firewall/security groups allow port 15001
- Try SSH tunnel if direct connection fails

**Slow training**
- Ensure vLLM is using GPU (not CPU)
- Check `nvidia-smi` for GPU utilization
- Increase `num_envs` for more parallel rollouts
- Use `padding_free: true` and `bf16` precision

**Model download fails**
- Set `HF_TOKEN` environment variable if model is gated
- Or download manually and set `model_path` to local directory

### What You Still Need

To reproduce ARPO fully from this codebase, you need:

1. **GPU cluster access** with 8-16 A100/H100 GPUs
2. **Docker-capable machines** for OSWorld environments (AWS EC2 works well)
3. **The filtered task lists** (`test_success_uitars1.5_wo_impossible.json` for 128 tasks) - you may need to generate these by running baseline evaluation
4. **The UI-TARS-1.5-7B model** (for paper-matching results) or UI-TARS-2B-SFT (for smaller experiments)
5. **Network connectivity** between GPU cluster and env servers
6. **Weights & Biases account** (optional, for experiment tracking)

### What This Codebase Already Provides

- Complete VERL training framework with ARPO modifications
- Experience replay buffer implementation
- GRPO advantage estimation with all the paper's hyperparameters
- Remote environment server for distributed rollout
- GUI agent action parsing (UI-TARS format)
- FSDP distributed training with vLLM inference
- Multiple configuration presets for different hardware
- Checkpoint management and model merging tools
- OSWorld integration (as git submodule)

---

## 13. Known Issues: Training Stuck / Loss Not Decreasing

> **Context**: When running with the smoke remote env configs (`smoke_remote_env.yaml`),
> training gets stuck - loss fails to decrease, the agent fails every task, and even the
> first evaluation produces zero reward. This section documents the root causes found
> via code-level investigation.

### 13.1 Do We Need SFT First?

**No.** The paper does NOT perform additional SFT. The training pipeline is:

```
Qwen2.5-VL (base) --[ByteDance SFT]--> UI-TARS-1.5 (SFT'd) --[ARPO (RL)]--> improved model
                    ^^^^^^^^^^^^^^^^^^^                         ^^^^^^^^^^^^^^^
                    Done by ByteDance                          Done by the paper
                    before releasing                           (what this repo does)
```

- **UI-TARS-1.5-7B** (paper model): Already SFT'd by ByteDance on large-scale GUI data
- **ByteDance-Seed/UI-TARS-2B-SFT** (this repo's default): Also already SFT'd (the "-SFT" suffix)
- The ARPO paper authors took UI-TARS-1.5 off the shelf and applied RL directly
- Evidence: `examples/osworld_full_arpo.sh` line 3 uses `MODEL_PATH=UI-TARS-1.5-7B` with no SFT step
- The codebase has no SFT training code - `main.py` goes straight to `RayPPOTrainer.fit()`

> **IMPORTANT: Do NOT confuse with "ARPO-SFT-54K"**
>
> There are **two completely unrelated papers** that both use the "ARPO" acronym:
>
> | | This Repo's ARPO (JIA-Lab) | The Other ARPO (dongguanting) |
> |---|---|---|
> | **Full Name** | Agentic **Replay** Policy Optimization | Agentic **Reinforced** Policy Optimization |
> | **Paper** | arxiv 2505.16282 | arxiv 2507.19849 |
> | **Domain** | GUI agents (OSWorld, desktop automation) | Multi-turn LLM tool-use agents |
> | **Base Model** | UI-TARS-1.5-7B (ByteDance, already SFT'd) | Different models |
> | **SFT Dataset** | None — uses ByteDance's pre-SFT'd model | ARPO-SFT-54K (HuggingFace) |
> | **Key Innovation** | Experience replay buffer for GRPO | Entropy-based adaptive rollout |
>
> The `dongguanting/ARPO-SFT-54K` dataset on HuggingFace is from the **other** paper and is
> **not relevant** to this codebase. This repo implements JIA-Lab's ARPO, which requires no
> additional SFT — the base UI-TARS model was already SFT'd by ByteDance before release.

**The problem is NOT missing SFT.** The problems are the aggressive config reductions and the
2B vs 7B model gap, detailed below.

### 13.2 Root Cause 1 (CRITICAL): Token Truncation Destroys Training Signal

The smoke remote env config uses `max_prompt_length: 2048`. The paper uses `64000`.

**Token budget for a single rollout step:**

| Component | Approximate Tokens |
|-----------|--------------------|
| System prompt (`gui_agent.py:19-45`) | ~150-200 |
| Task instruction | ~30-80 |
| One 1920x1080 screenshot (Qwen2-VL vision tokens at `max_pixels: 2116800`) | ~1200-1500 |
| Previous action text (per turn) | ~50-100 |

After just **1 screenshot + system prompt**, you're at ~1500-1800 tokens. After step 2
(second screenshot), the total exceeds 2048.

**What happens with right truncation** (`torch_functional.py:194-196`):

```python
elif truncation == "right":
    input_ids = input_ids[..., :max_length]  # keeps beginning, cuts end
```

The sequence is ordered: `[system prompt, user instruction, screenshot, assistant response, ...]`.
Right truncation **cuts from the end**, which means:
- For GRPO training (`OSWorldGRPODataset` in `osworld.py:500-509`): the assistant's response
  tokens (the action labels to train on) are at the end and get **truncated away**
- The `labels` tensor is truncated too: no response tokens = no training signal
- The model trains on empty/incomplete responses -> **zero meaningful gradient**

**For rollout** (`prepare_vllm_inputs_full` in `ray_trainer.py:675`): the `fast_rollout=True`
path uses empty placeholder tensors, so truncation doesn't directly block generation. But the
accumulating message history sent to vLLM grows beyond what the model can handle coherently.

After 2-3 environment steps, the model cannot see the current screenshot because the context
is dominated by old history, producing incoherent actions.

### 13.3 Root Cause 2 (CRITICAL): Empty Replay Buffer = Cold Start Problem

The replay buffer starts **completely empty** (`replay_buffer.py:11-17`):

```python
def __init__(self, json_path, buffer_size):
    self.pos_dataset = defaultdict(list)
    # JSON loading code is COMMENTED OUT (lines 16-25)
```

The injection logic in `ray_trainer.py:784-792`:

```python
if cur_reward_std < 0.05 and cur_reward_mean < 0.2:  # all failed
    pos_batch = self.replay.get_pos(task_id, num_samples=1)
    # Returns empty DataProto because pos_dataset is empty
else:
    pos_batch = []

if len(pos_batch) > 0:  # Always False when buffer is empty
    final_batch.append(pos_batch)  # Never reached
```

**Chicken-and-egg problem**: The replay buffer needs successful trajectories to inject, but
without injection there may never be successes (especially with the 2B model on hard tasks).
The buffer only gets populated via `update_replay_buffer_batch` at line 799, which requires
at least one `eval_result > 0.1` first.

### 13.4 Root Cause 3 (CRITICAL): GRPO Degeneracy With All-Zero Rewards

With `rollout.n: 2` (smoke config) and both rollouts failing (`core_algos.py:171-183`):

```python
# rewards = [0.0, 0.0]
id2mean[idx] = torch.mean(sample_scores)   # = 0
id2std[idx] = torch.std(sample_scores)      # = 0

# Then for each sample:
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + 1e-6)
# = (0 - 0) / (0 + 1e-6) = 0
```

**Advantages are zero -> policy loss is zero -> no gradient -> no learning.**

The paper uses `n=8` for good reason: with 8 rollouts there's a much higher chance at least
one succeeds, creating non-zero variance in rewards.

### 13.5 Root Cause 4 (HIGH): 2B Model Too Weak for Selected Tasks

The paper's task filtering was done with the **7B** model:
- Evaluate UI-TARS-1.5-7B on all 369 OSWorld tasks (16 rollouts each)
- Keep tasks where 7B succeeds at least once -> 128 "trainable" tasks

The 2B model has significantly less capacity:
- Much lower baseline success rate on OSWorld
- Many of the 128 tasks filtered for 7B are likely **impossible for 2B**
- Even the 4 smoke test Chrome tasks may be too difficult for the 2B model

If the model cannot solve ANY task even once, the entire ARPO mechanism breaks:
no successes -> empty replay buffer -> zero advantages -> no learning.

### 13.6 Root Cause 5 (HIGH): vLLM Memory Starvation

`smoke_remote_env.yaml` sets `gpu_memory_utilization: 0.05`. This gives vLLM only **5% of
GPU memory** for its KV cache. On a 40GB A100, that's ~2GB. For multi-turn conversations
with vision tokens, this severely limits:
- Maximum sequence length vLLM can process
- Batch processing capability
- The model may silently produce garbage or very short responses

### 13.7 Comparison: Smoke Config vs Paper Config

| Parameter | `smoke_remote_env.yaml` | Paper (Full) | Reduction |
|-----------|------------------------|--------------|-----------|
| `max_prompt_length` | 2048 | 64000 | **97%** |
| `max_response_length` | 512 | 8192 | **94%** |
| `max_num_batched_tokens` | 10240 | 128000 | **92%** |
| `gpu_memory_utilization` | 0.05 | 0.6 | **92%** |
| `rollout.n` | 2 | 8 | **75%** |
| `num_envs` | 1 | 128 | **99%** |
| `limit_images` | 10 | 15 | 33% |
| Model | 2B | 7B | **71% fewer params** |

The smoke configs were aggressively reduced to avoid OOM, but this crossed the threshold
where training becomes non-functional.

### 13.8 Why Even the First Evaluation Fails

Multiple compounding factors:

1. **Truncated context**: After system prompt + 1 screenshot, there are barely any tokens
   left for the action. After 2-3 steps, the model literally cannot see the current screenshot.

2. **2B model weakness**: Much weaker than the 7B model the paper uses. Near-zero baseline
   success on complex Chrome tasks.

3. **vLLM memory starvation**: 5% GPU memory -> KV cache too small for multi-turn + vision
   sequences. Model may generate nonsensical outputs.

4. **Evaluation uses temperature 0.5** (`val_override_config.temperature: 0.5` in
   `smoke_remote_env.yaml:77`): Combined with a confused model, outputs may be repetitive
   or degenerate.

### 13.9 Recommended Fixes

**Priority 1: Increase token budgets** (most critical fix):

```yaml
data:
  max_prompt_length: 16384    # minimum viable; 32000+ preferred
  max_response_length: 2048   # enough for chain-of-thought + action
worker:
  rollout:
    max_num_batched_tokens: 65536   # or higher
    gpu_memory_utilization: 0.4     # at minimum
```

**Priority 2: Increase rollout.n** for GRPO to function:

```yaml
worker:
  rollout:
    n: 4    # ideally 8; minimum 4 for meaningful group variance
```

**Priority 3: Switch to the 7B model** if GPU memory allows:

```yaml
worker:
  actor:
    model:
      model_path: UI-TARS-1.5-7B
```

**Priority 4: Pre-populate the replay buffer** - uncomment the JSON loading code in
`replay_buffer.py:16-25` and provide a JSON file with seed successful trajectories. Or
run a few evaluation passes first, save any successes, and use them to bootstrap the buffer.

**Priority 5: Use easier tasks or filter for 2B** - if staying with 2B, run baseline
evaluation to find tasks where 2B can sometimes succeed, and train only on those.

**Priority 6: If OOM forces reductions**, reduce the number of parallel tasks/envs rather
than starving each task of tokens. It is better to train on 1 task with full 64K context
than 4 tasks with 2K context that produces zero signal.

### 13.10 Minimum Viable Config Adjustments

If you must use the 2B model and have limited GPU memory, here is the minimum viable
configuration that should at least produce non-zero training signal:

```yaml
data:
  max_prompt_length: 16384
  max_response_length: 2048

worker:
  actor:
    global_batch_size: 4
    fsdp:
      enable_cpu_offload: true
      enable_rank0_init: false
    offload:
      offload_params: true
      offload_optimizer: true
  rollout:
    n: 4
    gpu_memory_utilization: 0.3
    max_num_batched_tokens: 32768
    limit_images: 10

env:
  num_envs: 1
```

This trades throughput for correctness: fewer parallel tasks, but each task gets enough
context to actually produce meaningful actions and training gradients.

### 13.11 Diagnostic Checklist

When debugging stuck training, check these in order:

1. **Check reward logs**: Look for `num_invalid_group` in console output. If it equals
   the total number of groups, ALL groups have zero reward variance -> no learning.

2. **Check `reward_stds_list`**: Printed at `ray_trainer.py:985`. If all values are 0.0
   or < 0.01, GRPO has zero advantages everywhere.

3. **Check `Evaluation results` print**: At `ray_trainer.py:1010`. If all eval_results
   are 0.0, the model never succeeds at any task.

4. **Check format_rewards**: Also printed at line 1010. If many are -1.0, the model is
   generating unparseable actions (syntax errors in its output).

5. **Check `prepare_vllm_inputs_time`**: If this is very fast but generation is slow,
   vLLM may be memory-starved and swapping.

6. **Check actual response text**: Decode `action_batch_output.batch['responses']` and
   inspect whether the model produces valid `Thought: ... Action: ...` format.

---

## Appendix: Quick Command Reference

```bash
# Clone
git clone https://github.com/hanshengzhu0001/arpo_remote_env.git
cd arpo_remote_env && git submodule update --init --recursive

# Install (GPU cluster)
pip install -r requirements.txt && pip install vllm flash-attn --no-build-isolation

# Install (env server)
pip install -r requirements.txt && pip install -r OSWorld/requirements.txt

# Start env server (on AWS/Docker machine)
python scripts/remote_env_server.py

# Health check
curl http://ENV_SERVER_IP:15001/health

# Smoke test (local envs, 4 GPUs)
python -m verl.trainer.main config=configs/smoke_4gpu.yaml

# Smoke test (remote env)
python -m verl.trainer.main config=configs/smoke_remote_env.yaml

# Full training (paper reproduction)
bash examples/osworld_full_arpo.sh

# Merge checkpoints
python scripts/model_merger.py --checkpoint_dir CKPT_DIR --output_dir OUTPUT_DIR

# Ray management
ray start --head --num-gpus=8
ray status
ray stop
```
