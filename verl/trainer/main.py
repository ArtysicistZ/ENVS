# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.

python -m verl.trainer.main config=configs/smoke_remote_env_8gpu_a100.yaml

python -m verl.trainer.main config=configs/smoke_remote_env_16gpu_a100.yaml

python -m verl.trainer.main config=configs/8gpu_a100_n8_env16.yaml

python -m verl.trainer.main config=configs/8gpu_a100_n8_env32.yaml
python -m verl.trainer.main config=configs/16gpu_a100_n8_env48.yaml

sudo -E /home/kevinzyz/yincheng/arpo/.venv/bin/python -m verl.trainer.main config=configs/8gpu_a100_n8_env32.yaml

sudo -E /home/kevinzyz/yincheng/arpo/.venv/bin/python -m verl.trainer.main config=configs/inference_306tasks_n8_env64.yaml

bash scripts/training/run_smoke_remote_env_8gpu_a100.sh

bash scripts/training/run_smoke_remote_env_8gpu_a100_paper_candidate.sh
"""

import json
import os

# Load .env from repo root so OPENAI_API_KEY etc. are available (e.g. for generation)
def _load_dotenv():
    try:
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
        if os.path.isfile(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

_load_dotenv()

# Add OSWorld to path so desktop_env is importable (git submodule)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_osworld = os.path.join(_repo_root, "OSWorld")
if os.path.isdir(_osworld):
    import sys
    if _osworld not in sys.path:
        sys.path.insert(0, _osworld)

import ray
from ray.exceptions import RayActorError, RayTaskError
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import CustomRewardManager
from .config import PPOConfig
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        config.deep_post_init()
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        reward_fn = CustomRewardManager(tokenizer=tokenizer, config=config.worker.reward)
        val_reward_fn = CustomRewardManager(tokenizer=tokenizer, config=config.worker.reward)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        # Keep trainer on the actor instance so Python GC doesn't trigger
        # vLLM/CUDA __del__ cleanup (which can segfault) when run() returns.
        self._trainer = trainer
        trainer.init_workers()
        trainer.fit()
        return "completed"


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config = OmegaConf.to_object(ppo_config)

    # Resolve data paths to absolute so Ray workers (possibly different cwd) can find files
    if getattr(ppo_config.data, "train_files", None) and not os.path.isabs(ppo_config.data.train_files):
        ppo_config.data.train_files = os.path.abspath(ppo_config.data.train_files)
    if getattr(ppo_config.data, "val_files", None) and not os.path.isabs(ppo_config.data.val_files):
        ppo_config.data.val_files = os.path.abspath(ppo_config.data.val_files)

    if not ray.is_initialized():
        # For multi-node: connect to existing Ray cluster via address="auto".
        # For single-node: address="local" starts a fresh local cluster.
        if ppo_config.trainer.nnodes > 1:
            ray_address = "auto"
        else:
            # Single-node: start fresh local cluster. Clear RAY_ADDRESS to avoid
            # connecting to a stale cluster whose session dir may have been removed.
            # Only strip RAY_ADDRESS — preserve other RAY_* vars (e.g. RAY_TMPDIR
            # for /mnt-backed sessions on hosts with small /tmp).
            ray_address = "local"
            os.environ.pop("RAY_ADDRESS", None)
        ray.init(
            address=ray_address,
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
                "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            }},
        )
    
    print(ray.cluster_resources().keys())

    runner = Runner.remote()
    _exit_code = 0
    try:
        ray.get(runner.run.remote(ppo_config))
        print("Training/evaluation completed successfully.")
    except (RayActorError, RayTaskError) as e:
        # Runner crashed unexpectedly (OOM, SIGSEGV, etc.)
        _exit_code = 1
        err_str = str(e).lower()
        if "owner" in err_str and ("died" in err_str or "crashed" in err_str):
            print("Runner died because the driver/head node crashed (likely OOM).")
            print("Check logs:")
            print("  /tmp/ray/session_latest/logs/python-core-driver-*.log")
            print("  /tmp/ray/session_latest/logs/python-core-worker-*.log")
        else:
            print(f"Runner crashed: {type(e).__name__}: {e}")
        if isinstance(e, RayTaskError) and e.cause is not None:
            print(f"Cause:\n{e.cause}")
    # Exit immediately from the driver to avoid vLLM/CUDA segfault during
    # Python cleanup. os._exit() from the driver kills the entire Ray process
    # tree cleanly (no SYSTEM_ERROR messages, unlike os._exit from an actor).
    os._exit(_exit_code)


if __name__ == "__main__":
    main()
