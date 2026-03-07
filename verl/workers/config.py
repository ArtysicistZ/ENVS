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
ActorRolloutRef config
"""

import copy
from dataclasses import dataclass, field

from .actor import ActorConfig, FSDPConfig, ModelConfig, OptimConfig, RefConfig
from .critic import CriticConfig
from .reward import RewardConfig
from .rollout import RolloutConfig


__all__ = [
    "ActorConfig",
    "CriticConfig",
    "FSDPConfig",
    "ModelConfig",
    "OptimConfig",
    "RefConfig",
    "RewardConfig",
    "RolloutConfig",
    "WorkerConfig",
]


@dataclass
class WorkerConfig:
    hybrid_engine: bool = True
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    def post_init(self):
        self.ref.micro_batch_size_per_device_for_experience = self.actor.micro_batch_size_per_device_for_experience
        self.ref.padding_free = self.actor.padding_free
        self.ref.ulysses_sequence_parallel_size = self.actor.ulysses_sequence_parallel_size
        self.ref.use_torch_compile = self.actor.use_torch_compile

        # If the critic model path is omitted, treat the critic as an actor-shaped value model by
        # default so critic-based training inherits the same model family and low-VRAM SP settings.
        if self.critic.model.model_path is None:
            self.critic.strategy = self.actor.strategy
            self.critic.global_batch_size = self.actor.global_batch_size
            self.critic.micro_batch_size_per_device_for_update = self.actor.micro_batch_size_per_device_for_update
            self.critic.micro_batch_size_per_device_for_experience = (
                self.actor.micro_batch_size_per_device_for_experience
            )
            self.critic.max_grad_norm = self.actor.max_grad_norm
            self.critic.ppo_epochs = self.actor.ppo_epochs
            self.critic.padding_free = self.actor.padding_free
            self.critic.ulysses_sequence_parallel_size = self.actor.ulysses_sequence_parallel_size
            self.critic.model = copy.deepcopy(self.actor.model)
            if self.critic.model.tokenizer_path is None:
                self.critic.model.tokenizer_path = self.critic.model.model_path
            self.critic.optim = copy.deepcopy(self.actor.optim)
            self.critic.fsdp = copy.deepcopy(self.actor.fsdp)
            self.critic.offload = copy.deepcopy(self.actor.offload)
