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

from typing import Any, Dict, List

import numpy as np
import torch

from ..protocol import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> Dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)

    # prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    # response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()
    prompt_mask = (torch.logical_and(batch.batch["attention_mask"], batch.batch["labels"] == -100)).bool()
    # Use the batch's pre-computed response_mask (shifted to match responses/advantages)
    # rather than recomputing from labels (which has full seqlen, not shifted seqlen-1).
    response_mask = batch.batch["response_mask"].bool() if "response_mask" in batch.batch.keys() else (batch.batch["labels"] != -100).bool()

    max_prompt_length = prompt_mask.size(-1)
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()
    num_images = (batch.batch["input_ids"] == 151655).bool().sum(-1).float() // 2691  # image_pad
    # Avoid div-by-zero when batch has no image tokens (e.g. empty rollouts after failed reset)
    num_images = torch.clamp(num_images, min=1.0)
    response_length = response_length / num_images  # average response length per action

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    # For tiny smoke-test batches it is possible that response_mask is all zeros,
    # which makes valid_adv/valid_returns (and valid_values) empty tensors.
    # Directly calling max/min/mean on empty tensors raises runtime errors, so we
    # guard these statistics and fall back to 0.0 when there is no valid data.
    def _safe_stats(x: torch.Tensor) -> tuple[float, float, float]:
        if x.numel() == 0:
            return 0.0, 0.0, 0.0
        return (
            torch.mean(x).detach().item(),
            torch.max(x).detach().item(),
            torch.min(x).detach().item(),
        )

    adv_mean, adv_max, adv_min = _safe_stats(valid_adv)
    ret_mean, ret_max, ret_min = _safe_stats(valid_returns)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        val_mean, val_max, val_min = _safe_stats(valid_values)
        if valid_returns.numel() > 0 and valid_values.numel() > 0:
            return_diff_var = torch.var(valid_returns - valid_values)
            return_var = torch.var(valid_returns)
            vf_explained_var = (1.0 - return_diff_var / (return_var + 1e-5)).detach().item()
        else:
            vf_explained_var = 0.0

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": adv_mean,
        "critic/advantages/max": adv_max,
        "critic/advantages/min": adv_min,
        # returns
        "critic/returns/mean": ret_mean,
        "critic/returns/max": ret_max,
        "critic/returns/min": ret_min,
        **(
            {
                # values
                "critic/values/mean": val_mean,
                "critic/values/max": val_max,
                "critic/values/min": val_min,
                # vf explained var
                "critic/vf_explained_var": vf_explained_var,
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/num_images": torch.mean(num_images).detach().item(),
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    timing_metrics: Dict[str, Any] = {f"timing_s/{name}": value for name, value in timing_raw.items()}
    for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys()):
        denom = num_tokens_of_section[name]
        if denom > 0:
            timing_metrics[f"timing_per_token_ms/{name}"] = timing_raw[name] * 1000 / denom
    return timing_metrics


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }
