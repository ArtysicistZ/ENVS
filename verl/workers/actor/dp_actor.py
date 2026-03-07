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
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils.multimodal_sp import register_multimodal_sp_input_hook, use_multimodal_sp_input_hook
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        # TorchDynamo/torch.compile can produce fake-tensor shape mismatches for cross_entropy
        # in dynamic remote-env / GRPO batches (e.g. different symbolic batch dims for logits vs labels).
        # For stability in PPO training (including smoke tests), always use the eager implementation.
        self.log_probs_from_logits = VF.log_probs_from_logits

    def _maybe_empty_cache(self) -> None:
        if self.config.empty_cache_policy == "aggressive":
            torch.cuda.empty_cache()

    def _select_pixel_values_for_sp_rank(
        self,
        input_ids_rmpad: torch.Tensor,
        multi_modal_inputs: dict,
        sp_size: int,
    ) -> dict:
        """Select pixel_values/image_grid_thw for images whose tokens fall in this SP rank's slice.

        With Ulysses SP, each rank processes only [total_nnz / sp_size] tokens. The vision encoder
        must receive only the pixel_values for images present in that rank's slice — otherwise
        masked_scatter assigns image 1's features to image 3's token positions.

        NOTE: image_grid_thw stores PRE-merge dimensions (t, h, w). Each image contributes
        t*(h//merge_size)*(w//merge_size) tokens in input_ids (post-merge). Rather than relying on
        image_grid_thw to count post-merge tokens, we detect contiguous blocks of image_token_id
        directly — this is robust to any merge_size.
        pixel_values rows per image = t*h*w (pre-merge), matching image_grid_thw as-is.
        """
        import torch.distributed as dist

        if not multi_modal_inputs or "pixel_values" not in multi_modal_inputs:
            return multi_modal_inputs

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        pixel_values = multi_modal_inputs["pixel_values"]
        if image_grid_thw is None or len(image_grid_thw) == 0:
            return multi_modal_inputs

        sp_rank = dist.get_rank() % sp_size
        total_nnz = input_ids_rmpad.shape[1]
        pad_size = (sp_size - total_nnz % sp_size) % sp_size
        slice_size = (total_nnz + pad_size) // sp_size
        rank_start = sp_rank * slice_size
        rank_end = min((sp_rank + 1) * slice_size, total_nnz)

        # Get image_token_id from model config; fall back to Qwen2-VL/Qwen2.5-VL default.
        try:
            image_token_id = self.actor_module.config.image_token_id
        except AttributeError:
            image_token_id = 151655

        flat_ids = input_ids_rmpad.squeeze(0)  # [total_nnz]
        is_image = flat_ids == image_token_id

        if not is_image.any():
            return multi_modal_inputs

        # Find contiguous blocks of image_token_id — one block per image.
        # Avoids using image_grid_thw to count post-merge tokens (image_grid_thw is pre-merge).
        false_pad = torch.zeros(1, dtype=torch.bool, device=is_image.device)
        padded = torch.cat([false_pad, is_image, false_pad])
        block_starts = (padded[1:] & ~padded[:-1]).nonzero(as_tuple=True)[0]
        block_ends = (~padded[1:] & padded[:-1]).nonzero(as_tuple=True)[0] - 1

        num_images = len(image_grid_thw)
        if len(block_starts) != num_images:
            # Block count mismatch — safe fallback: pass all pixel_values unchanged.
            return multi_modal_inputs

        # pixel_values rows per image = t*h*w (pre-merge from image_grid_thw).
        pv_per_image = [
            int(image_grid_thw[i, 0] * image_grid_thw[i, 1] * image_grid_thw[i, 2])
            for i in range(num_images)
        ]

        # Which image blocks overlap with this rank's slice [rank_start, rank_end)?
        rank_image_indices = [
            i for i in range(num_images)
            if int(block_starts[i]) < rank_end and int(block_ends[i]) >= rank_start
        ]

        if len(rank_image_indices) == num_images:
            return multi_modal_inputs  # all images in this rank — no filtering needed

        if len(rank_image_indices) == 0:
            # No images in this slice; remove pixel_values so vision encoder is skipped.
            result = dict(multi_modal_inputs)
            result.pop("pixel_values", None)
            result.pop("image_grid_thw", None)
            return result

        # Build filtered pixel_values and image_grid_thw for this rank's images.
        rank_image_set = set(rank_image_indices)
        pv_chunks = []
        pv_offset = 0
        for i, n_pv in enumerate(pv_per_image):
            if i in rank_image_set:
                pv_chunks.append(pixel_values[pv_offset : pv_offset + n_pv])
            pv_offset += n_pv

        rank_pixel_values = torch.cat(pv_chunks, dim=0)
        rank_grid_thw = image_grid_thw[rank_image_indices]
        return {**multi_modal_inputs, "pixel_values": rank_pixel_values, "image_grid_thw": rank_grid_thw}

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            _sp_hook = None
            if self.config.ulysses_sequence_parallel_size > 1:
                sp_size = self.config.ulysses_sequence_parallel_size
                if use_multimodal_sp_input_hook(multi_modal_inputs):
                    _sp_hook = register_multimodal_sp_input_hook(
                        self.actor_module,
                        input_ids_rmpad,
                        multi_modal_inputs,
                        sp_size,
                    )

                # Slice position_ids and label tokens per rank (same for both paths)
                _, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=sp_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, sp_size
                )
                # For text-only SP: also slice input_ids
                if not use_multimodal_sp_input_hook(multi_modal_inputs):
                    input_ids_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, None, sp_size
                    )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            try:
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
            finally:
                if _sp_hook is not None:
                    _sp_hook.remove()
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            # In multi-turn ARPO, responses = input_ids[:, 1:] so response_length ≈ seqlen-1.
            # logits_to_keep=response_length+1 would keep the full sequence (useless), so we omit it.
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits  # [bsz, seqlen, vocab_size]
            del output  # release model output object; logits base tensor stays alive via grad_fn
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)

            # Memory-efficient log_prob: log_prob(label) = logit[label] - logsumexp(logits over vocab).
            # F.cross_entropy internally allocates a full [bsz, response_length, vocab] log_softmax
            # intermediate (16+ GiB for 45K-token multi-turn sequences), causing OOM.
            # torch.logsumexp reduces over vocab dim in a streaming pass with O(bsz*response_length)
            # output — no 16 GiB intermediate. Numerically identical to cross_entropy. Gradients flow.
            logsumexp = torch.logsumexp(logits, dim=-1)  # [bsz, response_length]
            gathered = logits.gather(-1, responses.unsqueeze(-1)).squeeze(-1)  # [bsz, response_length]
            del logits  # free the ~16 GiB logits tensor before computing log_probs
            log_probs = gathered - logsumexp  # [bsz, response_length]

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
            # Ensure optimizer states are on the same device as parameters before step
            # When FSDP CPU offload is enabled, parameters are moved to GPU during forward/backward,
            # but optimizer states might have been loaded to GPU separately, causing device mismatch
            if self.actor_optimizer is not None:
                # Synchronize optimizer state device with parameter device
                # FSDP manages parameter placement, so we ensure optimizer states match
                for param_group in self.actor_optimizer.param_groups:
                    for param in param_group["params"]:
                        if param.grad is not None:
                            param_device = param.device
                            state = self.actor_optimizer.state[param]
                            for key, value in state.items():
                                if isinstance(value, torch.Tensor) and value.device != param_device:
                                    # Move optimizer state tensor to match parameter device
                                    state[key] = value.to(param_device, non_blocking=True)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=2)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "response_mask"]
        # select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # NOTE: In remote smoke-test settings we can have very small batches (e.g. len(data)==1) and
        # world_size > len(data), which can lead to an effective global_batch_size_per_device of 0.
        # That causes DataProto.split(chunk=0) to raise ZeroDivisionError. For these tiny batches,
        # just treat the whole batch as a single mini-batch.
        data_selected = data.select(select_keys, non_tensor_select_keys)
        chunks = self.config.global_batch_size_per_device
        if chunks <= 0 or len(data_selected) <= chunks:
            mini_batches = [data_selected]
        else:
            mini_batches = data_selected.split(chunks)

        print("data size: ", len(data), len(mini_batches))
        print('Global batch Size per device:', self.config.global_batch_size_per_device)
        print('micro batch size per device for update:', self.config.micro_batch_size_per_device_for_update)
        if self.config.global_batch_size_per_device > 0:
            print('Gradient accumulation:', self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=2)

            for mini_batch in mini_batches:
                micro_size = self.config.micro_batch_size_per_device_for_update
                gradient_accumulation = (
                    max(1, self.config.global_batch_size_per_device)
                    // max(1, micro_size)
                )
                micro_batches = mini_batch.split(micro_size)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=3)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    # response_mask and advantages are already shifted to (seqlen-1) in the trainer
                    # (responses = input_ids[:, 1:], response_mask = (labels != -100)[:, 1:])
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    self._maybe_empty_cache()
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                    # Align shapes when log_probs and response_mask differ (e.g. variable-length VL batches / padding)
                    seq_len = min(log_probs.size(1), response_mask.size(1))
                    if log_probs.size(1) != response_mask.size(1):
                        log_probs = log_probs[:, :seq_len].contiguous()
                        response_mask = response_mask[:, :seq_len].contiguous()
                        old_log_probs = old_log_probs[:, :seq_len].contiguous()
                        advantages = advantages[:, :seq_len].contiguous()
                    # Guard against all-zero response_mask (no valid tokens) => skip to avoid NaN
                    if response_mask.sum() == 0:
                        continue
                    entropy_loss = -VF.masked_mean(log_probs, response_mask)

                    pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl = core_algos.compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                    )
                    if "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"][:, :seq_len].contiguous()
                        # compute kl loss
                        kld = core_algos.compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = VF.masked_mean(kld, response_mask)
                        pg_loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef

                    loss = pg_loss / gradient_accumulation
                    print(f'pg_loss: {pg_loss}')
                    self._maybe_empty_cache()
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
