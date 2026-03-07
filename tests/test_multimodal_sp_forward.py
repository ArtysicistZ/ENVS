import os
import sys
import types
import unittest
from types import SimpleNamespace

import torch
from torch import nn


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.utils import ulysses
import verl.workers.actor.dp_actor as dp_actor_module
import verl.workers.critic.dp_critic as dp_critic_module
from verl.utils.multimodal_sp import _locate_multimodal_inner_module
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.critic.dp_critic import DataParallelPPOCritic


IMAGE_TOKEN_ID = 99
HIDDEN_SIZE = 8
VOCAB_SIZE = 128


class FakeInnerMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.recorded_calls = []

    def get_input_embeddings(self):
        return self.embed

    def get_image_features(self, pixel_values, image_grid_thw):
        features = []
        for grid in image_grid_thw:
            t, h, w = [int(v.item()) for v in grid]
            token_count = (t * h * w) // 4
            features.append(torch.ones(token_count, HIDDEN_SIZE, device=pixel_values.device, dtype=pixel_values.dtype))
        return features

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_thw=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_features = torch.cat(self.get_image_features(pixel_values, image_grid_thw), dim=0)
                image_mask = input_ids.squeeze(0) == IMAGE_TOKEN_ID
                if int(image_mask.sum()) != image_features.shape[0]:
                    raise RuntimeError("full multimodal path saw mismatched image token and feature counts")
                inputs_embeds[0, image_mask] = image_features.to(inputs_embeds.dtype)

        seq_len = inputs_embeds.shape[1]
        pos_len = position_ids.shape[-1] if position_ids is not None else seq_len
        if seq_len != pos_len:
            raise RuntimeError(f"sequence length {seq_len} != position length {pos_len}")

        self.recorded_calls.append(
            {
                "seq_len": seq_len,
                "pos_len": pos_len,
                "used_inputs_embeds": inputs_embeds is not None,
                "pixel_values_cleared": pixel_values is None,
                "input_ids_cleared": input_ids is None,
            }
        )
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class FakeActorModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.model = FakeInnerMultimodalModel()
        self._fsdp_wrapped_module = types.SimpleNamespace(model=self.model)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        return SimpleNamespace(logits=self.lm_head(outputs.last_hidden_state))


class FakeCriticModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.model = FakeInnerMultimodalModel()
        self._fsdp_wrapped_module = types.SimpleNamespace(model=self.model)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1, bias=False)

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        return SimpleNamespace(logits=self.value_head(outputs.last_hidden_state))


class OuterWrapperWithOwnMethods(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeInnerMultimodalModel()

    # This outer wrapper intentionally exposes the same multimodal methods as the backbone.
    # The locator should still prefer `self.model`, which is the correct hook target under FSDP.
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_image_features(self, pixel_values, image_grid_thw):
        return self.model.get_image_features(pixel_values, image_grid_thw)


class FSDPStyleProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self._fsdp_wrapped_module = OuterWrapperWithOwnMethods()

    # Simulate FSDP attribute forwarding: the proxy itself also exposes the multimodal methods,
    # but the hook must still be registered on the wrapped backbone rather than this outer proxy.
    def get_input_embeddings(self):
        return self._fsdp_wrapped_module.model.get_input_embeddings()

    def get_image_features(self, pixel_values, image_grid_thw):
        return self._fsdp_wrapped_module.model.get_image_features(pixel_values, image_grid_thw)


class ShardedLookingEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulate a pre-forward FSDP/use_orig_params view that is not a materialized 2-D table yet.
        self.weight = nn.Parameter(torch.zeros(HIDDEN_SIZE))

    def forward(self, input_ids):
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], HIDDEN_SIZE, device=input_ids.device)


class StructurallyCorrectButShardedLookingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = ShardedLookingEmbedding()

    def get_input_embeddings(self):
        return self.embed

    def get_image_features(self, pixel_values, image_grid_thw):
        return [torch.zeros(1, HIDDEN_SIZE, device=pixel_values.device, dtype=pixel_values.dtype)]


def build_micro_batch():
    input_ids = torch.tensor(
        [[1, 2, 3, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).repeat(3, 1).unsqueeze(0)
    responses = input_ids[:, -4:].clone()
    pixel_values = torch.zeros(16, 3, 14, 14)
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
        "multi_modal_inputs": [{"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}],
    }


class TestMultimodalSequenceParallelForward(unittest.TestCase):
    def setUp(self):
        self._orig_get_world_size = ulysses.dist.get_world_size
        self._orig_get_rank = ulysses.dist.get_rank
        self._orig_group = ulysses.get_ulysses_sequence_parallel_group()
        self._orig_actor_gather = dp_actor_module.gather_outputs_and_unpad
        self._orig_critic_gather = dp_critic_module.gather_outputs_and_unpad
        ulysses.dist.get_world_size = lambda group=None: 8
        ulysses.dist.get_rank = lambda group=None: 3
        ulysses.set_ulysses_sequence_parallel_group(object())

        def _fake_gather(x, gather_dim, **kwargs):
            repeats = [1] * x.dim()
            repeats[gather_dim] = 8
            return x.repeat(*repeats)

        dp_actor_module.gather_outputs_and_unpad = _fake_gather
        dp_critic_module.gather_outputs_and_unpad = _fake_gather

    def tearDown(self):
        ulysses.dist.get_world_size = self._orig_get_world_size
        ulysses.dist.get_rank = self._orig_get_rank
        ulysses.set_ulysses_sequence_parallel_group(self._orig_group)
        dp_actor_module.gather_outputs_and_unpad = self._orig_actor_gather
        dp_critic_module.gather_outputs_and_unpad = self._orig_critic_gather

    def test_actor_multimodal_sp_uses_local_embeds_and_local_positions(self):
        actor = DataParallelPPOActor(
            config=SimpleNamespace(padding_free=True, ulysses_sequence_parallel_size=8),
            actor_module=FakeActorModule(),
        )
        actor.log_probs_from_logits = lambda logits, labels: logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        log_probs = actor._forward_micro_batch(build_micro_batch(), temperature=1.0)

        self.assertEqual(tuple(log_probs.shape), (1, 4))
        recorded = actor.actor_module.model.recorded_calls[-1]
        self.assertEqual(recorded["seq_len"], 2)
        self.assertEqual(recorded["pos_len"], 2)
        self.assertTrue(recorded["input_ids_cleared"])
        self.assertTrue(recorded["pixel_values_cleared"])

    def test_critic_multimodal_sp_uses_local_embeds_and_local_positions(self):
        critic = DataParallelPPOCritic(
            config=SimpleNamespace(padding_free=True, ulysses_sequence_parallel_size=8, max_grad_norm=1.0),
            critic_module=FakeCriticModule(),
            critic_optimizer=torch.optim.Adam([nn.Parameter(torch.zeros(1))]),
        )

        values = critic._forward_micro_batch(build_micro_batch())

        self.assertEqual(tuple(values.shape), (1, 4))
        recorded = critic.critic_module.model.recorded_calls[-1]
        self.assertEqual(recorded["seq_len"], 2)
        self.assertEqual(recorded["pos_len"], 2)
        self.assertTrue(recorded["input_ids_cleared"])
        self.assertTrue(recorded["pixel_values_cleared"])

    def test_locator_prefers_backbone_over_outer_wrapper(self):
        wrapper = OuterWrapperWithOwnMethods()
        located = _locate_multimodal_inner_module(wrapper)

        self.assertIs(located, wrapper.model)

    def test_locator_skips_fsdp_style_proxy(self):
        wrapper = FSDPStyleProxy()
        located = _locate_multimodal_inner_module(wrapper)

        self.assertIs(located, wrapper._fsdp_wrapped_module.model)

    def test_locator_accepts_structurally_correct_module_with_sharded_looking_embedding(self):
        model = StructurallyCorrectButShardedLookingModel()
        located = _locate_multimodal_inner_module(model)

        self.assertIs(located, model)


if __name__ == "__main__":
    unittest.main()
