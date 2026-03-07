from typing import Dict, Optional

import torch
from torch import nn

from .ulysses import get_ulysses_sequence_parallel_rank


DEFAULT_IMAGE_TOKEN_ID = 151655


def use_multimodal_sp_input_hook(multi_modal_inputs: Dict[str, torch.Tensor]) -> bool:
    return multi_modal_inputs.get("pixel_values") is not None


def register_multimodal_sp_input_hook(
    module: nn.Module,
    input_ids_rmpad: torch.Tensor,
    multi_modal_inputs: Dict[str, torch.Tensor],
    sp_size: int,
):
    """Pre-embed the full multimodal sequence, then slice embeds for the local SP rank.

    Ulysses SP slices the token sequence across ranks. For image inputs that can straddle a rank
    boundary, the stock Qwen2.5-VL path cannot be fed rank-local `input_ids` together with the
    full-image `pixel_values`, because the model enforces that image token count matches image
    feature count. We therefore compute the full text+image embeddings once, then hand each rank
    only its local `inputs_embeds` slice and clear the multimodal placeholders from the forwarded
    kwargs.
    """

    if not use_multimodal_sp_input_hook(multi_modal_inputs):
        return None

    inner_module = _locate_multimodal_inner_module(module)
    if inner_module is None:
        raise RuntimeError("Unable to locate an inner multimodal model for sequence-parallel pre-embedding.")

    total_nnz = input_ids_rmpad.shape[1]
    pad_size = (sp_size - total_nnz % sp_size) % sp_size
    slice_size = (total_nnz + pad_size) // sp_size
    rank_start = get_ulysses_sequence_parallel_rank() * slice_size
    image_token_id = getattr(getattr(module, "config", None), "image_token_id", DEFAULT_IMAGE_TOKEN_ID)

    def _embed_and_slice_hook(inner, args, kwargs):
        kw_ids = kwargs.get("input_ids")
        if kw_ids is None or kwargs.get("pixel_values") is None:
            return args, kwargs

        full_embeds = inner.get_input_embeddings()(kw_ids)
        image_features = torch.cat(
            inner.get_image_features(kwargs["pixel_values"].to(full_embeds.dtype), kwargs.get("image_grid_thw")),
            dim=0,
        ).to(full_embeds.device, full_embeds.dtype)

        image_mask = kw_ids.squeeze(0) == image_token_id
        image_token_count = int(image_mask.sum())
        if image_token_count != image_features.shape[0]:
            raise RuntimeError(
                "Multimodal SP pre-embed mismatch: "
                f"image token count {image_token_count} != image feature count {image_features.shape[0]}"
            )

        full_embeds[0, image_mask] = image_features

        if pad_size > 0:
            full_embeds = torch.cat(
                [full_embeds, full_embeds.new_zeros(1, pad_size, full_embeds.shape[-1])],
                dim=1,
            )

        new_kwargs = dict(kwargs)
        new_kwargs["inputs_embeds"] = full_embeds[:, rank_start : rank_start + slice_size, :]
        new_kwargs["input_ids"] = None
        new_kwargs["pixel_values"] = None
        new_kwargs["image_grid_thw"] = None
        # Slice position_ids to match the sliced inputs_embeds so rotary_emb produces
        # cos/sin of the correct sequence length (slice_size, not total_nnz).
        pos = kwargs.get("position_ids")
        if pos is not None:
            if pos.shape[-1] > slice_size:
                if pad_size > 0:
                    pad_pos = pos.new_zeros(*pos.shape[:-1], pad_size)
                    pos = torch.cat([pos, pad_pos], dim=-1)
                pos = pos[..., rank_start : rank_start + slice_size]
                new_kwargs["position_ids"] = pos
            if pos.shape[-1] != new_kwargs["inputs_embeds"].shape[1]:
                raise RuntimeError(
                    "Multimodal SP pre-embed mismatch: "
                    f"position length {pos.shape[-1]} != embed length {new_kwargs['inputs_embeds'].shape[1]}"
                )
        return args, new_kwargs

    return inner_module.register_forward_pre_hook(_embed_and_slice_hook, with_kwargs=True)


def _locate_multimodal_inner_module(module: nn.Module) -> Optional[nn.Module]:
    seen = set()

    def _has_multimodal_methods(candidate: nn.Module) -> bool:
        return hasattr(candidate, "get_input_embeddings") and hasattr(candidate, "get_image_features")

    def _visit(candidate: Optional[nn.Module]) -> Optional[nn.Module]:
        if candidate is None or not isinstance(candidate, nn.Module) or id(candidate) in seen:
            return None
        seen.add(id(candidate))

        wrapped = getattr(candidate, "_fsdp_wrapped_module", None)
        if wrapped is not None and wrapped is not candidate:
            found = _visit(wrapped)
            if found is not None:
                return found

        module_attr = getattr(candidate, "module", None)
        if module_attr is not None and module_attr is not candidate:
            found = _visit(module_attr)
            if found is not None:
                return found

        for attr in ("model", "base_model"):
            inner = getattr(candidate, attr, None)
            found = _visit(inner)
            if found is not None:
                return found

        # Choose the deepest actual multimodal module. We intentionally do not inspect embedding
        # weights here: under FSDP with use_orig_params=True (used when freezing the vision tower),
        # the correct module can expose sharded-looking parameters before forward, but FSDP
        # materializes valid weights by the time this inner module's pre-hook runs.
        if _has_multimodal_methods(candidate):
            return candidate
        return None

    return _visit(module)
