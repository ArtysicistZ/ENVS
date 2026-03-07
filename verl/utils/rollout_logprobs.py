from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..protocol import DataProto


class RolloutLogProbAlignmentError(RuntimeError):
    pass


def _extract_selected_logprobs(
    token_ids: Sequence[int],
    position_logprobs: Optional[Sequence[Optional[Dict[int, Any]]]],
    *,
    label: str,
) -> List[float]:
    if position_logprobs is None:
        raise RolloutLogProbAlignmentError(f"{label}: missing logprobs.")
    if len(position_logprobs) != len(token_ids):
        raise RolloutLogProbAlignmentError(
            f"{label}: token/logprob length mismatch ({len(token_ids)} vs {len(position_logprobs)})."
        )

    selected: List[float] = []
    for idx, (token_id, token_logprobs) in enumerate(zip(token_ids, position_logprobs)):
        if token_logprobs is None:
            raise RolloutLogProbAlignmentError(f"{label}: position {idx} has no selected-token logprob.")
        if token_id not in token_logprobs:
            raise RolloutLogProbAlignmentError(f"{label}: token_id={token_id} missing at position {idx}.")
        selected.append(float(token_logprobs[token_id].logprob))
    return selected


def build_rollout_step_metadata(
    *,
    prompt_token_ids: Sequence[int],
    prompt_logprobs: Optional[Sequence[Optional[Dict[int, Any]]]],
    response_token_ids: Sequence[int],
    response_logprobs: Optional[Sequence[Optional[Dict[int, Any]]]],
    assistant_prefix_token_ids: Sequence[int],
) -> Dict[str, List[float] | List[int]]:
    prefix_len = len(assistant_prefix_token_ids)
    if prefix_len == 0:
        raise RolloutLogProbAlignmentError("assistant prefix token ids must not be empty.")
    if len(prompt_token_ids) < prefix_len:
        raise RolloutLogProbAlignmentError("prompt shorter than assistant prefix.")

    prompt_suffix = list(prompt_token_ids[-prefix_len:])
    if prompt_suffix != list(assistant_prefix_token_ids):
        raise RolloutLogProbAlignmentError("prompt does not end with the assistant generation prefix.")

    prefix_logprobs = _extract_selected_logprobs(
        assistant_prefix_token_ids,
        prompt_logprobs[-prefix_len:] if prompt_logprobs is not None else None,
        label="assistant_prefix",
    )
    response_logprob_values = _extract_selected_logprobs(
        response_token_ids,
        response_logprobs,
        label="response",
    )

    return {
        "assistant_token_ids": list(assistant_prefix_token_ids) + list(response_token_ids),
        "assistant_token_logprobs": prefix_logprobs + response_logprob_values,
    }


def assemble_old_log_probs_from_sample(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    rollout_step_metadata: Sequence[Dict[str, Sequence[float] | Sequence[int]]],
) -> torch.Tensor:
    if input_ids.dim() != 1 or labels.dim() != 1:
        raise RolloutLogProbAlignmentError("assemble_old_log_probs_from_sample expects 1D input_ids/labels.")
    if input_ids.shape != labels.shape:
        raise RolloutLogProbAlignmentError("input_ids and labels must have the same shape.")

    valid_mask = labels[1:] != -100
    dense = torch.zeros(input_ids.size(0) - 1, dtype=torch.float32, device=input_ids.device)
    if not bool(valid_mask.any()):
        return dense

    flat_token_ids: List[int] = []
    flat_logprobs: List[float] = []
    for step_idx, step_meta in enumerate(rollout_step_metadata):
        if not isinstance(step_meta, dict):
            raise RolloutLogProbAlignmentError(f"step {step_idx}: rollout metadata must be a dict.")
        token_ids = step_meta.get("assistant_token_ids")
        token_logprobs = step_meta.get("assistant_token_logprobs")
        if token_ids is None or token_logprobs is None:
            raise RolloutLogProbAlignmentError(f"step {step_idx}: incomplete rollout metadata.")
        if len(token_ids) != len(token_logprobs):
            raise RolloutLogProbAlignmentError(f"step {step_idx}: token/logprob length mismatch.")
        flat_token_ids.extend(int(token_id) for token_id in token_ids)
        flat_logprobs.extend(float(logprob) for logprob in token_logprobs)

    target_ids = input_ids[1:][valid_mask].tolist()
    if len(flat_token_ids) < len(target_ids):
        raise RolloutLogProbAlignmentError(
            f"rollout metadata shorter than labeled assistant tokens ({len(flat_token_ids)} < {len(target_ids)})."
        )

    mismatch_idx = next(
        (idx for idx, (target_id, rollout_id) in enumerate(zip(target_ids, flat_token_ids)) if target_id != rollout_id),
        None,
    )
    if mismatch_idx is not None:
        raise RolloutLogProbAlignmentError(
            "token mismatch while aligning rollout old_log_probs at "
            f"position {mismatch_idx}: target={target_ids[mismatch_idx]} rollout={flat_token_ids[mismatch_idx]}."
        )

    dense[valid_mask] = torch.tensor(flat_logprobs[: len(target_ids)], dtype=torch.float32, device=input_ids.device)
    return dense


def build_old_log_probs_from_batch(batch: DataProto) -> Tuple[torch.Tensor, Dict[str, int]]:
    if "labels" not in batch.batch.keys():
        raise RolloutLogProbAlignmentError("batch is missing labels required for rollout old_log_probs alignment.")
    if "rollout_step_metadata" not in batch.non_tensor_batch:
        raise RolloutLogProbAlignmentError("batch is missing rollout_step_metadata.")

    old_log_probs = []
    aligned_tokens = 0
    for sample_idx in range(len(batch)):
        rollout_step_metadata = batch.non_tensor_batch["rollout_step_metadata"][sample_idx]
        dense = assemble_old_log_probs_from_sample(
            input_ids=batch.batch["input_ids"][sample_idx],
            labels=batch.batch["labels"][sample_idx],
            rollout_step_metadata=rollout_step_metadata,
        )
        aligned_tokens += int((batch.batch["labels"][sample_idx, 1:] != -100).sum().item())
        old_log_probs.append(dense)

    return torch.stack(old_log_probs, dim=0), {"aligned_samples": len(old_log_probs), "aligned_tokens": aligned_tokens}


def get_old_log_probs_with_fallback(
    *,
    batch: DataProto,
    source: str,
    recompute_fn: Callable[[], DataProto],
) -> Tuple[DataProto, Dict[str, Any]]:
    if source not in {"auto", "rollout", "recompute"}:
        raise ValueError(f"Unknown old_logprob_source: {source}")

    if source != "recompute":
        try:
            old_log_probs, stats = build_old_log_probs_from_batch(batch)
            return (
                DataProto.from_dict(tensors={"old_log_probs": old_log_probs}),
                {
                    "used_rollout_old_logprobs": 1,
                    "used_recomputed_old_logprobs": 0,
                    "rollout_old_logprob_fallback": 0,
                    **stats,
                },
            )
        except RolloutLogProbAlignmentError as exc:
            if source == "rollout":
                raise
            recomputed = recompute_fn()
            return (
                recomputed,
                {
                    "used_rollout_old_logprobs": 0,
                    "used_recomputed_old_logprobs": 1,
                    "rollout_old_logprob_fallback": 1,
                    "rollout_old_logprob_error": str(exc),
                },
            )

    recomputed = recompute_fn()
    return (
        recomputed,
        {
            "used_rollout_old_logprobs": 0,
            "used_recomputed_old_logprobs": 1,
            "rollout_old_logprob_fallback": 0,
        },
    )


def to_object_array(values: Iterable[Any]) -> np.ndarray:
    return np.array(list(values), dtype=object)
