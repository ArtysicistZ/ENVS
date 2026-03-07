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
"""Utils for tokenization."""

import json
import os
import tempfile
import traceback
from typing import Optional

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin


def _patch_size_dict_qwen2vl(size: dict) -> None:
    """
    Ensure size has shortest_edge/longest_edge (from min_pixels/max_pixels) and remove
    min_pixels/max_pixels so transformers get_size_dict() accepts it (allowed key sets only).
    """
    default_shortest = 5656
    default_longest = 28281280
    if "shortest_edge" not in size and "min_pixels" in size:
        size["shortest_edge"] = size["min_pixels"]
    elif "shortest_edge" not in size:
        size["shortest_edge"] = default_shortest
    if "longest_edge" not in size and "max_pixels" in size:
        size["longest_edge"] = size["max_pixels"]
    elif "longest_edge" not in size:
        size["longest_edge"] = default_longest
    # Validation only allows specific key sets; extra keys (min_pixels, max_pixels) cause ValueError
    size.pop("min_pixels", None)
    size.pop("max_pixels", None)


def _patch_preprocessor_config_for_qwen2vl(config_path: str) -> None:
    """
    Patch preprocessor_config.json or preprocessor.json so transformers 4.51+ accept it.
    Some Qwen2-VL models use min_pixels/max_pixels in size; newer transformers require
    shortest_edge/longest_edge and reject extra keys. See: huggingface/transformers#37811
    """
    with open(config_path, "r") as f:
        data = json.load(f)

    def patch_all_size_dicts(obj):
        if isinstance(obj, dict):
            if "size" in obj and isinstance(obj["size"], dict):
                _patch_size_dict_qwen2vl(obj["size"])
            for v in obj.values():
                patch_all_size_dicts(v)
        elif isinstance(obj, list):
            for item in obj:
                patch_all_size_dicts(item)

    patch_all_size_dicts(data)
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)


def get_tokenizer(model_path: str, **kwargs) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        # the EOS token in gemma2 & gemma3 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(model_path: str, **kwargs) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor.
    On ValueError about shortest_edge/longest_edge (transformers 4.51+ vs Qwen2-VL config),
    downloads model to a temp dir, patches preprocessor_config.json, and loads from there.
    """
    try:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    except ValueError as e:
        err_msg = str(e)
        if "shortest_edge" in err_msg and "longest_edge" in err_msg:
            try:
                from huggingface_hub import snapshot_download
                with tempfile.TemporaryDirectory(prefix="verl_processor_") as tmpdir:
                    snapshot_download(model_path, local_dir=tmpdir)
                    for name in ("preprocessor_config.json", "preprocessor.json"):
                        config_path = os.path.join(tmpdir, name)
                        if os.path.isfile(config_path):
                            _patch_preprocessor_config_for_qwen2vl(config_path)
                    processor = AutoProcessor.from_pretrained(tmpdir, **kwargs)
            except Exception as fallback_e:
                print(f"get_processor: fallback load after config patch failed: {type(fallback_e).__name__}: {fallback_e}")
                traceback.print_exc()
                processor = None
        else:
            processor = None
    except Exception:
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        print(f"get_processor: class {processor.__class__.__name__} does not contain 'Processor'; returning None.")
        processor = None

    return processor
