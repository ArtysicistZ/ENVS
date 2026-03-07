import os
import sys
import unittest

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.models.monkey_patch import apply_ulysses_patch
from verl.models.transformers.flash_attention_utils import flash_attention_forward


class TestApplyUlyssesPatch(unittest.TestCase):
    def test_padding_free_patch_overrides_sdpa_and_flash_attention(self):
        original_sdpa = ALL_ATTENTION_FUNCTIONS.get("sdpa")
        original_flash = ALL_ATTENTION_FUNCTIONS.get("flash_attention_2")
        try:
            apply_ulysses_patch("qwen2_5_vl")
            self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], flash_attention_forward)
            self.assertIs(ALL_ATTENTION_FUNCTIONS["flash_attention_2"], flash_attention_forward)
        finally:
            if original_sdpa is not None:
                ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa
            if original_flash is not None:
                ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = original_flash


if __name__ == "__main__":
    unittest.main()
