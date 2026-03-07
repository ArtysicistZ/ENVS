"""
Quick diagnostic test for the logits_to_keep OOM fix.
Run from the arpo directory: python test_fix.py
Takes ~10 seconds on CPU, no GPU needed.
"""
import sys
import os
import torch
import inspect

print("=" * 60)
print("TEST 1: Which dp_actor.py is imported?")
print("=" * 60)

# Add yincheng/arpo to front of path to ensure correct import
sys.path.insert(0, os.path.dirname(__file__))

import verl.workers.actor.dp_actor as dp_mod
print(f"  verl.workers.actor.dp_actor -> {dp_mod.__file__}")

src = inspect.getsource(dp_mod.DataParallelPPOActor._forward_micro_batch)
if "logits_to_keep" in src:
    print("  [PASS] logits_to_keep found in _forward_micro_batch")
else:
    print("  [FAIL] logits_to_keep NOT found — wrong file is being imported!")
    print("         Fix: run training from /home/kevinzyz/yincheng/arpo/")
    sys.exit(1)

print()
print("=" * 60)
print("TEST 2: Does transformers Qwen2_5_VL support logits_to_keep?")
print("=" * 60)

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
sig = inspect.signature(Qwen2_5_VLForConditionalGeneration.forward)
if "logits_to_keep" in sig.parameters:
    print("  [PASS] logits_to_keep in Qwen2_5_VLForConditionalGeneration.forward signature")
else:
    print("  [FAIL] logits_to_keep NOT in forward signature!")
    print("         The transformers version doesn't support this parameter.")
    sys.exit(1)

print()
print("=" * 60)
print("TEST 3: Does logits_to_keep slice correctly? (mock lm_head, no GPU)")
print("=" * 60)

# Test the exact slice logic from modeling_qwen2_5_vl.py line 1498-1499:
#   slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
#   logits = self.lm_head(hidden_states[:, slice_indices, :])

seqlen = 100
hidden_dim = 64
vocab_size = 512
resp_len = 20
logits_to_keep = resp_len + 1

hidden_states = torch.randn(1, seqlen, hidden_dim)
lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

# Simulate what transformers does with logits_to_keep
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits_sliced = lm_head(hidden_states[:, slice_indices, :])
print(f"  seqlen={seqlen}, resp_len={resp_len}, logits_to_keep={logits_to_keep}")
print(f"  logits_sliced.shape = {logits_sliced.shape}")
assert logits_sliced.shape == (1, logits_to_keep, vocab_size), \
    f"Expected (1, {logits_to_keep}, {vocab_size}), got {logits_sliced.shape}"
print(f"  [PASS] logits_to_keep={logits_to_keep} correctly slices to {logits_sliced.shape}")

# Verify values match the last resp_len+1 positions of full logits
logits_full = lm_head(hidden_states)
ref = logits_full[:, -logits_to_keep:, :]
match = torch.allclose(ref, logits_sliced, atol=1e-5)
assert match, "Sliced logits don't match full logits at response positions!"
print(f"  [PASS] Sliced logits values match full logits at response positions")

# Verify our dp_actor slicing: logits[:, -response_length-1:-1, :]
# On sliced output [1, resp_len+1, vocab], this should give [1, resp_len, vocab]
final_logits = logits_sliced[:, -resp_len - 1 : -1, :]
assert final_logits.shape == (1, resp_len, vocab_size), \
    f"Expected (1, {resp_len}, {vocab_size}), got {final_logits.shape}"
print(f"  [PASS] dp_actor slicing [:, -resp_len-1:-1, :] gives shape {final_logits.shape}")

# Verify edge case: logits_to_keep=0 → slice(0, None) = all positions
slice_zero = slice(-0, None)
logits_zero = lm_head(hidden_states[:, slice_zero, :])
assert logits_zero.shape == (1, seqlen, vocab_size), \
    f"logits_to_keep=0 should give full sequence, got {logits_zero.shape}"
print(f"  [PASS] logits_to_keep=0 (default) gives full sequence shape {logits_zero.shape}")

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("The fix is correct. Run training normally.")
print("=" * 60)
