"""
GPU OOM test for the logits_to_keep fix.
Loads the actual UI-TARS-1.5-7B model on a single GPU and runs the exact
forward + backward that caused OOM during training (no Ray, no env needed).
Runtime: ~2-3 minutes.

Run: python test_gpu_oom.py
"""
import os
import sys
import torch
import gc

# Ensure we import from the correct arpo directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "ByteDance-Seed/UI-TARS-1.5-7B"
MAX_PROMPT_LEN = 64000   # matches config max_prompt_length
MAX_RESP_LEN   = 8192    # matches config max_response_length
DEVICE = "cuda:0"
torch.cuda.set_device(0)  # initialize CUDA before any stats calls

def vram(label=""):
    alloc = torch.cuda.memory_allocated(0) / 1024**3
    peak  = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"  VRAM [{label}]: allocated={alloc:.1f} GiB, peak={peak:.1f} GiB")

print("=" * 60)
print("Verifying logits_to_keep fix on actual model")
print(f"  Model: {MODEL_PATH}")
print(f"  seqlen={MAX_PROMPT_LEN}, resp_len={MAX_RESP_LEN}")
print("=" * 60)

# ── 1. Verify the dp_actor fix is present ───────────────────────────────────
import inspect
import verl.workers.actor.dp_actor as dp_mod
src = inspect.getsource(dp_mod.DataParallelPPOActor._forward_micro_batch)
assert "logits_to_keep" in src, \
    f"Fix missing from {dp_mod.__file__}! Apply the logits_to_keep change first."
print(f"\n[PASS] logits_to_keep fix found in {dp_mod.__file__}")

# ── 2. Load model on one GPU ─────────────────────────────────────────────────
print(f"\nLoading {MODEL_PATH} onto {DEVICE} in bf16...")
from transformers import AutoModelForVision2Seq

torch.cuda.reset_peak_memory_stats(0)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map=DEVICE,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
# Enable gradient checkpointing exactly as training does
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()
vram("model loaded")

# ── 3. Fake batch (same shapes as training micro-batch) ──────────────────────
print(f"\nBuilding fake batch  input_ids=[1,{MAX_PROMPT_LEN}]  responses=[1,{MAX_RESP_LEN}]")
input_ids      = torch.randint(0, 5000, (1, MAX_PROMPT_LEN), device=DEVICE)
attention_mask = torch.ones(1, MAX_PROMPT_LEN, dtype=torch.long, device=DEVICE)
responses      = torch.randint(0, 5000, (1, MAX_RESP_LEN), device=DEVICE)
response_length = responses.size(-1)  # 8192

# ── 4. Forward WITH fix ──────────────────────────────────────────────────────
print(f"\nForward WITH logits_to_keep={response_length+1} ...")
torch.cuda.reset_peak_memory_stats(0)
try:
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=response_length + 1,   # THE FIX
    )
    logits = output.logits
    vram("after forward")

    expected_shape = (1, response_length + 1, model.config.text_config.vocab_size)
    actual_shape   = tuple(logits.shape)
    if actual_shape != expected_shape:
        print(f"  [FAIL] logits shape = {actual_shape}, expected {expected_shape}")
        print("         logits_to_keep is NOT being applied by the model!")
        sys.exit(1)
    print(f"  logits.shape = {actual_shape}  [PASS — only response positions]")

    # Simulate the dp_actor loss (same as training)
    logits_resp = logits[:, -response_length - 1 : -1, :]  # [1, 8192, vocab]
    loss = -logits_resp.float().log_softmax(-1)[:, :, 0].mean()   # fake log-prob loss

    print(f"\nBackward ...")
    loss.backward()
    vram("after backward")

    peak_gib = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"\n[PASS] Peak VRAM = {peak_gib:.1f} GiB  (A100 limit = 79.25 GiB)")
    if peak_gib > 79.0:
        print("  [WARN] Peak is very close to limit — may still OOM under FSDP overhead.")
    else:
        print("  Comfortable headroom for FSDP sharding overhead.")

except torch.cuda.OutOfMemoryError as e:
    peak_gib = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"  [FAIL] OOM after {peak_gib:.1f} GiB peak: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED — fix is working. Run training normally.")
print("=" * 60)
