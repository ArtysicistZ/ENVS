"""
Quick standalone inference test for UI-TARS-2B-SFT.
Uses transformers (no vLLM, no Ray) to verify the model produces valid GUI actions.
Run: python scripts/test_model_inference.py
"""
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image

MODEL = "ByteDance-Seed/UI-TARS-2B-SFT"

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='xxx')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait()
finished(content='xxx')

## Note
- Use English in `Thought` and `Action` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

def create_dummy_screenshot(width=1920, height=1080):
    """Create a simple dummy desktop screenshot with some UI elements."""
    img = Image.new("RGB", (width, height), color=(30, 30, 46))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 40], fill=(50, 50, 70))
    draw.text((20, 10), "File  Edit  View  Help", fill=(200, 200, 200))
    draw.rectangle([50, 200, 400, 600], fill=(40, 40, 60))
    draw.text((60, 210), "Desktop", fill=(180, 180, 200))
    draw.rectangle([width - 300, height - 50, width, height], fill=(50, 50, 70))
    draw.text((width - 280, height - 40), "Start  |  Terminal  |  Browser", fill=(180, 180, 200))
    return img


def main():
    print(f"Loading model: {MODEL}")
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    instruction = "Open the terminal application"
    screenshot = create_dummy_screenshot()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Your are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": SYSTEM_PROMPT.format(instruction=instruction)}]},
        {"role": "user", "content": [{"type": "image", "image": screenshot}]},
    ]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\n--- Prompt (first 500 chars) ---\n{prompt_text[:500]}\n---\n")

    inputs = processor(images=[screenshot], text=[prompt_text], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"Input shape: input_ids={inputs['input_ids'].shape}")

    print("\nGenerating with temperature=0.7 ...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    response_ids = out[0][inputs["input_ids"].shape[1]:]
    response_text = processor.tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"\n=== Model Output (temp=0.7) ===\n{response_text}\n{'=' * 40}")

    print("\nGenerating with temperature=0.0 (greedy) ...")
    with torch.no_grad():
        out2 = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    response_ids2 = out2[0][inputs["input_ids"].shape[1]:]
    response_text2 = processor.tokenizer.decode(response_ids2, skip_special_tokens=True)
    print(f"\n=== Model Output (greedy) ===\n{response_text2}\n{'=' * 40}")

    valid = "Action:" in response_text or "Action:" in response_text2
    print(f"\n{'PASS' if valid else 'FAIL'}: Model {'produces' if valid else 'does NOT produce'} valid Action: output")


if __name__ == "__main__":
    main()
