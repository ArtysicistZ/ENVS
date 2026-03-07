#!/usr/bin/env python3
"""
ARPO UI-TARS 7B Inference Server
GPU-optimized server for ARPO-trained UITARS 1.5 7B model

Model: Fanbin/ARPO_UITARS1.5_7B
Performance: 83.9% on 128 OSWorld tasks
Requires: CUDA GPU with 16GB+ VRAM
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import time
import sys

app = Flask(__name__)

print("="*70)
print("ARPO UI-TARS 7B Inference Server (GPU)")
print("="*70)

# Check CUDA availability
if not torch.cuda.is_available():
    print("❌ ERROR: CUDA not available!")
    print("This server requires GPU. For CPU, use uitars_2b_server.py")
    sys.exit(1)

print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Model configuration
MODEL_NAME = "Fanbin/ARPO_UITARS1.5_7B"

print(f"\n📥 Loading ARPO UITARS 7B model (full precision)...")
print("This will take 1-2 minutes on first run...")

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load model (use AutoModelForVision2Seq for generation)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for A100
)
model.eval()

print(f"✅ Model loaded!")
print(f"📍 Device: {model.device}")
print(f"💾 GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("="*70)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    try:
        data = request.json
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 256)
        temperature = data.get('temperature', 0.7)
        
        # Convert messages to model format
        model_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                continue  # Skip system messages
            
            content = msg.get('content', [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            
            # Handle images (decode base64 if needed)
            processed_content = []
            for item in content:
                if item['type'] == 'image_url':
                    image_url = item['image_url']['url']
                    if image_url.startswith('data:image'):
                        # Extract base64 data
                        base64_data = image_url.split(',')[1]
                        image_data = base64.b64decode(base64_data)
                        image = Image.open(BytesIO(image_data))
                        
                        # Optional: resize if too large
                        max_dim = 1920
                        if image.width > max_dim or image.height > max_dim:
                            ratio = max_dim / max(image.width, image.height)
                            new_size = (int(image.width * ratio), int(image.height * ratio))
                            image = image.resize(new_size, Image.Resampling.LANCZOS)
                            print(f"Resized image to {new_size}")
                        
                        processed_content.append({"type": "image", "image": image})
                    else:
                        processed_content.append(item)
                else:
                    processed_content.append(item)
            
            model_messages.append({
                "role": msg['role'],
                "content": processed_content
            })
        
        num_images = len([c for m in model_messages for c in m['content'] if 'image' in c])
        print(f"\n{'='*70}")
        print(f"[{time.strftime('%H:%M:%S')}] REQUEST: {num_images} images, max_tokens={max_tokens}")
        
        # Tokenize input
        print(f"[{time.strftime('%H:%M:%S')}] Tokenizing...")
        inputs = processor.apply_chat_template(
            model_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_tokens = inputs["input_ids"].shape[-1]
        print(f"[{time.strftime('%H:%M:%S')}] Input: {input_tokens} tokens")
        
        # Generate
        print(f"[{time.strftime('%H:%M:%S')}] Generating (max {max_tokens} tokens)...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
            )
        
        gen_time = time.time() - start_time
        
        # Decode
        response_text = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        output_tokens = len(outputs[0]) - input_tokens
        total_time = time.time() - start_time
        
        print(f"[{time.strftime('%H:%M:%S')}] COMPLETE: {input_tokens} in → {output_tokens} out in {total_time:.2f}s")
        print(f"Response: {response_text[:100]}...")
        print(f"{'='*70}\n")
        
        # Return OpenAI-compatible response
        return jsonify({
            "id": "chatcmpl-" + str(int(time.time())),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "arpo-uitars-7b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "inference_time": total_time
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        "object": "list",
        "data": [{
            "id": "arpo-uitars-7b",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local"
        }]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "arpo-uitars-7b",
        "device": str(model.device),
        "gpu_memory_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    })

if __name__ == '__main__':
    print()
    print("="*70)
    print("ARPO UI-TARS 7B Server Ready!")
    print("="*70)
    print(f"Server: http://localhost:9000")
    print(f"API: http://localhost:9000/v1/chat/completions")
    print(f"Health: http://localhost:9000/health")
    print()
    print("⚡ GPU Inference: ~2-5 seconds per request")
    print("💾 Memory efficient with 4-bit quantization")
    print("="*70)
    print()
    
    app.run(host='0.0.0.0', port=9000, debug=False, threaded=False)
