#!/usr/bin/env python3
"""
UI-TARS-2B Inference Server
Provides OpenAI-compatible API for UI-TARS-2B model
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import time

app = Flask(__name__)

print("Loading UI-TARS-2B model...")
MODEL_NAME = "ByteDance-Seed/UI-TARS-2B-SFT"

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu",
)
model.eval()

print(f"✓ Model loaded on CPU")

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
                    # Handle base64 encoded images
                    image_url = item['image_url']['url']
                    if image_url.startswith('data:image'):
                        # Extract base64 data
                        base64_data = image_url.split(',')[1]
                        image_data = base64.b64decode(base64_data)
                        image = Image.open(BytesIO(image_data))
                        
                        # Resize for faster CPU inference (trade quality for speed)
                        max_size = (800, 600)  # Reduce from 1920x1080
                        if image.width > max_size[0] or image.height > max_size[1]:
                            image.thumbnail(max_size, Image.Resampling.LANCZOS)
                            print(f"Resized image to {image.size} for faster CPU inference")
                        
                        processed_content.append({"type": "image", "image": image})
                    else:
                        processed_content.append(item)
                else:
                    processed_content.append(item)
            
            model_messages.append({
                "role": msg['role'],
                "content": processed_content
            })
        
        # Generate response
        num_images = len([c for m in model_messages for c in m['content'] if 'image' in c])
        print(f"\n{'='*70}")
        print(f"[{time.strftime('%H:%M:%S')}] REQUEST: {num_images} images, max_tokens={max_tokens}")
        
        print(f"[{time.strftime('%H:%M:%S')}] Step 1/4: Tokenizing input...")
        tokenize_start = time.time()
        inputs = processor.apply_chat_template(
            model_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        print(f"[{time.strftime('%H:%M:%S')}] Tokenization took {time.time()-tokenize_start:.2f}s")
        
        input_tokens = inputs["input_ids"].shape[-1]
        print(f"[{time.strftime('%H:%M:%S')}] Input size: {input_tokens} tokens")
        
        start_time = time.time()
        
        # Limit max_tokens for faster CPU inference
        max_tokens = min(max_tokens, 128)  # Cap at 128 for speed
        
        print(f"[{time.strftime('%H:%M:%S')}] Step 2/4: Generating {max_tokens} tokens (this is the slow part)...")
        gen_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                num_beams=1,  # Greedy for faster inference
            )
        
        gen_time = time.time() - gen_start
        print(f"[{time.strftime('%H:%M:%S')}] Generation took {gen_time:.2f}s ({gen_time/60:.1f} min)")
        
        print(f"[{time.strftime('%H:%M:%S')}] Step 3/4: Decoding output...")
        decode_start = time.time()
        response_text = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        print(f"[{time.strftime('%H:%M:%S')}] Decoding took {time.time()-decode_start:.2f}s")
        
        inference_time = time.time() - start_time
        output_tokens = len(outputs[0]) - input_tokens
        
        print(f"[{time.strftime('%H:%M:%S')}] Step 4/4: Sending response")
        print(f"COMPLETE: {input_tokens} in → {output_tokens} out in {inference_time:.2f}s ({inference_time/60:.1f} min)")
        print(f"Response preview: {response_text[:80]}...")
        print(f"{'='*70}\n")
        
        # Return OpenAI-compatible response
        return jsonify({
            "id": "chatcmpl-" + str(int(time.time())),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "ui-tars-2b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": inputs["input_ids"].shape[-1],
                "completion_tokens": len(outputs[0]) - inputs["input_ids"].shape[-1],
                "total_tokens": len(outputs[0])
            },
            "inference_time": inference_time
        })
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        "object": "list",
        "data": [{
            "id": "ui-tars-2b",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local"
        }]
    })

if __name__ == '__main__':
    print()
    print("="*70)
    print("UI-TARS-2B Inference Server")
    print("="*70)
    print(f"Server: http://localhost:9000")
    print(f"API: http://localhost:9000/v1/chat/completions")
    print(f"Health: http://localhost:9000/health")
    print()
    print("⚠️  NOTE: CPU inference is VERY slow (5-30+ min per request)")
    print("   Consider using GPU for practical training")
    print("="*70)
    print()
    app.run(host='0.0.0.0', port=9000, debug=False, threaded=False)  # Single-threaded to avoid queue buildup
