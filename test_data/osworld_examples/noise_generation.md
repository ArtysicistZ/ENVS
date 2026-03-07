# OSWorld Noise Generator

Generate noisy versions of OSWorld task JSON files using OpenAI's GPT models with few-shot learning.

## Folder Structure
```
osworld_examples/
├── tasks/
│   ├── chrome/              # Clean Chrome task JSONs (input)
│   ├── gimp/                # Clean GIMP task JSONs (input)
│   └── ...                  # Other app folders
├── noisy_example_seed/      # Example noisy files for few-shot learning
│   ├── 0d8b7de3-e8de-4d86-b9fd-dd2dce58a217.json
│   ├── 0d8b7de3-e8de-4d86-b9fd-dd2dce58a217_noise.json
│   ├── 2e6f678f-472d-4c55-99cc-8e7c5c402a71.json
│   └── 2e6f678f-472d-4c55-99cc-8e7c5c402a71_noise.json
├── noisy_tasks/
│   ├── chrome/              # Generated noisy files (output)
│   ├── gimp/                # Generated noisy files (output)
│   └── ...                  # Other app folders
└── fewshot_noise_generator_openai.py
```

## Setup

1. Install OpenAI Python SDK:
```bash
pip3 install openai
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

Navigate to the `osworld_examples` directory:
```bash
cd /path/to/osworld_examples
```

Run the generator with few-shot examples:
```bash
python3 fewshot_noise_generator_openai.py \
  --inputs tasks \
  --out noisy_tasks \
  --examples \
    noisy_example_seed/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217.json:noisy_example_seed/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217_noise.json \
    noisy_example_seed/2e6f678f-472d-4c55-99cc-8e7c5c402a71.json:noisy_example_seed/2e6f678f-472d-4c55-99cc-8e7c5c402a71_noise.json \
  --model gpt-4o \
  --temperature 0.2 \
  --noise_level light \
  --suffix _noise
```

## Parameters

- `--inputs`: Input folder with clean JSONs (default: `osworld_examples/tasks`)
- `--out`: Output folder for noisy JSONs (default: `osworld_examples/noisy_tasks`)
- `--examples`: Clean:noisy file pairs for few-shot learning (can provide multiple)
- `--model`: OpenAI model (default: `gpt-4o`)
- `--temperature`: Creativity level (0.0-1.0, default: 0.2)
- `--noise_level`: `light` | `medium` | `heavy` (default: `light`)
- `--suffix`: Output filename suffix (default: `_noise`)

## Noise Types

Based on your requirements, the generator adds:

1. **Web Security / Phishing** (web security dataset spoofing)
   - Extra browser tabs with distractor URLs
   - Example: example.com, ads.example.com

2. **Advertisement Popups**
   - Additional tabs simulating ads
   - Extra URLs in `chrome_open_tabs`

3. **VPN**
   - System notifications: `notify-send "VPN Connection" "Connected to server"`

4. **Network issues**
   - *Note: Not implemented as it could break VM connectivity*

5. **Environmental Noise** - Different difficulty levels
   - **Light**: 1-2 notifications
   - **Medium**: 2-3 background apps + notifications + extra tabs/files
   - **Heavy**: 3+ apps + multiple notifications + many distractors

## Example Workflow

Input: `tasks/chrome/44ee5668-ecd5-4366-a6ce-c1c9b8d4e938.json`
Output: `noisy_tasks/chrome/44ee5668-ecd5-4366-a6ce-c1c9b8d4e938_noise.json`

The script:
1. Reads clean task from `tasks/chrome/`
2. Uses examples from `noisy_example_seed/` for few-shot prompting
3. Generates noisy version with Ubuntu-compatible configs
4. Saves to `noisy_tasks/chrome/` with `_noise.json` suffix
5. Preserves subfolder structure automatically

## What It Does

1. Reads clean task JSONs from `tasks/[subfolder]/`
2. Uses example pairs to learn noise patterns via few-shot prompting
3. Adds realistic GUI noise:
   - Ubuntu apps: `gedit`, `gnome-calculator`, `nautilus`, `eog`
   - System notifications: `notify-send`
   - Extra browser tabs/files
4. Preserves instruction and evaluator fields
5. Outputs to `noisy_tasks/[subfolder]/[original_name]_noise.json`

## Constraints

The generator ensures:
- ✅ Original `instruction` unchanged
- ✅ `evaluator.func` and `evaluator.expected` unchanged
- ✅ Original config steps preserved in order
- ✅ Only Ubuntu-compatible commands (safe for VM)
- ✅ No credentials or unsafe content
- ✅ Minimal target keyword introduction
- ✅ Subfolder structure maintained

## Output Format

Each generated file includes metadata:
```json
{
  "id": "...",
  "instruction": "...",
  "config": [...],
  "noise_meta": {
    "generated_by": "fewshot_noise_generator_openai.py",
    "noise_level": "light",
    "model": "gpt-4o",
    "temperature": 0.2
  }
}
```