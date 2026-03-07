# Data Flow: Colab GPU ↔ Mac OSWorld

## 🔄 Complete Architecture

### The Colab GPU Server Does NOT Access Results

**Important**: The Colab GPU server is **only an inference service**. It:
- ✅ Receives screenshots from Mac
- ✅ Returns action predictions
- ❌ **Does NOT** access VM results
- ❌ **Does NOT** read result files
- ❌ **Does NOT** need to know task outcomes

---

## 📊 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Step-by-Step Flow                             │
└─────────────────────────────────────────────────────────────────────┘

Step 1: OSWorld VM Captures Screenshot
┌─────────────────────────────────────────────────────────────────────┐
│ Mac: OSWorld VM (Ubuntu)                                             │
│   └─> Takes screenshot of current desktop state                     │
│   └─> Encodes as base64 image                                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: Send Screenshot to Colab GPU
┌─────────────────────────────────────────────────────────────────────┐
│ Mac: OSWorld Agent                                                   │
│   └─> POST https://xxxx.ngrok.io/v1/chat/completions               │
│   └─> Body: {messages: [{role: "user", content: [image, text]}]}   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: Colab GPU Processes (Inference Only)
┌─────────────────────────────────────────────────────────────────────┐
│ Colab: UI-TARS 7B Model                                              │
│   └─> Loads image from request                                       │
│   └─> Tokenizes input (screenshot + history)                         │
│   └─> Generates action prediction (~2-5 seconds)                    │
│   └─> Returns: "LEFT_CLICK(x=100, y=200)"                           │
│                                                                       │
│ ⚠️  Colab has NO knowledge of:                                       │
│     - Task success/failure                                           │
│     - Previous results                                               │
│     - Result files                                                   │
│     - VM state                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 4: Mac Receives Action & Executes
┌─────────────────────────────────────────────────────────────────────┐
│ Mac: OSWorld VM                                                      │
│   └─> Receives: "LEFT_CLICK(x=100, y=200)"                         │
│   └─> Executes action in Ubuntu VM                                  │
│   └─> Waits for result                                              │
│   └─> Gets reward (0.0 or 1.0)                                      │
│   └─> Saves to: results/gpu_eval/.../traj.jsonl                     │
│                                                                       │
│ 📁 Results saved locally on Mac:                                     │
│    results/gpu_eval/                                                 │
│    └─ pyautogui/                                                    │
│       └─ screenshot/                                                │
│          └─ arpo-uitars-7b/                                         │
│             └─ chrome/                                              │
│                └─ {task_id}/                                         │
│                   ├─ traj.jsonl      ← Step-by-step log             │
│                   ├─ result.txt      ← Final score (0.0 or 1.0)    │
│                   ├─ step_1_*.png    ← Screenshots                 │
│                   ├─ step_2_*.png                                  │
│                   └─ recording.mp4   ← Video of task                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 5: Repeat Until Task Complete
┌─────────────────────────────────────────────────────────────────────┐
│ Mac: OSWorld Loop                                                    │
│   └─> If not done: Go to Step 1 (capture next screenshot)          │
│   └─> If done: Save final result.txt                                │
│   └─> Move to next task                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Points

### What Colab GPU Does:
- ✅ **Model inference**: Processes screenshots → generates actions
- ✅ **API service**: Provides OpenAI-compatible endpoint
- ✅ **Fast**: 2-5 seconds per inference (vs 60 min on CPU)

### What Colab GPU Does NOT Do:
- ❌ **No file access**: Cannot read/write files on Mac
- ❌ **No result storage**: Doesn't save anything
- ❌ **No task knowledge**: Doesn't know if task succeeded
- ❌ **No VM access**: Cannot interact with Ubuntu VM directly

### What Mac OSWorld Does:
- ✅ **Task execution**: Runs tasks in VMware Ubuntu VM
- ✅ **Result storage**: Saves all results locally
- ✅ **Evaluation**: Checks task completion, computes scores
- ✅ **File management**: Creates traj.jsonl, result.txt, screenshots

---

## 📁 Where Results Are Saved

**Location**: `/Users/hanszhu/Desktop/ARPO_replicate/results/gpu_eval/`

**Structure**:
```
results/gpu_eval/
└─ pyautogui/
   └─ screenshot/
      └─ arpo-uitars-7b/
         └─ chrome/
            ├─ 44ee5668-ecd5-4366-a6ce-c1c9b8d4e938/
            │  ├─ traj.jsonl          ← All steps logged here
            │  ├─ result.txt          ← Final score: 1.0 or 0.0
            │  ├─ step_1_20260117@123456.png
            │  ├─ step_2_20260117@123457.png
            │  └─ recording.mp4
            ├─ f3b19d1e-2d48-44e9-b4e1-defcae1a0197/
            │  └─ ...
            └─ ...
```

**To view results**:
```bash
cd /Users/hanszhu/Desktop/ARPO_replicate

# View all scores
python OSWorld/show_result.py \
    --action_space pyautogui \
    --observation_type screenshot \
    --model arpo-uitars-7b \
    --result_dir results/gpu_eval/

# View specific task trajectory
cat results/gpu_eval/pyautogui/screenshot/arpo-uitars-7b/chrome/{task_id}/traj.jsonl
```

---

## 🔍 Why This Architecture?

### Separation of Concerns:
1. **Colab GPU**: Fast model inference (what it's good at)
2. **Mac OSWorld**: Task execution & result storage (what it's good at)

### Benefits:
- ✅ **No complex setup**: Colab doesn't need OSWorld installed
- ✅ **No file sync**: Results stay on Mac (where you need them)
- ✅ **No network storage**: No need to upload/download results
- ✅ **Simple**: Just HTTP API calls (screenshot → action)

### Alternative (Not Recommended):
If you wanted Colab to access results, you'd need:
- ❌ File sharing (Google Drive mount)
- ❌ Complex sync logic
- ❌ Slower (network I/O)
- ❌ More failure points

**Current setup is optimal!** 🎯

---

## 📊 Summary

| Component | Responsibility | Data Location |
|-----------|---------------|---------------|
| **Colab GPU** | Model inference | No persistent data |
| **Mac OSWorld** | Task execution | `results/gpu_eval/` |
| **ngrok** | Network tunnel | Temporary (2-hour sessions) |

**Colab GPU is just a "black box" API**:
- Input: Screenshot + instruction
- Output: Action prediction
- **No access to results needed!**

---

## ✅ Verification

After running evaluation, check results on Mac:

```bash
# Check if results exist
ls -la results/gpu_eval/pyautogui/screenshot/arpo-uitars-7b/chrome/

# Count completed tasks
find results/gpu_eval -name "result.txt" | wc -l

# View average score
python OSWorld/show_result.py \
    --action_space pyautogui \
    --observation_type screenshot \
    --model arpo-uitars-7b \
    --result_dir results/gpu_eval/
```

**All results are on your Mac - Colab never sees them!** ✅
