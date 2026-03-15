# Complete ARPO Training Guide - UI-TARS-2B on Mac

## ✅ What You Have Now

### Step 3 & 4 Complete!

**VERL Framework Examined**:
- ✅ `verl/trainer/main.py` - Training entry point
- ✅ `verl/trainer/gui_agent.py` - GUI agent rollout logic
- ✅ `verl/trainer/replay_buffer.py` - Experience replay (ARPO key feature!)
- ✅ `verl/trainer/core_algos.py` - GRPO algorithm
- ✅ `examples/osworld_subset32.sh` - Reference training script

**Adapted for UI-TARS-2B**:
- ✅ `config_uitars_2b_mac.yaml` - Complete training config for Mac
- ✅ `train_uitars_2b_arpo.sh` - Full ARPO training script
- ✅ `uitars_2b_server.py` - Model inference server
- ✅ `test_server.sh` - Server testing script
- ✅ `test_osworld_uitars.sh` - Integration testing script

---

## 🚀 How to Run ARPO Training

### Phase 1: Testing (30 minutes)

#### Terminal 1: Start UI-TARS-2B Server
```bash
conda activate arpo
cd /Users/hanszhu/Desktop/ARPO_replicate
python uitars_2b_server.py
```

Wait for:
```
✓ Model loaded on CPU
Server: http://localhost:9000
Ready to accept requests!
```

#### Terminal 2: Test Server
```bash
cd /Users/hanszhu/Desktop/ARPO_replicate
bash test_server.sh
```

Expected: ✓ Server responds to requests

#### Terminal 2: Test OSWorld Integration
```bash
bash test_osworld_uitars.sh
```

Expected:
- VM starts
- Screenshots sent to server
- Actions predicted and executed
- Task completes (or times out)

---

### Phase 2: Full ARPO Training (5-10 hours)

Once testing passes:

#### Terminal 1: Keep Server Running
```bash
# Server should still be running from Phase 1
# If not, restart: python uitars_2b_server.py
```

#### Terminal 2: Start Training
```bash
cd /Users/hanszhu/Desktop/ARPO_replicate
bash train_uitars_2b_arpo.sh
```

This will:
1. Check server is running
2. Start Ray cluster (if not running)
3. Create 8-task training subset
4. Run ARPO training with VERL framework:
   - 8 tasks
   - 1 VMware VM
   - 5 epochs
   - Experience replay enabled
   - GRPO optimization
   - No KL divergence

#### Terminal 3: Monitor (Optional)
```bash
# Watch training logs
tail -f logs/*.log

# Check Ray dashboard
ray dashboard
# Open: http://127.0.0.1:8265

# Monitor system resources
htop  # or Activity Monitor
```

---

## 📊 Training Progress

### What to Expect:

**Epoch 1** (~1-2 hours):
- Initial rollouts (slow, ~3-5 min per task)
- Experience replay buffer populating
- Success rate: ~10-20%

**Epoch 2-3** (~1-2 hours each):
- Faster rollouts (model improving)
- Replay buffer helping with sparse rewards
- Success rate: ~30-50%

**Epoch 4-5** (~1-2 hours each):
- Model converging
- Most tasks have cached successes
- Success rate: ~50-70%

**Total**: ~5-10 hours for 8 tasks, 5 epochs

### Files Generated:

```
checkpoints_2b/
├── epoch_0/
│   └── model.safetensors
├── epoch_2/
│   └── model.safetensors
└── epoch_4/
    └── model.safetensors

results_2b/
├── training_metrics.json
├── replay_buffer_stats.json
└── task_results/
    ├── task_1_results.json
    └── ...

logs/
├── training.log
└── test_osworld_uitars.log
```

---

## 🔍 Key Configuration Details

### `config_uitars_2b_mac.yaml`

**ARPO-Specific Settings**:
```yaml
algorithm:
  enable_replay: true      # ← Experience replay buffer (ARPO!)
  disable_kl: true         # ← No KL divergence
  kl_coef: 0
  adv_estimator: grpo      # ← GRPO algorithm

worker:
  actor:
    clip_ratio_low: 0.2    # ← From paper
    clip_ratio_high: 0.3   # ← From paper
```

**Mac CPU Optimizations**:
```yaml
env:
  num_envs: 1              # ← Single VM
  provider: vmware         # ← VMware for Mac

worker:
  actor:
    global_batch_size: 2   # ← Minimal for CPU
  rollout:
    temperature: 0.7       # ← Lower = faster inference
    limit_images: 10       # ← Reduced context
    use_external_server: true  # ← Use our Flask server
    server_url: "http://localhost:9000/v1"

trainer:
  n_gpus_per_node: 0       # ← CPU only
  use_cpu: true
  total_episodes: 5        # ← Quick iteration
```

---

## 🐛 Troubleshooting

### Server Issues

**Server won't start**:
```bash
# Check port
lsof -i :9000

# If occupied
kill -9 <PID>
```

**Server too slow**:
- Expected: 10-30 seconds per inference
- If >60 seconds: Check Activity Monitor, CPU may be throttling

### OSWorld Issues

**VM won't start**:
```bash
# Check VMware
vmrun -T fusion list

# Restart if needed
vmrun -T fusion stop <vm_path>
vmrun -T fusion start <vm_path>
```

**Connection timeout**:
- VM takes time to boot (~30 seconds)
- Check VM IP: `vmrun -T fusion getGuestIPAddress <vm_path>`

### Training Issues

**Ray not starting**:
```bash
# Stop and restart
ray stop
ray start --head --port=2468
```

**Out of memory**:
- Close other applications
- Reduce `limit_images` to 5
- Use single environment only

---

## 📈 Monitoring Training

### View Live Progress:

```bash
# Training logs
tail -f logs/training.log

# Task completion
tail -f results_2b/training_metrics.json

# Replay buffer stats
cat results_2b/replay_buffer_stats.json
```

### Check Metrics:

After each epoch, look for:
- **Average reward**: Should increase
- **Success rate**: Should improve
- **Replay buffer size**: Should grow
- **Loss**: Should decrease

### Expected Metrics:

| Epoch | Avg Reward | Success Rate | Replay Buffer |
|-------|-----------|--------------|---------------|
| 1 | ~0.15 | ~15% | 1-2 tasks |
| 2 | ~0.30 | ~30% | 2-3 tasks |
| 3 | ~0.45 | ~45% | 3-4 tasks |
| 4 | ~0.60 | ~60% | 4-5 tasks |
| 5 | ~0.70 | ~70% | 5-6 tasks |

---

## 🎯 Success Criteria

### Minimum (Learning is happening):
- [ ] Training runs without errors
- [ ] Reward increases over epochs
- [ ] Replay buffer populates
- [ ] At least 1-2 tasks solved

### Good (ARPO is working):
- [ ] Success rate improves each epoch
- [ ] Replay buffer prevents vanishing gradients
- [ ] Final success rate >50% on training tasks

### Excellent (Paper-like results):
- [ ] Final success rate >70% on 8 training tasks
- [ ] Clear improvement: ARPO > baseline
- [ ] Replay buffer improves ~10-15%

---

## 📚 Next Steps After Training

1. **Evaluate trained model**:
   - Test on held-out tasks
   - Compare vs baseline (no ARPO)
   - Measure improvement from replay buffer

2. **Scale up** (if successful):
   - Increase to 16 tasks
   - Try 2 VMs if memory allows
   - Run more epochs (10-15)

3. **Upgrade to UI-TARS-7B** (with GPU):
   - Transfer learnings
   - Use same config with 7B model
   - Scale to 32 or 128 tasks
   - Replicate paper results (83.9%)

---

## 🎉 Summary

You now have complete ARPO training setup:

**Testing**:
- `test_server.sh` - Test model server
- `test_osworld_uitars.sh` - Test OSWorld integration

**Training**:
- `config_uitars_2b_mac.yaml` - Complete VERL config
- `train_uitars_2b_arpo.sh` - Full ARPO training script

**Everything adapted for**:
- ✅ UI-TARS-2B (not 7B)
- ✅ Mac CPU (not GPU)
- ✅ VMware (not Docker)
- ✅ Single environment (not 256)
- ✅ Experience replay enabled
- ✅ Local inference server

**Ready to train!** 🚀

Run the test scripts first, then start full training!
