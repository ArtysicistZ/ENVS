# ARPO Troubleshooting Guide

Common issues and solutions when replicating ARPO training.

---

## Installation Issues

### 1. Submodule Not Cloned

**Error**:
```
OSWorld directory is empty or missing
```

**Solution**:
```bash
# If you cloned without --recurse-submodules
git submodule update --init --recursive

# Or clone properly from scratch
git clone --recurse-submodules https://github.com/JIA-Lab-research/ARPO.git
```

### 2. Ray Installation Fails

**Error**:
```
ERROR: Could not find a version that satisfies the requirement ray
```

**Solution**:
```bash
# Make sure you're using Python 3.10
python --version  # Should show 3.10.x

# Update pip
pip install --upgrade pip

# Install ray with specific version
pip install ray[default]==2.9.0  # or whatever version ARPO requires
```

### 3. OSWorld Dependencies Missing

**Error**:
```
ModuleNotFoundError: No module named 'desktop_env'
```

**Solution**:
```bash
cd OSWorld
pip install -e .
cd ..

# If still failing, install dependencies manually
pip install pillow opencv-python pyautogui
```

---

## Docker Issues

### 4. Docker Permission Denied

**Error**:
```
docker: Got permission denied while trying to connect to the Docker daemon socket
```

**Solution**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Apply changes
newgrp docker

# Verify
docker ps  # Should work without sudo
```

### 5. Docker Images Not Found

**Error**:
```
Error: No such image: osworld/ubuntu:latest
```

**Solution**:
```bash
# Run initial evaluation to download images
cd OSWorld
python run_multienv_uitars.py \
    --headless \
    --num_envs 1 \
    --max_steps 5 \
    --test_all_meta_path ./evaluation_examples/test_all.json

# This will download and cache Docker images
```

### 6. Out of Disk Space

**Error**:
```
Error: No space left on device
```

**Solution**:
```bash
# Check disk usage
df -h

# Clean Docker system
docker system prune -a

# Remove old images
docker images
docker rmi <image_id>

# Check OSWorld cache
du -sh OSWorld/cache
# Consider removing old cache files if needed
```

---

## Ray Cluster Issues

### 7. Ray Cluster Not Starting

**Error**:
```
ConnectionError: Ray cluster is not started
```

**Solution**:
```bash
# Check if Ray is running
ray status

# If not, start Ray
RAY_PORT=2468
RAY_HEAD_IP=127.0.0.1
ray start --head --port=$RAY_PORT --resources='{"docker:'$RAY_HEAD_IP'": 128}'

# Verify
ray status
```

### 8. Ray Cluster Out of Memory

**Error**:
```
RayOutOfMemoryError: More than 95% of the memory is used
```

**Solution**:
```bash
# Reduce number of parallel environments
# Edit config: num_envs = 2 (instead of 4 or 8)

# Or increase object store memory
ray start --head --object-store-memory=10000000000  # 10GB

# Monitor memory usage
ray status
```

### 9. Ray Port Already in Use

**Error**:
```
OSError: [Errno 48] Address already in use
```

**Solution**:
```bash
# Stop existing Ray cluster
ray stop

# Or use different port
RAY_PORT=6379  # Try different port
ray start --head --port=$RAY_PORT
```

---

## Training Issues

### 10. Model Download Fails

**Error**:
```
HTTPError: 403 Client Error: Forbidden for url: https://huggingface.co/...
```

**Solution**:
```bash
# Login to Hugging Face
huggingface-cli login

# Enter your HF token
# Get token from: https://huggingface.co/settings/tokens

# Retry download
```

### 11. CUDA Out of Memory (Even with CPU)

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
```bash
# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Verify
python -c "import torch; print(torch.cuda.is_available())"  # Should be False

# If still failing, model might auto-detect GPU
# Edit training script to explicitly use device='cpu'
```

### 12. Training Script Not Found

**Error**:
```
bash: ./examples/osworld_subset32.sh: No such file or directory
```

**Solution**:
```bash
# Check if file exists
ls examples/

# If missing, the repo might not have this exact file
# Use the generated script from notebook instead
bash ./train_cpu_subset32.sh

# Or examine what scripts are available
ls examples/*.sh
```

### 13. Gradient Vanishing (All Rewards Zero)

**Symptom**:
- Training loss stays constant
- Rewards remain at 0
- No improvement over epochs

**Solution**:
```python
# This is exactly what ARPO solves with experience replay!
# Make sure replay buffer is enabled:

# In training config:
--use_replay_buffer \
--replay_buffer_size 128

# Also try:
# 1. Reduce task difficulty (use easier subset)
# 2. Increase rollouts per task
# 3. Lower temperature for less exploration
# 4. Check if any tasks are solvable at all
```

### 14. Very Slow Training

**Symptom**:
- Single epoch takes >6 hours
- CPU usage is low

**Solution**:
```bash
# 1. Reduce rollout parameters
num_envs = 2  # Minimum
rollouts_per_task = 1  # Minimum

# 2. Use smaller model (if available)
# 3. Cache environment states more aggressively
# 4. Reduce max_steps to 10 (from 15)
# 5. Use fewer tasks (16 instead of 32)

# Monitor CPU usage
htop  # Or top on macOS

# If CPU is maxed out, training is just slow on CPU
# Consider using GPU or cloud instance
```

---

## Evaluation Issues

### 15. Evaluation Errors

**Error**:
```
AssertionError: Task evaluation failed
```

**Solution**:
```bash
# Check OSWorld evaluation examples
ls evaluation_examples/

# Try running single task evaluation
cd OSWorld
python run_multienv_uitars.py \
    --headless \
    --num_envs 1 \
    --max_steps 15 \
    --observation_type screenshot \
    --result_dir ./test_results/ \
    --test_all_meta_path ./evaluation_examples/test_all.json

# Check test_results/ for errors
cat test_results/*.log
```

### 16. Screenshots Not Captured

**Error**:
```
FileNotFoundError: Screenshot file not found
```

**Solution**:
```bash
# Ensure headless mode is working
# Install virtual display
sudo apt-get install xvfb  # Linux

# Or for macOS, ensure Docker has display access

# Check Docker container logs
docker ps
docker logs <container_id>
```

---

## Symlink Issues

### 17. Broken Symlinks

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'cache_dirs/cache_0'
```

**Solution**:
```bash
# Remove broken symlinks
rm -f cache_dirs/cache_0
rm -f evaluation_examples

# Recreate correctly
ln -sf $(pwd)/OSWorld/evaluation_examples ./evaluation_examples
mkdir -p cache_dirs/
ln -sf $(pwd)/OSWorld/cache ./cache_dirs/cache_0

# Verify
ls -la cache_dirs/cache_0  # Should show -> ../OSWorld/cache
```

---

## Debugging Tips

### General Debugging Strategy:

1. **Check Logs**:
```bash
# Training logs
tail -f results/*.log

# Ray logs
tail -f /tmp/ray/session_latest/logs/*.log

# Docker logs
docker ps
docker logs <container_id>
```

2. **Verify Setup**:
```bash
# Check Ray status
ray status

# Check Docker
docker ps

# Check disk space
df -h

# Check memory
free -h  # Linux
vm_stat  # macOS
```

3. **Test Components Individually**:
```bash
# Test model loading
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Zhenyu00/UITars-1.5')"

# Test OSWorld
cd OSWorld
python -c "import desktop_env; print('OSWorld OK')"

# Test Ray
python -c "import ray; ray.init(); print('Ray OK')"
```

4. **Reduce Complexity**:
- Use 1 environment instead of 4
- Train on 4 tasks instead of 32
- Reduce max_steps to 5
- Use temperature=0 for deterministic behavior

5. **Check Versions**:
```bash
python --version  # Should be 3.10
pip list | grep ray
pip list | grep torch
pip list | grep transformers
```

---

## Getting Help

If you're still stuck:

1. **Check GitHub Issues**: https://github.com/JIA-Lab-research/ARPO/issues
2. **Search for Error**: Google the exact error message
3. **Ask on Forums**:
   - Ray Discourse: https://discuss.ray.io/
   - Hugging Face Forums: https://discuss.huggingface.co/
4. **Create Minimal Reproducible Example**:
   - Isolate the problem
   - Create minimal code that reproduces it
   - Share with error message and environment details

---

## Environment Information Template

When asking for help, provide:

```
**Environment**:
- OS: [e.g., Ubuntu 22.04, macOS 14.0]
- Python: [e.g., 3.10.12]
- CUDA: [e.g., 11.8, or N/A for CPU]
- Ray: [e.g., 2.9.0]
- Torch: [e.g., 2.1.0]

**Setup**:
- RAM: [e.g., 32GB]
- CPU: [e.g., Intel i9, 8 cores]
- GPU: [e.g., N/A, NVIDIA A100]
- Disk: [e.g., 100GB free]

**Error**:
[Paste full error message with traceback]

**What I tried**:
1. [Step 1]
2. [Step 2]
...
```

---

## Common Warnings (Safe to Ignore)

These warnings are usually safe to ignore:

```
Warning: Using cache directory...
→ This is expected behavior

FutureWarning: The behavior of DataFrame...
→ Internal library warning, doesn't affect training

UserWarning: Creating a tensor from a list of numpy arrays...
→ Performance warning, training still works
```

---

## Quick Fixes Checklist

Before asking for help, try:

- [ ] Restart Ray cluster: `ray stop && ray start --head`
- [ ] Clear cache: `rm -rf cache_dirs/cache_0/*`
- [ ] Restart Docker: `docker restart $(docker ps -q)`
- [ ] Free up memory: Close other applications
- [ ] Update dependencies: `pip install --upgrade -r requirements.txt`
- [ ] Check disk space: `df -h` (need at least 20GB free)
- [ ] Verify symlinks: `ls -la cache_dirs/ evaluation_examples/`
- [ ] Test with minimal config: 1 env, 1 task, 5 steps
- [ ] Read error message carefully (often suggests fix)
- [ ] Search error on GitHub issues

---

Good luck with your ARPO replication! 🚀
