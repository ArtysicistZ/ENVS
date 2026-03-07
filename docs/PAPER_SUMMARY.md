# ARPO Paper Summary

**Title**: ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay  
**Authors**: Fanbin Lu, Zhisheng Zhong, Shu Liu, Chi-Wing Fu, Jiaya Jia  
**Institutions**: CUHK, SmartMore, HKUST  
**Link**: https://github.com/JIA-Lab-research/ARPO

## TL;DR

ARPO improves GUI agent training by adding an **experience replay buffer** to GRPO (Group Relative Policy Optimization). This prevents vanishing gradients in sparse reward scenarios, achieving **83.9%** success on training tasks vs **72.9%** with vanilla GRPO (+11% improvement).

---

## Problem Statement

Training vision-language GUI agents with reinforcement learning is challenging because:
1. **Sparse Rewards**: Most rollouts fail (reward = 0), especially for complex desktop tasks
2. **Vanishing Gradients**: When all rollouts in a group fail, GRPO advantages become zero
3. **Sample Inefficiency**: Successful trajectories are rare but informative, yet discarded after one use

---

## Method Overview

### 1. Base Architecture: UITars-1.5

- **Model**: Qwen2.5-VL 7B (vision-language model)
- **Context**: 64K tokens, up to 15 screenshots (1080P)
- **Input**: Full trajectory history {s₀, a₀, s₁, a₁, ..., sₜ, aₜ}
- **Output**: Chain-of-Thought actions (thinking + solution)

**Action Space**:
- Primitive: LEFT_CLICK, RIGHT_CLICK, TYPE_TEXT, PRESS_HOTKEY, SCROLL
- Meta: WAIT, FINISH, FAIL, CALL_USER

### 2. GRPO (Group Relative Policy Optimization)

**Objective**:
```
J_GRPO(θ) = (1/G) Σᵢ (1/|oᵢ|) Σₜ min(
    ratio * Âᵢ,ₜ,
    clip(ratio, 1-ε, 1+ε) * Âᵢ,ₜ
)

where:
- ratio = πθ(oᵢ(t)|oᵢ,<t) / πold(oᵢ(t)|oᵢ,<t)
- Âᵢ,ₜ = (rᵢ - μ) / σ  (group-normalized advantage)
```

**Key Properties**:
- No value function (unlike PPO)
- Group-normalized advantages from trajectory rewards
- Token-level optimization
- Simpler than PPO (only policy network)

### 3. Experience Replay Buffer (Key Innovation)

**Problem**: 
When all rollouts fail (all rewards = 0):
- Mean μ = 0, Std σ = 0
- All advantages Â = 0
- **No gradient signal → training stalls**

**ARPO Solution**:
1. Maintain per-task replay buffer of successful trajectories
2. When σ(rewards) = 0 (all same reward):
   - Randomly replace one failed trajectory with cached success
   - Now: advantages ≠ 0, gradients flow!
3. Update buffer with new successes during training
4. Fixed-size buffer with FIFO eviction

**Pseudo-code**:
```python
def handle_replay(task_id, trajectories):
    rewards = [r for _, _, r in trajectories]
    
    # Check if all failed
    if all(r == 0 for r in rewards):
        # Inject success from buffer
        if task_id in replay_buffer:
            trajectories[0] = replay_buffer[task_id]
    
    # Store new successes
    for trajectory in trajectories:
        if trajectory.reward > 0:
            replay_buffer[task_id] = trajectory
    
    return trajectories
```

### 4. Distributed Rollout

- **Architecture**: Centralized inference (VLLM) + distributed environments
- **Parallelism**: 256 Docker VMs (full training), 4-8 (CPU subset)
- **Efficiency**: Batch action prediction across all environments
- **Latency**: Parallel rollout minimizes GPU idle time

### 5. Task Filtering

Not all OSWorld tasks are trainable with current agents. ARPO filters for "valuable" tasks:

**Process**:
1. Evaluate each OSWorld task with UI-Tars-1.5 baseline
2. Run 16 rollouts per task
3. Keep task if ≥1 success (agent can sometimes solve it)
4. Result: **128 trainable tasks** from 369 total

**Rationale**: 
- Tasks with 0/16 success are too hard → no learning signal
- Tasks with some success → agent can explore and improve

---

## Training Configuration

### Full Training (Paper):
- **Tasks**: 128 (filtered from OSWorld)
- **Parallel Envs**: 256 Docker VMs
- **Rollouts per Task**: 8
- **Epochs**: 15
- **Batch Size**: 32 (rollout), 8 (optimization)
- **Learning Rate**: 1e-6 (AdamW)
- **Temperature**: 1.0 (rollout), 0.6 (eval)
- **Clipping**: ε_low=0.2, ε_high=0.3
- **Max Steps**: 15 per trajectory
- **Gradient Accumulation**: 4 steps
- **No KL Divergence**: Removes reference model requirement

### CPU-Optimized (Subset):
- **Tasks**: 32
- **Parallel Envs**: 4-8
- **Rollouts per Task**: 2
- **Batch Size**: 8 (rollout), 2 (optimization)

---

## Results

### Performance on OSWorld Benchmark:

| Model | 128 Training Tasks | OSWorld Overall (369) |
|-------|-------------------|-----------------------|
| UI-Tars-1.5 (Base) | 68.7% | 23.5% |
| UI-Tars-1.5 + GRPO | 72.9% | 26.0% |
| **UI-Tars-1.5 + ARPO** | **83.9%** | **29.9%** |

### Key Findings:

1. **+11% on training tasks** (83.9% vs 72.9%)
   - Experience replay effectively prevents vanishing gradients
   - Agent learns better from sparse rewards

2. **+3.9% on all tasks** (29.9% vs 26.0%)
   - Improved generalization to unseen tasks
   - Better sample efficiency → better policies

3. **Trajectory reward increases consistently**
   - Steady improvement over 15 epochs
   - No catastrophic forgetting

---

## Why ARPO Works

### Problem: Vanishing Gradients in Sparse Rewards

**Example Scenario**:
- Task: "Save the document using keyboard shortcut"
- Rollouts: 8 attempts, all fail (r = 0, 0, 0, 0, 0, 0, 0, 0)
- GRPO advantages: Â = (0-0)/0 = NaN → all zero
- Gradient: ∇L = 0 → **no learning**

### ARPO Solution: Inject Success

**After Replay Buffer Injection**:
- Rollouts: (1, 0, 0, 0, 0, 0, 0, 0) ← injected success
- Mean: μ = 0.125, Std: σ = 0.33
- Advantages: Â = (+2.65, -0.38, -0.38, ...) ← non-zero!
- Gradient: ∇L ≠ 0 → **learning continues**

### Mathematical Insight

Standard GRPO loss:
```
L = Σᵢ ratio_i * Â_i

When all r_i = 0:
  Â_i = (0 - 0) / 0 = 0 for all i
  L = 0 → ∇L = 0
```

ARPO with replay:
```
After injecting r_1 = 1:
  Â_1 = (1 - 0.125) / 0.33 = +2.65 (positive!)
  Â_i = (0 - 0.125) / 0.33 = -0.38 (negative)
  L ≠ 0 → ∇L ≠ 0 → training signal!
```

---

## Implementation Details

### Reward Design

1. **Trajectory Reward** (r_t):
   - r_t = 1.0 if task completed successfully
   - r_t = 0.0 otherwise
   - Evaluated by OSWorld's rule-based checker

2. **Action Format Reward** (r_f):
   - r_f = -1.0 if action fails to parse
   - r_f = 0.0 if action is valid
   - Encourages syntactically correct outputs

3. **Total Reward**: r = r_t + r_f

### Training Loop

```python
for epoch in range(15):
    for task_batch in sample_tasks(batch_size=32):
        # 1. Distributed rollout
        trajectories = []
        for task in task_batch:
            task_trajectories = rollout(task, num_envs=8)
            
            # 2. Experience replay
            task_trajectories = handle_replay(task.id, task_trajectories)
            trajectories.extend(task_trajectories)
        
        # 3. Compute GRPO loss
        loss = compute_grpo_loss(trajectories)
        
        # 4. Update policy
        loss.backward()
        optimizer.step()
        
        # 5. Log metrics
        log(loss, avg_reward, success_rate)
```

### Hyperparameter Choices

- **Temperature 1.0 (rollout)**: High exploration during data collection
- **Temperature 0.6 (eval)**: Lower for more deterministic evaluation
- **Clipping [0.2, 0.3]**: Asymmetric clipping (from DAPO)
  - Larger upper clip (0.3) allows more positive updates
  - Smaller lower clip (0.2) prevents large negative updates
- **No KL penalty**: Simplifies training, no reference model needed
- **AdamW optimizer**: Better generalization than Adam

---

## Comparison to Other Methods

### vs. Vanilla GRPO
- **+11%** on training tasks
- ARPO: Experience replay buffer
- GRPO: All trajectories discarded after use

### vs. PPO
- Both: Clipped policy gradients
- ARPO (GRPO-based): Group-normalized advantages, no critic
- PPO: Temporal difference advantages, requires value function

### vs. Dynamic Sampling (DS-GRPO)
- DS-GRPO: Removes groups with σ=0 (vanishing gradient)
- ARPO: Injects successful trajectory instead
- **Why ARPO is better**: Preserves training data, doesn't waste rollouts

---

## Limitations and Future Work

### Current Limitations:
1. **Task filtering required**: Many tasks still too hard
2. **Expensive rollouts**: Docker environments have latency
3. **Limited context**: 15 images may not be enough for very long tasks
4. **Binary rewards**: No partial credit for progress

### Future Directions:
1. **Curriculum learning**: Gradually increase task difficulty
2. **Hierarchical RL**: Decompose long tasks into subtasks
3. **Self-supervised pretraining**: Learn from unlabeled GUI interactions
4. **Multi-modal rewards**: Combine rule-based + learned rewards
5. **Cross-platform transfer**: Train on Linux, test on macOS/Windows

---

## Key Takeaways

1. **Experience replay solves sparse reward problem** in GUI agent training
2. **GRPO is effective for VLM RL** when combined with proper handling of edge cases
3. **Long context is crucial** for multi-turn GUI interaction (15 images, 64K tokens)
4. **Task filtering is important** to focus on learnable tasks
5. **Distributed rollout** enables scalable training across many environments

---

## Citation

```bibtex
@article{lu2024arpo,
  title={ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay},
  author={Lu, Fanbin and Zhong, Zhisheng and Liu, Shu and Fu, Chi-Wing and Jia, Jiaya},
  journal={arXiv preprint},
  year={2024}
}
```

---

## Resources

- **Paper**: [Link to paper]
- **Code**: https://github.com/JIA-Lab-research/ARPO
- **Model**: https://huggingface.co/Zhenyu00/UITars-1.5
- **Benchmark**: https://github.com/xlang-ai/OSWorld
- **Framework**: https://github.com/volcengine/verl (VERL)
