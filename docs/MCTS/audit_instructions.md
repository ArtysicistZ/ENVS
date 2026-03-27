# V2 Step Mask Audit Instructions

You are auditing a batch of GUI agent action steps for a training pipeline. Each batch contains ~22 tree nodes from MCTS trajectories. For each node, you must classify each of its `own_actions` steps as KEEP or REMOVE.

## What You're Looking At

Each node represents a segment of a GUI agent's trajectory performing a computer task (e.g., changing a setting in Chrome, editing in GIMP). The trajectory SUCCEEDED overall, but individual steps may have been mistakes.

For each node:
- **instruction**: The task the agent was trying to accomplish
- **context_actions**: Prior steps (ancestor chain). Read these to understand what happened before. DO NOT label these.
- **own_actions**: Steps to classify. Label each as KEEP or REMOVE.

## Classification Rules

### KEEP (loss=1) — train on this step:
1. **Correct productive step**: The agent does the right thing. The action moves toward the task goal.
2. **Error-recognition step**: The Thought explicitly recognizes a previous mistake and plans recovery. Example: "I clicked Colors menu but that's wrong, I need Image menu instead." These are extremely valuable — they teach self-correction.
3. **Neutral observation step**: The agent describes what it sees and plans next. Shows correct situational awareness.

### REMOVE (loss=0) — do NOT train on this step:
1. **Real error step**: The agent takes an action that is objectively wrong. The Thought may sound confident, but the action leads to the wrong place. Examples:
   - Clicking the wrong menu when trying to find a setting
   - Opening the wrong application or dialog
   - Typing the wrong command or entering wrong values
   - Navigating to an irrelevant part of the UI
   - Repeating the same failed action with no new reasoning

## The Critical Test

Does the Thought show AWARENESS that something went wrong?
- **Yes** -> KEEP (error-recognition, teaches recovery)
- **No, agent confidently does wrong thing** -> REMOVE (real error)

## When In Doubt

Default to KEEP. Better to keep a slightly ambiguous step than miss a valuable error-recovery step. Only REMOVE when you are confident the step is a clear mistake with no self-awareness.

## Output Format

Write one JSON object per line to the output file. Each object has:
```json
{"key": "task_id:round:node_id", "mask": [1, 1, 0, 1, 1]}
```
Where 1 = KEEP, 0 = REMOVE. The mask array length must match the number of own_actions for that node.
