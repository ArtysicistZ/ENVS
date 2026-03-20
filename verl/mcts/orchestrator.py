"""MCTS Orchestrator: the main loop for action-level MCTS trajectory collection.

Architecture:
- All VMs pre-setup for the same task at start (identical initial state)
- 1 VM starts stepping. Tree grows on demand at high-entropy nodes.
- Step 0 uses the SAME branching logic as all other steps.
- Hard VM cap (config.max_active_vms) prevents exceeding resource limits.
- Per-node K probing (not shared budget) ensures diversity is always detectable.
"""

import copy
import logging
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
    sys.path.insert(0, os.path.join(PROJ_ROOT, "OSWorld"))

from verl.mcts.config import MCTSConfig
from verl.mcts.tree import BranchBudget, MCTSTree, TreeNode
from verl.mcts.clustering import (
    cluster_by_fingerprint,
    compute_type_entropy,
    representative,
    should_branch,
)
from verl.mcts.trajectory_io import make_mcts_trajectory


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
type(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
fail() # Use when you think the task is not feasible or cannot be completed.

## Note
- Use English in `Thought` and `Action` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}"""

LIMIT_IMAGES = 3


class MCTSOrchestrator:
    """Orchestrates MCTS trajectory collection for one task."""

    def __init__(self, config: MCTSConfig, vllm_pool, processor, tokenizer):
        self.config = config
        self.vllm_pool = vllm_pool
        self.processor = processor
        self.tokenizer = tokenizer

    def run_task(
        self,
        task_config: Dict[str, Any],
        env_workers: List,
    ) -> Tuple[MCTSTree, List[Dict[str, Any]]]:
        """Run MCTS exploration for one task.

        All env_workers are pre-setup with the same task.
        The tree grows on demand — VMs claimed when entropy demands branching.
        """
        import ray

        instruction = task_config.get("instruction", "")
        waiting = list(range(len(env_workers)))  # all VMs available
        tree = MCTSTree()
        active_nodes: List[TreeNode] = []
        node_counter = [0]

        def _make_node_id():
            node_counter[0] += 1
            return f"node_{node_counter[0]:03d}"

        def _n_active():
            return len([n for n in active_nodes if not n.done])

        def _can_spawn():
            """Check if we can spawn more VMs (hard cap)."""
            return bool(waiting) and _n_active() < self.config.max_active_vms

        # ---- Get initial screenshot (all VMs identical) ----
        init_screenshot_b64 = ray.get(env_workers[0].get_obs_screenshot.remote())
        if not init_screenshot_b64:
            logger.error("Failed to get initial screenshot")
            return tree, []

        # ============================================================
        # UNIFIED STEP LOOP (step 0 uses same logic as all other steps)
        # ============================================================
        for step in range(self.config.max_steps):
            alive = [n for n in active_nodes if not n.done]

            # ---- Step 0: bootstrap from initial screenshot ----
            if step == 0 and not alive:
                # Create 1 root node to start
                if not waiting:
                    break
                vm_idx = waiting.pop(0)
                root = TreeNode(
                    node_id=_make_node_id(),
                    vm_slot_id=vm_idx,
                    depth=0,
                    budget=BranchBudget(self.config.max_branch_per_explorer),
                    instruction=instruction,
                    current_screenshot_b64=init_screenshot_b64,
                )
                tree.add_root(root)
                active_nodes.append(root)
                alive = [root]

            if not alive:
                logger.info("Step %d: no active nodes, stopping", step)
                break

            logger.info("Step %d: %d active, %d waiting, cap=%d",
                        step, len(alive), len(waiting), self.config.max_active_vms)

            # ---- Phase 1: Get screenshots for all active nodes ----
            # Step 0: screenshot already set from init
            if step > 0:
                for node in alive:
                    obs = ray.get(env_workers[node.vm_slot_id].get_obs_screenshot.remote())
                    if obs:
                        node.current_screenshot_b64 = obs
                    else:
                        logger.warning("  %s: no screenshot at step %d", node.node_id, step)

            # ---- Phase 2: Generate K candidates per node ----
            k = self.config.k_step0 if step == 0 else self.config.k_per_node
            for node in alive:
                if not node.current_screenshot_b64:
                    node.candidates = []
                    continue
                messages = self._build_messages_for_node(node)
                node.candidates = self._generate_candidates(messages, k)
                if not node.candidates:
                    logger.warning("  %s: empty candidates at step %d", node.node_id, step)

            # ---- Phase 3: Branch decisions ----
            for node in alive:
                if len(node.candidates) < 2:
                    # Not enough candidates to branch — use majority vote
                    node.action = node.get_majority_action() or (node.candidates[0] if node.candidates else None)
                    continue

                if should_branch(node.candidates, step, node, self.config):
                    clusters = cluster_by_fingerprint(
                        node.candidates,
                        grid_size=self.config.spatial_grid_size,
                        min_cluster_size=self.config.min_cluster_size,
                    )
                    # Node takes majority action
                    node.action = representative(node.candidates, clusters[0])

                    # Spawn VMs for minority clusters (up to max_branches_per_step)
                    for minority_cluster in clusters[1:self.config.max_branches_per_step + 1]:
                        if not _can_spawn():
                            logger.info("  VM cap reached (%d active), skipping branch",
                                        _n_active())
                            break

                        vm_idx = waiting.pop(0)
                        child_action = representative(node.candidates, minority_cluster)

                        child = TreeNode(
                            node_id=_make_node_id(),
                            vm_slot_id=vm_idx,
                            depth=step,
                            budget=BranchBudget(self.config.child_branch_budget),
                            instruction=instruction,
                            branch_action=child_action,
                            just_spawned=True,
                        )
                        tree.add_child(node, child)
                        active_nodes.append(child)

                        # Replay the parent's PHYSICAL action sequence on the new VM
                        # (not the logical tree path — only what was actually on the parent's VM)
                        parent_physical = node.get_physical_action_sequence()
                        child.replay_prefix = list(parent_physical)  # store for grandchildren
                        if parent_physical:
                            replay_timeout = max(60, len(parent_history) * 5 + 30)
                            logger.info("  Spawning %s: replay %d physical actions on VM %d",
                                        child.node_id, len(parent_physical), vm_idx)
                            try:
                                ray.get(env_workers[vm_idx].replay.remote(
                                    parent_physical,
                                    replay_pause_sec=self.config.replay_pause_sec,
                                ), timeout=replay_timeout)
                            except Exception as e:
                                logger.error("  Replay failed for %s: %s", child.node_id, e)
                                child.done = True
                        # else: step 0, no replay needed (VM already at initial state)

                        # Child inherits parent's current screenshot
                        child.screenshot_history = []
                        child.current_screenshot_b64 = node.current_screenshot_b64

                    # Only consume budget if we actually spawned at least 1 child
                    if any(c.parent == node and c.depth == step for c in active_nodes if c != node):
                        node.budget.use()
                else:
                    # No branching — use majority vote
                    node.action = node.get_majority_action() or (node.candidates[0] if node.candidates else None)

            # Handle just-spawned nodes (they use branch_action)
            for node in active_nodes:
                if node.just_spawned and not node.done:
                    node.action = node.branch_action
                    node.just_spawned = False

            # ---- Phase 4: Execute actions on all active VMs ----
            self._execute_step(
                [n for n in active_nodes if not n.done and n.action],
                env_workers, ray,
            )

            # ---- Phase 5: Prune stuck/terminal VMs ----
            for node in active_nodes:
                if node.done:
                    continue
                if node.is_terminal():
                    node.done = True
                    logger.info("  %s: terminal at step %d", node.node_id, step)
                elif node.is_stuck(
                    repeat_limit=self.config.stuck_repeat_limit,
                    wait_limit=self.config.stuck_wait_limit,
                ):
                    node.done = True
                    logger.info("  %s: stuck at step %d", node.node_id, step)

        # ---- Evaluate ----
        logger.info("Evaluating %d nodes...", len(tree.all_nodes()))
        self._evaluate_nodes(tree, env_workers, ray)

        # ---- Build trajectories ----
        trajectories = []
        for node in tree.all_nodes():
            if len(node.action_history) == 0:
                continue
            traj = make_mcts_trajectory(node, task_config, limit_images=self.config.limit_images)
            trajectories.append(traj)

        logger.info("Tree summary: %s", tree.summary())
        logger.info("Trajectories: %d total, %d successful",
                     len(trajectories),
                     sum(1 for t in trajectories if t.get("eval_result", 0) > 0))

        return tree, trajectories

    # ================================================================
    # Message building
    # ================================================================

    def _build_messages_for_node(self, node: TreeNode) -> List[Dict]:
        """Build conversation messages for a node's current state."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Your are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "text", "text": SYSTEM_PROMPT.format(instruction=node.instruction)},
            ]},
        ]

        screenshots = node.get_full_screenshot_history()
        actions = node.get_action_history()
        for i in range(len(actions)):
            if i < len(screenshots):
                messages.append({
                    "role": "user",
                    "content": [{"type": "image",
                                 "image": f"data:image/jpeg;base64,{screenshots[i]}"}],
                })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": actions[i]}],
            })

        if node.current_screenshot_b64:
            messages.append({
                "role": "user",
                "content": [{"type": "image",
                             "image": f"data:image/jpeg;base64,{node.current_screenshot_b64}"}],
            })

        return messages

    # ================================================================
    # Generation
    # ================================================================

    def _generate_candidates(self, messages: List[Dict], k: int) -> List[str]:
        """Generate K candidate responses using the vLLM pool."""
        from qwen_vl_utils import process_vision_info

        if k <= 0:
            return []

        messages_copy = copy.deepcopy(messages)
        messages_copy = _truncate_to_last_n_images(messages_copy, LIMIT_IMAGES)

        for msg in messages_copy:
            content = msg.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "image":
                        c.pop("min_pixels", None)
                        c.pop("max_pixels", None)

        prompt = self.processor.apply_chat_template(
            messages_copy, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, _, _ = process_vision_info(messages_copy, return_video_kwargs=True)
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        vllm_input = {"prompt_token_ids": raw_prompt_ids}
        if image_inputs:
            vllm_input["multi_modal_data"] = {"image": image_inputs}

        batch = [copy.deepcopy(vllm_input) for _ in range(k)]

        try:
            if hasattr(self.vllm_pool, 'generate_batch'):
                candidates = self.vllm_pool.generate_batch(
                    batch,
                    temperature=self.config.probe_temperature,
                    max_tokens=self.config.generation_max_tokens,
                )
            else:
                from vllm import SamplingParams
                sp = SamplingParams(
                    n=1, temperature=self.config.probe_temperature,
                    max_tokens=self.config.generation_max_tokens)
                outputs = self.vllm_pool.generate(batch, sampling_params=sp)
                candidates = [out.outputs[0].text for out in outputs]
        except Exception as e:
            logger.error("Generation failed: %s", e)
            candidates = []

        return candidates

    # ================================================================
    # Execution
    # ================================================================

    def _execute_step(self, nodes: List[TreeNode], env_workers, ray) -> None:
        """Execute actions on all given nodes in parallel."""
        futures = {}
        for node in nodes:
            if node.action is None:
                continue
            node.record_action(node.action, node.current_screenshot_b64)
            futures[node.node_id] = (
                node,
                env_workers[node.vm_slot_id].step.remote(node.action),
            )

        for node_id, (node, future) in futures.items():
            try:
                result = ray.get(future, timeout=60)
                if result.get("is_done", False):
                    node.done = True
                obs_messages = result.get("obs_messages")
                if obs_messages:
                    screenshot_b64 = _extract_screenshot_from_wire(obs_messages)
                    if screenshot_b64:
                        node.current_screenshot_b64 = screenshot_b64
            except Exception as e:
                logger.error("Step failed for %s: %s", node_id, e)
                node.done = True

    # ================================================================
    # Evaluation
    # ================================================================

    def _evaluate_nodes(self, tree: MCTSTree, env_workers, ray) -> None:
        """Evaluate all nodes that executed at least 1 step."""
        futures = {}
        for node in tree.all_nodes():
            if len(node.action_history) == 0:
                continue
            futures[node.node_id] = (
                node,
                env_workers[node.vm_slot_id].evaluate.remote(),
            )

        for node_id, (node, future) in futures.items():
            try:
                result = ray.get(future, timeout=350)
                score = result if isinstance(result, (int, float)) else result.get("score", 0.0)
                node.eval_score = float(score)
            except Exception as e:
                logger.error("Eval failed for %s: %s", node_id, e)
                node.eval_score = 0.0


# ================================================================
# Utilities
# ================================================================

def _truncate_to_last_n_images(messages, max_images):
    """Keep only the last max_images images in messages."""
    messages = copy.deepcopy(messages)
    image_positions = []
    for mi, msg in enumerate(messages):
        content = msg.get("content") or []
        if not isinstance(content, list):
            content = [content]
        for ci, c in enumerate(content):
            if isinstance(c, dict) and "image" in c:
                image_positions.append((mi, ci))
    if len(image_positions) <= max_images:
        return messages
    drop_set = set(image_positions[: len(image_positions) - max_images])
    out = []
    for mi, msg in enumerate(messages):
        content = msg.get("content") or []
        if not isinstance(content, list):
            content = [content]
        new_content = [
            c for ci, c in enumerate(content)
            if not (isinstance(c, dict) and "image" in c and (mi, ci) in drop_set)
        ]
        if new_content:
            out.append({**msg, "content": new_content})
    return out


def _extract_screenshot_from_wire(obs_messages) -> Optional[str]:
    """Extract the last screenshot base64 from wire-format messages."""
    if not obs_messages:
        return None
    for msg in reversed(obs_messages):
        content = msg.get("content", [])
        if isinstance(content, list):
            for c in reversed(content):
                if isinstance(c, dict):
                    if c.get("type") == "image" and "b64" in c:
                        return c["b64"]
                    if c.get("type") == "image" and "image" in c:
                        img_str = c["image"]
                        if img_str.startswith("data:image"):
                            return img_str.split(",", 1)[1]
    return None
