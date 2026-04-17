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
from verl.trainer.gui_agent import add_box_token

# Noise support (v3 noisy MCTS)
try:
    from OSWorld.evaluation_examples.noise_generation.runtime_sampler import (
        RuntimeNoiseSampler,
        fires_for_sr_mcts,
        feasibility_constrained_fire_steps,
    )
    _NOISE_AVAILABLE = True
except ImportError:
    _NOISE_AVAILABLE = False


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

    def __init__(self, config: MCTSConfig, vllm_pool, processor, tokenizer,
                 task_sr_map: Optional[Dict[str, float]] = None):
        self.config = config
        self.vllm_pool = vllm_pool
        self.processor = processor
        self.tokenizer = tokenizer
        # Per-task clean SR from v2 collection (used for noise fire count)
        self._task_sr_map: Dict[str, float] = task_sr_map or {}

    # ----------------------------------------------------------------
    # Noise helpers (v3 noisy MCTS)
    # ----------------------------------------------------------------

    def _noise_fire_count_for_task(self, task_id: str) -> int:
        """Determine noise fire count based on task's clean SR."""
        sr = self._task_sr_map.get(task_id, 0.0)
        if not _NOISE_AVAILABLE or not self.config.enable_noise:
            return 0
        return fires_for_sr_mcts(sr)

    def _should_branch_have_noise(self, rng) -> bool:
        """Decide if a branch gets noise (vs. clean control)."""
        return rng.random() < self.config.noise_branch_probability

    def _build_noise_task_config(self, task_config: Dict, node: TreeNode) -> Dict:
        """Build a task_config with per-branch noise settings for env reset.

        If noise is disabled for this node, returns config with enable_noise=False.
        Otherwise, sets a unique noise_step_seed so the env samples a unique schedule.
        """
        cfg = dict(task_config)
        if not node.noise_enabled:
            cfg["enable_noise"] = False
            return cfg

        task_id = task_config.get("id", "")
        sr = self._task_sr_map.get(task_id, 0.0)
        cfg["enable_noise"] = True
        cfg["noise_mode"] = self.config.noise_mode
        cfg["noise_step_seed"] = node.noise_seed
        cfg["noise_success_rate"] = sr
        cfg["noise_observed_sr"] = sr
        cfg["noise_probability"] = 0.1  # vestigial for v4 scheduler
        cfg["noise_max_steps"] = self.config.max_steps
        return cfg

    def _build_clean_task_config(self, task_config: Dict) -> Dict:
        """Build a task_config with noise disabled (for replay)."""
        cfg = dict(task_config)
        cfg["enable_noise"] = False
        return cfg

    def _assign_noise_to_node(self, node: TreeNode, task_id: str,
                               branch_idx: int = 0) -> None:
        """Assign noise metadata to a node (seed, fire count, schedule)."""
        import random as _rng_mod

        if not self.config.enable_noise or not _NOISE_AVAILABLE:
            node.noise_enabled = False
            return

        rng = _rng_mod.Random(hash((task_id, node.node_id, branch_idx)) & 0xFFFFFFFF)
        if not self._should_branch_have_noise(rng):
            node.noise_enabled = False
            return

        fire_count = self._noise_fire_count_for_task(task_id)
        if fire_count <= 0:
            node.noise_enabled = False
            return

        node.noise_enabled = True
        node.noise_seed = hash((task_id, node.node_id, branch_idx)) & 0xFFFFFFFF
        node.noise_fire_count = fire_count

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
                task_id = task_config.get("id", "")
                self._assign_noise_to_node(root, task_id, branch_idx=0)
                if root.noise_enabled:
                    logger.info("Root node %s: noise_seed=%d, fire_count=%d",
                                root.node_id, root.noise_seed, root.noise_fire_count)
                tree.add_root(root)
                active_nodes.append(root)
                alive = [root]

            if not alive:
                logger.info("Step %d: no active nodes, stopping", step)
                break

            logger.info("Step %d: %d active, %d waiting, cap=%d",
                        step, len(alive), len(waiting), self.config.max_active_vms)

            # ---- Phase 1: Get screenshots (PARALLEL — fire all, then collect) ----
            if step > 0:
                ss_futures = [
                    (node, env_workers[node.vm_slot_id].get_obs_screenshot.remote())
                    for node in alive
                ]
                for node, future in ss_futures:
                    try:
                        obs = ray.get(future, timeout=30)
                        if obs:
                            node.current_screenshot_b64 = obs
                        else:
                            logger.warning("  %s: no screenshot at step %d", node.node_id, step)
                    except Exception as e:
                        logger.warning("  %s: screenshot failed: %s", node.node_id, e)

            # ---- Phase 2: Generate K candidates (BATCHED — one generate_batch call) ----
            n_alive = len(alive)
            n_remaining = len(waiting)

            can_branch = n_remaining > 0 and _n_active() < self.config.max_active_vms
            if not can_branch:
                k = 1
                logger.info("  K=1 (cap reached or no VMs remaining)")
            else:
                k = max(1, min(self.config.k_max,
                               -(-self.config.total_probe_budget // n_alive)))
                logger.info("  K=%d (%d active, %d remaining, cap=%d)",
                            k, n_alive, n_remaining, self.config.max_active_vms)

            # Prepare all vllm_inputs on CPU, then generate in ONE batch
            all_vllm_inputs = []
            node_ranges = []  # (node, start_idx, end_idx)
            for node in alive:
                if not node.current_screenshot_b64:
                    node.candidates = []
                    continue
                messages = self._build_messages_for_node(node)
                vllm_input = self._prepare_vllm_input(messages)
                if vllm_input is None:
                    node.candidates = []
                    continue
                start = len(all_vllm_inputs)
                # Shallow copy: share multi_modal_data refs (vLLM doesn't mutate inputs)
                for _ in range(k):
                    all_vllm_inputs.append({
                        "prompt_token_ids": vllm_input["prompt_token_ids"],
                        "multi_modal_data": vllm_input.get("multi_modal_data"),
                    })
                node_ranges.append((node, start, start + k))

            if all_vllm_inputs:
                logger.info("  Generating %d prompts (%d nodes × K=%d)",
                            len(all_vllm_inputs), len(node_ranges), k)
                all_results = self._generate_batch_raw(all_vllm_inputs)
                for node, start, end in node_ranges:
                    node.candidates = all_results[start:end]
                    if not node.candidates:
                        logger.warning("  %s: empty candidates at step %d", node.node_id, step)

            # ---- Phase 3: Branch decisions (sequential) + replay (PARALLEL) ----
            replay_tasks = []  # (child, future, timeout)
            for node in alive:
                if len(node.candidates) < 2:
                    node.action = node.get_majority_action() or (node.candidates[0] if node.candidates else None)
                    continue

                if should_branch(node.candidates, step, node, self.config):
                    clusters = cluster_by_fingerprint(
                        node.candidates,
                        grid_size=self.config.spatial_grid_size,
                        min_cluster_size=self.config.min_cluster_size,
                    )
                    node.action = representative(node.candidates, clusters[0])

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
                        task_id = task_config.get("id", "")
                        self._assign_noise_to_node(child, task_id,
                                                    branch_idx=node_counter[0])
                        tree.add_child(node, child)
                        active_nodes.append(child)

                        # Snapshot parent's logical history at branch time (frozen)
                        child.parent_action_snapshot = list(node.get_action_history())
                        child.parent_screenshot_snapshot = list(node.get_full_screenshot_history())
                        child.parent_steps_at_branch = len(node.action_history)

                        parent_physical = node.get_physical_action_sequence()
                        child.replay_prefix = list(parent_physical)
                        child.screenshot_history = []
                        child.current_screenshot_b64 = node.current_screenshot_b64

                        # Fire replay async — collect later
                        # NOTE: replay uses noise-free config (noise starts after branch)
                        if parent_physical:
                            replay_timeout = max(60, len(parent_physical) * 5 + 30)
                            noise_tag = (f", noise_seed={child.noise_seed}"
                                         if child.noise_enabled else ", clean")
                            logger.info("  Spawning %s: replay %d actions on VM %d (async%s)",
                                        child.node_id, len(parent_physical), vm_idx, noise_tag)
                            future = env_workers[vm_idx].replay.remote(
                                parent_physical,
                                replay_pause_sec=self.config.replay_pause_sec,
                            )
                            replay_tasks.append((child, future, replay_timeout))

                    if any(c.parent == node and c.depth == step for c in active_nodes if c != node):
                        node.budget.use()
                else:
                    node.action = node.get_majority_action() or (node.candidates[0] if node.candidates else None)

            # Collect ALL replays in parallel (must finish before Phase 4)
            if replay_tasks:
                logger.info("  Waiting for %d replays in parallel...", len(replay_tasks))
                for child, future, timeout in replay_tasks:
                    try:
                        ray.get(future, timeout=timeout)
                    except Exception as e:
                        logger.error("  Replay failed for %s: %s", child.node_id, e)
                        child.done = True

            # Handle just-spawned nodes
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

    def _prepare_vllm_input(self, messages: List[Dict]) -> Optional[Dict]:
        """Prepare a single vllm_input dict from messages (CPU work only)."""
        from qwen_vl_utils import process_vision_info

        messages_copy = copy.deepcopy(messages)
        messages_copy = _truncate_to_last_n_images(messages_copy, LIMIT_IMAGES)

        for msg in messages_copy:
            content = msg.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "image":
                        c["max_pixels"] = 2116800
                        c["min_pixels"] = 256

        try:
            prompt = self.processor.apply_chat_template(
                messages_copy, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, _, _ = process_vision_info(messages_copy, return_video_kwargs=True)
            raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        except Exception as e:
            logger.error("Prompt preparation failed: %s", e)
            return None

        n_images = len(image_inputs) if image_inputs else 0
        img_sizes = [f"{img.size}" for img in image_inputs] if image_inputs else []
        est_img_tokens = sum((img.size[0]//14)*(img.size[1]//14)//4 for img in image_inputs) if image_inputs else 0
        logger.info("  Prompt: %d text_ids + %d images %s ≈ %d img_tokens → total ≈ %d",
                    len(raw_prompt_ids), n_images, img_sizes,
                    est_img_tokens, len(raw_prompt_ids) + est_img_tokens)
        # DEBUG: dump first prompt with >2000 text tokens
        if len(raw_prompt_ids) > 2000:
            logger.warning("  LARGE PROMPT DUMP (%d text tokens):\n%s", len(raw_prompt_ids), prompt[:3000])

        vllm_input = {"prompt_token_ids": raw_prompt_ids}
        if image_inputs:
            vllm_input["multi_modal_data"] = {"image": image_inputs}
        return vllm_input

    def _generate_batch_raw(self, vllm_inputs: List[Dict]) -> List[str]:
        """Generate responses for a flat list of vllm_inputs in one batch."""
        if not vllm_inputs:
            return []
        try:
            if hasattr(self.vllm_pool, 'generate_batch'):
                return self.vllm_pool.generate_batch(
                    vllm_inputs,
                    temperature=self.config.probe_temperature,
                    max_tokens=self.config.generation_max_tokens,
                )
            else:
                from vllm import SamplingParams
                sp = SamplingParams(
                    n=1, temperature=self.config.probe_temperature,
                    max_tokens=self.config.generation_max_tokens)
                outputs = self.vllm_pool.generate(vllm_inputs, sampling_params=sp)
                return [out.outputs[0].text for out in outputs]
        except Exception as e:
            logger.error("Batch generation failed: %s", e)
            return [""] * len(vllm_inputs)

    # ================================================================
    # Execution
    # ================================================================

    def _execute_step(self, nodes: List[TreeNode], env_workers, ray) -> None:
        """Execute actions on all given nodes in parallel."""
        futures = {}
        for node in nodes:
            if node.action is None:
                continue
            node.record_action(add_box_token(node.action), node.current_screenshot_b64)
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
                # Capture noise burden (v3 noisy MCTS)
                noise_burden = result.get("noise_burden")
                if noise_burden and node.noise_enabled:
                    fired_this_step = noise_burden.get("events_fired_this_step", 0)
                    if fired_this_step > 0:
                        node.noise_events_fired.append({
                            "step": node.current_step() - 1,
                            "events": noise_burden.get("events", []),
                            "step_recovery_cost": noise_burden.get("step_recovery_cost", 0),
                        })
                        node.noise_total_recovery_cost += noise_burden.get("step_recovery_cost", 0)
                        logger.info("  %s: noise fired at step %d (cost=%d)",
                                    node_id, node.current_step() - 1,
                                    noise_burden.get("step_recovery_cost", 0))
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
