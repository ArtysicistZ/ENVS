# Noise-Generation Redesign: Workflow-Interrupting, Action-Triggered, Capability-Adaptive

## Context

We built a static, timing-based noise pipeline at [OSWorld/evaluation_examples/noise_generation/](../../OSWorld/evaluation_examples/noise_generation/) that produced 472 noise elements across 86 OSWorld training tasks. After end-to-end smoke testing and architectural review, four problems emerged:

1. **Timing is unreliable on VMs.** 85% of elements wrap commands in `(sleep $((RANDOM % N + M)) && CMD) &`. VM clock behavior under load is unpredictable — noise may fire after the agent has already finished, or not at all.

2. **Most noise is cosmetic, not interruptive.** The current rubric (per review of `noise_generation/prompts.py:148-287`) treats noise as "messy desk" decoration: notifications that fade, window-geometry tweaks that preserve layout, filesystem clutter the agent never sees. The goal is the opposite: **simulate a concurrent human user who is genuinely competing with the agent for focus, screen real estate, and keyboard input** — forcing the agent to recognize the interruption and recover (e.g., "Chrome opened above my LibreOffice Impress — I need to click Impress in the taskbar").

3. **Noise must be ADDITIVE, never SUBTRACTIVE.** A concurrent human opens *their own* new window above the agent's work. They do **not** close the agent's terminal, kill the process the agent is running, press `Esc` while the agent has a menu open, delete the file the agent is editing, or `pkill -STOP` the app the agent is using. Any action that **destroys the agent's existing state** is sabotage, not noise. The test: *if the agent did nothing and the noise's effect expired/was dismissed, would the task still be completable?* If no, it's sabotage.

4. **Noise is an OPT-IN feature, not a default override.** Clean (noise-free) training and eval must remain bit-for-bit unchanged when the feature flag is off. All noise machinery (scheduler, curriculum, wandb logging, env kwargs) activates only when `algorithm.enable_noise=true`. Default is `false`. Existing configs (`arpo_8gpu.yaml`, `grpo_8gpu.yaml`, `sft_eval_*.yaml`, all eval configs) continue to work exactly as today without any edits.

This plan replaces the pipeline with (a) a redesigned rubric that produces workflow-interrupting but non-sabotaging noise, (b) action-triggered execution via an env-side scheduler, and (c) a trainer-side curriculum controller that adjusts trigger probability per task based on the agent's current capability — all gated behind a single config flag.

## Guiding principles for the new noise rubric

These are the rules a new Claude subagent prompt must enforce. The existing `noise_generation/prompts.py:21-56` ("MESSY DESK PRINCIPLE") must be replaced.

**The headline rule (the only test that matters):** *every noise element must be reversible by the agent using ≤ 3 standard agent actions* (pyautogui click / type / hotkey). Anything above 3 steps of recovery is sabotage, not noise, and must be rejected — regardless of tier.

A "standard agent action" is what's in the agent's existing action space:
- `CLICK(x, y)` at coordinates visible on screen
- `TYPING(text)` / `PRESS(key)` / `HOTKEY(...)` to the focused window
- `SCROLL(dx, dy)` / `DRAG_TO(x1, y1, x2, y2)`

A noise element is legal **if and only if** there exists a sequence of ≤ 3 such actions that restores the agent to a state functionally equivalent to "noise never fired." The subagent must declare this sequence in the element's `recovery_actions` field (see format below) — if it can't write a concrete ≤ 3-step recovery plan, reject.

**Corollaries of the 3-step rule (derived, not separate invariants):**

*Allowed* (always recoverable in ≤ 3 steps, works on the agent's own app OR any other):
- Move a window partially off-screen (at least one edge still visible or taskbar handle present). Recovery: 1 drag, or 1 click in taskbar + 1 drag = ≤ 2 steps.
- Resize a window smaller. Recovery: 1 drag on a corner = 1 step (or ignored entirely; cost 0 if agent works in reduced area).
- Steal focus to a noise-owned new window. Recovery: 1 click on the agent's app or 1 Alt-Tab = 1 step.
- Overlap the agent's app with a non-fullscreen window. Recovery: 1 click on agent's app in taskbar = 1 step.
- Modal dialog (`zenity`). Recovery: 1 click on OK/Cancel = 1 step.
- Passive notification. Recovery: 0 steps (agent ignores; toast fades on its own).
- Fullscreen a *non-target* window (agent recovers via 1 Alt-Tab + 1 click = 2 steps).

*Forbidden* (cannot be recovered in ≤ 3 steps OR cannot be recovered by any standard agent action):
- `pkill`, `kill -9`, `pkill -STOP` on any process. Recovery requires process restart, which the agent's action space cannot express → reject.
- `wmctrl -c`, `xdotool windowclose`, `xdotool windowunmap` — closing/unmapping a window. Same reason.
- Move a window *entirely* off-screen with no taskbar handle. Recovery needs `wmctrl -r ... -e` which the agent can't invoke → reject.
- Fullscreen-cover the *agent's own app* such that its taskbar entry is hidden and Alt-Tab cycles away from it on XFCE — recovery path ambiguous → reject. (Fullscreening a non-target window is still fine; agent recovers from fullscreen with Alt-Tab.)
- Screen lock / logout / session-switch — needs password, no agent action can recover → reject.
- Keystroke/mouse injection into a window the noise did **not** itself just open and focus — lands on the agent's input and corrupts its state, unrecoverable → reject. Specifically: `pyautogui.press('esc')` / `hotkey('alt','f4')` / `typewrite(...)` with no preceding explicit focus-set to a noise-owned window.
- File deletion (`rm`), rename (`mv`), chmod, or any write to a path referenced in the task's `evaluator` block — destroys state the agent needs to succeed → reject.
- Any element whose `recovery_actions` field has length > 3.

**Interruption severity (what we now want):**
- **Occlude the agent's working window from ABOVE.** The agent's app remains open and fully functional; it is just visually covered. Recovery = click the agent's app in the taskbar, or Alt+Tab. Use: `wmctrl -r <noise_window> -b add,above` on a window the noise itself just opened; or launch a maximized / fullscreen new window (`wmctrl -r <noise_window> -b add,fullscreen`).
- **Modal dialog from a non-target source that blocks clicks until dismissed.** `zenity --info --text '...'` or `zenity --question --text '...'`. `zenity` is already installed. Recovery = click OK/Cancel on the zenity dialog; agent's work underneath is untouched.
- **Focus-steal to a new noise-owned window.** Noise opens its own window (e.g., a new `gedit` untitled buffer) and the WM gives it focus. Agent's app still exists, just no longer has keyboard focus. Recovery = click the agent's app to refocus. Must NOT send keystrokes while the noise owns focus — the stolen focus is the interruption, not a keystroke-injection attack.
- **Passive notifications (kept as low-severity layer).** `notify-send` still allowed; they are "background life" the agent can ignore.

**Severity budget per task (replaces current element-count-only budget):**

Each noise element's `recovery_cost` is the *length* of the agent's recovery sequence — the number of standard agent actions (click/type/hotkey/scroll/drag) required to return to a state functionally equivalent to "noise never fired." **Per the headline rule, any element with recovery_cost > 3 is rejected outright.**

- cost 0: passive — agent recovers by ignoring (notification fades, Desktop file agent never opens).
- cost 1: one agent action — click to refocus, click to dismiss modal, drag to pull a partially off-screen window back.
- cost 2: two agent actions — Alt-Tab + click, drag + resize-back.
- cost 3: three agent actions — maximum allowed per element.
- cost ≥ 4: **not allowed as noise; must be rejected by the validator.**

The validator additionally requires each element to carry a concrete `recovery_actions` list (one entry per agent action, written in the agent's action-space vocabulary). If the subagent cannot produce a ≤3-step recovery plan, the element is not noise.

Tier budgets are **cumulative recovery-cost ceilings per task**, not element counts:
- `very_hard` (success_rate ≤ 5%): total cost ≤ **1** (one cost-1 element, or passive notifications only). Preserves training signal.
- `hard` (≤ 15%): total cost ≤ **2**.
- `medium` (≤ 40%): total cost ≤ **4**.
- `easy` (≤ 75%): total cost ≤ **6**.
- `very_easy` (> 75%): total cost ≤ **9**.

A "medium" task can have (1 non-target occlusion cost 2) + (1 modal cost 1) + (1 focus-steal cost 1) + unlimited cost-0 passive notifications — total cost 4, within budget.

## Feature flag and opt-in contract

A single new config field gates the entire feature:

```yaml
algorithm:
  enable_noise: false   # default off; opt-in only
  noise_p_init: 0.10    # ignored when enable_noise=false
  noise_target_sr: 0.5  # ignored when enable_noise=false
  noise_p_max: 0.30     # ignored when enable_noise=false
```

**Contract (verified in tests):**
- When `enable_noise=false`: `desktop_env.reset()` does NOT instantiate `NoiseScheduler`; `desktop_env.step()` does NOT read the `noise_probability` kwarg; `ray_trainer` does NOT create `NoiseCurriculum`; no wandb `noise/*` keys are logged; no `noise_meta` lookups. Training behavior is byte-identical to pre-feature.
- When `enable_noise=true` and a task has no `noise_meta`: scheduler is a no-op; no crash, no warning spam.
- When `enable_noise=true` and task has `noise_meta`: full machinery engages.
- Existing 300-task clean eval configs MUST continue to produce identical numbers pre- vs post-feature. This is a regression test.

## Architecture changes (four layers)

### Layer 1 — Regenerate the 86 noise sets under the new rubric

Modify the prompt-builder pipeline so the 86 per-task Claude subagent prompts reflect the new rubric, then re-dispatch.

**Files to modify:**
- [OSWorld/evaluation_examples/noise_generation/prompts.py](../../OSWorld/evaluation_examples/noise_generation/prompts.py) — rewrite system prompt + taxonomy to emphasize interruption severity, non-sabotage rules, `related_apps` awareness, and the agent-step recovery-cost budget.
- [OSWorld/evaluation_examples/noise_generation/difficulty.py](../../OSWorld/evaluation_examples/noise_generation/difficulty.py) — replace element-count budget (`min_elements`/`max_elements`) with cost-ceiling budget (`max_recovery_cost`).
- [OSWorld/evaluation_examples/noise_generation/validation.py](../../OSWorld/evaluation_examples/noise_generation/validation.py) — add new checks: (a) no command references any string in `related_apps` of the clean task, (b) no command in the forbidden-action list (`pkill`, `rm`, `mv`, `chmod`, `wmctrl -c`, `xdotool windowclose`, `pyautogui.press`/`typewrite` without explicit focus-target argument, etc.), (c) sum of declared per-element `recovery_cost` ≤ tier ceiling.
- [OSWorld/evaluation_examples/noise_generation/build_prompts.py](../../OSWorld/evaluation_examples/noise_generation/build_prompts.py) — extract `related_apps` from clean task and pass into prompt context.

**Element format in the regenerated `_noise.json` (matches Layer 2 scheduler):**
```json
"noise_meta": {
  "tier": "medium",
  "success_rate_used": 0.22,
  "rubric_version": 2,
  "total_recovery_cost": 3,
  "elements": [
    {
      "id": "n0",
      "category": "modal_dialog",
      "severity": "medium",
      "recovery_cost": 1,
      "recovery_actions": ["click the OK button on the zenity dialog"],
      "touches_target_app": false,
      "once": true,
      "description": "Zenity dialog from a non-task source; agent dismisses with one click.",
      "command": ["bash", "-c", "zenity --info --title 'System Update' --text 'A new version is available.' 2>/dev/null"]
    },
    {
      "id": "n1",
      "category": "target_window_shove",
      "severity": "medium",
      "recovery_cost": 1,
      "recovery_actions": ["drag the LibreOffice Impress titlebar back to a visible position"],
      "touches_target_app": true,
      "once": true,
      "description": "Shove the task's Impress window partially off the right edge; title bar remains visible.",
      "command": ["bash", "-c", "wmctrl -r 'LibreOffice Impress' -e 0,1400,200,1200,800"]
    }
  ]
}
```
The `touches_target_app` field replaces the previous `targets_own_app` (now allowed when the interaction is reversible in ≤ 3 steps). The `recovery_actions` field is required — it is the subagent's proof that the element satisfies the headline rule.

Write regenerated files fresh; archive the current action-triggered v1 outputs to `_noise.json.v1_bak` before overwriting. Keep the existing `*.pretrigger_bak` files untouched for deeper rollback.

**Taxonomy (new, replacing the 10-category list):**

Categories are organized by whether they touch the agent's own app (`target_*`) or only other windows (`non_target_*` / passive). Both families are legal provided the ≤3-step recovery rule holds.

| Category | Touches target? | Typical cost | Example | Recovery |
|---|---|---|---|---|
| `passive_notification` | no | 0 | `notify-send 'Slack' 'New message'` | agent ignores; toast fades |
| `filesystem_clutter` | no | 0 | `touch ~/Desktop/scratch_$RANDOM.txt` (not in evaluator paths) | agent ignores |
| `background_window_open` | no | 0–1 | `gedit &` (no focus grab) | agent ignores or 1 click if focus leaked |
| `non_target_focus_steal` | no | 1 | `gedit &` → WM focuses it | 1 click on agent's app to refocus |
| `non_target_occlusion` | no | 2 | `(gnome-calculator &); wmctrl -r Calculator -b add,above,fullscreen` | 1 Alt-Tab + 1 click |
| `modal_dialog` | no | 1 | `zenity --info --text '...'` | 1 click OK |
| `target_window_shove` | yes | 1 | `wmctrl -r <TargetApp> -e 0,1400,200,1200,800` (partially off right edge; titlebar visible) | 1 drag to bring back on-screen |
| `target_window_shrink` | yes | 0–1 | `wmctrl -r <TargetApp> -e 0,100,100,800,600` (shrink to 800×600) | 0 (work in reduced area) or 1 drag on corner |
| `target_focus_drop` | yes | 1 | `(zenity --info ...) &` steals focus from target | 1 click on target window to refocus |
| `target_partial_overlap` | yes | 1 | `(gedit &); wmctrl -r gedit -e 0,0,0,800,600` overlapping half of target | 1 click on target's visible half, or 1 click in taskbar |

Forbidden (no legal example, always rejected):
- `pkill`, `kill -9`, `pkill -STOP` of any process.
- `wmctrl -c`, `xdotool windowclose/unmap` of any window.
- Full-screen cover of the *target* window with no taskbar recovery.
- Move the target window entirely off-screen with no visible handle.
- Keystroke/mouse injection into any window the noise did not itself just open and focus.
- `rm`/`mv`/`chmod` of any file; any write to evaluator-referenced paths.
- Screen lock / logout / session switch.
- Previous cosmetic-only categories that reliably fail the ≥0-cost-meaningful bar: `panel_toggle`, `view_mode_change`, `scroll_displacement` (they're either cost-0 ignorable or cost-unclear; merge their use-cases into `background_window_open` or drop).

### Layer 2 — Env-side `NoiseScheduler` (action-triggered, not timing-based)

New module at `OSWorld/desktop_env/noise_scheduler.py` (create).

**Responsibilities:**
- Parse `noise_meta.elements` from the task JSON on env reset.
- Expose `on_step(controller, probability)` called after each agent action.
- For each unfired (or `once=false` repeatable) element, roll `rng.random() < probability`. On hit, execute its `command` via the existing `controller._execute_setup(command, shell=False)` ([OSWorld/desktop_env/controllers/setup.py:341](../../OSWorld/desktop_env/controllers/setup.py)) — reuse, do not reimplement.
- Track `fired` state so `once=true` elements do not re-trigger.
- Log firings with task_id, element_id, step_idx, probability for wandb introspection.

**Integration points:**
- [OSWorld/desktop_env/desktop_env.py](../../OSWorld/desktop_env/desktop_env.py): in `reset()`, if `enable_noise` and task JSON has `noise_meta`, instantiate `NoiseScheduler`. In `step(action, **kwargs)`, accept a kwarg `noise_probability: float` and call `self.noise_scheduler.on_step(...)` after action execution.
- [verl/trainer/ray_trainer.py](../../verl/trainer/ray_trainer.py) line ~1500–1800 (rollout chunk runner): when `enable_noise=true`, plumb a per-task `noise_probability` value through to each env worker's `step()` call. Start as a constant per-task; Layer 3 makes it adaptive.

**No VM-side code changes.** Commands fire through the existing `/setup/execute` HTTP endpoint that smoke-tested clean.

### Layer 3 — Trainer-side `NoiseCurriculum` (capability-adaptive `p_t`)

New module at `verl/workers/noise_curriculum.py` (create).

```python
class NoiseCurriculum:
    target_success_rate: float = 0.5
    lr: float = 0.05
    p_min: float = 0.0
    p_max: float = 0.3

    def update(self, task_id, rolling_success_rate): ...
    def get(self, task_id) -> float: ...
```

**Update rule:**
`p[task_id] += lr * (rolling_success_rate - target_success_rate)`, clamped to `[p_min, p_max]`.
- If agent solves the task reliably above the target → `p` climbs → noise fires more often → interruption pressure rises.
- If agent is failing → `p` drops toward 0 → noise gets out of the way, preserving training signal.

**Rolling success rate:** maintain an EMA per task over the last N rollouts (N=8, matching `rollout.n`). Reset on checkpoint load.

**Wandb logging:** log `noise/p/<task_id>` per step plus `noise/p_mean`, `noise/firing_rate`, `noise/recovery_cost_consumed`. Enables direct inspection of the curriculum's behavior.

**Integration in fit loop:** after the per-step rollout reward compute in `ray_trainer.py` (~line 1900 where `traj_reward` is aggregated), update the curriculum per task with that step's per-task success. On the next step, pass `curriculum.get(task_id)` through the rollout chunk to the env scheduler.

### Layer 4 — OSWorld image fix (owned by this implementation)

From smoke test, three tools are missing from `osworld:latest`: `libnotify-bin` (provides `notify-send`), `gnome-calculator`, `xdotool`. `zenity` (critical for new `modal_dialog` category) IS already installed.

Rebuild the image with:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnotify-bin gnome-calculator xdotool \
  && rm -rf /var/lib/apt/lists/*
```

Steps:
1. Add the line to the image build recipe.
2. Rebuild locally on 10.100.4.8 (~10 min).
3. `docker save | ssh … docker load` to distribute to 10.100.4.4 and 10.100.4.6.
4. Roll existing `osworld-slot-*` containers one-by-one (DockerProvider will recreate on demand).

Rollback path: prior image is tagged `osworld:pre_noise` before rebuild, so `docker tag osworld:pre_noise osworld:latest` reverts. Important: since noise is opt-in, the new image is fully backward-compatible — all three missing binaries become available but unused until `enable_noise=true`.

## Reuse of existing code (avoid reinvention)

- **Command execution:** reuse `Setup._execute_setup` and `Setup._launch_setup` in [OSWorld/desktop_env/controllers/setup.py:317-425](../../OSWorld/desktop_env/controllers/setup.py). Both map `type=execute`/`type=launch` to the `/setup/execute` HTTP endpoint that already works end-to-end (smoke-test verified).
- **Validation framework:** extend [OSWorld/evaluation_examples/noise_generation/validation.py](../../OSWorld/evaluation_examples/noise_generation/validation.py). Keep `_config_subsequence_ok`, `_extract_protected_strings`, `_walk_strings` — add `_check_non_sabotage`, `_check_related_apps_disjoint`, `_check_recovery_cost_budget`.
- **Tier/success-rate mapping:** keep `rate_to_tier` in `difficulty.py`; only replace the per-tier `*_BUDGETS` dict.
- **Subagent dispatch:** keep the existing Phase 2 workflow (orchestrator launches Claude subagents reading `prompts/<domain>/<task_id>.prompt.md` and writing two output files). Only the prompt *content* changes.
- **Backups and migration tooling:** keep [OSWorld/evaluation_examples/noise_generation/migrate_to_action_triggered.py](../../OSWorld/evaluation_examples/noise_generation/migrate_to_action_triggered.py) for any residual format migration. Its `_unwrap_timing` / `_cleanup_tail` logic stays useful for subagent-output edge cases.

## Files to create / modify

**Modify:**
- `OSWorld/evaluation_examples/noise_generation/prompts.py` — rewrite system prompt, taxonomy, severity rubric.
- `OSWorld/evaluation_examples/noise_generation/difficulty.py` — replace element budgets with recovery-cost ceilings.
- `OSWorld/evaluation_examples/noise_generation/validation.py` — add non-sabotage + related-apps + cost-budget checks.
- `OSWorld/evaluation_examples/noise_generation/build_prompts.py` — include `related_apps` in per-task prompt context.
- `OSWorld/desktop_env/desktop_env.py` — accept `noise_probability` kwarg on `step()`, init `NoiseScheduler` on `reset()` (only when enabled).
- `verl/trainer/ray_trainer.py` — plumb `noise_probability` per task through the rollout loop (only when enabled); add `NoiseCurriculum` updates after per-step reward compute.
- `verl/trainer/config.py` (or wherever `AlgorithmConfig` lives) — add `enable_noise`, `noise_p_init`, `noise_target_sr`, `noise_p_max` fields with opt-in defaults.
- `OSWorld/Dockerfile` (or image build recipe) — add `libnotify-bin gnome-calculator xdotool`.

**Create:**
- `OSWorld/desktop_env/noise_scheduler.py` — the `NoiseScheduler` class.
- `verl/workers/noise_curriculum.py` — the `NoiseCurriculum` class.

**Archive (do not delete):**
- Every `OSWorld/evaluation_examples/noise_generation/<domain>/<task_id>_noise.json` is copied to `<task_id>_noise.json.v1_bak` before full regeneration. The existing `*.pretrigger_bak` files (original timing-based outputs) are preserved as-is for deeper rollback.

**Do not create new:**
- No new command-execution path; reuse `_execute_setup`.
- No new HTTP endpoints on the VM server.
- No new config-step type; element commands run through existing `type=execute`.

## Verification (end-to-end, robust)

**A. Feature-flag regression (must pass before any noise change is shipped):**
- With `enable_noise=false` (default), run the existing ARPO step-105 n=1 greedy eval on 300 tasks. Compare `eval_results_at_105.json` to the pre-feature output. Expect **byte-identical** results. Any drift means noise machinery is leaking into clean-eval path — blocker.
- Same for n=8 t=1 eval. Same for SFT v2.1 eval configs.
- Unit-test: import `desktop_env` and assert that when `enable_noise=false` is in config, no `NoiseScheduler` is constructed even if `noise_meta` is present in the task JSON.

**B. Rubric-layer verification (before subagent dispatch):**
- Unit-test `validation.py` on hand-crafted `_noise.json` samples:
  - `pkill chrome`, `rm file`, `wmctrl -c`, any `pyautogui.press` with no explicit preceding focus-set: flagged.
  - Element with `recovery_cost > 3` or with `recovery_actions` missing / empty / length > 3: flagged.
  - Element whose cumulative task cost exceeds tier ceiling: flagged.
  - Full-screen cover of the task's target app (identified via `related_apps`) with no taskbar handle: flagged.
  - Valid workflow-interrupting element that touches target app within ≤3-step recovery (e.g. `target_window_shove` with titlebar visible + `recovery_actions: ["drag titlebar"]`): **passes**.
  - Valid cross-app element (zenity modal + `recovery_actions: ["click OK"]`): passes.
- Regression against the existing additive contract: `validate_additive` must still pass.

**C. Regeneration pilot:**
- Dispatch 5 subagents (one per tier) using the new prompt. Manually inspect each generated `<task_id>.md` explanation. Every element must declare:
  - `recovery_cost: <int 0..3>`
  - `recovery_actions: [<1..3 actions in agent-action-space vocabulary>]`
  - `touches_target_app: true|false`
  - human-language justification of how the agent would recover (which screenshot it would see, which UI element it would click)
- Run new `validation.py` — zero violations required.
- If pilot passes, dispatch remaining 81 tasks. Archive v1 outputs to `*.v1_bak`. Total subagent dispatch cost ≈ same as original pipeline.

**D. Env scheduler unit tests (noise enabled):**
- Mock controller, instantiate `NoiseScheduler` with a 3-element task.
- `on_step(mock_ctrl, 1.0)` three times with `once=true` elements; each fires exactly once.
- With `once=false`; repeatable elements fire on every step.
- With `p=0.0`; zero firings.
- With `enable_noise=false` in config; scheduler is never constructed (covered by test A).
- Fault injection: an element whose command exits non-zero → scheduler marks it fired (don't retry infinitely), logs a warning, continues.

**E. End-to-end single-task rollout (on rebuilt image):**
- One rollout of one medium-tier LibreOffice Impress task at `noise_probability=0.3`. Verify:
  1. Agent's target app window (Impress) remains open and reachable throughout all noise firings — check screenshots before/after each firing. Target-touching elements (shove / shrink / partial-overlap / focus-drop) are allowed; forbidden is Impress being closed, killed, or rendered unrecoverable.
  2. For every firing, the element's declared `recovery_actions` sequence is actually executable from the post-firing screenshot (spot-check 3–5 firings).
  3. Wandb receives `noise/*` keys with expected cardinality.
  4. Agent completes OR hits 15-step budget — no VM crash, no hang.
- Repeat on a chrome task (different `related_apps`) to confirm taxonomy specializes correctly per task.

**F. Curriculum verification:**
- Run 20 steps with a task the agent always solves; confirm `p_t` monotonically rises toward `p_max`.
- Run 20 steps with a task the agent always fails; confirm `p_t` monotonically drops toward 0.
- Run 20 steps with a stochastic task (SR ≈ 0.5 = target); confirm `p_t` hovers near `p_init`.
- Verify EMA does not oscillate wildly under noise; no NaN/inf even with zero-success edge cases.
- `load_checkpoint` + resume: curriculum state is restored correctly (or reset cleanly if not persisted — document which).

**G. Paper-signal experiment (final check, longer-running):**
- Train one GRPO (noise-free) baseline and one GRPO (noise-enabled) run on the same 86 tasks for matched step count (e.g., 30 steps).
- Eval both checkpoints on: (1) 86 clean, (2) 86 noisy at fixed `p=0.1`, (3) 214 held-out clean, (4) 214 held-out at fixed `p=0.1` (requires noise generation for 214; deferred unless time allows).
- Expected signal: noisy-trained ≥ clean-trained on noisy eval; parity or small gain on clean eval; any improvement on held-out is a bonus.
- If noise-enabled run degrades clean eval by > 2 pts, the rubric is letting through too much interference — revisit cost ceilings.

## What is deliberately NOT in this plan

- **Noise on held-out 214 tasks.** Not needed for robustness-during-training; would be needed only if we want to eval "robustness generalization" as a separate axis. Defer.
- **Multi-category per-element probabilities.** We start with a single scalar `p_t` per task; per-category probabilities can be added later if logs show one category dominating / being useless.
- **Online noise regeneration.** The subagent-produced elements stay fixed per task; only the trigger probability is adaptive. Full online regeneration is interesting but expensive (each task's noise set would need another Claude pass) and not required for the "self-adjusting curriculum" paper claim.
