# OSWorld Noise Generation Pipeline (86 trainable tasks)

Generate additive noise variants of the 86 trainable OSWorld tasks for robustness training, using independent **Claude Opus Code subagents** — one per several tasks — launched via the `Agent` tool from the main Claude Code session.

## Why this exists

Real desktops are messy: notifications interrupt workflows, background apps clutter the screen, files pile up on the desktop. Agents trained only on clean benchmark tasks learn brittle priors that break on realistic environments. This pipeline produces a noisy variant of each trainable task so agents can learn to route around irrelevant distractions.

Two design choices:

1. **Per-task independent subagents.** Each task gets its own Claude Code subagent with only that task's data. No cross-referencing between tasks, no few-shot examples, no shared memory. Each subagent reads its pre-built prompt file and writes its own two output files directly.
2. **Inverse-difficulty noise budget.** A task's noise budget is scaled inversely to its MCTS rollout success rate (from [checkpoints/mcts_trajectories_v2/combined_all/collection_results.json](../../../checkpoints/mcts_trajectories_v2/combined_all/collection_results.json)). Near-impossible tasks get minimal noise so they remain solvable and preserve training signal; easy tasks get maximum noise so they become non-trivial.

## Difficulty tiers

| Tier | Success rate | Count in 86 | Noise budget |
|---|---|---|---|
| `very_hard` | 0.00 – 0.05 | 15 | 1 element, 1 category (single notification) |
| `hard` | 0.05 – 0.15 | 21 | 2–3 elements, 1–2 categories |
| `medium` | 0.15 – 0.40 | 30 | 4–5 elements, 2–3 categories |
| `easy` | 0.40 – 0.75 | 15 | 6–8 elements, 3–4 categories |
| `very_easy` | 0.75 – 1.00 | 5 | 8–12 elements, ALL applicable categories |

Implemented in [difficulty.py](difficulty.py).

## Additive-only contract

Every output `<task_id>_noise.json` is a strict superset of the input task JSON:

- **All top-level fields preserved byte-identical** (`id`, `snapshot`, `instruction`, `source`, `trajectory`, `related_apps`, `evaluator`, `proxy`, `fixed_ip`, `possibility_of_env_change`, etc.).
- **All original `config` steps preserved in their original relative order.** Noise steps are inserted between them or appended, never substituted.
- **Only two new things may appear**: (1) new config steps, (2) a new top-level `noise_meta` field describing what was added.
- **Evaluator untouchable**: `evaluator.func`, `evaluator.expected`, `evaluator.result`, `evaluator.postconfig` are all byte-identical to the input. No noise step may reference any path/URL/window mentioned inside the evaluator.

This contract is enforced by [validation.py](validation.py), which can be run against any generated `_noise.json` and attaches violations to `noise_meta.validation_violations`.

## File layout

```
noise_generation/
├── README.md                           # this file
├── build_prompts.py                    # prep script: writes 86 per-task prompt files
├── prompts.py                          # shared system prompt + user prompt builder library
├── difficulty.py                       # success rate → tier → budget
├── validation.py                       # additive-contract validator
├── task_metadata.json                  # per-task precomputed metadata
├── prompts_index.json                  # generated: maps task_id → prompt_file + output paths
├── prompts/
│   └── <domain>/
│       └── <task_id>.prompt.md         # 86 self-contained prompts (input to subagents)
└── <domain>/
    ├── <task_id>.md                    # written by subagent: human-readable explanation
    └── <task_id>_noise.json            # written by subagent: drop-in noisy task
```

## Two-phase workflow

### Phase 1 — Prep (Python, no API / no subagents)

Generates 86 self-contained prompt files under `prompts/<domain>/`. Each prompt contains the full task JSON, success rate, assigned tier and budget, noise taxonomy hints, additive-only contract, and explicit instructions for the subagent to write its two outputs.

```bash
cd OSWorld/evaluation_examples/noise_generation
python build_prompts.py                             # generates all 86 prompts
python build_prompts.py --task_filter <uuid>        # single task
python build_prompts.py --domain_filter chrome      # single domain
```

Output:
- `prompts/<domain>/<task_id>.prompt.md` × 86
- `prompts_index.json` — index used by the orchestrator to dispatch subagents

### Phase 2 — Dispatch (Claude Code main session launches subagents)

The main Claude Code session reads `prompts_index.json` and launches subagents via the `Agent` tool. Each subagent is given:

- A short instruction to read a specific prompt file
- The path to the clean task JSON
- The two output paths it must write (`.md` + `_noise.json`)

Each subagent works independently: reads its prompt file, performs the noise analysis, writes both output files directly with the `Write` tool, and returns a brief summary. The main session can launch subagents in parallel batches (e.g. 8 at a time) and collect results as they complete.

There is no Python driver for Phase 2 — the main Claude Code session IS the orchestrator. This gives each subagent access to the full Claude Code toolchain (Read, Write, Bash) so it can validate its own outputs if needed.

### Phase 3 — Post-run validation (Python)

After subagents finish, run [validation.py](validation.py) against each generated `_noise.json` to confirm the additive-only contract holds. Violations are written into the JSON's `noise_meta.validation_violations` field (non-fatal — the human reviewer decides what to do with each one).

## Human review workflow

1. Once subagents have finished, each task has two output files.
2. Start by opening a few `<task_id>.md` files across different tiers. Read the "Proposed Noise Additions" + "Safety Checklist" sections.
3. Cross-check the companion `<task_id>_noise.json` against the MD — every listed addition should correspond to a new config step in the JSON.
4. Check `noise_meta.validation_violations` — non-empty lists indicate a constraint breach the human needs to adjudicate.
5. Approved `_noise.json` files become drop-in training inputs.

## Independence guarantees

- Each prompt file contains ONLY one task's data. No manifest summaries, no other task IDs, no few-shot examples, no cross-references.
- Each subagent starts fresh with no conversation history beyond its own prompt.
- The system prompt explicitly forbids cross-referencing other tasks or agents.

## References

- Task manifest: [test_86tasks_trainable.json](../test_86tasks_trainable.json)
- Canonical task definitions: [examples/](../examples/)
- Success-rate source: [checkpoints/mcts_trajectories_v2/combined_all/collection_results.json](../../../checkpoints/mcts_trajectories_v2/combined_all/collection_results.json)
- Legacy few-shot generator (historical reference only): [test_data/osworld_examples/fewshot_noise_generator_openai.py](../../../test_data/osworld_examples/fewshot_noise_generator_openai.py)
