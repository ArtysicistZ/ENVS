#!/usr/bin/env python3
"""
build_prompts.py

Prep script (no API calls, no subagents): generates one fully-formed prompt
file per task under `prompts/<domain>/<task_id>.prompt.md`. Each prompt is
self-contained — it includes the task JSON, success rate, tier, budget,
noise taxonomy hints, additive-only contract, and exact output format.

A subagent launched later in Claude Code can be given just the prompt file
path as its task, and it will have everything it needs to produce:
  - <task_id>.md          (markdown analysis)
  - <task_id>_noise.json  (additive-only noisy task JSON)

Usage:
    cd OSWorld/evaluation_examples/noise_generation
    python build_prompts.py                            # generate all 86 prompts
    python build_prompts.py --task_filter <uuid>       # single task
    python build_prompts.py --domain_filter chrome     # single domain
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from difficulty import (
    TIERS,
    load_success_rates,
    rate_to_tier,
    tier_to_budget,
)
from prompts import build_user_prompt, SYSTEM_PROMPT, OUTPUT_DELIMITER


def _parse_filter(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def _build_subagent_wrapper(
    task_id: str,
    domain: str,
    user_prompt: str,
    md_out_abs: Path,
    json_out_abs: Path,
) -> str:
    """
    Wrap the user prompt with explicit instructions for a Claude Code subagent:
    where to write its two outputs, what the output contract is.
    """
    return f"""You are a noise-generation analyst for OSWorld benchmark task `{task_id}`.

You must produce TWO output files at these exact paths (write them directly with the Write tool):

1. **Markdown analysis**: `{md_out_abs}`
2. **Noisy task JSON**: `{json_out_abs}`

The task details, success rate, difficulty tier, noise budget, noise taxonomy hints, additive-only contract, and exact output template are all in the prompt below. Follow it precisely.

---

## SYSTEM PROMPT (treat this as authoritative)

{SYSTEM_PROMPT}

---

## TASK PROMPT

{user_prompt}

---

## IMPORTANT OUTPUT INSTRUCTIONS

The prompt above asks for a two-part response with a `{OUTPUT_DELIMITER}` delimiter. You must write the two halves to disk using the **Bash tool** (NOT the Write tool — Write requires interactive permission that is unavailable in this context).

**Step 1**: Create the output directory:
```
mkdir -p {md_out_abs.parent}
```

**Step 2**: Write the markdown analysis to `{md_out_abs}` using Bash with a heredoc:
```
cat << 'NOISE_MD_EOF' > {md_out_abs}
<your markdown content here>
NOISE_MD_EOF
```

**Step 3**: Write the noisy task JSON to `{json_out_abs}` using Bash with a heredoc:
```
cat << 'NOISE_JSON_EOF' > {json_out_abs}
<your pretty-printed JSON here>
NOISE_JSON_EOF
```

**Step 4**: Verify both files exist and have content:
```
ls -la {md_out_abs} {json_out_abs} && echo "OK"
```

IMPORTANT: Use single-quoted heredoc delimiters ('NOISE_MD_EOF' and 'NOISE_JSON_EOF') to prevent shell expansion of $, `, and other special characters in your content. The JSON must be pretty-printed with 2-space indentation.

After writing both files, respond with a brief one-paragraph summary: which difficulty tier this task received, how many noise elements you added, and whether any safety-checklist items are unchecked.

You MUST NOT reference any other task, any other agent, or any existing output files. Work in complete isolation based only on the prompt above.
"""


def main() -> None:
    here = Path(__file__).resolve().parent
    eval_examples = here.parent
    repo_root = eval_examples.parent.parent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(eval_examples / "test_86tasks_trainable.json"))
    ap.add_argument("--examples_root", default=str(eval_examples / "examples"))
    ap.add_argument(
        "--success_rates",
        default=str(
            repo_root
            / "checkpoints"
            / "mcts_trajectories_v2"
            / "combined_all"
            / "collection_results.json"
        ),
    )
    ap.add_argument("--out", default=str(here))
    ap.add_argument("--domain_filter", default=None, help="Comma-separated domain allowlist")
    ap.add_argument("--task_filter", default=None, help="Comma-separated task_id allowlist")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    examples_root = Path(args.examples_root)
    success_rates_path = Path(args.success_rates)
    out_root = Path(args.out)

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not examples_root.exists():
        raise SystemExit(f"Examples root not found: {examples_root}")
    if not success_rates_path.exists():
        raise SystemExit(f"Success rates not found: {success_rates_path}")

    manifest: Dict[str, List[str]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    success_rates = load_success_rates(success_rates_path)
    domain_filter = _parse_filter(args.domain_filter)
    task_filter = _parse_filter(args.task_filter)

    prompts_dir = out_root / "prompts"
    index: Dict[str, Dict[str, Any]] = {}
    tier_counts: Counter = Counter()
    missing_inputs: List[str] = []
    missing_rates: List[str] = []

    for domain, task_ids in manifest.items():
        if domain_filter and domain not in domain_filter:
            continue
        for task_id in task_ids:
            if task_filter and task_id not in task_filter:
                continue

            task_path = examples_root / domain / f"{task_id}.json"
            if not task_path.exists():
                missing_inputs.append(str(task_path))
                continue

            rate_info = success_rates.get(task_id)
            if rate_info is None:
                missing_rates.append(task_id)
                continue

            clean_task = json.loads(task_path.read_text(encoding="utf-8"))
            success_rate = rate_info["success_rate"]
            total_runs = rate_info["total_runs"]
            successful_trajectories = rate_info["successful_trajectories"]
            tier = rate_to_tier(success_rate)
            budget = tier_to_budget(tier)

            user_prompt = build_user_prompt(
                task_id=task_id,
                domain=domain,
                task_json=clean_task,
                success_rate=success_rate,
                total_runs=total_runs,
                successful_trajectories=successful_trajectories,
                tier=tier,
                budget_description=budget["description"],
            )

            md_out = (out_root / domain / f"{task_id}.md").resolve()
            json_out = (out_root / domain / f"{task_id}_noise.json").resolve()

            wrapper = _build_subagent_wrapper(
                task_id=task_id,
                domain=domain,
                user_prompt=user_prompt,
                md_out_abs=md_out,
                json_out_abs=json_out,
            )

            prompt_path = prompts_dir / domain / f"{task_id}.prompt.md"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(wrapper, encoding="utf-8")

            index[task_id] = {
                "domain": domain,
                "tier": tier,
                "success_rate": success_rate,
                "total_runs": total_runs,
                "successful_trajectories": successful_trajectories,
                "budget": budget,
                "prompt_file": str(prompt_path.resolve()),
                "md_out": str(md_out),
                "json_out": str(json_out),
                "clean_task_path": str(task_path.resolve()),
            }
            tier_counts[tier] += 1

    # Write the index file — this is what the main Claude session uses to know
    # which tasks to dispatch subagents for.
    index_path = out_root / "prompts_index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Generated {len(index)} prompt file(s) under {prompts_dir}")
    print(f"Index → {index_path}")
    print()
    print("Tier distribution:")
    for t in TIERS:
        print(f"  {t:10s}: {tier_counts.get(t, 0)}")
    if missing_inputs:
        print(f"\nWARNING: {len(missing_inputs)} missing task JSON(s):")
        for p in missing_inputs[:10]:
            print(f"  - {p}")
    if missing_rates:
        print(f"\nWARNING: {len(missing_rates)} task(s) missing from collection_results.json:")
        for t in missing_rates[:10]:
            print(f"  - {t}")


if __name__ == "__main__":
    main()
