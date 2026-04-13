"""
migrate_to_action_triggered.py

Converts timing-based noise (`_noise.json` files) to action-triggered format.

Before:
    - Added config steps contain shell commands wrapped in `(sleep X && CMD) &`
    - Noise fires after a fixed wall-clock delay inside the VM
    - Unreliable on VMs where timing is subtle

After:
    - `config` is reverted to the clean baseline (byte-identical to input task JSON)
    - `noise_meta.elements[]` lists bare commands (no timing wrappers)
    - At training time, the env scheduler rolls dice per element per agent step
    - Trigger probability is passed in from the trainer (capability-adaptive)

For each source added-step, the migration:
  1. Extracts the bash payload
  2. Strips `(sleep ... &&` and trailing `) &` / `>/dev/null 2>&1` wrappers
  3. Splits `&& sleep ... &&` chains into multiple elements (so each original
     queued command becomes a separate dice-roll candidate)
  4. Detects `while true; do sleep X; CMD; done` → single element with
     `once: false` (repeatable; fires independently per action)
  5. Tags each element with its source category (from noise_meta.categories_applied
     where available) and a stable `id = "n<index>"`

Outputs:
  - Overwrites each `<task_id>_noise.json` in place (after a one-shot backup to
    `<task_id>_noise.json.pretrigger_bak` for rollback)
  - Report printed to stdout: tasks migrated, elements extracted, any steps
    that could not be parsed.

Usage:
    cd OSWorld/evaluation_examples/noise_generation
    python migrate_to_action_triggered.py --dry-run   # preview
    python migrate_to_action_triggered.py             # apply in place
    python migrate_to_action_triggered.py --restore   # undo from backups
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


NG_ROOT = Path(__file__).parent
BASE_EXAMPLES = NG_ROOT.parent / "examples"


# Patterns for stripping timing wrappers.
# Matches `(sleep $((RANDOM % N + M)) && REST)` and captures REST.
_SLEEP_PREFIX_RE = re.compile(
    r"^\(\s*sleep\s+\$?\(?\(?RANDOM[^)]*\)?\)?[^&]*&&\s*(.+?)\)\s*(?:>/dev/null)?\s*(?:2>&1)?\s*&?\s*$"
)
# Matches a literal `sleep N` prefix (no RANDOM).
_SLEEP_LITERAL_PREFIX_RE = re.compile(
    r"^\(\s*sleep\s+[\d.]+\s*&&\s*(.+?)\)\s*(?:>/dev/null)?\s*(?:2>&1)?\s*&?\s*$"
)
# Matches a `while true; do sleep X; CMD; done` loop. Captures CMD.
_WHILE_LOOP_RE = re.compile(
    r"^\(\s*while\s+(?:true|:)\s*;\s*do\s+sleep\s+[^;]+;\s*(.+?);\s*done\s*\)\s*(?:>/dev/null)?\s*(?:2>&1)?\s*&?\s*$"
)
# Matches a bare foreground/background command with redirection, like `gedit >/dev/null 2>&1 &`.
_BARE_CMD_RE = re.compile(
    r"^(.+?)\s*(?:>/dev/null)?\s*(?:2>&1)?\s*&?\s*$"
)
# Matches an intermediate `sleep X &&` (for splitting nested sleeps).
_INTERNAL_SLEEP_RE = re.compile(
    r"&&\s*sleep\s+\$?\(?\(?(?:RANDOM|[\d.]+)[^&]*\)?\)?\s*&&"
)


_WHILE_LOOP_INNER_RE = re.compile(
    r"\(\s*while\s+(?:true|:)\s*;\s*do\s+sleep\s+[^;]+;\s*(.+?);\s*done\s*\)\s*(?:>/dev/null)?\s*(?:2>&1)?\s*&?"
)
_LAZY_SLEEP_WRAPPER_RE = re.compile(
    r"\(\s*sleep\s+\$?\(?\(?(?:RANDOM|[\d.]+)[^)]*\)?\)?\s*&&\s*([^()]+?)\)\s*(?:>/dev/null)?\s*(?:2>&1)?\s*&?"
)


def _extract_while_loops(s: str) -> Tuple[str, List[str]]:
    """
    Strip any embedded `(while true; do sleep X; CMD; done)` subexpressions out
    of `s`. Returns (remaining_text, [CMDs_extracted]).

    In the new action-triggered design, these loops are redundant: the env
    scheduler rolls dice per agent step, so a repeatable noise element fires
    many times naturally across a rollout. We keep the inner CMD but drop the
    loop wrapper.
    """
    extracted: List[str] = []
    def _sub(match):
        extracted.append(match.group(1).strip())
        return ""
    new = _WHILE_LOOP_INNER_RE.sub(_sub, s)
    return new, extracted


def _extract_sleep_wrappers(s: str) -> Tuple[str, List[str]]:
    """
    Strip any embedded `(sleep X && CMD)` subexpressions, returning inner CMDs.
    """
    extracted: List[str] = []
    def _sub(match):
        extracted.append(match.group(1).strip())
        return ""
    new = _LAZY_SLEEP_WRAPPER_RE.sub(_sub, s)
    return new, extracted


def _cleanup_tail(s: str) -> str:
    """Trim trailing `&& ` / `; ` / `|| true` / redirections / backgrounding / stray fd digits."""
    s = s.strip()
    # Peel off repeated trailing junk.
    for _ in range(8):
        original = s
        s = re.sub(r"\s*\|\|\s*true\s*$", "", s)
        s = re.sub(r"\s*>/dev/null\s*$", "", s)
        s = re.sub(r"\s*2>&1\s*$", "", s)
        s = re.sub(r"\s*2>/dev/null\s*$", "", s)
        # Stray trailing "2" left over from partially-stripped `2>/dev/null`.
        s = re.sub(r"\s+2\s*$", "", s)
        s = re.sub(r"\s*&\s*$", "", s)
        s = re.sub(r"\s*;\s*$", "", s)
        s = re.sub(r"\s*&&\s*$", "", s)
        s = s.strip()
        if s == original:
            break
    return s


def _unwrap_timing(shell_src: str) -> Tuple[List[str], bool]:
    """
    Given a shell string, return (list_of_bare_commands, is_repeatable).

    Handles:
      - `(sleep X && CMD) & ...` → ["CMD"], once=True
      - `(sleep X && A && sleep Y && B) &` → ["A", "B"], once=True
      - `(while true; do sleep X; CMD; done) &` → ["CMD"], once=False
      - `CMD1 && (while true; do sleep X; CMD2; done)` → ["CMD1", "CMD2"], once=False
        (first is once=True, second once=False — caller must split; here we
        return the mix and rely on the caller to use once=False when any
        while-loop was present)
      - `CMD &` or plain `CMD` → ["CMD"], once=True
    """
    s = shell_src.strip()

    # Whole-string while-loop form → single repeatable.
    m = _WHILE_LOOP_RE.match(s)
    if m:
        return [_cleanup_tail(m.group(1))], False

    # Extract embedded while-loops first (they turn into repeatable elements).
    s, while_cmds = _extract_while_loops(s)
    # Extract embedded `(sleep X && CMD)` subexpressions.
    s, lazy_cmds = _extract_sleep_wrappers(s)

    # Now strip the top-level (sleep X && ...) wrapper if present.
    m = _SLEEP_PREFIX_RE.match(s) or _SLEEP_LITERAL_PREFIX_RE.match(s)
    if m:
        inner = m.group(1).strip()
    else:
        m = _BARE_CMD_RE.match(s)
        inner = m.group(1).strip() if m else s

    # Split on internal `&& sleep X &&` chains.
    parts = _INTERNAL_SLEEP_RE.split(inner)
    parts = [_cleanup_tail(p) for p in parts]
    parts = [p for p in parts if p]

    # Append stripped-out embedded commands.
    parts.extend(_cleanup_tail(c) for c in lazy_cmds if _cleanup_tail(c))
    parts.extend(_cleanup_tail(c) for c in while_cmds if _cleanup_tail(c))

    # If we saw a while-loop, everything becomes repeatable.
    once = len(while_cmds) == 0
    return parts, once


def _find_added_steps(clean_cfg: List[Any], noisy_cfg: List[Any]) -> List[Any]:
    """Greedy j-pointer walk: noisy steps not in clean subsequence = added."""
    added: List[Any] = []
    j = 0
    for step in clean_cfg:
        while j < len(noisy_cfg) and noisy_cfg[j] != step:
            added.append(noisy_cfg[j])
            j += 1
        if j < len(noisy_cfg) and noisy_cfg[j] == step:
            j += 1
        else:
            return []  # subsequence broken; migration can't proceed
    while j < len(noisy_cfg):
        added.append(noisy_cfg[j])
        j += 1
    return added


# Heuristic category tagger — infers category from the command if noise_meta
# does not attach per-step categories. Used as a fallback.
_CATEGORY_HINTS = [
    ("random_notifications", re.compile(r"\bnotify-send\b")),
    ("filesystem_clutter",   re.compile(r"\b(?:touch|mkdir)\b.*(?:Desktop|Documents|Downloads|/tmp)")),
    ("window_geometry",      re.compile(r"\bwmctrl\b.*-e\b")),
    ("focus_stealing",       re.compile(r"\bwmctrl\b\s+-a\b")),
    ("window_occlusion",     re.compile(r"\bwmctrl\b.*-b\b\s*add")),
    ("view_mode_change",     re.compile(r"pyautogui|xdotool.*key")),
    ("panel_toggle",         re.compile(r"gsettings|xfconf-query")),
    ("dialog_popup",         re.compile(r"\bzenity\b|\bnotify-send\b.*--urgency=critical")),
    ("scroll_displacement",  re.compile(r"xdotool\s+click\s+[45]")),
    ("background_apps",      re.compile(r"\b(?:gedit|nautilus|firefox|libreoffice|file-roller|thunderbird|vlc|gnome-calculator|gnome-system-monitor|gnome-terminal|gnome-text-editor)\b")),
]


def _infer_category(shell: str) -> Optional[str]:
    for cat, rx in _CATEGORY_HINTS:
        if rx.search(shell):
            return cat
    return None


def _step_to_elements(step: Dict[str, Any], start_id: int, default_cat: Optional[str]) -> List[Dict[str, Any]]:
    """
    Convert one added config step into one or more noise elements.
    Returns a list of element dicts.
    """
    params = step.get("parameters") or {}
    cmd = params.get("command") or []
    if not isinstance(cmd, list) or len(cmd) == 0:
        return []
    # Only the trailing string carries the shell payload for ["bash", "-c", "..."].
    if cmd[0] in ("bash", "sh") and len(cmd) >= 3 and cmd[1] in ("-c", "-lc"):
        shell = cmd[-1]
        bares, once = _unwrap_timing(shell)
    else:
        # Non-shell command (e.g. ["python3", "-c", "..."]) — treat as single
        # atomic element, no timing to strip.
        return [{
            "id": f"n{start_id}",
            "category": default_cat or _infer_category(" ".join(str(x) for x in cmd)) or "unknown",
            "once": True,
            "command": cmd,
        }]

    out: List[Dict[str, Any]] = []
    for i, bare in enumerate(bares):
        cat = _infer_category(bare) or default_cat or "unknown"
        out.append({
            "id": f"n{start_id + i}",
            "category": cat,
            "once": once,
            "command": ["bash", "-c", bare],
        })
    return out


def migrate_one(task_id: str, domain: str, dry_run: bool) -> Dict[str, Any]:
    noise_path = NG_ROOT / domain / f"{task_id}_noise.json"
    clean_path = BASE_EXAMPLES / domain / f"{task_id}.json"
    backup_path = NG_ROOT / domain / f"{task_id}_noise.json.pretrigger_bak"

    if not noise_path.exists() or not clean_path.exists():
        return {"status": "missing", "task_id": task_id, "domain": domain}

    with clean_path.open("r", encoding="utf-8") as f:
        clean = json.load(f)
    with noise_path.open("r", encoding="utf-8") as f:
        noisy = json.load(f)

    # Already migrated?
    existing_meta = noisy.get("noise_meta") or {}
    if "elements" in existing_meta:
        return {"status": "already_migrated", "task_id": task_id, "domain": domain,
                "elements": len(existing_meta.get("elements", []))}

    clean_cfg = clean.get("config", [])
    noisy_cfg = noisy.get("config", [])

    added = _find_added_steps(clean_cfg, noisy_cfg)
    if not added and len(clean_cfg) != len(noisy_cfg):
        return {"status": "subsequence_broken", "task_id": task_id, "domain": domain}

    cats_applied = (existing_meta.get("categories_applied") or [])
    default_cat = cats_applied[0] if len(cats_applied) == 1 else None

    elements: List[Dict[str, Any]] = []
    for step in added:
        new_els = _step_to_elements(step, start_id=len(elements), default_cat=default_cat)
        elements.extend(new_els)

    if not elements and added:
        return {"status": "no_elements_extracted", "task_id": task_id, "domain": domain,
                "n_added": len(added)}

    new_meta = dict(existing_meta)
    new_meta["elements"] = elements
    new_meta["trigger_mode"] = "action_triggered_probabilistic"
    new_meta["migration_note"] = (
        "Timing-based wrappers removed. Commands fire per-action via env "
        "scheduler using a probability passed from the trainer. "
        f"Original added-step count: {len(added)}; elements produced: {len(elements)}."
    )

    new_noisy = dict(noisy)
    new_noisy["config"] = list(clean_cfg)  # revert to clean
    new_noisy["noise_meta"] = new_meta

    if dry_run:
        return {"status": "ok_dry_run", "task_id": task_id, "domain": domain,
                "n_added": len(added), "n_elements": len(elements)}

    # Backup once.
    if not backup_path.exists():
        shutil.copy2(noise_path, backup_path)
    with noise_path.open("w", encoding="utf-8") as f:
        json.dump(new_noisy, f, ensure_ascii=False, indent=2)
    return {"status": "migrated", "task_id": task_id, "domain": domain,
            "n_added": len(added), "n_elements": len(elements)}


def restore_one(task_id: str, domain: str) -> Dict[str, Any]:
    noise_path = NG_ROOT / domain / f"{task_id}_noise.json"
    backup_path = NG_ROOT / domain / f"{task_id}_noise.json.pretrigger_bak"
    if not backup_path.exists():
        return {"status": "no_backup", "task_id": task_id}
    shutil.copy2(backup_path, noise_path)
    return {"status": "restored", "task_id": task_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    idx = json.loads((NG_ROOT / "prompts_index.json").read_text())
    results: List[Dict[str, Any]] = []
    for tid, entry in idx.items():
        dom = entry["domain"]
        if args.restore:
            results.append(restore_one(tid, dom))
        else:
            results.append(migrate_one(tid, dom, dry_run=args.dry_run))

    # Summary.
    from collections import Counter
    status_counts = Counter(r["status"] for r in results)
    total_elements = sum(r.get("n_elements", 0) for r in results if r["status"] in ("migrated", "ok_dry_run"))
    total_added = sum(r.get("n_added", 0) for r in results if r["status"] in ("migrated", "ok_dry_run"))

    print(f"\n=== Migration summary ===")
    print(f"Tasks processed: {len(results)}")
    for s, n in sorted(status_counts.items()):
        print(f"  {s}: {n}")
    if total_added:
        print(f"Source added-steps: {total_added}")
        print(f"Extracted elements: {total_elements}")
        print(f"Avg elements per task: {total_elements / len(results):.2f}")

    # Fail loudly if any task couldn't be migrated.
    bad_statuses = {"missing", "subsequence_broken", "no_elements_extracted"}
    bad = [r for r in results if r["status"] in bad_statuses]
    if bad:
        print(f"\nERRORS in {len(bad)} tasks:")
        for r in bad[:10]:
            print(f"  {r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
