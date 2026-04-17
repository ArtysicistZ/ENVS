"""
validation.py

Additive-contract validator for noise-generation outputs.

Checks that a noisy task JSON is strictly additive over the clean input:
  - All non-config top-level fields byte-identical
  - Original config steps appear as a subsequence of noisy config in the same
    relative order
  - Instruction, evaluator.func, evaluator.expected unchanged
  - No noise step references any path/URL/window protected by the evaluator
  - Target-keyword count not increased (preserves keyword-based eval checks)
  - noise_meta.noise_element_count matches the actual delta

Violations are returned as a list of strings. Empty list = OK. The driver
attaches violations to noise_meta.validation_violations (non-fatal) so the
human reviewer can decide what to do with each one.

Logic for the instruction/evaluator/subsequence/target-keyword checks is
adapted from test_data/osworld_examples/fewshot_noise_generator_openai.py:80-124.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Set


# Fields that the driver is allowed to ADD to the noisy JSON (not present in
# clean). These are whitelisted so the top-level field-equality check does not
# flag them as violations.
_ALLOWED_NEW_TOP_LEVEL_FIELDS = {"noise_meta"}


def _json_minify(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _extract_target_keywords(task: Dict[str, Any]) -> List[str]:
    """
    If evaluator.expected.rules.type == 'keywords', return those keywords
    lowercased. Noise must not increase their occurrence count.
    """
    evaluator = task.get("evaluator") or {}
    expected = evaluator.get("expected")

    def _from_rules(rules: Any) -> List[str]:
        if isinstance(rules, dict) and rules.get("type") == "keywords":
            kw = rules.get("keywords")
            if isinstance(kw, list):
                return [str(x).lower() for x in kw]
        return []

    if isinstance(expected, dict):
        return _from_rules(expected.get("rules") or {})
    if isinstance(expected, list):
        for item in expected:
            if isinstance(item, dict):
                found = _from_rules(item.get("rules") or {})
                if found:
                    return found
    return []


def _walk_strings(obj: Any) -> Iterable[str]:
    """Recursively yield all string leaves of a JSON-like structure."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


# Heuristic path/URL extraction — anything that looks like an absolute file
# path, a URL, or a window name is considered "protected".
_PATH_LIKE_RE = re.compile(r"(?:https?://[^\s\"']+|/[A-Za-z0-9_./\-]{3,})")


def _extract_protected_strings(evaluator: Dict[str, Any]) -> Set[str]:
    """
    Walk evaluator.result, evaluator.expected, evaluator.postconfig and extract
    all "protected" substrings that noise must not reference in command args.
    """
    protected: Set[str] = set()
    for key in ("result", "expected", "postconfig"):
        sub = evaluator.get(key)
        if sub is None:
            continue
        for s in _walk_strings(sub):
            # Whole-string forms: full URLs, absolute paths, window names
            if s.startswith("http://") or s.startswith("https://") or s.startswith("/"):
                if len(s) >= 4:
                    protected.add(s)
            # Regex-based: find embedded path-like substrings
            for m in _PATH_LIKE_RE.findall(s):
                if len(m) >= 4:
                    protected.add(m)
    return protected


def _config_subsequence_ok(clean_cfg: List[Any], noisy_cfg: List[Any]) -> bool:
    """
    Check that every element of clean_cfg appears in noisy_cfg in the same
    relative order. Adapted from fewshot_noise_generator_openai.py:102-113.
    """
    j = 0
    for step in clean_cfg:
        found = False
        while j < len(noisy_cfg):
            if noisy_cfg[j] == step:
                found = True
                j += 1
                break
            j += 1
        if not found:
            return False
    return True


def _find_added_steps(clean_cfg: List[Any], noisy_cfg: List[Any]) -> List[Any]:
    """
    Return the noisy steps that are NOT part of the clean subsequence.

    Uses a greedy j-pointer walk that matches clean steps against noisy in
    order; everything in noisy that isn't consumed by a clean match is an
    added (noise) step.
    """
    added: List[Any] = []
    j = 0
    for step in clean_cfg:
        while j < len(noisy_cfg) and noisy_cfg[j] != step:
            added.append(noisy_cfg[j])
            j += 1
        if j < len(noisy_cfg) and noisy_cfg[j] == step:
            j += 1
        else:
            # clean step not found; the subsequence check already flags this
            # separately, so we bail on added-step accounting.
            return []
    while j < len(noisy_cfg):
        added.append(noisy_cfg[j])
        j += 1
    return added


def validate_additive(clean: Dict[str, Any], noisy: Dict[str, Any]) -> List[str]:
    """
    Return a list of violation strings. Empty list = valid.
    """
    violations: List[str] = []

    # 1. Instruction immutability.
    if noisy.get("instruction") != clean.get("instruction"):
        violations.append("instruction_changed")

    # 2. Evaluator func/expected immutability.
    clean_ev = clean.get("evaluator") or {}
    noisy_ev = noisy.get("evaluator") or {}
    if noisy_ev.get("func") != clean_ev.get("func"):
        violations.append("evaluator_func_changed")
    if noisy_ev.get("expected") != clean_ev.get("expected"):
        violations.append("evaluator_expected_changed")
    if clean_ev.get("result") != noisy_ev.get("result"):
        violations.append("evaluator_result_changed")
    if clean_ev.get("postconfig") != noisy_ev.get("postconfig"):
        violations.append("evaluator_postconfig_changed")

    # 3. Every top-level field of clean (except `config` which may grow) must
    #    be present in noisy with byte-identical value.
    for key, value in clean.items():
        if key == "config":
            continue
        if key not in noisy:
            violations.append(f"missing_top_level_field:{key}")
        elif noisy[key] != value:
            violations.append(f"top_level_field_changed:{key}")

    # 4. Noisy may introduce only whitelisted new top-level fields.
    for key in noisy.keys():
        if key in clean:
            continue
        if key not in _ALLOWED_NEW_TOP_LEVEL_FIELDS:
            violations.append(f"unexpected_new_top_level_field:{key}")

    # 5. Config must be a list and must preserve clean as a subsequence.
    clean_cfg = clean.get("config", [])
    noisy_cfg = noisy.get("config", [])
    if not isinstance(clean_cfg, list) or not isinstance(noisy_cfg, list):
        violations.append("config_not_list")
        return violations  # nothing else makes sense if config shape is wrong
    if not _config_subsequence_ok(clean_cfg, noisy_cfg):
        violations.append("clean_config_step_missing_or_reordered")

    # 6. Evaluator-path blacklist: noise-only steps must not reference
    #    any protected string from the evaluator.
    added_steps = _find_added_steps(clean_cfg, noisy_cfg)
    protected = _extract_protected_strings(clean_ev)
    if protected and added_steps:
        for idx, step in enumerate(added_steps):
            step_str = _json_minify(step)
            for p in protected:
                if p in step_str:
                    violations.append(
                        f"noise_step_{idx}_touches_protected_path:{p[:80]}"
                    )
                    break

    # 7. Target-keyword count must not increase.
    target_kw = _extract_target_keywords(clean)
    if target_kw:
        noisy_str = _json_minify(noisy).lower()
        clean_str = _json_minify(clean).lower()
        for k in target_kw:
            if noisy_str.count(k) > clean_str.count(k):
                violations.append(f"introduced_target_keyword:{k}")

    # 8. noise_meta.noise_element_count must match the real delta.
    noise_meta = noisy.get("noise_meta") or {}
    declared = noise_meta.get("noise_element_count")
    actual = len(noisy_cfg) - len(clean_cfg)
    if declared is not None and declared != actual:
        violations.append(
            f"noise_element_count_mismatch:declared={declared},actual={actual}"
        )

    return violations
