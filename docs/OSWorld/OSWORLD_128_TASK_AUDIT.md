# OSWorld 128 Training Task Audit

**Date:** 2026-03-08
**Scope:** All 128 tasks in `test_data/osworld_examples/train_all_128.json`

## Task Distribution

| App | Count | Infeasible |
|-----|-------|------------|
| libreoffice_calc | 12 | 1 |
| vlc | 6 | 3 |
| multi_apps | 11 | 1 |
| chrome | 18 | 2 |
| vs_code | 19 | 5 |
| os | 12 | 4 |
| libreoffice_impress | 13 | 0 |
| libreoffice_writer | 11 | 1 |
| thunderbird | 7 | 1 |
| gimp | 19 | 8 |
| **Total** | **128** | **26 (20%)** |

---

## 1. Reset Mechanism Analysis

### Two Reset Strategies

**A. Full Snapshot Revert (Docker/VMware)**
- Destroys and recreates the VM/container from a clean image
- Resets ALL state (filesystem, processes, configs, packages)
- Slow (~30-90s depending on provider)

**B. Soft Reset via Port 5001 Daemon (AWS)**

The reset daemon (`osworld-resetd.service`, port 5001) performs:

1. **Stop 5000 server** (OSWorld control plane)
2. **Stop X11 display session** (graphical desktop)
3. **Kill ALL user processes** (SIGKILL, 15s timeout)
4. **Wipe & remount OverlayFS** at `/home/user/` — restores entire home directory to baseline
5. **Clear `/tmp`, `/var/tmp`** (user-owned files only)
6. **Clear user crontab**
7. **Clear `/run/user/{uid}/`** (D-Bus, PulseAudio sockets, etc.)
8. **Restore dconf** (GNOME desktop settings from baseline snapshot)
9. **Restart X11 display + 5000 server**

Total time: ~5-8 seconds.

### What Soft Reset Covers

| State Category | Location | Covered? |
|---------------|----------|----------|
| Files in `/home/user/` | OverlayFS | ✅ Wiped completely |
| Chrome profile/settings | `~/.config/google-chrome/` | ✅ Inside home |
| VS Code settings/extensions | `~/.config/Code/`, `~/.vscode/` | ✅ Inside home |
| VLC config (vlcrc) | `~/.config/vlc/` | ✅ Inside home |
| GIMP config (gimprc, sessionrc) | `~/.config/GIMP/` | ✅ Inside home |
| Thunderbird profile | `~/.thunderbird/` | ✅ Inside home |
| LibreOffice config | `~/.config/libreoffice/` | ✅ Inside home |
| Desktop files | `~/Desktop/` | ✅ Inside home |
| Git repos created by tasks | `~/projects/` etc. | ✅ Inside home |
| GNOME settings (gsettings) | dconf database | ✅ dconf restore |
| PulseAudio (volume) | `~/.config/pulse/` + `/run/user/` | ✅ Both cleared |
| Default apps (xdg-mime) | `~/.local/share/applications/` | ✅ Inside home |
| Temp files | `/tmp`, `/var/tmp` | ✅ Cleared |
| User processes | All killed | ✅ SIGKILL |

### What Soft Reset Does NOT Cover

| State Category | Location | Covered? |
|---------------|----------|----------|
| System packages (apt/snap) | `/usr/`, `/snap/`, `/var/lib/dpkg/` | ❌ Persists |
| System config | `/etc/` | ❌ Persists |
| Kernel parameters | `/proc/sys/` | ❌ Persists |
| System services | systemd state | ❌ Persists |
| Network config | `/etc/netplan/` etc. | ❌ Persists |

---

## 2. Per-Task Reset Safety Assessment

### ✅ SAFE with Soft Reset (125 of 128 tasks)

The vast majority of tasks only modify state within `/home/user/` or dconf:
- All **libreoffice_calc** (12): Download file to home → edit → save
- All **vlc** (6): Config in `~/.config/vlc/`
- All **chrome** (18): Profile in `~/.config/google-chrome/` (2 setup tasks install `jq` but that's idempotent)
- All **vs_code** (19): Settings in `~/.config/Code/`
- 11 of 12 **os**: gsettings (dconf), files on Desktop, trash
- All **multi_apps** except 1: Files in home, git repos
- All **libreoffice_impress** (13): Files on Desktop
- All **libreoffice_writer** (11): Files on Desktop
- All **thunderbird** (7): Profile in `~/.thunderbird/`
- All **gimp** (19): Config in `~/.config/GIMP/`, images on Desktop

### ⚠️ NEEDS ATTENTION (3 of 128 tasks)

| Task ID | App | Issue | Risk |
|---------|-----|-------|------|
| `94d95f96` | os | Agent is asked to **install Spotify** (system package via apt/snap). Persists across soft resets. | **HIGH** — Once installed, subsequent runs find Spotify already present. Evaluator (`which spotify`) would pass even without agent action. |
| `3299584d` | chrome | Setup runs `sudo apt install jq`. Idempotent; jq stays installed. | **LOW** — jq is a harmless utility; doesn't affect any other task. |
| `9656a811` | chrome | Setup runs `sudo apt install jq`. Same as above. | **LOW** — Same as 3299584d. |

### Recommendation for `94d95f96` (Install Spotify)

This is the only task where soft reset is insufficient. Options:
1. **Accept the risk** — In training, if the agent succeeds once, Spotify persists. On subsequent rollouts the evaluator will always return 1.0 (free reward). This corrupts the training signal.
2. **Add cleanup** — After evaluation, run `sudo snap remove spotify` or equivalent as a post-evaluation step.
3. **Use full relaunch** for this specific task — Flag it so the provider does a hard relaunch instead of soft reset.
4. **Remove from training set** — Replace with a less problematic task.

**Recommendation:** Option 2 or 4. Option 2 is simplest — add a post-evaluation cleanup step. Option 4 avoids the issue entirely and is safest for training stability.

### Recommendation for `3299584d` and `9656a811` (apt install jq)

No action needed. `jq` is a small JSON utility installed during setup. It's idempotent (re-running `apt install jq` when already installed is a no-op) and doesn't affect any other task.

---

## 3. Action Execution Bugs

### BUG 1: `press` action references undefined `hotkey` variable (CRITICAL) — FIXED

**File:** `verl/trainer/gui_agent.py` and `osworld_patches/uitars_agent.py`

**Was:** Arrow key conversion in the `press` block referenced `hotkey` (from the `hotkey` branch) instead of `key_to_press`, causing NameError crashes and dead code.

**Fix applied:** Replaced all `hotkey` references with `key_to_press` in the press block. Verified all arrow key conversions and space work correctly.

### BUG 2: `str(None)` truthy for click with missing `start_box` (HIGH) — FIXED

**File:** `verl/trainer/gui_agent.py` and `osworld_patches/uitars_agent.py`

**Was:** `str(None)` produces `"None"` (truthy), then `eval("None")` returns Python `None`, and `len(None)` crashes.

**Fix applied:** Added `if start_box is None: pass` guard before `str()`/`eval()` conversion. Click actions with missing coordinates are now silently skipped.

### BUG 3: `pyperclip.copy()` string injection (MEDIUM) — FIXED

**File:** `verl/trainer/gui_agent.py` and `osworld_patches/uitars_agent.py`

**Was:** Content with backslashes or quotes inside `f"\npyperclip.copy('{content}')"` produced malformed Python code.

**Fix applied:** Replaced `'{stripped_content}'` with `{repr(stripped_content)}` in both `pyperclip.copy()` and `pyautogui.write()` calls. Verified with quotes and backslash content.

### BUG 4: `{` in pyautogui commands breaks PythonController (MEDIUM) — FIXED

**File:** `OSWorld/desktop_env/controllers/python.py` line 142

**Was:** `self.pkgs_prefix.format(command=command)` could crash if the pyautogui command contained `{` or `}` characters.

**Fix applied:** Replaced `.format(command=command)` with `.replace("{command}", command)` which is safe for arbitrary command content.

### BUG 5: `<|box_start|>` markers not stripped from coordinates (CRITICAL) — FIXED

**File:** `verl/trainer/gui_agent.py` and `osworld_patches/uitars_agent.py`

**Was:** `parse_action_to_structure_output` parsed `start_box`/`end_box` coordinate strings by removing `(` and `)` then splitting on `,`, but did NOT strip `<|box_start|>` and `<|box_end|>` markers. This caused `float('<|box_start|>500')` → `ValueError`, crashing all scroll and drag actions (and potentially click/double-click if markers were present).

**Discovered by:** End-to-end VM pipeline test (`scripts/test_e2e_vm_pipeline.py`).

**Fix applied:** Added `ori_box = ori_box.replace("<|box_start|>", "").replace("<|box_end|>", "")` before coordinate parsing in both files.

---

## 4. Per-App Task Issues

### LibreOffice Calc (12 tasks)

| Task ID | Issue | Severity |
|---------|-------|----------|
| `1e8df695` | Instruction typo "CGOS" instead of "COGS" | LOW |
| `357ef137` | Missing `expected` field in evaluator — may cause KeyError during evaluation | MEDIUM |
| `2bd59342` | Infeasible (sparklines not in LibreOffice) | OK |
| `6e99a1ad` | Postconfig runs `libreoffice --convert-to csv` while LO instance is open — single-instance lock conflict | MEDIUM |
| `4e6fcf72` | Date-sensitive age calculation | LOW |

### VLC (6 tasks)
No issues. 3 infeasible tasks correctly marked.

### Chrome (18 tasks)

| Task ID | Issue | Severity |
|---------|-------|----------|
| `a728a36e` | `proxy: false` but accesses external website dmv.virginia.gov | HIGH |
| `b4f95342` | `proxy: false` but accesses recreation.gov; `possibility_of_env_change: high` | HIGH |
| `2ad9387a` | Bookmark evaluator strict set equality — fails if pre-existing folders | MEDIUM |
| `e1e75309` | PDF filename must exactly match browser default (special chars like `'`) | MEDIUM |
| `af630914` | Font size evaluator too lenient (accepts >16 for "largest") | LOW |

### VS Code (19 tasks)

| Task ID | Issue | Severity |
|---------|-------|----------|
| `5e2d93d8` | Instruction says save workspace at `/home/user/` but evaluator checks `/home/user/project/` — **MISMATCH** | HIGH |
| `c6bf789c` | `check_json_settings` uses exact dict equality for `files.exclude` — fails if VS Code has default patterns | HIGH |
| `e2b5e914` | Same nested dict equality issue for `diagnosticSeverityOverrides` | MEDIUM |
| `ec71221e` | Postconfig window title may not match VS Code actual title | MEDIUM |
| `0512bb38`, `4e60007a`, `eabc805a` | Evaluator uses pipe `\|` in command array — works only if executor joins as shell | MEDIUM |

### OS (12 tasks)

| Task ID | Issue | Severity |
|---------|-------|----------|
| `94d95f96` | Install Spotify requires internet AND persists across soft resets | HIGH |
| `e0df059f` | Directory created with `sudo` — agent may face permission issues renaming | MEDIUM |

### Multi-Apps (11 tasks)

| Task ID | Issue | Severity |
|---------|-------|----------|
| `acb0f96b` | `proxy: false` but needs GitHub for `git clone`; gold file is static `ls -R` | HIGH |
| `2b9493d7` | Possible evaluator bash syntax error (extra `]` bracket) | MEDIUM |
| `3a93cae4` | Ambiguous timetable instruction (no course name) | LOW |

### LibreOffice Impress (13 tasks)

| Task ID | Issue | Severity |
|---------|-------|----------|
| `a097acff` | Relative path `Downloads/` instead of `/home/user/Downloads/` | MEDIUM |
| `9cf05d24` | "Green" is ambiguous — exact shade must match gold | MEDIUM |
| `39be0d19`, `70bca0cc`, `73c99fb9`, `af23762e` | Full pptx comparison, no relaxed options — very strict | LOW |

### LibreOffice Writer (11 tasks)
No significant issues. 1 infeasible task correctly marked.

### Thunderbird (7 tasks)

| Task ID | Issue | Severity |
|---------|-------|----------|
| `dd84e895` | Evaluator checks `sum(1) > 0` — passes if JUST ONE email starred, not ALL | HIGH |
| `dfac9ee8` | Tarball extraction without `--recursive-unlink` may cause profile merge issues | MEDIUM |
| `a1af9f1c` | Infeasible (debatable — SMTP-only is possible in some versions) | LOW |

### GIMP (19 tasks)

| Task ID | Issue | Severity |
|---------|-------|----------|
| `554785e9`, `72f83cdc`, `7a4deb26`, `e2dd0213`, `f723c744` | **Fragile postconfig export** — uses pyautogui to trigger Export As dialog; fails if GIMP state is unexpected | HIGH |
| `d52d6308` | Evaluator checks `hide-docks=yes` (all docks) but instruction says "left dock" only | MEDIUM |
| `b148e375` | Evaluator only checks gimprc for last-used layer name, not actual layer existence | MEDIUM |

---

## 5. Can Soft Reset (Restarting 5000 & 5001) Replace Full Snapshot Revert?

### Answer: YES, for 125 of 128 tasks.

The soft reset via port 5001 is **sufficient** for almost all tasks because:

1. **OverlayFS wipe restores `/home/user/` completely** — All app configs (Chrome, VS Code, VLC, GIMP, Thunderbird, LibreOffice) live under `/home/user/.config/` or `/home/user/.local/`, which are fully wiped and restored to baseline.

2. **dconf restore resets GNOME settings** — gsettings changes (volume, screen lock, notifications, dim screen) are all stored in dconf and explicitly restored from a baseline snapshot.

3. **Process kill eliminates running apps** — All user processes (LibreOffice, Chrome, GIMP, VLC, Thunderbird, VS Code, socat) are killed with SIGKILL.

4. **X11 restart gives fresh desktop** — The display manager restarts, providing a clean graphical session.

5. **Temp/runtime cleanup** — `/tmp`, `/var/tmp`, `/run/user/` are cleared.

### Exceptions (3 tasks):

| Task ID | Why soft reset is insufficient | Mitigation |
|---------|-------------------------------|------------|
| `94d95f96` | Installs Spotify to system `/usr/` or `/snap/` | Add post-eval uninstall or remove from training set |
| `3299584d` | Setup installs `jq` via apt | No action needed — idempotent, harmless |
| `9656a811` | Setup installs `jq` via apt | No action needed — idempotent, harmless |

### Performance Advantage

| Reset Type | Time | When to Use |
|-----------|------|-------------|
| Soft reset (5001) | ~5-8s | Default for 125/128 tasks |
| Full relaunch (terminate + new AMI) | ~89s | Fallback when soft reset fails, or for `94d95f96` |

### Implementation

The current code in `OSWorld/desktop_env/providers/aws/provider.py` already implements this two-tier approach:
1. Try soft reset via 5001 first
2. Fall back to full relaunch if soft reset fails

No code changes are needed for the reset mechanism itself. The only actionable items are:
1. Handle the Spotify task (`94d95f96`) — either remove it or add cleanup
2. Fix the action execution bugs in `verl/trainer/gui_agent.py`

---

## 6. Summary of All HIGH/CRITICAL Issues

| # | Issue | Scope | Category |
|---|-------|-------|----------|
| 1 | `press` references undefined `hotkey` variable | All tasks | Code bug |
| 2 | `str(None)` truthy — click crash | All tasks | Code bug |
| 3 | `94d95f96` Spotify install persists across soft resets | 1 task | Reset |
| 4 | `5e2d93d8` instruction/evaluator path mismatch | 1 task | Task config |
| 5 | `c6bf789c` nested dict equality fails with default excludes | 1 task | Evaluator |
| 6 | `a728a36e`, `b4f95342` need internet but `proxy: false` | 2 tasks | Task config |
| 7 | `acb0f96b` needs internet but `proxy: false`; static gold | 1 task | Task config |
| 8 | `dd84e895` evaluator only checks 1 starred email | 1 task | Evaluator |
| 9 | 5 GIMP tasks with fragile postconfig export via pyautogui | 5 tasks | Evaluator |

---

## 7. Infeasible Task List (26 tasks)

These tasks require the agent to recognize impossibility and output a FAIL action:

**VLC (3):** `5ac2891a` (auto-close), `7882ed6e` (DRM content), `cb130f0d` (auto brightness)
**Chrome (2):** `3720f614` (fictional language), `ae78f875` (search results per page)
**VS Code (5):** `7aeae0e2` (visualize numpy), `7c4cc09e` (Arabic without extensions), `847a96b6` (two workspaces), `971cbb5b` (auto-create file), `dcbe20e8` (background image)
**OS (4):** `4783cc41` (undefined $targetDir), `a462a795` (user Charles), `c288e301` (Python4), `fe41f596` (battery on VM)
**Multi-Apps (1):** `6d72aad6` (Impress to video)
**LibreOffice Calc (1):** `2bd59342` (sparklines)
**LibreOffice Writer (1):** `bb8ccc78` (real-time collaboration)
**Thunderbird (1):** `a1af9f1c` (send-only without incoming)
**GIMP (8):** `045bf3ff` (CMYK), `2e6f678f` (batch brightness), `38f48d40` (trim video), `5ca86c6f` (download logo), `62f7fd55` (PNG to SVG), `dbbf4b99` (RAW to JPEG), `e19bd559` (no image provided), `fbb548ca` (Blue theme)
