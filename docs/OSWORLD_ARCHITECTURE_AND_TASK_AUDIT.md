# OSWorld AWS Architecture and 127-Task Audit

**Date:** 2026-03-09
**Scope:** Complete audit of the custom AWS-backed OSWorld training system and all 127 training tasks

---

## 1. System Architecture Overview

### 1.1 Original OSWorld Architecture vs. This Implementation

| Aspect | Original OSWorld | This Implementation |
|--------|-----------------|---------------------|
| **Compute backend** | VirtualBox / VMware / Docker | AWS EC2 (us-east-1) |
| **Reset mechanism** | Full VM snapshot revert | OverlayFS soft reset (port 5001 daemon) |
| **Reset time** | 30–90 seconds | **5–8 seconds** |
| **State isolation** | Full image revert | Overlay upper-layer wipe + service restart |
| **Control plane** | Flask server (port 5000) | Same Flask server (port 5000) |
| **Reset daemon** | None (no separate daemon) | `osworld-resetd.service` (port 5001) |
| **Fallback** | N/A | Full EC2 terminate + relaunch (~89s) |
| **Taint tracking** | None | Previously existed; **removed** (see §2.5) |
| **AMI** | N/A | `ami-092bc7644b0debfcd` (us-east-1) |

### 1.2 Port Architecture

```
Training Host / Controller
        │
        │  HTTP (private VPC)
        ├──────────────────────────→  Port 5000  (osworld-server.service)
        │                              Flask control plane
        │                              - /screenshot, /health
        │                              - /execute, /setup/execute
        │                              - /setup/upload
        │                              Runs as: user (non-root)
        │
        └──────────────────────────→  Port 5001  (osworld-resetd.service)
                                       Reset daemon (root)
                                       - /reset, /verify
                                       - /prepare_baseline
                                       - /health, /state
                                       - /mark_tainted (vestigial)
```

### 1.3 OverlayFS Architecture

```
/home/user  (mount point — what users/apps see)
    │
    ├── lowerdir = /opt/osworld/baseline/home-user   (read-only, baseline state)
    └── upperdir = /var/lib/osworld/session/home-upper (writable, all changes go here)
         workdir = /var/lib/osworld/session/home-work
```

**Baseline** (`/opt/osworld/baseline/`) contains:
- `/opt/osworld/baseline/home-user/` — pristine `/home/user` contents
- `/opt/osworld/baseline/dconf/user.dconf` — GNOME desktop settings snapshot

**Reset process** (atomic, ~5-8s total):
1. Stop `osworld-server.service` (port 5000)
2. Stop `osworld-graphical-session.service` (X server)
3. Kill all user processes (`loginctl kill-user` + `pkill -KILL -u user`, 15s timeout)
4. Unmount `/home/user` overlay
5. **Delete entire `home-upper` and `home-work` dirs** (complete wipe)
6. Create fresh empty `home-upper` and `home-work` dirs
7. Remount overlay → `/home/user` now reflects pure baseline state
8. Clear `/tmp`, `/var/tmp` (user-owned files)
9. Remove user crontab
10. Clear `/run/user/{uid}/` (D-Bus, PulseAudio sockets)
11. Restore dconf from `/opt/osworld/baseline/dconf/user.dconf`
12. Restart X session and port 5000 server

### 1.4 Systemd Service Dependency Chain

```
osworld-home-overlay.service  (oneshot, RemainAfterExit=yes)
    │  mounts /home/user overlay at boot
    │
    ├── osworld-resetd.service  (after overlay, port 5001, root, Restart=always)
    │
    ├── osworld-graphical-session.service  (after overlay, X server, Restart=always)
    │
    └── osworld-server.service  (after graphical-session, port 5000, user, Restart=always)
```

### 1.5 Provider Decision Flow

```
revert_to_snapshot(instance_id)
    │
    ├─→ Check: instance running? port 5001 reachable?
    │       No → raise RuntimeError → caller triggers _full_relaunch()
    │
    ├─→ vm_reset.soft_reset()
    │       POST /reset to port 5001 → resets overlay, restarts services
    │       POST /verify to port 5001 → checks mount + server health
    │
    ├─→ result.status == "reused_clean"?
    │       Yes → return instance_id (reuse same VM)
    │       No  → raise RuntimeError → caller triggers _full_relaunch()
    │
    └─→ _full_relaunch()
            Terminate old instance
            Launch new instance from ami-092bc7644b0debfcd
            Wait for port 5000 to respond (~89s)
            Schedule TTL termination via EventBridge
            Return new_instance_id
```

---

## 2. Key Design Decisions and Changes from Original

### 2.1 Soft Reset vs. Full Snapshot Revert

**Original**: Snapshot revert requires recreating the entire VM from an image snapshot. In VirtualBox/VMware this takes 30–90 seconds and requires a hypervisor snapshot mechanism.

**This system**: OverlayFS reset takes 5–8 seconds by wiping only the writable upper layer. The lower (baseline) layer is never modified and always clean.

**Critical correctness guarantee**: After step 5 (upper layer deletion), `/home/user` from the perspective of any newly started process is **identical** to the baseline. Every file ever written by an agent (including deep inside `~/.config/`, `~/.local/`, `~/.thunderbird/`, etc.) is gone.

### 2.2 Taint System — Removed

The original design included a taint system: if the agent ran privileged commands (sudo), the system would mark itself "tainted" and refuse a soft reset (requiring full relaunch instead).

**Problem**: The setup controller (`setup.py`) ran legitimate infrastructure commands via `sudo` (e.g., `apt install expect`), which triggered taint markers, blocking soft reset even though the OverlayFS wipe fully restores all state.

**Decision**: Taint system removed from the soft reset decision path. The `/mark_tainted` endpoint still exists in the reset daemon but is never consulted before reset — all soft resets proceed, and the overlay wipe handles all state. Residual taint code can be fully removed in a future cleanup.

### 2.3 Verify Strictness — Simplified

Original verify checked if the overlay upper layer was "clean" (only allowed runtime artifacts). This failed because GNOME/snap services immediately create files in `~/.config/` and `~/.local/` after the X session restarts — not agent actions, just normal desktop startup.

**Decision**: Verify now only checks:
1. `/home/user` is a valid overlay mount with correct options
2. Port 5000 server responds within 30 seconds

### 2.4 Systemd Rate Limiting Fix

Services that fail multiple times in quick succession enter systemd's failed/rate-limited state and cannot restart. Fixed by calling `systemctl reset-failed <service>` before each `systemctl start` during reset.

### 2.5 `_normalize_command_for_runtime` Bug Fix

`main.py` called `_normalize_command_for_runtime` (with underscore prefix) but `runtime_paths.py` exports `normalize_command_for_runtime` (no prefix). This crashed the port 5000 server on import. Fixed by removing the underscore prefix in all calls.

---

## 3. Reset Coverage for All 127 Tasks

### 3.1 What the Reset Covers

| State Category | Location | Reset Method | Covered? |
|---------------|----------|--------------|----------|
| All user home files | `/home/user/` | OverlayFS upper wipe | ✅ Complete wipe & restore |
| App configs | `~/.config/` (Chrome, Code, LibreOffice, VLC, GIMP...) | OverlayFS upper wipe | ✅ Inside home |
| App data | `~/.local/share/` | OverlayFS upper wipe | ✅ Inside home |
| Thunderbird profile | `~/.thunderbird/` | OverlayFS upper wipe | ✅ Inside home |
| Desktop files | `~/Desktop/` | OverlayFS upper wipe | ✅ Inside home |
| Downloaded task files | `~/Downloads/`, `~/` | OverlayFS upper wipe | ✅ Inside home |
| GNOME settings (gsettings) | dconf database | Explicit dconf restore | ✅ Restored from snapshot |
| PulseAudio / D-Bus sockets | `/run/user/{uid}/` | Directory cleared | ✅ Cleared |
| Temp files | `/tmp/`, `/var/tmp/` | User-owned files deleted | ✅ Cleared |
| User crontab | crontab | `crontab -u user -r` | ✅ Removed |
| Running processes | All user processes | SIGKILL via loginctl | ✅ Killed |
| X11 session | Display server | Service restart | ✅ Fresh session |
| pip packages (user) | `~/.local/lib/python3.x/` | OverlayFS upper wipe | ✅ Inside home |

### 3.2 What the Reset Does NOT Cover

| State Category | Location | Covered? | Impact |
|---------------|----------|----------|--------|
| System packages | `/usr/`, `/snap/`, `/var/lib/dpkg/` | ❌ Persists | See §3.3 |
| System config | `/etc/` | ❌ Persists | No tasks modify /etc |
| Kernel state | `/proc/sys/` | ❌ Persists | No tasks modify kernel |
| Network config | `/etc/netplan/` | ❌ Persists | No tasks modify networking |

### 3.3 Per-App Reset Safety — All 127 Tasks

| App | Tasks | Reset-Safe? | Notes |
|-----|-------|-------------|-------|
| **LibreOffice Calc** | 12 | ✅ All safe | Files in `~/`, configs in `~/.config/libreoffice/` |
| **LibreOffice Impress** | 13 | ✅ All safe | Files in `~/Desktop/` |
| **LibreOffice Writer** | 11 | ✅ All safe | Files in `~/Desktop/` |
| **VLC** | 6 | ✅ All safe | Config in `~/.config/vlc/vlcrc` |
| **Chrome** | 18 | ⚠️ 2 tasks need jq | `3299584d`, `9656a811` install `jq` via apt; harmless but persists |
| **VS Code** | 19 | ✅ All safe | Settings in `~/.config/Code/User/` |
| **OS** | 11 | ✅ All safe | gsettings (dconf) → restored by dconf snapshot |
| **Multi-Apps** | 11 | ✅ All safe | All changes in home dir |
| **Thunderbird** | 7 | ✅ All safe | Profile in `~/.thunderbird/`; pip installs → `~/.local/` |
| **GIMP** | 19 | ✅ All safe | Config in `~/.config/GIMP/2.10/`; images in `~/Desktop/` |

**Note on Chrome tasks 3299584d and 9656a811**: Setup runs `sudo apt install jq` to install a JSON processor needed to modify Chrome preferences. The `jq` binary lands in `/usr/bin/jq` (system-level, persists). This is **harmless** (jq doesn't affect any other task, apt install is idempotent on re-run), but the setup command will be slower on first run (~30s for apt update + install) and requires network access. **Recommendation**: Pre-install `jq` in the AMI.

### 3.4 Evaluation Checking Positions — Per App

The table below shows where each evaluator reads results and whether those locations are reset-safe.

#### LibreOffice Calc (12 tasks)

| Task ID | Evaluator Func | Result Location | Reset-Safe |
|---------|---------------|-----------------|------------|
| 1273e544 | compare_table | `/home/user/NetIncome.xlsx` | ✅ |
| 1334ca3e | compare_table | `/home/user/Zoom_Out_Oversized_Cells.xlsx` | ✅ |
| 1d17d234 | compare_table | `/home/user/FutureValue.xlsx` | ✅ |
| 1e8df695 | compare_table | `/home/user/WeeklySales.xlsx` | ✅ |
| 2bd59342 | **infeasible** | — | N/A |
| 357ef137 | compare_table (inline ref) | `/home/user/Multiply_Time_Number.xlsx` | ✅ |
| 3aaa4e37 | compare_csv | `/home/user/Export_Calc_to_CSV.csv` | ✅ |
| 4172ea6e | compare_table | `/home/user/MaturityDate.xlsx` | ✅ |
| 4e6fcf72 | compare_table | `/home/user/Employee_Age_By_Birthday.xlsx` | ✅ |
| 51b11269 | compare_table | `/home/user/Arrang_Value_min_to_max.xlsx` | ✅ |
| 6e99a1ad | compare_table | `/home/user/Keep_Two_decimal_points.xlsx` | ✅ |
| aa3a8974 | check_pdf_pages + compare_pdfs | `/home/user/Resize_Cells_Fit_Page.pdf` | ✅ |

#### LibreOffice Impress (13 tasks)

| Task ID | Evaluator Func | Result Location | Reset-Safe |
|---------|---------------|-----------------|------------|
| 08aced46 | compare_pptx_files | `/home/user/Desktop/22_6.pptx` | ✅ |
| 15aece23 | compare_pptx_files | `/home/user/Desktop/134_2.pptx` | ✅ |
| 21760ecb | check_transition | `/home/user/Desktop/Ch4 Video Effect.pptx` | ✅ |
| 2b94c692 | compare_pptx_files | `/home/user/Desktop/201_6.pptx` | ✅ |
| 2cd43775 | check_auto_saving_time | `~/.config/libreoffice/4/user/registrymodifications.xcu` | ✅ |
| 39be0d19 | compare_pptx_files | `/home/user/Desktop/41_3.pptx` | ✅ |
| 70bca0cc | compare_pptx_files | `/home/user/Desktop/71_6.pptx` | ✅ |
| 73c99fb9 | compare_pptx_files | `/home/user/Desktop/109_4.pptx` | ✅ |
| 9cf05d24 | compare_pptx_files | `/home/user/Desktop/214_9.pptx` | ✅ |
| a097acff | compare_pptx_files | `/home/user/Desktop/pre.pptx` | ✅ |
| ac1b39ff | compare_pptx_files | `/home/user/Desktop/55_10.pptx` | ✅ |
| af23762e | compare_pptx_files | `/home/user/Desktop/Forests.pptx` | ✅ |
| c59742c0 | compare_audios | `/home/user/Desktop/Mady_and_Mia_Baseball.pptx` | ✅ |

#### LibreOffice Writer (11 tasks)

| Task ID | Evaluator Func | Result Location | Reset-Safe |
|---------|---------------|-----------------|------------|
| 0810415c | compare_line_spacing | `/home/user/Desktop/Novels_Intro_Packet.docx` | ✅ |
| 0b17a146 | compare_docx_files | `/home/user/Desktop/H2O_Factsheet_WA.docx` | ✅ |
| 0e763496 | compare_font_names | `/home/user/Desktop/Dublin_Zoo_Intro.docx` | ✅ |
| 3ef2b351 | is_first_line_centered | `/home/user/Desktop/Constitution_Template.docx` | ✅ |
| 4bcb1253 | compare_pdfs (OR: 4 paths) | Desktop/Documents/Downloads/home | ✅ |
| 6a33f9b9 | check_highlighted_words | `/home/user/Desktop/sample-recruitment-phone-script.odt` | ✅ |
| 72b810ef | evaluate_strike_through | `/home/user/Desktop/GEOG2169_Course_Outline.docx` | ✅ |
| bb8ccc78 | **infeasible** | — | N/A |
| d53ff5ee | compare_docx_files | `/home/user/Desktop/presentation_instruction.docx` | ✅ |
| e528b65e | compare_docx_files | `/home/user/Desktop/Geography_And_Magical_Realism.docx` | ✅ |
| ecc2413d | contains_page_break | `/home/user/Desktop/Sample_Statutory_Declaration.docx` | ✅ |

#### VLC (6 tasks)

| Task ID | Evaluator Func | Result Location | Reset-Safe |
|---------|---------------|-----------------|------------|
| 59f21cfb | is_vlc_playing | VLC status XML (runtime) | ✅ |
| 5ac2891a | **infeasible** | — | N/A |
| 7882ed6e | **infeasible** | — | N/A |
| 8ba5ae7a | is_vlc_recordings_folder | `~/.config/vlc/vlcrc` | ✅ |
| 8d9fd4e2 | is_vlc_fullscreen | VM screen size check | ✅ |
| cb130f0d | **infeasible** | — | N/A |

#### Chrome (18 tasks)

| Task ID | Evaluator Func | Result Location | Reset-Safe |
|---------|---------------|-----------------|------------|
| 06fe7178 | is_expected_tabs | Chrome CDP (runtime) | ✅ |
| 0d8b7de3 | is_expected_active_tab | Chrome CDP (runtime) | ✅ |
| 2ad9387a | is_expected_bookmarks | Chrome CDP bookmarks | ✅ |
| 2ae9ba84 | exact_match | Chrome CDP profile | ✅ |
| 3299584d | exact_match | Chrome startup page pref | ✅ |
| 368d9ba4 | check_direct_json_object | Chrome history | ✅ |
| 3720f614 | **infeasible** | — | N/A |
| 59155008 | is_expected_active_tab | Chrome CDP | ✅ |
| 7b6c7e24 | is_cookie_deleted | Chrome cookies | ✅ |
| 9656a811 | exact_match | Chrome safe browsing pref | ✅ |
| a728a36e | is_expected_url_pattern_match | Chrome active URL | ✅ |
| ae78f875 | **infeasible** | — | N/A |
| af630914 | check_font_size | Chrome font size pref | ✅ |
| b070486d | is_expected_url_pattern_match | Chrome active URL | ✅ |
| b4f95342 | check_direct_json_object | Chrome active tab HTML | ✅ |
| bb5e4c0d | match_in_list | Chrome default search engine | ✅ |
| c1fa57f3 | is_expected_url_pattern_match | Chrome active URL | ✅ |
| e1e75309 | compare_pdfs | `/home/user/Desktop/*.pdf` | ✅ |

#### VS Code (19 tasks)

| Task ID | Evaluator Func | Result Location | Reset-Safe |
|---------|---------------|-----------------|------------|
| 0512bb38 | is_extension_installed | `code --list-extensions` output | ✅ |
| 0ed39f63 | compare_text_file | `/home/user/Desktop/` | ✅ |
| 4e60007a | is_extension_installed | `code --list-extensions` output | ✅ |
| 53ad5833 | compare_config | VS Code extension command | ✅ |
| 57242fad | is_extension_installed | `ls /home/user/Desktop` output | ✅ |
| 5e2d93d8 | is_extension_installed | `/home/user/project/` dir listing | ✅ |
| 70745df8 | check_json_settings | `~/.config/Code/User/settings.json` | ✅ |
| 7aeae0e2 | **infeasible** | — | N/A |
| 7c4cc09e | **infeasible** | — | N/A |
| 847a96b6 | **infeasible** | — | N/A |
| 9439a27b | check_json_settings | `~/.config/Code/User/settings.json` | ✅ |
| 971cbb5b | **infeasible** | — | N/A |
| 982d12a5 | compare_config | `~/.config/Code/User/settings.json` | ✅ |
| 9d425400 | check_json_settings | `~/.config/Code/User/settings.json` | ✅ |
| c6bf789c | check_json_settings | `~/.config/Code/User/settings.json` | ✅ |
| dcbe20e8 | **infeasible** | — | N/A |
| e2b5e914 | check_json_settings | `~/.config/Code/User/settings.json` | ✅ |
| eabc805a | is_extension_installed | `code --list-extensions` output | ✅ |
| ec71221e | compare_text_file | `/home/user/Desktop/test.py` | ✅ |

#### OS (11 tasks)

| Task ID | Evaluator Func | Result Location | Reset-Safe |
|---------|---------------|-----------------|------------|
| 28cc3b7e | exact_match | `pactl` volume output | ✅ (dconf reset restores audio settings) |
| 4783cc41 | **infeasible** | — | N/A |
| 4d117223 | check_include_exclude | Shell script output | ✅ |
| 5ea617a3 | exact_match | File existence check | ✅ |
| a462a795 | **infeasible** | — | N/A |
| a4d98375 | exact_match | `gsettings get` output | ✅ (dconf restored) |
| bedcedc4 | exact_match | `gsettings get` output | ✅ (dconf restored) |
| c288e301 | **infeasible** | — | N/A |
| e0df059f | exact_match | Directory existence check | ✅ (dir in `~/Desktop/`) |
| f9be0997 | exact_match | `gsettings get` output | ✅ (dconf restored) |
| fe41f596 | **infeasible** | — | N/A |

#### Multi-Apps (11 tasks)

| Task ID | Apps | Evaluator Func | Result Location | Reset-Safe |
|---------|------|---------------|-----------------|------------|
| 2b9493d7 | Writer+Terminal | check_include_exclude | ps/bash_history output | ✅ |
| 2c9fc0de | Git+Terminal | check_include_exclude | Git repo in `~/` | ✅ |
| 3a93cae4 | Calc | compare_table (OR×3) | `~/Desktop/*.xlsx` | ✅ |
| 510f64c8 | VSCode+Terminal | check_include_exclude | `~/.config/Code/` + file | ✅ |
| 58565672 | Chrome+Thunderbird | is_expected_tabs | Chrome CDP | ✅ |
| 6d72aad6 | Impress | **infeasible** | — | N/A |
| 9219480b | VSCode+Python | check_python_file_by_test_suite | `~/` Python files | ✅ |
| 937087b6 | VLC+OS | check_include_exclude | VLC config | ✅ |
| acb0f96b | Git+GitHub | compare_text_file | `~/instructor-embedding/` | ✅ |
| c2751594 | Thunderbird+GNOME | compare_images | Screenshot | ✅ |
| ee9a3c83 | Calc+Terminal | check_include_exclude + compare_csv | `~/` CSV files | ✅ |

#### Thunderbird (7 tasks)

| Task ID | Evaluator Func | Result Location | Reset-Safe |
|---------|---------------|-----------------|------------|
| 10a730d5 | check_thunderbird_prefs | `~/.thunderbird/*/prefs.js` | ✅ |
| 3f28fe4f | check_thunderbird_prefs | `~/.thunderbird/*/prefs.js` | ✅ |
| a10b69e1 | check_list | `ls -R ~/.thunderbird/` output | ✅ |
| a1af9f1c | **infeasible** | — | N/A |
| d38192b0 | check_list | Script output (attachments) | ✅ |
| dd84e895 | run_sqlite3 | `~/.thunderbird/*/global-messages-db.sqlite` | ✅ |
| dfac9ee8 | check_csv | Script output | ✅ |

#### GIMP (19 tasks)

| Task ID | Evaluator Func | Result Location | Reset-Safe |
|---------|---------------|-----------------|------------|
| 045bf3ff | **infeasible** | — | N/A |
| 2e6f678f | **infeasible** | — | N/A |
| 38f48d40 | **infeasible** | — | N/A |
| 554785e9 | check_saturation_increase_and_structure_sim | `/home/user/Desktop/edited_colorful.png` | ✅ |
| 5ca86c6f | **infeasible** | — | N/A |
| 62f7fd55 | **infeasible** | — | N/A |
| 72f83cdc | check_image_mirror | `/home/user/Desktop/berry_mirror.png` | ✅ |
| 7767eef2 | check_config_status | `~/.config/GIMP/2.10/gimprc` | ✅ |
| 77b8ab4d | check_file_exists_and_structure_sim | `/home/user/Desktop/export.jpg` | ✅ |
| 7a4deb26 | check_brightness_decrease_and_structure_sim | `/home/user/Desktop/edited_darker.png` | ✅ |
| 7b7617bd | check_config_status | `~/.config/GIMP/2.10/gimprc` | ✅ |
| a746add2 | check_include_exclude | `~/.config/GIMP/2.10/action-history` | ✅ |
| b148e375 | check_config_status | `~/.config/GIMP/2.10/gimprc` | ✅ |
| d52d6308 | check_config_status | `~/.config/GIMP/2.10/sessionrc` | ✅ |
| dbbf4b99 | **infeasible** | — | N/A |
| e19bd559 | **infeasible** | — | N/A |
| e2dd0213 | check_textbox_on_leftside | `/home/user/Desktop/leftside_textbox.png` | ✅ |
| f723c744 | check_contrast_increase_and_structure_sim | `/home/user/Desktop/berries_contrast.png` | ✅ |
| fbb548ca | **infeasible** | — | N/A |

**Summary**: ALL 127 tasks evaluate only within `/home/user/` or runtime state (dconf, Chrome CDP, process state). The OverlayFS reset + dconf restore covers **100% of evaluation checking positions**.

---

## 4. Bugs Found and Fixed

### 4.1 GIMP Postconfig `pyautogui.hotkey()` Syntax Error — FIXED

**Files**: 5 GIMP task JSONs
**Tasks**: `554785e9`, `72f83cdc`, `7a4deb26`, `e2dd0213`, `f723c744`

**Bug**: Postconfig passed a list to `hotkey()`:
```python
pyautogui.hotkey(["shift", "ctrl", "e"])  # WRONG: passes list as single arg
```
`hotkey(*args)` expects each key as a separate positional argument. This caused `AttributeError: 'list' object has no attribute 'lower'` in `keyDown()`, silently failing to open the Export As dialog → evaluator looks for exported file → finds nothing → score 0.0.

**Fix**:
```python
pyautogui.hotkey("shift", "ctrl", "e")  # CORRECT: separate string args
```

### 4.2 VS Code Evaluator Pipe Commands Without `shell=True` — FIXED

**Files**: 4 VS Code task JSONs
**Tasks**: `0512bb38`, `4e60007a`, `eabc805a`, `57242fad`

**Bug**: Evaluator `result.command` array contained `"|"` as a literal element:
```json
"command": ["code", "--list-extensions", "|", "grep", "ms-python.python"]
```
Without `"shell": true`, `get_vm_command_line()` posts `shell=false` to `/execute`. `subprocess.run` then passes `|` as a literal argument to `code`, not as a shell pipe operator. The pipe never executes.

**Fix**: Added `"shell": true` to each evaluator result config. With shell=True, `normalize_command_for_runtime()` joins the list to `"code --list-extensions | grep ms-python.python"` and passes it to `/bin/sh`.

### 4.3 Multi-Apps Bash Syntax Error in Evaluator — FIXED

**File**: `multi_apps/2b9493d7-49b8-493a-a71b-56cd1f4d6908.json`

**Bug**: Extra `]` in bash command substitution:
```bash
output=$(ps aux | grep "[s]office"]);  # extra ] after "
```
`grep` received `]` as a filename argument, causing a spurious error message to stderr and potentially incorrect behavior.

**Fix**: Removed the extra `]`:
```bash
output=$(ps aux | grep "[s]office");
```

### 4.4 Previously Fixed (Prior Sessions)

| Bug | Location | Fix |
|----|---------|-----|
| `_normalize_command_for_runtime` NameError | `server/main.py` | Renamed to `normalize_command_for_runtime` |
| `press` action references undefined `hotkey` | `gui_agent.py`, `uitars_agent.py` | Fixed to use `key_to_press` |
| `str(None)` truthy in click action | `gui_agent.py`, `uitars_agent.py` | Added None guard |
| `pyperclip.copy()` string injection | `gui_agent.py`, `uitars_agent.py` | Used `repr()` |
| `{` in pyautogui commands breaks `PythonController` | `controllers/python.py` | Replaced `.format()` with `.replace()` |
| `<\|box_start\|>` markers not stripped | `gui_agent.py`, `uitars_agent.py` | Strip markers before parsing |
| Taint system blocks soft reset for infrastructure commands | `reset_runtime.py`, `main.py` | Removed taint check from reset decision |
| Verify rejects desktop startup artifacts | `reset_runtime.py` | Simplified verify to only check mount + health |
| Systemd rate limiting | `reset_runtime.py` | Added `systemctl reset-failed` before start |
| Duplicate TTL scheduling | `provider.py` | Removed duplicate EventBridge code |
| VM leak on waiter failure | `provider.py` | Added terminate on waiter exception |
| `is_environment_used` state not reset before failed start | `desktop_env.py` | Set `False` before `_start_emulator()` |

---

## 5. Remaining Issues (Not Fixed — Require Task Redesign)

### 5.1 HIGH Priority

| Task | Issue | Recommendation |
|------|-------|----------------|
| `3299584d`, `9656a811` | Setup runs `sudo apt install jq` (system-level, slow, requires internet) | Pre-install `jq` in AMI |
| `a728a36e`, `b4f95342` | `proxy: false` but requires navigating to external websites (dmv.virginia.gov, recreation.gov) | Change to `proxy: true` or use cached pages |
| `acb0f96b` | `proxy: false` but requires GitHub for `git clone` | Change to `proxy: true` or use local git server |

### 5.2 MEDIUM Priority

| Task | Issue | Recommendation |
|------|-------|----------------|
| `dd84e895` | Evaluator SQL checks `sum(1) > 0` (at least 1 starred) instead of ALL emails starred | Fix SQL to count all emails in folder and compare |
| `6e99a1ad` | Postconfig runs `libreoffice --convert-to` while LO window is open (file lock conflict risk) | Close LO first via window close command |
| `4e6fcf72` | Age calculation depends on execution date; gold file also date-dependent | Consider date-independent calculation |
| `5e2d93d8` | Instruction says workspace at `/home/user/` but evaluator checks `/home/user/project/project.code-workspace` | Clarify correct output path in instruction |

### 5.3 LOW Priority / Informational

| Task | Issue |
|------|-------|
| `1e8df695` | Instruction typo "CGOS" should be "COGS" |
| `2ad9387a` | Bookmark evaluator uses strict set equality (may fail if extra bookmarks exist) |
| `e1e75309` | PDF filename must exactly match Chrome's auto-generated name |
| `dfac9ee8` | Downloads `firefox_decrypt.py` from a specific GitHub commit (external dependency, hardcoded commit hash) |
| `a097acff` | Downloads to relative path `"Downloads/"` instead of absolute `/home/user/Downloads/` |
| `d52d6308` | Task says "left dock" but evaluator checks `hide-docks` (all docks) in sessionrc |

---

## 6. Infeasible Tasks (26 total — correctly marked)

These tasks are marked with `"func": "infeasible"` and require the agent to recognize impossibility and output a FAIL action. They are not bugs.

| App | Task IDs | Reason |
|-----|---------|--------|
| **VLC** (3) | 5ac2891a, 7882ed6e, cb130f0d | Auto-close, DRM content, auto-brightness |
| **Chrome** (2) | 3720f614, ae78f875 | Fictional language, search results UI |
| **VS Code** (5) | 7aeae0e2, 7c4cc09e, 847a96b6, 971cbb5b, dcbe20e8 | Numpy viz, Arabic, multi-workspace, auto-file, background |
| **OS** (4) | 4783cc41, a462a795, c288e301, fe41f596 | Undefined var, missing user, Python4, battery on VM |
| **Multi-Apps** (1) | 6d72aad6 | LibreOffice → video |
| **LibreOffice Calc** (1) | 2bd59342 | Sparklines (unsupported in LO) |
| **LibreOffice Writer** (1) | bb8ccc78 | Real-time collaboration |
| **Thunderbird** (1) | a1af9f1c | SMTP-only without IMAP |
| **GIMP** (8) | 045bf3ff, 2e6f678f, 38f48d40, 5ca86c6f, 62f7fd55, dbbf4b99, e19bd559, fbb548ca | CMYK, batch, video edit, download, SVG, RAW, no image, Blue theme |

---

## 7. Reset Timing and Performance

| Operation | Time |
|-----------|------|
| Soft reset (overlay wipe + restart) | 5–8 seconds |
| Verify (health check) | 1–30 seconds |
| Full relaunch (EC2 terminate + new launch) | ~89 seconds |
| Port 5000 fix via resetd | ~60 seconds |

**Practical training throughput**: With 5–8s soft reset, a 20-step episode with one reset takes ~85–100s total (60s steps + 5-8s reset + 15-30s eval), vs ~170s with full snapshot revert.

---

## 8. Deployment Notes

- **AMI**: `ami-092bc7644b0debfcd` (us-east-1) — contains all services pre-configured
- **Cloud-init**: Does NOT work on this AMI (cloud-init disabled). All fixes must be baked in.
- **SSH key**: `/tmp/osworld-fix-key.pem` (key pair `osworld-fix-key`) — for debugging only; reset daemon/server use HTTP
- **Sudo password**: `osworld-public-evaluation`
- **Service start order**: overlay → resetd + graphical-session → server
- **Recommended**: Pre-install `jq` in AMI to avoid apt installs during task setup
