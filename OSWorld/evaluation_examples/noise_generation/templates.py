"""
noise_generation/templates.py — v3 template library.

DESIGN PHILOSOPHY
-----------------
Noise simulates a CONCURRENT HUMAN USER sharing the same desktop with the
agent. The human is doing *their own real work* in non-target apps —
opening windows, typing into them, navigating folders, running commands,
composing notes — just like the agent is doing its work in the target app.
Every noise element is a concrete, observable INTERACTION session, not a
background event.

The library is organized in three tiers:

  1. HUMAN TASK SESSIONS (primary — dominant use)
     Real interaction sessions in non-target apps: note-taking, calculator,
     terminal, file browsing, media, system monitoring. Each opens an app,
     focuses it, and performs a short meaningful action sequence
     (typing/clicking/shortcuts) that lands on the noise-owned window.

  2. AMBIENT BACKGROUND (secondary)
     Notifications, theme flickers, resource activity. Things that happen
     on any real desktop but aren't driven by an interactive user session.
     Kept small in count; used to spice up the ambient layer.

  3. ACCIDENTAL TARGET INTERFERENCE (rare)
     The simulated human occasionally bumps the agent's window — shoves,
     resizes, or partly covers it. Always ≤1-step recovery. Used sparingly.

SAFETY / NON-SABOTAGE INVARIANTS
--------------------------------
Every template returns a bash command that:
  - NEVER sends keystrokes or clicks to any window the noise didn't just
    open and focus itself (the agent's input is never corrupted).
  - NEVER closes, kills, stops, unmaps, or deletes anything the agent owns.
  - NEVER touches paths referenced by the task's evaluator.
  - Is recoverable by the agent in ≤ 3 standard actions (click/type/key).

RANDOMIZATION
-------------
Every template internally randomizes via bash arrays + $RANDOM. A single
element definition in a _noise.json produces a different concrete observed
event on every firing.

TOOLS USED
----------
Only tools already in osworld:latest: zenity, gedit, wmctrl, gnome-terminal,
nautilus, gnome-system-monitor, gnome-control-center (if present), vlc,
python3 (for pyautogui), touch, mkdir, gsettings, paplay, gio, ln, dd.
Tools requiring the planned image rebuild (libnotify-bin, gnome-calculator,
xdotool) are marked requires_image_rebuild=True and used optionally.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

# Ensure sibling modules (variants, compositional, human_extra,
# app_browser_os_extra, recovery_diverse) are importable when this module is
# imported from the repo root or a driver script.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Visual-rendering diversity (xmessage, tkinter, browser-rendered).
from variants import (
    modal_xmessage_centered, modal_xmessage_compact,
    modal_tkinter_info, modal_tkinter_warning, modal_tkinter_error,
    modal_tkinter_question, modal_tkinter_custom_banner,
    modal_browser_alert, modal_browser_dialog_element,
    notif_xmessage_timeout, notif_tkinter_corner_toast, notif_terminal_flash,
    overlay_tkinter_top_banner, overlay_tkinter_bottom_banner,
    overlay_tkinter_right_sidebar, overlay_tkinter_floating_widget,
    overlay_browser_cookie_real,
    focus_steal_tkinter_app, focus_steal_two_windows,
    focus_steal_scrolling_terminal,
)

# Compositional (multi-element) primitives.
from compositional import (
    comp_notif_then_modal, comp_three_notifs_staggered,
    comp_cookie_plus_popup, comp_focus_steal_plus_shove,
    comp_ad_banner_plus_chat, comp_audio_plus_visual_flicker,
    comp_cpu_spike_plus_resource_notif, comp_drag_plus_filesystem,
    comp_human_writing_plus_notif, comp_modal_then_decoy_notif,
    comp_shove_then_cover, comp_multi_modal_chain,
    comp_overlay_plus_focus_steal, comp_decoy_nothing,
    comp_notif_burst_plus_flicker,
)

# Extended human task sessions.
from human_extra import (
    human_journal_entry, human_sql_query_draft, human_readme_draft,
    human_essay_paragraph, human_config_file_edit,
    human_terminal_file_navigation, human_terminal_log_grep,
    human_terminal_network_check, human_terminal_python_repl,
    human_file_rename_desktop, human_file_search_nautilus,
    human_file_zoom_view, human_file_view_toggle,
    human_scroll_long_doc, human_select_all_copy, human_find_in_editor,
    human_split_editor_windows, human_preferences_dialog,
    human_help_browser, human_image_viewer_browse, human_archive_browse,
)

# Extended application / browser / OS prompts.
from app_browser_os_extra import (
    app_signin_required, app_trial_expired, app_first_run_tutorial,
    app_tip_of_the_day, app_feedback_request, app_whats_new_overlay,
    app_license_key_prompt, app_export_dialog,
    browser_autofill_suggestion, browser_certificate_warning,
    browser_location_permission, browser_microphone_permission,
    browser_camera_permission, browser_notifications_permission,
    browser_clipboard_permission, browser_fullscreen_exit_hint,
    browser_install_extension_banner, browser_not_responding_banner,
    browser_close_multiple_tabs, browser_clear_browsing_data,
    browser_reopen_closed_tabs, browser_proxy_reconnect_required,
    browser_proxy_auth_refresh, browser_dns_probe_recovering,
    os_disk_almost_full, os_wifi_connection_dialog, os_print_dialog_stuck,
    os_software_updater_banner, os_unattended_upgrades_notice,
    os_antivirus_scan_notice, os_firewall_block_notice,
    os_storage_quota_approaching, os_mail_arrived_toast,
    os_vpn_tunnel_reconnecting, os_secure_gateway_flap,
    os_power_adapter_disconnected,
)

# Diverse recovery-path templates.
from recovery_diverse import (
    recovery_wait_autodismiss, recovery_escape_only,
    recovery_drag_window_required, recovery_resize_required,
    recovery_specific_button, recovery_click_outside,
    recovery_scroll_to_dismiss, recovery_two_step,
    recovery_double_click, recovery_right_click_menu,
    recovery_type_to_close, recovery_click_specific_position,
    recovery_drag_to_corner, recovery_minimize_instead,
    recovery_ignore_decoy,
)


# ---------------------------------------------------------------------------
# Mapping: OSWorld related_apps key -> WM window title (used by target_* fns)
# ---------------------------------------------------------------------------

APP_TO_WM_TITLE: Dict[str, str] = {
    "chrome":              "Google Chrome",
    "gimp":                "GNU Image Manipulation Program",
    "libreoffice_calc":    "LibreOffice Calc",
    "libreoffice_impress": "LibreOffice Impress",
    "libreoffice_writer":  "LibreOffice Writer",
    "multi_apps":          "",
    "os":                  "",
    "thunderbird":         "Thunderbird",
    "vlc":                 "VLC",
    "vs_code":             "Visual Studio Code",
}


def _c(src: str) -> str:
    """Strip leading/trailing whitespace; preserve internal newlines so
    multi-line bash scripts (incl. python3 -c bodies) parse correctly."""
    return src.strip()


# ===========================================================================
# TIER 1 — CONCURRENT HUMAN TASK SESSIONS (the primary noise category)
# ===========================================================================
# Pattern: launch app → sleep → `wmctrl -a <NoiseAppTitle>` focuses it → send
# keystrokes/clicks that land on the noise-owned window.  Agent recovers with
# one click on its own app.

# ---------- Writing / note-taking sessions (gedit) ----------

def human_meeting_notes() -> str:
    """Open gedit and type a short meeting-notes draft (attendees, action
    items, next steps). Cost 1."""
    return _c(r"""
gedit --new-window /tmp/notes_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random
HEADS = ['Meeting:', 'Sync:', 'Standup:', 'Review:', '1:1:']
OWNERS = ['Alex', 'Maya', 'Priya', 'Tom', 'Chen', 'Sam']
TOPICS = ['Q2 planning', 'Bug triage', 'Launch prep', 'Roadmap', 'Hiring', 'Budget']
ACTIONS = ['Alex owns the report', 'Maya to draft proposal', 'Ping Priya re: vendor', 'Follow up Friday', 'Schedule a demo']
lines = [
    f'{random.choice(HEADS)} {random.choice(TOPICS)}',
    f'Attendees: {random.choice(OWNERS)}, {random.choice(OWNERS)}, {random.choice(OWNERS)}',
    '',
    '- ' + random.choice(ACTIONS),
    '- ' + random.choice(ACTIONS),
    '- ' + random.choice(ACTIONS),
    '',
    'Next steps: review by EOW',
]
for l in lines:
    pyautogui.typewrite(l, interval=0.02); pyautogui.press('enter')
" 2>/dev/null || true
""")


def human_todo_list() -> str:
    """Open gedit and write a todo/shopping/errands list. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/list_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random
TEMPLATES = [
  ('TODO today', ['review PR #142','finalize slides','reply to emails','book flights','renew insurance']),
  ('Shopping',   ['milk','bread','coffee','avocados','yogurt','bananas']),
  ('Errands',    ['post office','dry cleaner','pharmacy','library books','oil change']),
  ('Weekend',    ['laundry','clean kitchen','call mom','write blog post','plan trip']),
]
title, items = random.choice(TEMPLATES)
pyautogui.typewrite(title, interval=0.02); pyautogui.press('enter'); pyautogui.press('enter')
for i in items:
    pyautogui.typewrite('- ' + i, interval=0.02); pyautogui.press('enter')
" 2>/dev/null || true
""")


def human_compose_email() -> str:
    """Open gedit as a mock email-compose window and type a draft (To/Subject/
    body). Cost 1."""
    return _c(r"""
gedit --new-window /tmp/email_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random
TO = ['alex@example.com','team@corp.com','priya@org.net','client@vendor.io']
SUBJ = ['Quick update','RE: Meeting prep','FYI: documents','Status report','Q2 check-in','Proposal draft']
BODY = [
  ['Hi,','Checking in on the project status.','Let me know when you have a moment.','Thanks.'],
  ['Hey,','Here is the latest draft for review.','Happy to revise based on your thoughts.','Cheers.'],
  ['Team,','Quick note on Friday milestone.','I will circulate the slides tomorrow.','Regards.'],
]
pyautogui.typewrite(f'To: {random.choice(TO)}', interval=0.02); pyautogui.press('enter')
pyautogui.typewrite(f'Subject: {random.choice(SUBJ)}', interval=0.02); pyautogui.press('enter'); pyautogui.press('enter')
for l in random.choice(BODY):
    pyautogui.typewrite(l, interval=0.02); pyautogui.press('enter')
" 2>/dev/null || true
""")


def human_code_snippet() -> str:
    """Type a short code snippet into gedit — looks like a dev working. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/snippet_$RANDOM.py >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random
SNIPS = [
  ['def compute(x):','    return x * x + 2','','print(compute(5))'],
  ['import json','with open(\"data.json\") as f:','    d = json.load(f)','print(len(d))'],
  ['# quick check','for i in range(10):','    print(i, i**2)'],
  ['class Node:','    def __init__(self, v):','        self.v = v','        self.next = None'],
]
for l in random.choice(SNIPS):
    pyautogui.typewrite(l, interval=0.02); pyautogui.press('enter')
" 2>/dev/null || true
""")


def human_sticky_note() -> str:
    """Create a small gedit sticky-note window in a random corner with a
    short reminder. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/sticky_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 0.9;
CORNERS=("20 40" "1580 40" "20 780" "1580 780");
c=${CORNERS[$((RANDOM % ${#CORNERS[@]}))]};
x=$(echo $c | cut -d' ' -f1); y=$(echo $c | cut -d' ' -f2);
wmctrl -r 'gedit' -e "0,$x,$y,320,260" 2>/dev/null;
wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random
NOTES = [
  ('Reminder:', 'Check DB migration'),
  ('Call:', 'Sarah @ 3pm re: contract'),
  ('Buy:', 'lunch / coffee'),
  ('Urgent:', 'Deploy by EOD'),
  ('Ask:', 'vacation policy?'),
  ('Fix:', 'flaky test in CI'),
]
a, b = random.choice(NOTES)
pyautogui.typewrite(a, interval=0.03); pyautogui.press('enter')
pyautogui.typewrite(b, interval=0.03)
" 2>/dev/null || true
""")


# ---------- Terminal sessions ----------

def human_terminal_commands() -> str:
    """Open a terminal and run a short exploratory command; output stays
    visible. Cost 1."""
    return _c(r"""
CMDS=("ls -la ~" "ls /tmp" "ls ~/Desktop" "uname -a" "df -h" "free -h" "date" "ps aux | head -12" "whoami && pwd" "cat /etc/os-release" "ls -l /var/log | head -15");
c=${CMDS[$((RANDOM % ${#CMDS[@]}))]};
gnome-terminal --geometry=100x22 -- bash -c "printf 'user@host:~\$ %s\n' \"$c\"; eval $c; echo; read -p '[Enter to close]' ok" >/dev/null 2>&1 & disown
""")


def human_terminal_git_mock() -> str:
    """Run a short mock git session in a terminal (echoed output). Cost 1."""
    return _c(r"""
FLOWS=(
  "printf 'user@host:~/proj\$ git status\n'; printf 'On branch main\nnothing to commit, working tree clean\n'"
  "printf 'user@host:~/proj\$ git log --oneline -5\n'; for i in 1 2 3 4 5; do printf 'abc12%d feat: commit %d\n' \$i \$i; done"
  "printf 'user@host:~/proj\$ git diff --stat\n'; printf ' src/main.py | 12 ++++++++----\n src/utils.py |  3 +--\n'"
  "printf 'user@host:~/proj\$ git branch\n'; printf '  dev\n* main\n  feature/login\n  bugfix/cache\n'"
);
s=${FLOWS[$((RANDOM % ${#FLOWS[@]}))]};
gnome-terminal --geometry=100x22 -- bash -c "eval \"$s\"; echo; read -p '[Enter]' ok" >/dev/null 2>&1 & disown
""")


def human_terminal_tail_log() -> str:
    """Tail a synthetic log — scrolling output over 2-3s. Cost 1."""
    return _c(r"""
SCRIPTS=(
  "for i in \$(seq 1 22); do printf '[info] processing chunk %02d of 22\n' \$i; sleep 0.12; done"
  "for s in init load parse compile link sign package upload publish done; do echo '['\$(date +%H:%M:%S)']' \$s; sleep 0.25; done"
  "echo '[watcher] scanning ~/Documents...'; for i in \$(seq 1 18); do echo '  file_'\$i'.txt: ok'; sleep 0.15; done"
  "echo '[sync]'; for i in \$(seq 1 12); do printf 'transferring item_%02d ... %d%%\n' \$i \$((i*100/12)); sleep 0.2; done"
);
s=${SCRIPTS[$((RANDOM % ${#SCRIPTS[@]}))]};
gnome-terminal --geometry=100x22 -- bash -c "$s; echo; read -p '[Enter]' ok" >/dev/null 2>&1 & disown
""")


# ---------- File manager sessions ----------

def human_file_browser() -> str:
    """Open the file manager at a random common folder. Cost 1."""
    return _c(r"""
P=(/home/user /tmp /home/user/Desktop /home/user/Downloads /home/user/Documents /home/user/Pictures /home/user/.config);
for p in "${P[@]}"; do [[ -d "$p" ]] && V+=("$p"); done;
[[ ${#V[@]} -eq 0 ]] && exit 0;
nautilus "${V[$((RANDOM % ${#V[@]}))]}" >/dev/null 2>&1 & disown
""")


def human_create_folder_on_desktop() -> str:
    """Open nautilus on Desktop and create a new folder via Ctrl+Shift+N.
    Additive only; touches nothing existing. Cost 1."""
    return _c(r"""
nautilus /home/user/Desktop >/dev/null 2>&1 & disown;
sleep 1.0;
wmctrl -a 'Desktop' 2>/dev/null || wmctrl -a 'Files' 2>/dev/null || wmctrl -a 'Nautilus' 2>/dev/null;
sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random, time
pyautogui.hotkey('ctrl', 'shift', 'n'); time.sleep(0.6)
NAMES = ['Project', 'Drafts', 'Archive', 'Work', 'Research', 'Reviews', 'Ideas', 'Reports']
n = random.choice(NAMES) + '_' + str(random.randint(100, 999))
pyautogui.typewrite(n, interval=0.03); pyautogui.press('enter')
" 2>/dev/null || true
""")


def human_trash_scratch() -> str:
    """Move a scratch file into the user's Trash via `gio trash` AND surface
    a tiny 'Moved to trash' toast so the event is observable. Cost 0."""
    return _c(r"""
f=/tmp/trashable_$RANDOM.txt;
printf 'draft content %s\n' $RANDOM > "$f" 2>/dev/null;
gio trash "$f" 2>/dev/null || true;
bn=$(basename "$f");
DISPLAY=:0 python3 -c "
import tkinter as tk
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
W,H=280,44
r.geometry(f'{W}x{H}+{1920-W-16}+{1080-H-64}')
r.configure(bg='#1f2937')
tk.Label(r, text='🗑  Moved to Trash: ${bn}', bg='#1f2937', fg='white', font=('Sans',10)).pack(expand=True, fill='both', padx=10)
r.after(10000, r.destroy); r.mainloop()
" >/dev/null 2>&1 & disown
""")


def human_drop_markdown() -> str:
    """Save a short markdown draft to the Desktop — a real session artifact.
    Cost 0."""
    return _c(r"""
N=(notes draft memo brainstorm outline plan sketch);
TAG=$RANDOM; n=${N[$((RANDOM % ${#N[@]}))]}_$TAG.md;
f=/home/user/Desktop/$n;
mkdir -p /home/user/Desktop 2>/dev/null;
{
  echo "# Session $TAG";
  echo "";
  echo "- captured on $(date +%Y-%m-%d)";
  echo "- draft revision $(($RANDOM % 5 + 1))";
  echo "- TODO: flesh this out";
} > "$f" 2>/dev/null || true
""")


# ---------- System / utility sessions ----------

def human_system_monitor() -> str:
    """Open system monitor and cycle through its tabs. Cost 1."""
    return _c(r"""
gnome-system-monitor >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'System Monitor' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random, time
for _ in range(random.randint(1, 3)):
    pyautogui.hotkey('ctrl', 'Page_Down'); time.sleep(0.35)
" 2>/dev/null || true
""")


def human_open_settings() -> str:
    """Open the system settings panel (gnome-control-center or gnome-tweaks).
    Cost 1."""
    return _c(r"""
if command -v gnome-control-center >/dev/null 2>&1; then
  gnome-control-center >/dev/null 2>&1 & disown;
elif command -v gnome-tweaks >/dev/null 2>&1; then
  gnome-tweaks >/dev/null 2>&1 & disown;
else
  exit 0;
fi
""")


def human_vlc_session() -> str:
    """Open VLC media player's main window. Cost 1."""
    return _c(r"""
command -v vlc >/dev/null 2>&1 || exit 0;
vlc --no-video-title-show >/dev/null 2>&1 & disown
""")


# ---------- Window/workspace management sessions ----------

def human_alt_tab_cycle() -> str:
    """Rapidly Alt-Tab-equivalent through existing windows (wmctrl raises 3
    random non-target windows). Cost 1."""
    return _c(r"""
IDS=($(wmctrl -l | awk '{print $1}'));
[[ ${#IDS[@]} -lt 2 ]] && exit 0;
for _ in 1 2 3; do
  i=${IDS[$((RANDOM % ${#IDS[@]}))]};
  wmctrl -i -a "$i" 2>/dev/null; sleep 0.35;
done
""")


def human_drag_own_window() -> str:
    """Drag a noise-owned window along a 3-waypoint path — visibly moves
    across the screen. Cost 1 if it ends overlapping the target; 0 else."""
    return _c(r"""
APPS=("gedit --new-window /tmp/drag_$RANDOM.txt|gedit" "gnome-system-monitor|System Monitor" "nautilus /tmp|Files");
p=${APPS[$((RANDOM % ${#APPS[@]}))]};
cmd="${p%%|*}"; title="${p##*|}";
eval "$cmd >/dev/null 2>&1 & disown";
sleep 1.0;
for _ in 1 2 3; do
  X=$((RANDOM % 1400 + 100)); Y=$((RANDOM % 700 + 50));
  W=$((RANDOM % 400 + 500)); H=$((RANDOM % 300 + 400));
  wmctrl -r "$title" -e "0,$X,$Y,$W,$H" 2>/dev/null; sleep 0.25;
done
""")


def human_minimize_restore() -> str:
    """Open a noise-owned window, let it be VISIBLE for a few seconds, then
    minimize it. Next reset restores it. The visible phase is the observable
    part. Cost 0."""
    return _c(r"""
APPS=("gedit --new-window /tmp/toggle_$RANDOM.txt|gedit" "gnome-system-monitor|System Monitor" "nautilus /tmp|Files");
p=${APPS[$((RANDOM % ${#APPS[@]}))]};
cmd="${p%%|*}"; title="${p##*|}";
eval "$cmd >/dev/null 2>&1 & disown";
sleep 1.2;
wmctrl -a "$title" 2>/dev/null;
# Keep visible for ~12s, then minimize in the background.
(sleep 12; wmctrl -r "$title" -b add,hidden 2>/dev/null) & disown
""")


def human_right_click_desktop_menu() -> str:
    """Right-click in a noise-owned nautilus window, wait briefly, then press
    Escape. Click and Escape land on the noise-owned window. Cost 0."""
    return _c(r"""
nautilus /home/user/Desktop >/dev/null 2>&1 & disown;
sleep 1.0;
wmctrl -a 'Desktop' 2>/dev/null || wmctrl -a 'Files' 2>/dev/null;
sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random, time
x, y = random.randint(400, 1000), random.randint(300, 700)
pyautogui.click(x, y, button='right'); time.sleep(0.6)
pyautogui.press('escape')
" 2>/dev/null || true
""")


# ---------- Optional (requires image rebuild) ----------

def human_calc_session() -> str:
    """'Calculator' session. Prefers gnome-calculator when present; otherwise
    opens a gedit buffer and types an arithmetic ledger — same recovery
    affordance (close the window). Cost 1."""
    return _c(r"""
if command -v gnome-calculator >/dev/null 2>&1; then
  gnome-calculator >/dev/null 2>&1 & disown;
  sleep 0.9;
  wmctrl -a 'Calculator' 2>/dev/null || wmctrl -a 'gnome-calculator' 2>/dev/null;
  sleep 0.3;
  DISPLAY=:0 python3 -c "
import pyautogui, random, time
a, b = random.randint(10, 999), random.randint(10, 99)
op = random.choice(['+', '-', '*', '/'])
for c in f'{a}{op}{b}':
    pyautogui.press(c); time.sleep(0.05)
pyautogui.press('enter')
" 2>/dev/null || true;
else
  # Fallback: gedit-as-calculator session (visible, closable the same way).
  gedit --new-window /tmp/calc_$RANDOM.txt >/dev/null 2>&1 & disown;
  sleep 1.2;
  wmctrl -a 'gedit' 2>/dev/null;
  sleep 0.3;
  DISPLAY=:0 python3 -c "
import pyautogui, random, time
LINES = ['Quick ledger:', '']
for _ in range(random.randint(3, 6)):
    a, b = random.randint(10, 999), random.randint(10, 99)
    op = random.choice(['+','-','*','/'])
    LINES.append(f'{a} {op} {b} = ?')
LINES += ['', 'TODO: verify totals']
for l in LINES:
    pyautogui.typewrite(l, interval=0.02); pyautogui.press('enter'); time.sleep(0.05)
" 2>/dev/null || true;
fi
""")


# ===========================================================================
# TIER 2 — AMBIENT BACKGROUND (not interactive)
# ===========================================================================

def ambient_notification() -> str:
    """Realistic desktop notification — rendered as a corner toast tkinter
    overlay because `zenity --notification` relies on a notification daemon
    (notify-osd) that may be absent on minimal images. The overlay stays
    visible ~15s then fades, matching typical notify-osd behavior. Cost 0."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
K=random.randint(0,2)
if K==0:
    A=['Slack','Email','WhatsApp','Teams','Discord','LinkedIn','Reminders']
    P=['Alex','Maya','Tom','Priya','Jordan','Sam','Chen']
    S=['Quick question','Review this PR','Meeting moved','Urgent','New message','Follow up tomorrow','FYI']
    text=f'{random.choice(A)} — {random.choice(P)}: {random.choice(S)}'
elif K==1:
    M=['Security updates are ready','Wi-Fi reconnected','Battery at 15%','Backup completed','New device paired','Software updater finished']
    text=f'System — {random.choice(M)}'
else:
    E=['Team standup','1:1 with manager','Project review','Deadline: Q2 report','Doctor appointment']
    W=['in 5 minutes','in 15 minutes','at 2 PM','tomorrow morning']
    text=f'Calendar — {random.choice(E)} {random.choice(W)}'
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
W,H=360,72
# Top-right corner (classic notify-osd position)
r.geometry(f'{W}x{H}+{1920-W-24}+{48}')
r.configure(bg='#2d3748')
tk.Label(r, text=text, bg='#2d3748', fg='white', font=('Sans',10,'bold'),
         wraplength=W-24, justify='left', anchor='w').pack(fill='both', padx=12, pady=16)
r.after(15000, r.destroy); r.mainloop()
" >/dev/null 2>&1 & disown
""")


def ambient_modal() -> str:
    """A realistic modal dialog (update/backup/download/cert/license). Cost 1."""
    return _c(r"""
TTL=("Software Updates" "Backup Manager" "Downloads" "Certificate" "Storage" "Subscription");
TXT=("Updates ready to install." "Scheduled backup finished." "Download complete." "Certificate expires in 7 days." "Disk space below 20%% free." "Subscription renews next month.");
S=(info question warning);
t=${TTL[$((RANDOM % ${#TTL[@]}))]}; x=${TXT[$((RANDOM % ${#TXT[@]}))]}; s=${S[$((RANDOM % ${#S[@]}))]};
zenity --$s --title "$t" --text "$x" --width=340 >/dev/null 2>&1 &
""")


def ambient_cpu_burst() -> str:
    """Short python CPU burn with a tiny visible progress toast so the event
    is observable (a CPU spike alone is invisible in a screenshot). Cost 0."""
    return _c(r"""
N=$((RANDOM % 2000000 + 500000));
python3 -c "sum(i*i for i in range($N))" >/dev/null 2>&1 & disown;
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
W,H=260,44
corner=random.choice(['br','tr'])
X,Y=(1920-W-12, 1020-H) if corner=='br' else (1920-W-12, 44)
r.geometry(f'{W}x{H}+{X}+{Y}')
bg='#333333'
r.configure(bg=bg)
MSGS=['Background indexer running...','Scheduled scan in progress...','Compaction task active','Search index rebuilding...']
tk.Label(r, text=random.choice(MSGS), bg=bg, fg='#a7f3d0', font=('Sans',9)).pack(expand=True, fill='both', padx=8, pady=10)
r.after(12000, r.destroy); r.mainloop()
" >/dev/null 2>&1 & disown
""")


def ambient_audio_ding() -> str:
    """Play a short system sound AND show a tiny speaker-icon toast so a
    vision-only agent can observe the event. Cost 0."""
    return _c(r"""
S=(/usr/share/sounds/freedesktop/stereo/complete.oga /usr/share/sounds/freedesktop/stereo/bell.oga /usr/share/sounds/freedesktop/stereo/message-new-instant.oga /usr/share/sounds/freedesktop/stereo/dialog-information.oga);
for s in "${S[@]}"; do [[ -f "$s" ]] && V+=("$s"); done;
if [[ ${#V[@]} -gt 0 ]]; then
  paplay "${V[$((RANDOM % ${#V[@]}))]}" 2>/dev/null & disown;
fi;
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
W,H=120,44
r.geometry(f'{W}x{H}+{1920-W-12}+{12}')
r.configure(bg='#111111')
icons=['♪ ding','♪ chime','♪ alert','♪ ping','♪ beep']
tk.Label(r, text=random.choice(icons), bg='#111111', fg='#fbbf24', font=('Sans',11,'bold')).pack(expand=True, fill='both')
r.after(6000, r.destroy); r.mainloop()
" >/dev/null 2>&1 & disown
""")


def ambient_fake_download() -> str:
    """A .crdownload file grows briefly in ~/Downloads — looks like an active
    download. Also shows a small 'downloading...' bar toast so the event is
    visible. Cost 0."""
    return _c(r"""
N=(photos archive dataset bundle movie report);
n=${N[$((RANDOM % ${#N[@]}))]}_$RANDOM;
mkdir -p /home/user/Downloads 2>/dev/null;
f=/home/user/Downloads/${n}.crdownload;
dd if=/dev/zero of="$f" bs=1K count=$((RANDOM % 500 + 100)) 2>/dev/null || true;
DISPLAY=:0 python3 -c "
import tkinter as tk, random
fname=${n@Q}
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
W,H=320,54
# Bottom-right — typical Chrome download bar position
r.geometry(f'{W}x{H}+{1920-W-16}+{1080-H-16}')
r.configure(bg='#e5e7eb')
tk.Label(r, text=f'⬇ Downloading: {fname}.zip', bg='#e5e7eb', fg='#111827',
         font=('Sans',10,'bold'), anchor='w').pack(fill='x', padx=10, pady=4)
tk.Label(r, text='█' * random.randint(10, 28) + '░' * random.randint(2, 10),
         bg='#e5e7eb', fg='#2563eb', font=('Monospace',10)).pack(anchor='w', padx=10)
r.after(14000, r.destroy); r.mainloop()
" >/dev/null 2>&1 & disown
""")


def ambient_dark_mode_flicker() -> str:
    """Whole-screen palette blips via gsettings AND shows a 'Theme switching'
    toast so the event is observable even when gsettings doesn't propagate.
    Cost 0."""
    return _c(r"""
CUR=$(gsettings get org.gnome.desktop.interface color-scheme 2>/dev/null);
N=('default' 'prefer-dark' 'prefer-light');
gsettings set org.gnome.desktop.interface color-scheme "${N[$((RANDOM % ${#N[@]}))]}" 2>/dev/null;
(sleep $((RANDOM % 3 + 2)); gsettings set org.gnome.desktop.interface color-scheme $CUR 2>/dev/null) & disown;
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
W,H=240,44
corner=random.choice(['tl','tr'])
X,Y=(12,12) if corner=='tl' else (1920-W-12,12)
r.geometry(f'{W}x{H}+{X}+{Y}')
MSGS=['Theme switched','Appearance updated','Color scheme changed','High contrast toggled']
bg=random.choice(['#1f2937','#374151','#4b5563'])
r.configure(bg=bg)
tk.Label(r, text='◐ ' + random.choice(MSGS), bg=bg, fg='white', font=('Sans',10)).pack(expand=True, fill='both', padx=10)
r.after(8000, r.destroy); r.mainloop()
" >/dev/null 2>&1 & disown
""")


# ===========================================================================
# TIER 2B — APPLICATION / BROWSER / SYSTEM INTERRUPTIONS
# ===========================================================================
# Interruptions that appear autonomously, not driven by a concurrent human:
# app-internal prompts (styled to look like the target app), browser-style
# banners/toasts, OS device events, etc. Realistic GUI-task noise beyond
# "another person typing in gedit".

def app_recovery_dialog() -> str:
    """Modal styled as an application recovery dialog (LibreOffice/gedit/VS
    Code-style). Agent dismisses with one click. Cost 1."""
    return _c(r"""
APPS=("LibreOffice" "gedit" "Text Editor" "Code" "Firefox" "Thunderbird");
MSGS=(
  "The previous session ended unexpectedly. Recover unsaved changes?"
  "An autosaved version of your document was found. Restore it?"
  "Your last session was interrupted. Do you want to reopen the files that were previously open?"
  "Recovery data is available for a recent session."
);
a=${APPS[$((RANDOM % ${#APPS[@]}))]}; m=${MSGS[$((RANDOM % ${#MSGS[@]}))]};
zenity --question --title "$a — Recovery" --text "$m" --ok-label="Recover" --cancel-label="Discard" --width=380 >/dev/null 2>&1 &
""")


def app_update_prompt() -> str:
    """Modal styled as an app-internal "update available" prompt. Cost 1."""
    return _c(r"""
APPS=("Google Chrome" "LibreOffice" "Thunderbird" "VS Code" "GIMP");
VERS=("118.0.2" "7.4.1" "115.2" "1.85.0" "2.10.36" "24.3");
a=${APPS[$((RANDOM % ${#APPS[@]}))]}; v=${VERS[$((RANDOM % ${#VERS[@]}))]};
zenity --info --title "$a" --text "A new version ($v) is available.\nRestart $a to apply the update." --width=380 >/dev/null 2>&1 &
""")


def app_save_reminder() -> str:
    """Modal styled as "You have unsaved changes" reminder. Cost 1."""
    return _c(r"""
TTL=("Unsaved changes" "Document modified" "Save reminder");
TXT=(
  "You have unsaved changes in this document. Save before closing?"
  "Your document has been modified since the last save."
  "Autosave is disabled. Remember to save your work."
);
t=${TTL[$((RANDOM % ${#TTL[@]}))]}; x=${TXT[$((RANDOM % ${#TXT[@]}))]};
zenity --warning --title "$t" --text "$x" --width=360 >/dev/null 2>&1 &
""")


def browser_cookie_banner() -> str:
    """A persistent always-on-top 'cookie consent' banner styled like a web
    overlay, anchored at the bottom of the screen. Dismissed by one click on
    the banner's button. Cost 1."""
    return _c(r"""
MSGS=(
  "This site uses cookies to improve your experience. By continuing to browse, you accept our cookie policy."
  "We use cookies for analytics and personalized content. Manage preferences or continue with default settings."
  "Your privacy matters. This site stores cookies for session management and performance."
);
m=${MSGS[$((RANDOM % ${#MSGS[@]}))]};
( zenity --info --title "Cookie Notice" --text "$m" --width=640 --ok-label="Accept All" >/dev/null 2>&1 ) & disown;
sleep 0.5;
wmctrl -r "Cookie Notice" -b add,above 2>/dev/null;
wmctrl -r "Cookie Notice" -e "0,0,980,1920,100" 2>/dev/null;
""")


def browser_download_toast() -> str:
    """Small always-on-top toast in the bottom-right styled as a Chrome
    download-complete notification. Cost 1."""
    return _c(r"""
FILES=("invoice.pdf" "report_Q2.docx" "photos.zip" "presentation.pptx" "budget_2024.xlsx" "meeting_minutes.txt" "dataset.csv" "screenshot.png");
f=${FILES[$((RANDOM % ${#FILES[@]}))]};
( zenity --info --title "Downloads" --text "$f finished downloading.\nClick to open or show in folder." --width=320 >/dev/null 2>&1 ) & disown;
sleep 0.5;
wmctrl -r "Downloads" -e "0,1550,960,360,100" 2>/dev/null;
wmctrl -r "Downloads" -b add,above 2>/dev/null;
""")


def browser_password_save() -> str:
    """Modal styled as Chrome's 'Save password for this site?' prompt. Cost 1."""
    return _c(r"""
SITES=("example.com" "corp.internal" "github.com" "mail.service.net" "portal.app.io" "dashboard.tools.dev");
s=${SITES[$((RANDOM % ${#SITES[@]}))]};
zenity --question --title "Google Chrome" --text "Save password for $s?\n\nYour password will be stored securely in your Google Account." --ok-label="Save" --cancel-label="Never" --width=400 >/dev/null 2>&1 &
""")


def browser_popup_blocked() -> str:
    """Toast-style banner: 'Pop-up blocked — Always allow?' . Cost 1."""
    return _c(r"""
SITES=("example.com" "news-site.com" "vendor.io" "corp.internal" "tools.dev");
s=${SITES[$((RANDOM % ${#SITES[@]}))]};
( zenity --question --title "Google Chrome" --text "Pop-ups blocked on $s\n\nAlways allow pop-ups and redirects from $s?" --ok-label="Allow" --cancel-label="Keep blocked" --width=360 >/dev/null 2>&1 ) & disown;
sleep 0.3;
wmctrl -r "Google Chrome" -b add,above 2>/dev/null || true
""")


def browser_translate_prompt() -> str:
    """Modal styled as Chrome's translate prompt. Cost 1."""
    return _c(r"""
L1=(Spanish French German Japanese Korean Portuguese Chinese Arabic Russian);
l=${L1[$((RANDOM % ${#L1[@]}))]};
zenity --question --title "Google Chrome" --text "This page is in $l.\nTranslate to English?" --ok-label="Translate" --cancel-label="No thanks" --width=340 >/dev/null 2>&1 &
""")


def usb_attach_dialog() -> str:
    """OS-style 'Removable device detected' modal. Cost 1."""
    return _c(r"""
DEV=("SanDisk USB Drive" "Kingston USB Drive" "Seagate External Disk" "SD Card Reader" "Android Phone (MTP)" "iPhone");
d=${DEV[$((RANDOM % ${#DEV[@]}))]};
zenity --question --title "Removable Media" --text "$d has been connected.\n\nOpen the device in Files?" --ok-label="Open" --cancel-label="Ignore" --width=380 >/dev/null 2>&1 &
""")


def bluetooth_pair_prompt() -> str:
    """OS-style Bluetooth pairing request modal. Cost 1."""
    return _c(r"""
DEV=("Sony WH-1000XM4" "AirPods Pro" "Bose QC35" "Galaxy Buds" "JBL Go 3" "Logitech MX Master");
PIN=$((RANDOM % 900000 + 100000));
d=${DEV[$((RANDOM % ${#DEV[@]}))]};
zenity --question --title "Bluetooth Pairing Request" --text "Pair with $d?\n\nConfirmation code: $PIN" --ok-label="Confirm" --cancel-label="Cancel" --width=360 >/dev/null 2>&1 &
""")


def crash_reporter_dialog() -> str:
    """OS-style 'App crashed, send report?' modal. Cost 1."""
    return _c(r"""
APPS=("gedit" "Firefox" "Thunderbird" "Document Viewer" "Archive Manager" "Image Viewer");
a=${APPS[$((RANDOM % ${#APPS[@]}))]};
zenity --question --title "Problem Report" --text "$a stopped unexpectedly.\nWould you like to send a diagnostic report to help improve the software?" --ok-label="Send report" --cancel-label="Don't send" --width=380 >/dev/null 2>&1 &
""")


def chat_bubble_persistent() -> str:
    """A small always-on-top 'chat bubble' gedit window anchored bottom-right.
    Simulates Intercom/Drift support widgets seen on many websites. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/chat_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 0.9;
wmctrl -r 'gedit' -e "0,1620,820,280,180" 2>/dev/null;
wmctrl -r 'gedit' -b add,above 2>/dev/null;
wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random
CHATS=[
  'Support: Hi! Need help?',
  'Bot: Can I help you today?',
  'Agent: How can I help?',
  'Chat: Thanks for visiting — any questions?',
]
pyautogui.typewrite(random.choice(CHATS), interval=0.03)
" 2>/dev/null || true
""")


def persistent_ad_banner() -> str:
    """A thin always-on-top banner at the top of the screen styled like a
    promotional ad. Dismissed by one click. Cost 1."""
    return _c(r"""
ADS=(
  "Limited offer: 40%% off Premium — upgrade now!"
  "New arrivals — free shipping this week only."
  "Claim your free trial — 30 days, no credit card."
  "Flash sale ends tonight — don't miss out."
  "Save 25%% on your next subscription renewal."
);
a=${ADS[$((RANDOM % ${#ADS[@]}))]};
( zenity --info --title "Promotion" --text "$a" --width=720 --ok-label="Close" >/dev/null 2>&1 ) & disown;
sleep 0.5;
wmctrl -r "Promotion" -e "0,0,0,1920,60" 2>/dev/null;
wmctrl -r "Promotion" -b add,above 2>/dev/null;
""")


def newsletter_signup_popup() -> str:
    """Modal styled as a website newsletter-signup popup. Cost 1."""
    return _c(r"""
T=("Subscribe for updates" "Join our newsletter" "Get weekly tips" "Stay in the loop");
t=${T[$((RANDOM % ${#T[@]}))]};
zenity --question --title "$t" --text "Get curated content and exclusive offers.\n\nEnter your email to subscribe?" --ok-label="Subscribe" --cancel-label="No thanks" --width=360 >/dev/null 2>&1 &
""")


# ---------- State-drift primitives ----------

def external_file_modify() -> str:
    """Simulates another process modifying a file on disk AND shows a tiny
    'File modified' toast so the event is observable by a screenshot-only
    agent. Cost 0."""
    return _c(r"""
D=("/home/user/Documents" "/home/user/Desktop" "/tmp" "/home/user/Downloads");
d=${D[$((RANDOM % ${#D[@]}))]};
mkdir -p "$d" 2>/dev/null;
F=(notes.txt report.md todo.txt draft.txt agenda.md scratch.txt);
f=${F[$((RANDOM % ${#F[@]}))]};
touch "$d/$f" 2>/dev/null || true;
DISPLAY=:0 python3 -c "
import tkinter as tk
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
W,H=320,48
r.geometry(f'{W}x{H}+{1920-W-16}+{48}')
r.configure(bg='#374151')
tk.Label(r, text='✎  Modified on disk: ${f}', bg='#374151', fg='#fde68a',
         font=('Sans',10), anchor='w').pack(fill='both', padx=12, pady=12)
r.after(10000, r.destroy); r.mainloop()
" >/dev/null 2>&1 & disown
""")


def clock_tick_notification() -> str:
    """Realistic time-of-day reminder, rendered as a tkinter corner toast
    (notify-osd may be absent). Auto-dismisses after 15s. Cost 0."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
N=['Time for your hourly break','15 minutes until your next meeting','Pomodoro timer finished','End of workday approaching','Lunch hour started']
text='Clock — ' + random.choice(N)
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
W,H=340,64
r.geometry(f'{W}x{H}+{1920-W-24}+{48}')
r.configure(bg='#1a365d')
tk.Label(r, text=text, bg='#1a365d', fg='white', font=('Sans',10,'bold'),
         wraplength=W-24, justify='left', anchor='w').pack(fill='both', padx=12, pady=12)
r.after(15000, r.destroy); r.mainloop()
" >/dev/null 2>&1 & disown
""")


def page_zoom_indicator() -> str:
    """Zoom-level indicator (Chrome Ctrl+scroll artifact). Rendered as a
    small centered tkinter banner that fades after 10s. Cost 0."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
z=random.choice([75,90,110,125,150])
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
W,H=180,48
# Center-top, like a browser zoom indicator
r.geometry(f'{W}x{H}+{(1920-W)//2}+{12}')
r.configure(bg='#222222')
tk.Label(r, text=f'Zoom: {z}%', bg='#222222', fg='white', font=('Sans',12,'bold')).pack(expand=True, fill='both')
r.after(10000, r.destroy); r.mainloop()
" >/dev/null 2>&1 & disown
""")


def high_contrast_flicker() -> str:
    """Toggle high-contrast mode on and revert after a few seconds. Entire
    screen palette inverts/desaturates briefly. Cost 0."""
    return _c(r"""
CUR=$(gsettings get org.gnome.desktop.a11y.interface high-contrast 2>/dev/null);
gsettings set org.gnome.desktop.a11y.interface high-contrast true 2>/dev/null;
(sleep $((RANDOM % 3 + 2)); gsettings set org.gnome.desktop.a11y.interface high-contrast $CUR 2>/dev/null) & disown
""")


def session_expiry_warning() -> str:
    """Modal styled as a session-expiring-soon warning. Cost 1."""
    return _c(r"""
MINS=(2 5 10 15);
m=${MINS[$((RANDOM % ${#MINS[@]}))]};
zenity --warning --title "Session expiring" --text "Your session will expire in $m minutes due to inactivity.\nClick to extend." --width=340 >/dev/null 2>&1 &
""")


# ===========================================================================
# TIER 3 — ACCIDENTAL TARGET INTERFERENCE (rare; simulates accidental bumps)
# ===========================================================================

def target_accidental_shove(app_title: str) -> str:
    """Target window is shoved in a random direction by a random offset —
    like someone bumped it. Titlebar remains visible. Cost 1 (drag back)."""
    return _c(rf"""
DIR=$((RANDOM % 4));
W=$((RANDOM % 400 + 1000));
H=$((RANDOM % 300 + 650));
case $DIR in
  0) X=$((RANDOM % 600 + 900)); Y=$((RANDOM % 150 + 50)) ;;
  1) X=$((-(RANDOM % 600 + 300))); Y=$((RANDOM % 150 + 50)) ;;
  2) X=$((RANDOM % 400 + 100)); Y=$((RANDOM % 400 + 500)) ;;
  3) X=$((RANDOM % 400 + 100)); Y=$((-(RANDOM % 200 + 100))) ;;
esac
wmctrl -r '{app_title}' -e "0,$X,$Y,$W,$H" 2>/dev/null || true
""")


def target_accidental_shrink(app_title: str) -> str:
    """Target window is resized smaller — like someone dragged a corner.
    Cost 1 (drag corner back)."""
    return _c(rf"""
W=$((RANDOM % 500 + 650));
H=$((RANDOM % 350 + 450));
X=$((RANDOM % 400 + 80));
Y=$((RANDOM % 200 + 60));
wmctrl -r '{app_title}' -e "0,$X,$Y,$W,$H" 2>/dev/null || true
""")


def target_partial_overlap(app_title: str) -> str:
    """A noise-owned window is positioned partly over the target from above.
    Cost 1 (click target's visible half or taskbar entry)."""
    return _c(rf"""
PAIR=("gedit --new-window /tmp/overlap_$RANDOM.txt|gedit" "gnome-system-monitor|System Monitor" "nautilus /tmp|Files");
p=${{PAIR[$((RANDOM % ${{#PAIR[@]}}))]}};
cmd="${{p%%|*}}"; title="${{p##*|}}";
eval "$cmd >/dev/null 2>&1 & disown";
sleep 0.6;
OX=$((RANDOM % 700 + 100)); OY=$((RANDOM % 400 + 80));
OW=$((RANDOM % 300 + 500)); OH=$((RANDOM % 200 + 400));
wmctrl -r "$title" -e "0,$OX,$OY,$OW,$OH" 2>/dev/null;
wmctrl -r "$title" -b add,above 2>/dev/null;
""")


def target_accidental_cover_above() -> str:
    """A noise-owned window is maximized and pinned above-all — fully covers
    the target. Cost 2 (Alt-Tab + click). Use only in easy/very_easy tiers.
    (No target-app argument needed; uses a non-target window maximized above.)"""
    return _c(r"""
PAIR=("gedit --new-window /tmp/cover_$RANDOM.txt|gedit" "nautilus /home/user|Nautilus" "gnome-system-monitor|System Monitor");
p=${PAIR[$((RANDOM % ${#PAIR[@]}))]};
cmd="${p%%|*}"; title="${p##*|}";
eval "$cmd >/dev/null 2>&1 & disown";
sleep 0.6;
wmctrl -r "$title" -b add,maximized_vert,maximized_horz 2>/dev/null;
wmctrl -r "$title" -b add,above 2>/dev/null;
""")


# ===========================================================================
# CATALOG
# ===========================================================================

TEMPLATE_CATALOG: List[Dict] = [
    # ── TIER 1: HUMAN TASK SESSIONS (primary) ────────────────────────────
    # writing / notes
    {"name": "human_meeting_notes",         "fn": human_meeting_notes,         "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "human_todo_list",             "fn": human_todo_list,             "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "human_compose_email",         "fn": human_compose_email,         "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "human_code_snippet",          "fn": human_code_snippet,          "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "human_sticky_note",           "fn": human_sticky_note,           "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    # terminal
    {"name": "human_terminal_commands",     "fn": human_terminal_commands,     "cost": 1, "category": "human_terminal_session", "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "human_terminal_git_mock",     "fn": human_terminal_git_mock,     "cost": 1, "category": "human_terminal_session", "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "human_terminal_tail_log",     "fn": human_terminal_tail_log,     "cost": 1, "category": "human_terminal_session", "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    # file browsing
    {"name": "human_file_browser",              "fn": human_file_browser,              "cost": 1, "category": "human_file_session", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_create_folder_on_desktop",  "fn": human_create_folder_on_desktop,  "cost": 1, "category": "human_file_session", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_trash_scratch",             "fn": human_trash_scratch,             "cost": 0, "category": "human_file_session", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_drop_markdown",             "fn": human_drop_markdown,             "cost": 0, "category": "human_file_session", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    # system / utility
    {"name": "human_system_monitor",        "fn": human_system_monitor,        "cost": 1, "category": "human_system_session",   "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "human_open_settings",         "fn": human_open_settings,         "cost": 1, "category": "human_system_session",   "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "human_vlc_session",           "fn": human_vlc_session,           "cost": 1, "category": "human_system_session",   "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    # window management
    {"name": "human_alt_tab_cycle",         "fn": human_alt_tab_cycle,         "cost": 1, "category": "human_window_session",   "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "human_drag_own_window",       "fn": human_drag_own_window,       "cost": 1, "category": "human_window_session",   "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "human_minimize_restore",      "fn": human_minimize_restore,      "cost": 0, "category": "human_window_session",   "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "human_right_click_desktop_menu","fn": human_right_click_desktop_menu,"cost": 0,"category":"human_window_session", "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    # optional (image rebuild needed)
    {"name": "human_calc_session",          "fn": human_calc_session,          "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1, "requires_image_rebuild": True},

    # ── TIER 2: AMBIENT BACKGROUND (secondary) ───────────────────────────
    {"name": "ambient_notification",        "fn": ambient_notification,        "cost": 0, "category": "ambient_notification",   "touches_target": False, "once_default": False, "needs_target": False, "tier_group": 2},
    {"name": "ambient_modal",               "fn": ambient_modal,               "cost": 1, "category": "ambient_modal_dialog",   "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "ambient_cpu_burst",           "fn": ambient_cpu_burst,           "cost": 0, "category": "ambient_resource",       "touches_target": False, "once_default": False, "needs_target": False, "tier_group": 2},
    {"name": "ambient_audio_ding",          "fn": ambient_audio_ding,          "cost": 0, "category": "ambient_resource",       "touches_target": False, "once_default": False, "needs_target": False, "tier_group": 2},
    {"name": "ambient_fake_download",       "fn": ambient_fake_download,       "cost": 0, "category": "ambient_filesystem",     "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "ambient_dark_mode_flicker",   "fn": ambient_dark_mode_flicker,   "cost": 0, "category": "ambient_visual_flicker", "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},

    # ── TIER 2B: APPLICATION / BROWSER / SYSTEM INTERRUPTIONS ────────────
    {"name": "app_recovery_dialog",         "fn": app_recovery_dialog,         "cost": 1, "category": "app_internal_prompt",    "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "app_update_prompt",           "fn": app_update_prompt,           "cost": 1, "category": "app_internal_prompt",    "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "app_save_reminder",           "fn": app_save_reminder,           "cost": 1, "category": "app_internal_prompt",    "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "browser_cookie_banner",       "fn": browser_cookie_banner,       "cost": 1, "category": "browser_overlay",        "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "browser_download_toast",      "fn": browser_download_toast,      "cost": 1, "category": "browser_overlay",        "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "browser_password_save",       "fn": browser_password_save,       "cost": 1, "category": "browser_overlay",        "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "browser_popup_blocked",       "fn": browser_popup_blocked,       "cost": 1, "category": "browser_overlay",        "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "browser_translate_prompt",    "fn": browser_translate_prompt,    "cost": 1, "category": "browser_overlay",        "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "usb_attach_dialog",           "fn": usb_attach_dialog,           "cost": 1, "category": "os_device_event",        "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "bluetooth_pair_prompt",       "fn": bluetooth_pair_prompt,       "cost": 1, "category": "os_device_event",        "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "crash_reporter_dialog",       "fn": crash_reporter_dialog,       "cost": 1, "category": "os_device_event",        "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "chat_bubble_persistent",      "fn": chat_bubble_persistent,      "cost": 1, "category": "persistent_overlay",     "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "persistent_ad_banner",        "fn": persistent_ad_banner,        "cost": 1, "category": "persistent_overlay",     "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "newsletter_signup_popup",     "fn": newsletter_signup_popup,     "cost": 1, "category": "persistent_overlay",     "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "external_file_modify",        "fn": external_file_modify,        "cost": 0, "category": "state_drift",            "touches_target": False, "once_default": False, "needs_target": False, "tier_group": 2},
    {"name": "clock_tick_notification",     "fn": clock_tick_notification,     "cost": 0, "category": "state_drift",            "touches_target": False, "once_default": False, "needs_target": False, "tier_group": 2},
    {"name": "page_zoom_indicator",         "fn": page_zoom_indicator,         "cost": 0, "category": "state_drift",            "touches_target": False, "once_default": False, "needs_target": False, "tier_group": 2},
    {"name": "high_contrast_flicker",       "fn": high_contrast_flicker,       "cost": 0, "category": "accessibility_flicker",  "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},
    {"name": "session_expiry_warning",      "fn": session_expiry_warning,      "cost": 1, "category": "browser_overlay",        "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 2},

    # ── TIER 3: ACCIDENTAL TARGET INTERFERENCE (rare) ────────────────────
    {"name": "target_accidental_shove",       "fn": target_accidental_shove,       "cost": 1, "category": "target_accidental", "touches_target": True,  "once_default": True, "needs_target": True,  "tier_group": 3},
    {"name": "target_accidental_shrink",      "fn": target_accidental_shrink,      "cost": 1, "category": "target_accidental", "touches_target": True,  "once_default": True, "needs_target": True,  "tier_group": 3},
    {"name": "target_partial_overlap",        "fn": target_partial_overlap,        "cost": 1, "category": "target_accidental", "touches_target": True,  "once_default": True, "needs_target": True,  "tier_group": 3},
    {"name": "target_accidental_cover_above", "fn": target_accidental_cover_above, "cost": 2, "category": "target_accidental", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 3},

    # ═══ VISUAL-VARIANT modals (xmessage / tkinter / browser) ────────────
    {"name": "modal_xmessage_centered",       "fn": modal_xmessage_centered,       "cost": 1, "category": "modal_dialog_variant",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "modal_xmessage_compact",        "fn": modal_xmessage_compact,        "cost": 1, "category": "modal_dialog_variant",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "modal_tkinter_info",            "fn": modal_tkinter_info,            "cost": 1, "category": "modal_dialog_variant",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "modal_tkinter_warning",         "fn": modal_tkinter_warning,         "cost": 1, "category": "modal_dialog_variant",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "modal_tkinter_error",           "fn": modal_tkinter_error,           "cost": 1, "category": "modal_dialog_variant",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "modal_tkinter_question",        "fn": modal_tkinter_question,        "cost": 1, "category": "modal_dialog_variant",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "modal_tkinter_custom_banner",   "fn": modal_tkinter_custom_banner,   "cost": 1, "category": "modal_dialog_variant",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "modal_browser_alert",           "fn": modal_browser_alert,           "cost": 1, "category": "modal_dialog_variant",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "modal_browser_dialog_element",  "fn": modal_browser_dialog_element,  "cost": 1, "category": "modal_dialog_variant",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    # notification variants
    {"name": "notif_xmessage_timeout",        "fn": notif_xmessage_timeout,        "cost": 0, "category": "notification_variant",   "touches_target": False, "once_default": False, "needs_target": False, "tier_group": 2},
    {"name": "notif_tkinter_corner_toast",    "fn": notif_tkinter_corner_toast,    "cost": 0, "category": "notification_variant",   "touches_target": False, "once_default": False, "needs_target": False, "tier_group": 2},
    {"name": "notif_terminal_flash",          "fn": notif_terminal_flash,          "cost": 0, "category": "notification_variant",   "touches_target": False, "once_default": False, "needs_target": False, "tier_group": 2},
    # persistent-overlay variants
    {"name": "overlay_tkinter_top_banner",    "fn": overlay_tkinter_top_banner,    "cost": 1, "category": "persistent_overlay_variant", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "overlay_tkinter_bottom_banner", "fn": overlay_tkinter_bottom_banner, "cost": 1, "category": "persistent_overlay_variant", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "overlay_tkinter_right_sidebar", "fn": overlay_tkinter_right_sidebar, "cost": 1, "category": "persistent_overlay_variant", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "overlay_tkinter_floating_widget","fn": overlay_tkinter_floating_widget,"cost": 1,"category": "persistent_overlay_variant", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "overlay_browser_cookie_real",   "fn": overlay_browser_cookie_real,   "cost": 1, "category": "persistent_overlay_variant", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    # focus-steal variants
    {"name": "focus_steal_tkinter_app",       "fn": focus_steal_tkinter_app,       "cost": 1, "category": "focus_steal_variant",    "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "focus_steal_two_windows",       "fn": focus_steal_two_windows,       "cost": 1, "category": "focus_steal_variant",    "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},
    {"name": "focus_steal_scrolling_terminal","fn": focus_steal_scrolling_terminal,"cost": 1, "category": "focus_steal_variant",    "touches_target": False, "once_default": True,  "needs_target": False, "tier_group": 1},

    # ═══ COMPOSITIONAL (multi-element firings) ──────────────────────────
    {"name": "comp_notif_then_modal",            "fn": comp_notif_then_modal,            "cost": 1, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_three_notifs_staggered",      "fn": comp_three_notifs_staggered,      "cost": 0, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_cookie_plus_popup",           "fn": comp_cookie_plus_popup,           "cost": 2, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_focus_steal_plus_shove",      "fn": comp_focus_steal_plus_shove,      "cost": 1, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_ad_banner_plus_chat",         "fn": comp_ad_banner_plus_chat,         "cost": 2, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_audio_plus_visual_flicker",   "fn": comp_audio_plus_visual_flicker,   "cost": 0, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_cpu_spike_plus_resource_notif","fn": comp_cpu_spike_plus_resource_notif,"cost":0,"category": "compositional",      "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_drag_plus_filesystem",        "fn": comp_drag_plus_filesystem,        "cost": 1, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_human_writing_plus_notif",    "fn": comp_human_writing_plus_notif,    "cost": 1, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "comp_modal_then_decoy_notif",      "fn": comp_modal_then_decoy_notif,      "cost": 1, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_shove_then_cover",            "fn": comp_shove_then_cover,            "cost": 2, "category": "compositional",       "touches_target": True,  "once_default": True, "needs_target": True,  "tier_group": 3},
    {"name": "comp_multi_modal_chain",           "fn": comp_multi_modal_chain,           "cost": 3, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_overlay_plus_focus_steal",    "fn": comp_overlay_plus_focus_steal,    "cost": 2, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_decoy_nothing",               "fn": comp_decoy_nothing,               "cost": 0, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "comp_notif_burst_plus_flicker",    "fn": comp_notif_burst_plus_flicker,    "cost": 0, "category": "compositional",       "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},

    # ═══ EXTENDED HUMAN SESSIONS ────────────────────────────────────────
    {"name": "human_journal_entry",          "fn": human_journal_entry,          "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_sql_query_draft",        "fn": human_sql_query_draft,        "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_readme_draft",           "fn": human_readme_draft,           "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_essay_paragraph",        "fn": human_essay_paragraph,        "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_config_file_edit",       "fn": human_config_file_edit,       "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_terminal_file_navigation","fn": human_terminal_file_navigation,"cost": 1,"category": "human_terminal_session", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_terminal_log_grep",      "fn": human_terminal_log_grep,      "cost": 1, "category": "human_terminal_session", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_terminal_network_check", "fn": human_terminal_network_check, "cost": 1, "category": "human_terminal_session", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_terminal_python_repl",   "fn": human_terminal_python_repl,   "cost": 1, "category": "human_terminal_session", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_file_rename_desktop",    "fn": human_file_rename_desktop,    "cost": 1, "category": "human_file_session",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_file_search_nautilus",   "fn": human_file_search_nautilus,   "cost": 1, "category": "human_file_session",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_file_zoom_view",         "fn": human_file_zoom_view,         "cost": 1, "category": "human_file_session",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_file_view_toggle",       "fn": human_file_view_toggle,       "cost": 1, "category": "human_file_session",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_scroll_long_doc",        "fn": human_scroll_long_doc,        "cost": 1, "category": "human_window_session",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_select_all_copy",        "fn": human_select_all_copy,        "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_find_in_editor",         "fn": human_find_in_editor,         "cost": 1, "category": "human_writing_session",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_split_editor_windows",   "fn": human_split_editor_windows,   "cost": 1, "category": "human_window_session",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_preferences_dialog",     "fn": human_preferences_dialog,     "cost": 1, "category": "human_system_session",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_help_browser",           "fn": human_help_browser,           "cost": 1, "category": "human_system_session",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_image_viewer_browse",    "fn": human_image_viewer_browse,    "cost": 1, "category": "human_system_session",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},
    {"name": "human_archive_browse",         "fn": human_archive_browse,         "cost": 1, "category": "human_system_session",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 1},

    # ═══ EXTENDED APP / BROWSER / OS PROMPTS ────────────────────────────
    {"name": "app_signin_required",        "fn": app_signin_required,        "cost": 1, "category": "app_internal_prompt", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "app_trial_expired",          "fn": app_trial_expired,          "cost": 1, "category": "app_internal_prompt", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "app_first_run_tutorial",     "fn": app_first_run_tutorial,     "cost": 1, "category": "app_internal_prompt", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "app_tip_of_the_day",         "fn": app_tip_of_the_day,         "cost": 1, "category": "app_internal_prompt", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "app_feedback_request",       "fn": app_feedback_request,       "cost": 1, "category": "app_internal_prompt", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "app_whats_new_overlay",      "fn": app_whats_new_overlay,      "cost": 1, "category": "app_internal_prompt", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "app_license_key_prompt",     "fn": app_license_key_prompt,     "cost": 1, "category": "app_internal_prompt", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "app_export_dialog",          "fn": app_export_dialog,          "cost": 1, "category": "app_internal_prompt", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_autofill_suggestion","fn": browser_autofill_suggestion,"cost": 1, "category": "browser_overlay",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_certificate_warning","fn": browser_certificate_warning,"cost": 1, "category": "browser_overlay",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_location_permission","fn": browser_location_permission,"cost": 1, "category": "browser_overlay",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_microphone_permission","fn":browser_microphone_permission,"cost":1,"category": "browser_overlay",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_camera_permission",  "fn": browser_camera_permission,  "cost": 1, "category": "browser_overlay",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_notifications_permission","fn":browser_notifications_permission,"cost":1,"category":"browser_overlay","touches_target":False,"once_default":True,"needs_target":False,"tier_group":2},
    {"name": "browser_clipboard_permission","fn":browser_clipboard_permission,"cost": 1, "category": "browser_overlay",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_fullscreen_exit_hint","fn":browser_fullscreen_exit_hint,"cost": 0, "category": "browser_overlay",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_install_extension_banner","fn":browser_install_extension_banner,"cost":1,"category":"browser_overlay","touches_target":False,"once_default":True,"needs_target":False,"tier_group":2},
    {"name": "browser_not_responding_banner","fn":browser_not_responding_banner,"cost":1,"category":"browser_overlay",      "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_close_multiple_tabs","fn":browser_close_multiple_tabs, "cost": 1, "category": "browser_overlay",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_clear_browsing_data","fn":browser_clear_browsing_data, "cost": 1, "category": "browser_overlay",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_reopen_closed_tabs", "fn":browser_reopen_closed_tabs,  "cost": 1, "category": "browser_overlay",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "browser_proxy_reconnect_required","fn":browser_proxy_reconnect_required,"cost":2,"category":"network_proxy_event","touches_target":False,"once_default":True,"needs_target":False,"tier_group":3},
    {"name": "browser_proxy_auth_refresh", "fn": browser_proxy_auth_refresh, "cost": 2, "category": "network_proxy_event", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 3},
    {"name": "browser_dns_probe_recovering", "fn": browser_dns_probe_recovering, "cost": 2, "category": "network_proxy_event", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 3},
    {"name": "os_disk_almost_full",        "fn": os_disk_almost_full,        "cost": 1, "category": "os_device_event",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "os_wifi_connection_dialog",  "fn": os_wifi_connection_dialog,  "cost": 1, "category": "os_device_event",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "os_print_dialog_stuck",      "fn": os_print_dialog_stuck,      "cost": 1, "category": "os_device_event",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "os_software_updater_banner", "fn": os_software_updater_banner, "cost": 1, "category": "os_device_event",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "os_unattended_upgrades_notice","fn":os_unattended_upgrades_notice,"cost":0,"category":"state_drift",          "touches_target": False, "once_default": False,"needs_target": False, "tier_group": 2},
    {"name": "os_antivirus_scan_notice",   "fn": os_antivirus_scan_notice,   "cost": 0, "category": "state_drift",         "touches_target": False, "once_default": False,"needs_target": False, "tier_group": 2},
    {"name": "os_firewall_block_notice",   "fn": os_firewall_block_notice,   "cost": 0, "category": "state_drift",         "touches_target": False, "once_default": False,"needs_target": False, "tier_group": 2},
    {"name": "os_storage_quota_approaching","fn":os_storage_quota_approaching,"cost": 1, "category": "os_device_event",     "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "os_mail_arrived_toast",      "fn": os_mail_arrived_toast,      "cost": 0, "category": "ambient_notification","touches_target": False, "once_default": False,"needs_target": False, "tier_group": 2},
    {"name": "os_vpn_tunnel_reconnecting", "fn": os_vpn_tunnel_reconnecting, "cost": 2, "category": "network_proxy_event", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 3},
    {"name": "os_secure_gateway_flap", "fn": os_secure_gateway_flap, "cost": 2, "category": "network_proxy_event", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 3},
    {"name": "os_power_adapter_disconnected","fn":os_power_adapter_disconnected,"cost":0,"category":"ambient_notification", "touches_target": False, "once_default": False,"needs_target": False, "tier_group": 2},

    # ═══ DIVERSE RECOVERY PATHS ─────────────────────────────────────────
    {"name": "recovery_wait_autodismiss",       "fn": recovery_wait_autodismiss,       "cost": 0, "category": "recovery_variant_wait",    "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_escape_only",            "fn": recovery_escape_only,            "cost": 1, "category": "recovery_variant_escape",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_drag_window_required",   "fn": recovery_drag_window_required,   "cost": 1, "category": "recovery_variant_drag",    "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_resize_required",        "fn": recovery_resize_required,        "cost": 1, "category": "recovery_variant_resize",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_specific_button",        "fn": recovery_specific_button,        "cost": 1, "category": "recovery_variant_button",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_click_outside",          "fn": recovery_click_outside,          "cost": 1, "category": "recovery_variant_outside", "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_scroll_to_dismiss",      "fn": recovery_scroll_to_dismiss,      "cost": 1, "category": "recovery_variant_scroll",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_two_step",               "fn": recovery_two_step,               "cost": 2, "category": "recovery_variant_multi",   "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_double_click",           "fn": recovery_double_click,           "cost": 1, "category": "recovery_variant_dblclick","touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_right_click_menu",       "fn": recovery_right_click_menu,       "cost": 2, "category": "recovery_variant_rclick",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_type_to_close",          "fn": recovery_type_to_close,          "cost": 2, "category": "recovery_variant_type",    "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_click_specific_position","fn": recovery_click_specific_position,"cost": 1, "category": "recovery_variant_position","touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_drag_to_corner",         "fn": recovery_drag_to_corner,         "cost": 2, "category": "recovery_variant_drag",    "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_minimize_instead",       "fn": recovery_minimize_instead,       "cost": 1, "category": "recovery_variant_minimize","touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
    {"name": "recovery_ignore_decoy",           "fn": recovery_ignore_decoy,           "cost": 0, "category": "recovery_variant_ignore",  "touches_target": False, "once_default": True, "needs_target": False, "tier_group": 2},
]


RECOVERY_ACTIONS: Dict[str, List[str]] = {
    # Tier 1: every human session ends with the noise app focused; agent clicks
    # its own window to resume.
    "human_writing_session":   ["click on the target app window to restore keyboard focus"],
    "human_terminal_session":  ["click on the target app window to restore keyboard focus"],
    "human_file_session":      ["click on the target app window to restore keyboard focus"],
    "human_system_session":    ["click on the target app window to restore keyboard focus"],
    "human_window_session":    ["click on the target app window to restore keyboard focus"],
    # Tier 2
    "ambient_notification":    [],
    "ambient_modal_dialog":    ["click the dialog button to dismiss"],
    "ambient_resource":        [],
    "ambient_filesystem":      [],
    "ambient_visual_flicker":  [],
    # Tier 2B
    "app_internal_prompt":     ["click the app's recovery/update/save button to dismiss"],
    "browser_overlay":         ["click the banner's close/accept button to dismiss"],
    "os_device_event":         ["click the device dialog's Ignore/Cancel button"],
    "persistent_overlay":      ["click the overlay's close button to dismiss it from above other windows"],
    "state_drift":             [],
    "accessibility_flicker":   [],
    # Tier 3
    "target_accidental":       ["drag the target app's titlebar/corner back into place or Alt-Tab + click to it"],

    # Visual variants (re-use base category semantics)
    "modal_dialog_variant":        ["click the dialog button (or X close) to dismiss"],
    "notification_variant":        [],
    "persistent_overlay_variant":  ["click the overlay's close/accept button to dismiss"],
    "focus_steal_variant":         ["click the target app window to restore focus"],

    # Compositional: varies — two or three recoveries may be needed
    "compositional":               ["dismiss each component in turn (modal click, overlay close, refocus target as needed)"],

    # Diverse recovery paths — each category names its own required action
    "recovery_variant_wait":      [],  # correct action: do nothing; window auto-closes
    "recovery_variant_escape":    ["press the Escape key while the noise window has focus"],
    "recovery_variant_drag":      ["drag the noise window by its titlebar to the edge of the screen"],
    "recovery_variant_resize":    ["drag a corner of the noise window to resize it smaller"],
    "recovery_variant_button":    ["read the button labels; click the specific one named ('Maybe'/'Close'/etc.)"],
    "recovery_variant_outside":   ["click on the target app window — this dismisses the popup and refocuses"],
    "recovery_variant_scroll":    ["scroll the mouse wheel over the noise window to dismiss it"],
    "recovery_variant_multi":     ["click the first button ('Review'), then click the appearing 'Confirm' button"],
    "recovery_variant_dblclick":  ["double-click anywhere in the noise window to dismiss"],
    "recovery_variant_rclick":    ["right-click inside the noise window, then click 'Close panel' in the menu"],
    "recovery_variant_type":      ["click the input field if needed, type anything, press Enter"],
    "recovery_variant_position":  ["locate the small × at the bottom-right corner of the popup and click it"],
    "recovery_variant_minimize":  ["click the window's minimize button (or Super+H)"],
    "recovery_variant_ignore":    [],  # correct action: ignore it
}


SEVERITY_FOR_COST = {0: "low", 1: "medium", 2: "high", 3: "high"}


def resolve_app_title(related_apps: List[str]) -> Optional[str]:
    """Given a task's `related_apps` list, return the first recognized app's
    WM window title, or None if no single target is identifiable."""
    for key in (related_apps or []):
        title = APP_TO_WM_TITLE.get(key)
        if title:
            return title
    return None


def is_human_task(template: Dict) -> bool:
    return template.get("tier_group") == 1


def is_ambient(template: Dict) -> bool:
    return template.get("tier_group") == 2


def is_target_interference(template: Dict) -> bool:
    return template.get("tier_group") == 3
