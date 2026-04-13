"""
compositional.py — MULTI-ELEMENT COMPOSITIONS.

Each template here combines 2-3 primitive behaviors with randomized ordering
and intra-command delays. This prevents the agent from memorizing any fixed
firing pattern: even if it has seen every base element, a random composition
of two of them in a specific order is effectively a new event.

All compositions respect the ≤3-step recovery rule: the total recovery cost
of all elements in a composition is bounded by the tier cap of the target
task. Compositions are hand-curated so that any two behaviors in them are
compatible (e.g. notifications can chain with modals; occlusions don't
chain with each other).
"""

from __future__ import annotations


def _c(src: str) -> str:
    return " ".join(src.split())


# ============================================================================
# Paired compositions (2 elements, random order)
# ============================================================================

def comp_notif_then_modal() -> str:
    """A passive notification fires, then a dismissible modal appears 1s later.
    Agent must dismiss the modal (recovery path obscured by the notification
    that fired first). Cost 1."""
    return _c(r"""
APP=(Email Slack Drive Calendar); P=(Alex Maya Priya);
a=${APP[$((RANDOM % ${#APP[@]}))]}; p=${P[$((RANDOM % ${#P[@]}))]};
zenity --notification --window-icon=info --text "$a — $p: Please review" 2>/dev/null & disown;
sleep $((RANDOM % 2 + 1));
TTL=("Sync conflict" "Reconnect required" "Refresh needed" "Permission changed");
t=${TTL[$((RANDOM % ${#TTL[@]}))]};
zenity --warning --title "$t" --text "Action required — review settings." --width=320 >/dev/null 2>&1 &
""")


def comp_three_notifs_staggered() -> str:
    """Three notifications fire with random 0.5-1.5s gaps — simulates a burst
    arriving from different apps. Cost 0."""
    return _c(r"""
APPS=(Slack Email WhatsApp Signal Teams Calendar Drive LinkedIn);
SUBJ=("new message" "mention" "reply" "reminder" "alert" "update" "share" "ping");
for i in 1 2 3; do
  a=${APPS[$((RANDOM % ${#APPS[@]}))]}; s=${SUBJ[$((RANDOM % ${#SUBJ[@]}))]};
  zenity --notification --window-icon=info --text "$a — $s" 2>/dev/null & disown;
  sleep 0.$((RANDOM % 10 + 3));
done
""")


def comp_cookie_plus_popup() -> str:
    """A cookie banner appears at the bottom, then a pop-up-blocked notice
    appears 2s later. Two browser-style artifacts stacked. Cost 2."""
    return _c(r"""
( zenity --info --title 'Cookie Notice' --text 'This site uses cookies for analytics. By continuing you accept the policy.' --width=580 --ok-label='Accept' >/dev/null 2>&1 ) & disown;
sleep 0.4;
wmctrl -r 'Cookie Notice' -e "0,0,980,1920,100" 2>/dev/null;
wmctrl -r 'Cookie Notice' -b add,above 2>/dev/null;
sleep $((RANDOM % 2 + 2));
SITES=(news.example.com shop.vendor.io blog.tools.dev);
s=${SITES[$((RANDOM % ${#SITES[@]}))]};
zenity --question --title 'Google Chrome' --text "Pop-up blocked on $s.\nAlways allow from this site?" --ok-label='Allow' --cancel-label='Keep blocked' --width=340 >/dev/null 2>&1 &
""")


def comp_focus_steal_plus_shove() -> str:
    """A non-target window opens with focus, then a small wmctrl geometry shove
    of that noise-owned window (to make it even more visible). The target
    window is untouched. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/c_shove_$RANDOM.md >/dev/null 2>&1 & disown;
sleep 0.9;
wmctrl -a 'gedit' 2>/dev/null;
sleep 0.3;
X=$((RANDOM % 800 + 200)); Y=$((RANDOM % 400 + 100));
wmctrl -r 'gedit' -e "0,$X,$Y,640,480" 2>/dev/null
""")


def comp_ad_banner_plus_chat() -> str:
    """A top ad-banner appears, then a support-chat bubble appears bottom-right.
    Two persistent overlays at once. Cost 2."""
    return _c(r"""
( zenity --info --title 'Promotion' --text 'Spring sale — 30%% off selected items this week' --width=720 --ok-label='Close' >/dev/null 2>&1 ) & disown;
sleep 0.4;
wmctrl -r 'Promotion' -e "0,0,0,1920,56" 2>/dev/null;
wmctrl -r 'Promotion' -b add,above 2>/dev/null;
sleep 0.8;
gedit --new-window /tmp/c_chat_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 0.8;
wmctrl -r 'gedit' -e "0,1620,820,280,180" 2>/dev/null;
wmctrl -r 'gedit' -b add,above 2>/dev/null
""")


def comp_audio_plus_visual_flicker() -> str:
    """Audio ding plays simultaneously with a brief color-scheme flicker —
    sensory-pairing noise. Cost 0."""
    return _c(r"""
S=(/usr/share/sounds/freedesktop/stereo/complete.oga /usr/share/sounds/freedesktop/stereo/bell.oga /usr/share/sounds/freedesktop/stereo/dialog-information.oga);
for x in "${S[@]}"; do [[ -f "$x" ]] && V+=("$x"); done;
[[ ${#V[@]} -gt 0 ]] && paplay "${V[$((RANDOM % ${#V[@]}))]}" 2>/dev/null & disown;
CUR=$(gsettings get org.gnome.desktop.interface color-scheme 2>/dev/null);
N=('prefer-dark' 'prefer-light'); gsettings set org.gnome.desktop.interface color-scheme "${N[$((RANDOM % ${#N[@]}))]}" 2>/dev/null;
(sleep $((RANDOM % 3 + 2)); gsettings set org.gnome.desktop.interface color-scheme $CUR 2>/dev/null) & disown
""")


def comp_cpu_spike_plus_resource_notif() -> str:
    """Background CPU burn fires while a system notification announces the
    resource activity. Cost 0."""
    return _c(r"""
N=$((RANDOM % 2000000 + 800000));
python3 -c "sum(i*i for i in range($N))" >/dev/null 2>&1 & disown;
sleep 0.3;
M=("CPU usage elevated" "Background task started" "Scheduled scan initiated" "Indexing in progress");
m=${M[$((RANDOM % ${#M[@]}))]};
zenity --notification --window-icon=info --text "System — $m" 2>/dev/null &
""")


def comp_drag_plus_filesystem() -> str:
    """A noise-owned window drags along a random 3-waypoint path while new
    scratch files appear on the Desktop at each waypoint. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/c_drag_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 0.9;
N=(session workbench note); mkdir -p /home/user/Desktop 2>/dev/null;
for _ in 1 2 3; do
  X=$((RANDOM % 1400 + 100)); Y=$((RANDOM % 600 + 80));
  wmctrl -r 'gedit' -e "0,$X,$Y,640,420" 2>/dev/null;
  touch "/home/user/Desktop/${N[$((RANDOM % ${#N[@]}))]}_$RANDOM.txt" 2>/dev/null;
  sleep 0.3;
done
""")


def comp_human_writing_plus_notif() -> str:
    """Human starts typing in gedit, and midway a notification arrives. The
    writing session continues; the notification fires alongside. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/c_mix_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
(
  DISPLAY=:0 python3 -c "
import pyautogui, random, time
LINES = ['Working notes','- kickoff on Mon','- doc review Wed','- retro Fri','Open items:']
for l in LINES:
    pyautogui.typewrite(l, interval=0.02); pyautogui.press('enter'); time.sleep(0.1)
" 2>/dev/null
) &
sleep $((RANDOM % 2 + 1));
zenity --notification --window-icon=info --text 'Reminder — 1:1 starts in 10 min' 2>/dev/null &
wait
""")


def comp_modal_then_decoy_notif() -> str:
    """A modal fires, then a notification arrives WITHOUT needing response
    (decoy). The correct recovery is still just to dismiss the modal — not
    the notification. Trains the agent to distinguish. Cost 1."""
    return _c(r"""
TTL=("Session Notice" "Update Info"); t=${TTL[$((RANDOM % ${#TTL[@]}))]};
zenity --info --title "$t" --text "A routine maintenance notice — no action needed." --width=340 >/dev/null 2>&1 &
sleep 1.2;
zenity --notification --window-icon=info --text "System — Backup verification complete" 2>/dev/null &
""")


def comp_shove_then_cover(app_title: str) -> str:
    """Target window is first shoved to a random position, then a noise-owned
    window opens maximized above it. Agent faces two-level recovery: Alt-Tab
    then drag-back. Cost 2 (within easy+ tier budget)."""
    return _c(rf"""
DIR=$((RANDOM % 4));
case $DIR in
  0) X=$((RANDOM % 600 + 900)); Y=$((RANDOM % 150 + 50)) ;;
  1) X=$((-(RANDOM % 600 + 300))); Y=$((RANDOM % 150 + 50)) ;;
  2) X=$((RANDOM % 400 + 100)); Y=$((RANDOM % 400 + 500)) ;;
  3) X=$((RANDOM % 400 + 100)); Y=$((-(RANDOM % 200 + 100))) ;;
esac
wmctrl -r '{app_title}' -e "0,$X,$Y,1200,800" 2>/dev/null;
sleep 0.6;
PAIR=("gedit --new-window /tmp/cover_$RANDOM.txt|gedit" "nautilus /home/user|Nautilus" "gnome-system-monitor|System Monitor");
p=${{PAIR[$((RANDOM % ${{#PAIR[@]}}))]}}; cmd="${{p%%|*}}"; ttl="${{p##*|}}";
eval "$cmd >/dev/null 2>&1 & disown";
sleep 0.6;
wmctrl -r "$ttl" -b add,maximized_vert,maximized_horz 2>/dev/null;
wmctrl -r "$ttl" -b add,above 2>/dev/null
""")


def comp_multi_modal_chain() -> str:
    """Three sequential zenity dialogs fire with 2-3s gaps — but the agent can
    dismiss them in any order. Cost 3 (only for very_easy tier)."""
    return _c(r"""
T=("Sync status" "Cache updated" "Background tasks" "Indexing" "Session log");
M=("Some items have been refreshed." "Preferences synced to cloud." "Indexer finished pass 1." "Session log rotated." "Recent items list rebuilt.");
for i in 1 2 3; do
  t=${T[$((RANDOM % ${#T[@]}))]}; m=${M[$((RANDOM % ${#M[@]}))]};
  zenity --info --title "$t" --text "$m" --width=320 >/dev/null 2>&1 & disown;
  sleep $((RANDOM % 2 + 2));
done
""")


def comp_overlay_plus_focus_steal() -> str:
    """A persistent overlay appears, then a separate window steals focus. Two
    distinct recoveries needed (close overlay + refocus). Cost 2."""
    return _c(r"""
( zenity --info --title 'Notice' --text 'New features available — click to learn more.' --width=420 >/dev/null 2>&1 ) & disown;
sleep 0.4;
wmctrl -r 'Notice' -e "0,30,840,420,80" 2>/dev/null;
wmctrl -r 'Notice' -b add,above 2>/dev/null;
sleep 1.0;
gnome-terminal --geometry=90x20 -- bash -c "ls -la /home/user/Desktop; read -p 'Enter to close' ok" >/dev/null 2>&1 & disown
""")


def comp_decoy_nothing() -> str:
    """A composition that fires 2-3 very brief actions — file touches, a
    dark-mode flicker, a passive notification — but REQUIRES NO RECOVERY.
    The correct agent response is continue working. Critical for training
    the agent not to over-react. Cost 0."""
    return _c(r"""
touch /tmp/decoy_$RANDOM.txt 2>/dev/null;
CUR=$(gsettings get org.gnome.desktop.interface enable-animations 2>/dev/null);
NEW=true; [[ "$CUR" == "true" ]] && NEW=false;
gsettings set org.gnome.desktop.interface enable-animations $NEW 2>/dev/null;
(sleep 2; gsettings set org.gnome.desktop.interface enable-animations $CUR 2>/dev/null) & disown;
zenity --notification --window-icon=info --text 'Settings — Preferences were auto-saved' 2>/dev/null &
""")


def comp_notif_burst_plus_flicker() -> str:
    """Two notifications + a brief high-contrast flicker — sensory saturation
    but zero recovery needed. Agent should continue working. Cost 0."""
    return _c(r"""
CUR=$(gsettings get org.gnome.desktop.a11y.interface high-contrast 2>/dev/null);
gsettings set org.gnome.desktop.a11y.interface high-contrast true 2>/dev/null;
(sleep $((RANDOM % 2 + 2)); gsettings set org.gnome.desktop.a11y.interface high-contrast $CUR 2>/dev/null) & disown;
for i in 1 2; do
  zenity --notification --window-icon=info --text "Background — task $i completed" 2>/dev/null & disown;
  sleep 0.4;
done
""")
