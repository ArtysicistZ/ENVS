"""
human_extra.py — Extended HUMAN TASK SESSIONS.

Adds 20 new concrete human-interaction sessions that are distinct from the
core set in templates.py. Every template uses different content arrays,
different randomization, and different app-interaction patterns from
anything in the main library. The goal: the agent should never be able to
pattern-match a specific human-session rendering.
"""

from __future__ import annotations


def _c(src: str) -> str:
    return " ".join(src.split())


# ---------- Writing sessions with genuinely different content ----------

def human_journal_entry() -> str:
    """Human types a diary-style journal entry — long-form reflective prose.
    Cost 1."""
    return _c(r"""
gedit --new-window /tmp/journal_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random, datetime
openers = ['Today was', 'A quiet day', 'Busy morning', 'Looking back on', 'Wrapped up']
events = ['the commute', 'a long meeting', 'that coffee chat', 'the deployment', 'one intense review']
closers = ['Feeling settled.', 'Tomorrow looks lighter.', 'Need to sleep earlier.', 'Grateful for the team.', 'More tomorrow.']
pyautogui.typewrite(f'{datetime.date.today()}', interval=0.02); pyautogui.press('enter'); pyautogui.press('enter')
pyautogui.typewrite(f'{random.choice(openers)} {random.choice(events)}.', interval=0.02); pyautogui.press('enter')
pyautogui.typewrite(f'It went better than expected — small progress adds up.', interval=0.02); pyautogui.press('enter')
pyautogui.typewrite(random.choice(closers), interval=0.02)
" 2>/dev/null || true
""")


def human_sql_query_draft() -> str:
    """Human drafts a SQL query in gedit with syntax-like content. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/query_$RANDOM.sql >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random
QS = [
  ['-- latest active users','SELECT user_id, last_seen','FROM sessions','WHERE last_seen > NOW() - INTERVAL 7 DAY','ORDER BY last_seen DESC','LIMIT 100;'],
  ['-- order totals by month','SELECT DATE_TRUNC(month, order_ts) AS m, SUM(total) AS revenue','FROM orders','WHERE status = chr(39) + \"complete\" + chr(39)','GROUP BY m','ORDER BY m;'],
  ['-- deduplicate product table','WITH ranked AS (','  SELECT *, ROW_NUMBER() OVER (PARTITION BY sku ORDER BY updated_at DESC) rn','  FROM products',')','SELECT * FROM ranked WHERE rn = 1;'],
]
lines = random.choice(QS)
for l in lines:
    pyautogui.typewrite(l, interval=0.02); pyautogui.press('enter')
" 2>/dev/null || true
""")


def human_readme_draft() -> str:
    """Human writes a README.md draft — markdown-structured. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/readme_$RANDOM.md >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random
PROJECTS = ['tinytool', 'fast-sync', 'graph-kit', 'md-render', 'http-pool']
p = random.choice(PROJECTS)
lines = [
  f'# {p}',
  '',
  'A small utility for doing one thing well.',
  '',
  '## Installation',
  '',
  '    pip install ' + p,
  '',
  '## Usage',
  '',
  'Run with: ' + p + ' --help',
  '',
  '## License',
  '',
  'MIT',
]
for l in lines:
    pyautogui.typewrite(l, interval=0.02); pyautogui.press('enter')
" 2>/dev/null || true
""")


def human_essay_paragraph() -> str:
    """Human types a longer opinion-essay-style paragraph. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/essay_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random
OPEN = ['The argument for', 'Against common belief,', 'It is worth noting that', 'Recent events suggest']
TOPIC = ['remote work', 'focused time', 'small teams', 'written communication']
BODY = ['- small iterations compound', '- deep work produces compounding returns', '- a 3-person team can outpace a 10-person one', '- docs age faster than code but last longer']
END = ['In sum, the evidence is modest but consistent.', 'More work is needed, but the direction seems clear.']
pyautogui.typewrite(f'{random.choice(OPEN)} {random.choice(TOPIC)}:', interval=0.02); pyautogui.press('enter'); pyautogui.press('enter')
for _ in range(3):
    pyautogui.typewrite(random.choice(BODY), interval=0.02); pyautogui.press('enter')
pyautogui.press('enter'); pyautogui.typewrite(random.choice(END), interval=0.02)
" 2>/dev/null || true
""")


def human_config_file_edit() -> str:
    """Human pastes a YAML/INI config snippet into gedit. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/cfg_$RANDOM.yaml >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random
CFGS = [
  ['app:','  host: 0.0.0.0','  port: 8080','  workers: 4','logging:','  level: info','  format: json'],
  ['database:','  driver: postgres','  pool_size: 16','  timeout_s: 30','features:','  experimental: true','  beta_flags: []'],
  ['service:','  name: worker','  retries: 5','  backoff_ms: 250','metrics:','  enabled: true','  endpoint: /metrics'],
]
for l in random.choice(CFGS):
    pyautogui.typewrite(l, interval=0.02); pyautogui.press('enter')
" 2>/dev/null || true
""")


# ---------- Different terminal workflows ----------

def human_terminal_file_navigation() -> str:
    """Human moves through folders in a terminal — cd, ls, pwd. Cost 1."""
    return _c(r"""
gnome-terminal --geometry=100x22 -- bash -c "
cd /tmp; pwd; ls; sleep 0.3;
cd /home/user; pwd; ls; sleep 0.3;
cd /etc; pwd; ls | head -20; sleep 0.3;
echo; read -p '[Enter]' ok
" >/dev/null 2>&1 & disown
""")


def human_terminal_log_grep() -> str:
    """Human filters a mock log file with grep. Cost 1."""
    return _c(r"""
gnome-terminal --geometry=100x22 -- bash -c "
LEVELS=(INFO WARN ERROR DEBUG);
printf '[%s] request id=%d status=%d\n' {\$((RANDOM%4)),\$((RANDOM%9000)),200} > /tmp/mocklog_$$.log 2>/dev/null;
for _ in 1 2 3 4 5 6 7 8; do
  L=\${LEVELS[\$((RANDOM%4))]};
  printf '[%s] request id=%d status=%d\n' \$L \$((RANDOM%9000)) \$((200 + RANDOM%5 * 100)) >> /tmp/mocklog_$$.log;
done;
echo 'user@host:~\$ grep ERROR /tmp/mocklog_$$.log';
grep ERROR /tmp/mocklog_$$.log 2>/dev/null || echo '(no matches)';
rm /tmp/mocklog_$$.log 2>/dev/null;
read -p '[Enter]' ok
" >/dev/null 2>&1 & disown
""")


def human_terminal_network_check() -> str:
    """Human runs a small network-check sequence. Cost 1."""
    return _c(r"""
gnome-terminal --geometry=100x22 -- bash -c "
echo 'user@host:~\$ ip -4 addr show | grep inet';
ip -4 addr show 2>/dev/null | grep inet | head -5 || echo '(no addrs)';
echo; echo 'user@host:~\$ cat /etc/resolv.conf';
cat /etc/resolv.conf 2>/dev/null | head -5;
echo; echo 'user@host:~\$ ping -c 2 127.0.0.1';
ping -c 2 127.0.0.1 2>/dev/null;
read -p '[Enter]' ok
" >/dev/null 2>&1 & disown
""")


def human_terminal_python_repl() -> str:
    """Human drops into a Python REPL, computes some values, exits. Cost 1."""
    return _c(r"""
gnome-terminal --geometry=100x22 -- bash -c "
python3 -c \"
print('>>> sum(range(1000))')
print(sum(range(1000)))
print('>>> [x*x for x in range(10)]')
print([x*x for x in range(10)])
print('>>> import math; math.pi')
import math; print(math.pi)
\" 2>/dev/null;
read -p '[Enter]' ok
" >/dev/null 2>&1 & disown
""")


# ---------- Different file operations ----------

def human_file_rename_desktop() -> str:
    """Human renames a pre-existing scratch file on the Desktop via nautilus
    F2-press. Creates the source file first. Cost 1."""
    return _c(r"""
mkdir -p /home/user/Desktop 2>/dev/null;
SRC=/home/user/Desktop/draft_$RANDOM.txt;
touch "$SRC" 2>/dev/null;
nautilus /home/user/Desktop >/dev/null 2>&1 & disown;
sleep 1.0;
wmctrl -a 'Desktop' 2>/dev/null || wmctrl -a 'Files' 2>/dev/null;
sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random, time
pyautogui.press('f2'); time.sleep(0.5)
N = ['archive', 'review', 'final', 'revised', 'v2', 'draft']
pyautogui.typewrite(random.choice(N) + '_' + str(random.randint(100, 999)) + '.txt', interval=0.02)
pyautogui.press('enter')
" 2>/dev/null || true
""")


def human_file_search_nautilus() -> str:
    """Human opens a file manager and invokes search. Cost 1."""
    return _c(r"""
nautilus /home/user >/dev/null 2>&1 & disown;
sleep 1.0;
wmctrl -a 'Home' 2>/dev/null || wmctrl -a 'Files' 2>/dev/null || wmctrl -a 'Nautilus' 2>/dev/null;
sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random, time
pyautogui.hotkey('ctrl', 'f'); time.sleep(0.5)
QS = ['report', 'draft', 'notes', 'invoice', '2024', 'archive', 'final']
pyautogui.typewrite(random.choice(QS), interval=0.03)
" 2>/dev/null || true
""")


def human_file_zoom_view() -> str:
    """Human zooms the nautilus view with Ctrl+=/Ctrl+-. Cost 1."""
    return _c(r"""
nautilus /home/user/Desktop >/dev/null 2>&1 & disown;
sleep 1.0;
wmctrl -a 'Desktop' 2>/dev/null || wmctrl -a 'Files' 2>/dev/null;
sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random, time
for _ in range(random.randint(2, 3)):
    pyautogui.hotkey('ctrl', 'equal' if random.random() < 0.5 else 'minus'); time.sleep(0.3)
" 2>/dev/null || true
""")


def human_file_view_toggle() -> str:
    """Human toggles between list/grid view in nautilus with Ctrl+1/Ctrl+2. Cost 1."""
    return _c(r"""
nautilus /home/user >/dev/null 2>&1 & disown;
sleep 1.0;
wmctrl -a 'Home' 2>/dev/null || wmctrl -a 'Files' 2>/dev/null;
sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random, time
pyautogui.hotkey('ctrl', str(random.choice([1, 2]))); time.sleep(0.4)
pyautogui.hotkey('ctrl', str(random.choice([1, 2])))
" 2>/dev/null || true
""")


# ---------- Other creative human actions ----------

def human_scroll_long_doc() -> str:
    """Human opens a long pre-seeded gedit file and scrolls through it with
    Page Down. Cost 1."""
    return _c(r"""
F=/tmp/long_$RANDOM.txt;
for i in $(seq 1 80); do echo "Line $i — random content $(echo $RANDOM | md5sum 2>/dev/null | cut -c1-8)"; done > "$F" 2>/dev/null;
gedit --new-window "$F" >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random, time
for _ in range(random.randint(3, 6)):
    pyautogui.press('pagedown'); time.sleep(0.25)
" 2>/dev/null || true
""")


def human_select_all_copy() -> str:
    """Human opens gedit, types some text, then Ctrl+A + Ctrl+C. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/copy_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random, time
pyautogui.typewrite('Quick reference snippet:', interval=0.02); pyautogui.press('enter')
pyautogui.typewrite(random.choice(['alpha beta gamma delta','one two three four','eta theta iota kappa']), interval=0.02)
time.sleep(0.4)
pyautogui.hotkey('ctrl', 'a'); time.sleep(0.2); pyautogui.hotkey('ctrl', 'c')
" 2>/dev/null || true
""")


def human_find_in_editor() -> str:
    """Human opens gedit, types content, then Ctrl+F to search. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/find_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, random, time
BODY = ['the quick brown fox','jumps over the lazy dog','pack my box with five dozen liquor jugs','sphinx of black quartz judge my vow']
for b in BODY:
    pyautogui.typewrite(b, interval=0.02); pyautogui.press('enter')
time.sleep(0.3)
pyautogui.hotkey('ctrl', 'f'); time.sleep(0.4)
pyautogui.typewrite(random.choice(['quick','lazy','pack','sphinx']), interval=0.03)
" 2>/dev/null || true
""")


def human_split_editor_windows() -> str:
    """Human opens two gedit windows side-by-side — simulates comparing two
    documents. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/left_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 0.7;
wmctrl -r 'gedit' -e "0,20,120,900,800" 2>/dev/null;
gedit --new-window /tmp/right_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 0.8;
IDS=($(wmctrl -l | grep gedit | awk '{print $1}'));
[[ ${#IDS[@]} -ge 2 ]] && wmctrl -i -r "${IDS[-1]}" -e "0,960,120,900,800" 2>/dev/null
""")


def human_preferences_dialog() -> str:
    """Human opens gedit and presses Ctrl+, (preferences shortcut on many apps)
    — a preferences dialog appears. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/pref_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, time
pyautogui.hotkey('ctrl', 'comma'); time.sleep(0.5)
" 2>/dev/null || true
""")


def human_help_browser() -> str:
    """Human opens the help dialog of a noise-owned app (F1). Cost 1."""
    return _c(r"""
gedit --new-window /tmp/help_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
DISPLAY=:0 python3 -c "
import pyautogui, time
pyautogui.press('f1'); time.sleep(0.8)
" 2>/dev/null || true
""")


def human_image_viewer_browse() -> str:
    """Human opens the image viewer (eog) on a random system wallpaper.
    Requires eog; gracefully no-ops if missing. Cost 1."""
    return _c(r"""
command -v eog >/dev/null 2>&1 || exit 0;
P=(/usr/share/backgrounds/*.jpg /usr/share/backgrounds/*.png);
for p in "${P[@]}"; do [[ -f "$p" ]] && V+=("$p"); done;
[[ ${#V[@]} -eq 0 ]] && exit 0;
eog "${V[$((RANDOM % ${#V[@]}))]}" >/dev/null 2>&1 & disown
""")


def human_archive_browse() -> str:
    """Human opens the archive manager (file-roller) on a tarball if one
    exists. Graceful no-op if absent. Cost 1."""
    return _c(r"""
command -v file-roller >/dev/null 2>&1 || exit 0;
TAG=$RANDOM;
F=/tmp/arc_$TAG.tar.gz;
tar -czf "$F" /etc/hostname /etc/os-release 2>/dev/null || exit 0;
file-roller "$F" >/dev/null 2>&1 & disown
""")
