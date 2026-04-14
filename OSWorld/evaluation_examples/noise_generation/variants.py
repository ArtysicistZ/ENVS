"""
variants.py — VISUAL-RENDERING DIVERSITY for the same behaviors.

The risk of zenity-only noise is that the agent learns a pixel template:
"zenity OK button at position X = click to dismiss". Every behavior
class here is rendered in at least 3 distinct UI frameworks so the agent
must learn the *behavior* (modal, notification, overlay, focus-steal),
not the specific rendering.

Rendering frameworks used:
  - zenity                 (already-installed GTK-style)
  - xmessage               (plain X11 widget; radically different look)
  - python3 + tkinter      (custom-styled Tk windows — colors, fonts, layout)
  - browser-rendered HTML  (launches a non-target Chrome window showing a
                            data-URL modal/banner; pixel-identical to real
                            web popups)

All templates respect the ≤3-step recovery rule and produce bash commands
with internal $RANDOM variation.
"""

from __future__ import annotations


def _c(src: str) -> str:
    return src.strip()


# ============================================================================
# MODAL DIALOGS rendered in different UI frameworks
# ============================================================================

def modal_xmessage_centered() -> str:
    """Plain X11 xmessage dialog — very different look from GTK zenity. Cost 1."""
    return _c(r"""
MSGS=("System check completed — review the log." "Update is ready. Restart when convenient." "Your session has been active for 4 hours." "Reminder: submit timesheet today." "Configuration has been saved successfully." "Permission updated for selected items.");
BTNS=("OK" "Close" "Dismiss" "Got it" "Acknowledge");
m=${MSGS[$((RANDOM % ${#MSGS[@]}))]}; b=${BTNS[$((RANDOM % ${#BTNS[@]}))]};
xmessage -center -button "$b" "$m" >/dev/null 2>&1 &
""")


def modal_xmessage_compact() -> str:
    """Small xmessage with short text and unusual position. Cost 1."""
    return _c(r"""
M=("Done" "Saved" "Updated" "OK!" "Ready" "Complete");
X=$((RANDOM % 800 + 200)); Y=$((RANDOM % 500 + 100));
m=${M[$((RANDOM % ${#M[@]}))]};
xmessage -geometry 180x80+$X+$Y -button OK "$m" >/dev/null 2>&1 &
""")


def modal_tkinter_info() -> str:
    """Custom tkinter info dialog. Randomized background color, title bar,
    button text, and position — visually nothing like zenity. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk
from tkinter import messagebox
import random
TITLES=['Status','Notice','Information','Update','System']
MSGS=['Operation completed.','Background task finished.','Preferences synced.','Cache refreshed successfully.','Certificate validated.','Session token renewed.']
bgs=['#f0f0f0','#eef2f7','#fff8dc','#e6f7ff','#f5f5dc']
r=tk.Tk(); r.withdraw()
r.configure(bg=random.choice(bgs))
messagebox.showinfo(random.choice(TITLES), random.choice(MSGS))
r.destroy()
" >/dev/null 2>&1 &
""")


def modal_tkinter_warning() -> str:
    """Tkinter warning-styled dialog — yellow hues, warning icon. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
from tkinter import messagebox
MSGS=['Low disk space detected.','Configuration differs from default.','Unsaved changes found.','File permissions modified.','Signature not verified.','Network latency elevated.']
r=tk.Tk(); r.withdraw()
messagebox.showwarning('Warning', random.choice(MSGS))
r.destroy()
" >/dev/null 2>&1 &
""")


def modal_tkinter_error() -> str:
    """Tkinter error-styled dialog — red hues, error icon. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
from tkinter import messagebox
MSGS=['Connection timed out.','Permission denied.','Invalid configuration value.','File not found in expected path.','Module failed to load.','Unrecognized format.']
r=tk.Tk(); r.withdraw()
messagebox.showerror('Error', random.choice(MSGS))
r.destroy()
" >/dev/null 2>&1 &
""")


def modal_tkinter_question() -> str:
    """Tkinter yes/no question dialog. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
from tkinter import messagebox
QS=['Continue with default settings?','Apply changes to all selected items?','Refresh the view?','Clear recent history?','Enable auto-backup?','Install optional components?']
r=tk.Tk(); r.withdraw()
messagebox.askquestion('Confirm', random.choice(QS))
r.destroy()
" >/dev/null 2>&1 &
""")


def modal_tkinter_custom_banner() -> str:
    """Custom tkinter window: a dialog with a visible text label, close button
    at a random corner, custom colors. Radically different from all other
    modals. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
root=tk.Tk(); root.title(random.choice(['Dialog','Notice','Update','Confirm','Alert']))
W,H=random.randint(280,460), random.randint(120,220)
X,Y=random.randint(200,1200), random.randint(120,700)
root.geometry(f'{W}x{H}+{X}+{Y}')
bg=random.choice(['#ffffff','#fff0f0','#f0fff0','#f0f0ff','#fffaf0'])
root.configure(bg=bg)
msgs=['Your action has been recorded.','The service is now available.','Updates have been applied.','Sync completed successfully.','You are enrolled in beta features.','Review required for pending items.']
lbl=tk.Label(root, text=random.choice(msgs), wraplength=W-40, bg=bg, font=('Sans', random.choice([10,11,12])))
lbl.pack(pady=20)
btn_text=random.choice(['OK','Close','Dismiss','Got it','Acknowledge'])
btn=tk.Button(root, text=btn_text, command=root.destroy)
btn.pack(pady=8, padx=12, side=random.choice(['left','right','bottom']))
root.mainloop()
" >/dev/null 2>&1 &
""")


def modal_browser_alert() -> str:
    """Launches a non-target Chrome window with a data-URL containing a real
    JavaScript alert. Pixel-identical to a web-page alert — which is exactly
    the kind of thing a real user sees. Cost 1."""
    return _c(r"""
TITLES=("Message from site" "Notice from website" "Alert" "Information");
MSGS=("Please verify your details before continuing." "Your input contains special characters." "Session will expire soon." "Page has been updated. Refresh to see changes." "New content is available.");
t=${TITLES[$((RANDOM % ${#TITLES[@]}))]}; m=${MSGS[$((RANDOM % ${#MSGS[@]}))]};
URL="data:text/html,<html><head><title>$t</title></head><body><script>alert('$m')</script></body></html>";
google-chrome --new-window --window-size=640,360 "$URL" >/dev/null 2>&1 & disown
""")


def modal_browser_dialog_element() -> str:
    """Chrome window with a <dialog> element showing a styled in-page modal.
    Cost 1."""
    return _c(r"""
HTML='<html><head><style>body{font-family:sans-serif;padding:40px}dialog{border:2px solid #888;padding:24px;border-radius:6px}button{padding:6px 20px}</style></head><body><dialog open><h3>Notice</h3><p>This session is using new security settings.</p><form method="dialog"><button>OK</button></form></dialog></body></html>';
google-chrome --new-window --window-size=500,320 "data:text/html,$HTML" >/dev/null 2>&1 & disown
""")


# ============================================================================
# NOTIFICATIONS rendered in different frameworks
# ============================================================================

def notif_xmessage_timeout() -> str:
    """xmessage with timeout — auto-dismisses after a few seconds. Cost 0."""
    return _c(r"""
M=("Build completed." "Test suite green." "Deploy successful." "Backup finished." "Sync OK." "Monitor: healthy.");
m=${M[$((RANDOM % ${#M[@]}))]};
T=$((RANDOM % 4 + 3));
xmessage -center -timeout $T "$m" >/dev/null 2>&1 & disown
""")


def notif_tkinter_corner_toast() -> str:
    """Small tkinter window in random corner, auto-destroys after 3 seconds.
    Cost 0."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.overrideredirect(True)
W,H=260,70
corner=random.choice(['tl','tr','bl','br'])
SW,SH=1920,1080
if corner=='tl': X,Y=20,40
elif corner=='tr': X,Y=SW-W-20,40
elif corner=='bl': X,Y=20,SH-H-40
else: X,Y=SW-W-20,SH-H-40
r.geometry(f'{W}x{H}+{X}+{Y}')
bg=random.choice(['#2d3748','#1a365d','#22543d','#744210','#702459'])
r.configure(bg=bg)
MSGS=['Deploy: staging ok','Git: sync complete','Build #4721 passed','Coverage: 82%%','Lints: 0 errors','DB: query cached','Alerts: 0 active']
m=random.choice(MSGS)
lbl=tk.Label(r, text=m, bg=bg, fg='white', font=('Sans',10)); lbl.pack(padx=10, pady=18)
r.after(15000, r.destroy); r.mainloop()
" >/dev/null 2>&1 & disown
""")


def notif_terminal_flash() -> str:
    """Briefly opens a terminal, prints an important-looking message, and
    stays on-screen long enough to be observed by the next step's screenshot.
    Auto-closes after ~18s. Cost 0."""
    return _c(r"""
M=("[systemd] Job completed successfully." "[auditd] Policy updated." "[cron] Nightly backup finished." "[net] DHCP lease renewed." "[security] Rules reloaded." "[daemon] Healthcheck OK.");
m=${M[$((RANDOM % ${#M[@]}))]};
if command -v gnome-terminal >/dev/null 2>&1; then
  gnome-terminal --geometry=60x3 -- bash -c "echo '$m'; sleep 18" >/dev/null 2>&1 & disown;
else
  # Fallback: an xmessage auto-dismissed after 18s.
  ( xmessage -center -timeout 18 -buttons '' "$m" >/dev/null 2>&1 ) & disown;
fi
""")


# ============================================================================
# PERSISTENT OVERLAYS in different frameworks and positions
# ============================================================================

def overlay_tkinter_top_banner() -> str:
    """Thin always-on-top tkinter banner stretched across the top of the
    screen. Closes via a small X button. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.overrideredirect(True)
r.attributes('-topmost', True)
r.geometry('1920x50+0+0')
bg=random.choice(['#2b6cb0','#c05621','#2f855a','#702459','#744210'])
r.configure(bg=bg)
MSGS=['New feature announcement — click to learn more','Limited-time offer — ends in 3 days','Beta program now open — join today','Security advisory — review settings','System maintenance planned this weekend']
lbl=tk.Label(r, text=random.choice(MSGS), bg=bg, fg='white', font=('Sans',12))
lbl.pack(side='left', padx=20, pady=12)
tk.Button(r, text='×', command=r.destroy, bg=bg, fg='white', relief='flat').pack(side='right', padx=10)
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def overlay_tkinter_bottom_banner() -> str:
    """Thin always-on-top tkinter banner at bottom of screen. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
r.geometry('1920x48+0+1032')
bg=random.choice(['#1a202c','#2d3748','#4a5568'])
r.configure(bg=bg)
MSGS=['We use cookies to improve your experience','Enable notifications? Allow or Deny','Consent to telemetry updates','Privacy policy updated — click to review']
lbl=tk.Label(r, text=random.choice(MSGS), bg=bg, fg='white', font=('Sans',11))
lbl.pack(side='left', padx=20, pady=12)
tk.Button(r, text='Accept', command=r.destroy).pack(side='right', padx=8)
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def overlay_tkinter_right_sidebar() -> str:
    """Tall tkinter sidebar pinned to the right edge. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
r.geometry('260x700+1660+150')
bg=random.choice(['#f7fafc','#edf2f7','#e2e8f0'])
r.configure(bg=bg)
headers=['Related articles','Recommended','Also viewed','Recent updates','Quick links']
tk.Label(r, text=random.choice(headers), bg=bg, font=('Sans',13,'bold')).pack(pady=16)
items=['Getting started guide','Advanced tutorial','Troubleshooting tips','API reference','Release notes','Community forum','Support center']
for _ in range(random.randint(4,6)):
    tk.Label(r, text='• ' + random.choice(items), bg=bg, font=('Sans',10), anchor='w').pack(fill='x', padx=14, pady=2)
tk.Button(r, text='Close', command=r.destroy).pack(pady=16)
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def overlay_tkinter_floating_widget() -> str:
    """Small floating widget in mid-screen — styled like a support chat or
    newsletter signup. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.attributes('-topmost', True)
W,H=320,240
X,Y=random.randint(500,1200),random.randint(300,600)
r.geometry(f'{W}x{H}+{X}+{Y}')
r.title(random.choice(['Chat','Newsletter','Subscribe','Quick survey']))
tk.Label(r, text=random.choice(['Got a question? Chat with us.','Join our newsletter for updates.','Rate your experience?','We would love your feedback.']), wraplength=280, font=('Sans',11)).pack(pady=20)
btnframe=tk.Frame(r); btnframe.pack(pady=10)
tk.Button(btnframe, text=random.choice(['Start chat','Subscribe','Yes']), width=10).pack(side='left', padx=6)
tk.Button(btnframe, text='Close', command=r.destroy, width=10).pack(side='left', padx=6)
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def overlay_browser_cookie_real() -> str:
    """Cookie-consent banner rendered as a tkinter overlay (fast, no browser
    startup). Styled to match typical dark-themed web cookie bars. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
r.geometry('1920x74+0+1006')
r.configure(bg='#1f2937')
MSGS=['This site uses cookies for analytics and functionality. By continuing, you accept our cookie policy.',
      'We use cookies to improve your browsing experience and serve personalized ads.',
      'Cookie notice: we and our partners use cookies to enhance functionality and analytics.']
lbl=tk.Label(r, text=random.choice(MSGS), bg='#1f2937', fg='white', font=('Sans',11), anchor='w')
lbl.pack(side='left', padx=20, pady=16, fill='x', expand=True)
tk.Button(r, text='Accept', command=r.destroy, bg='#10b981', fg='white',
          font=('Sans',10,'bold'), relief='flat', padx=16, pady=6).pack(side='right', padx=20)
r.mainloop()
" >/dev/null 2>&1 & disown
""")


# ============================================================================
# FOCUS-STEAL variants (different apps, different geometries)
# ============================================================================

def focus_steal_tkinter_app() -> str:
    """A custom tkinter mini-app steals focus — looks nothing like gedit. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk()
r.title(random.choice(['Notes','Editor','Quick Memo','Scratchpad']))
W,H=random.randint(420,640), random.randint(280,440)
r.geometry(f'{W}x{H}+{random.randint(200,1200)}+{random.randint(150,600)}')
r.lift(); r.attributes('-topmost', True); r.after(500, lambda: r.attributes('-topmost', False))
txt=tk.Text(r, font=('Monospace', 11))
txt.pack(fill='both', expand=True, padx=8, pady=8)
SEEDS=['Working notes:','Draft session:','Scratch for later:','Ideas bucket:','Review checklist:']
txt.insert('1.0', random.choice(SEEDS) + chr(10)*2 + '- point A' + chr(10) + '- point B' + chr(10) + '- point C')
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def focus_steal_two_windows() -> str:
    """Opens two noise-owned windows in quick succession, cycling focus between
    them. More disruptive than single focus-steal. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/a_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 0.5;
gnome-system-monitor >/dev/null 2>&1 & disown;
sleep 0.5;
wmctrl -a 'System Monitor' 2>/dev/null || wmctrl -a 'gedit' 2>/dev/null
""")


def focus_steal_scrolling_terminal() -> str:
    """Terminal opens with a live-updating htop-like display. Cost 1."""
    return _c(r"""
gnome-terminal --geometry=80x24 -- bash -c "
for i in \$(seq 1 25); do
  clear
  printf '  PID  USER     %%CPU  %%MEM  COMMAND\n'
  for j in 1 2 3 4 5 6 7; do
    printf '%5d  user     %4d  %4d  process_%d\n' \$((RANDOM % 9000 + 1000)) \$((RANDOM % 60)) \$((RANDOM % 40)) \$j
  done
  sleep 0.4
done
" >/dev/null 2>&1 & disown
""")


# ============================================================================
# INSTALLATION / DEPENDENCY DETECTION
# ============================================================================

def _has_tkinter() -> bool:
    """tkinter is part of standard Python. Assumed available."""
    return True


def _has_xmessage() -> bool:
    """xmessage ships with x11-utils. Assumed available on XFCE/GNOME."""
    return True
