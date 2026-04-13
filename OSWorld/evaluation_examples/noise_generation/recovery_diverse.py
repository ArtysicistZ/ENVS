"""
recovery_diverse.py — Elements with NON-STANDARD recovery paths.

Most of our noise dismisses via a single click on an OK button. That lets
the agent memorize "always click OK". This file contains elements that
require DIFFERENT recovery actions from the agent — the correct response
varies per element, forcing the agent to read context rather than
pattern-match.

Recovery paths covered:
  - Wait (auto-dismiss; correct action is DO NOTHING)
  - Press Escape (from a noise-owned focused window)
  - Double-click
  - Scroll away
  - Drag the noise window
  - Resize the noise window
  - Click outside (click-away dismissal)
  - Multi-step (two clicks in sequence)
  - Action is a specific non-OK button
"""

from __future__ import annotations


def _c(src: str) -> str:
    return " ".join(src.split())


def recovery_wait_autodismiss() -> str:
    """Modal with timeout — auto-dismisses after 4-6 seconds. Correct agent
    action: do nothing, keep working. Cost 0."""
    return _c(r"""
T=("Routine check" "Status update" "Background info" "Informational");
t=${T[$((RANDOM % ${#T[@]}))]};
M=("Maintenance finished." "Cache reloaded." "Settings synced." "Connection restored.");
m=${M[$((RANDOM % ${#M[@]}))]};
S=$((RANDOM % 3 + 4));
zenity --info --title "$t" --text "$m\n(This message will close automatically)" --timeout=$S >/dev/null 2>&1 &
""")


def recovery_escape_only() -> str:
    """Focus-steal to a noise-owned xmessage that only dismisses with Escape
    key. Agent must recognize the window is focused and press Escape
    (safely — the keystroke lands on the noise window, not the target).
    Cost 1."""
    return _c(r"""
M=("Press Escape to close" "Hit Esc to dismiss" "Tap Esc to continue");
m=${M[$((RANDOM % ${#M[@]}))]};
xmessage -center -buttons '' "$m" >/dev/null 2>&1 & disown
""")


def recovery_drag_window_required() -> str:
    """Opens a noise-owned window that sits at an awkward fixed position
    obscuring part of the taskbar. The agent either drags it or Alt-Tabs
    past it — no button click helps. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk()
r.title('Layout hint')
W,H=400,80
X=random.randint(200,1200); Y=1020-H
r.geometry(f'{W}x{H}+{X}+{Y}')
tk.Label(r, text='Drag me out of the way to restore layout.', font=('Sans',11)).pack(pady=16)
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def recovery_resize_required() -> str:
    """Noise-owned window opens at a large size covering content. Correct
    recovery is to resize (drag corner), not close — window has no close
    button. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk()
r.title('Resize panel')
W,H=random.randint(500,700), random.randint(300,450)
X,Y=random.randint(200,900),random.randint(150,500)
r.geometry(f'{W}x{H}+{X}+{Y}')
tk.Label(r, text='Grab a corner to resize. This panel has no close button.', wraplength=W-30, font=('Sans',11)).pack(pady=30)
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def recovery_specific_button() -> str:
    """Modal with Accept/Decline/Maybe buttons — wrong buttons produce
    nothing; only 'Maybe' dismisses. Trains the agent to read button labels
    rather than pattern-match positions. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.title('Preference check')
W,H=380,180; r.geometry(f'{W}x{H}+{random.randint(400,900)}+{random.randint(300,500)}')
tk.Label(r, text='Confirm your preference:', pady=14, font=('Sans',12)).pack()
bf=tk.Frame(r); bf.pack(pady=14)
# Shuffle buttons so position doesn't encode identity
import random as rd
labels=['Accept','Decline','Maybe']
rd.shuffle(labels)
for lbl in labels:
    def mk(l=lbl):
        def fn(): r.destroy() if l=='Maybe' else None
        return fn
    tk.Button(bf, text=lbl, command=mk(), width=10).pack(side='left', padx=6)
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def recovery_click_outside() -> str:
    """A tkinter popup that dismisses when the user clicks anywhere OUTSIDE
    it (focus-out). The agent's natural recovery — click on target app —
    also dismisses the popup as a side effect. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.title('Hover menu')
W,H=300,120
X,Y=random.randint(500,1200),random.randint(200,500)
r.geometry(f'{W}x{H}+{X}+{Y}')
r.bind('<FocusOut>', lambda e: r.destroy())
tk.Label(r, text='Click anywhere outside this window to dismiss.', wraplength=W-20, font=('Sans',11)).pack(pady=30)
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def recovery_scroll_to_dismiss() -> str:
    """A tkinter banner that dismisses on scroll-wheel events from anywhere
    on its window. Agent must scroll over it. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.title('Scroll away')
W,H=500,100
X,Y=random.randint(200,1100),random.randint(200,700)
r.geometry(f'{W}x{H}+{X}+{Y}')
r.attributes('-topmost', True)
r.bind('<MouseWheel>', lambda e: r.destroy())
r.bind('<Button-4>', lambda e: r.destroy())
r.bind('<Button-5>', lambda e: r.destroy())
tk.Label(r, text='Scroll over this bar to dismiss it.', font=('Sans',12)).pack(pady=30)
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def recovery_two_step() -> str:
    """Modal requires two sequential clicks: first 'Review', then 'Confirm'.
    Cost 2 (two agent actions)."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.title('Action required')
r.geometry(f'360x180+{random.randint(500,1000)}+{random.randint(300,500)}')
state={'seen': False}
lbl=tk.Label(r, text='Please review before confirming.', pady=30, font=('Sans',12)); lbl.pack()
def on_review():
    state['seen']=True
    lbl.config(text='Review complete. Confirm to proceed.')
    btn.config(text='Confirm', command=r.destroy)
btn=tk.Button(r, text='Review', width=14, command=on_review); btn.pack(pady=10)
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def recovery_double_click() -> str:
    """Noise-owned tkinter window dismisses ONLY on double-click. Single
    clicks do nothing — trains the agent to vary its click rhythm. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.title('Double click to close')
r.geometry(f'320x120+{random.randint(600,1100)}+{random.randint(200,600)}')
lbl=tk.Label(r, text='Double-click anywhere in this window to close.', wraplength=280, pady=20, font=('Sans',11))
lbl.pack()
r.bind('<Double-Button-1>', lambda e: r.destroy())
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def recovery_right_click_menu() -> str:
    """Tkinter window with a right-click context menu that has a 'Close' item.
    Must right-click, then click Close. Cost 2."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.title('Context action')
r.geometry(f'340x140+{random.randint(400,1100)}+{random.randint(250,550)}')
tk.Label(r, text='Right-click to open the context menu.', pady=30, font=('Sans',11)).pack()
menu=tk.Menu(r, tearoff=0)
menu.add_command(label='Close panel', command=r.destroy)
menu.add_command(label='Keep open', command=lambda: None)
r.bind('<Button-3>', lambda e: menu.post(e.x_root, e.y_root))
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def recovery_type_to_close() -> str:
    """Noise-owned window with a text field — dismisses when the user types
    anything + Enter. Tests type-then-Enter recovery. Cost 2."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.title('Quick confirmation')
r.geometry(f'360x160+{random.randint(500,1100)}+{random.randint(250,550)}')
tk.Label(r, text='Type any text and press Enter to dismiss:', pady=14).pack()
e=tk.Entry(r, width=32); e.pack(pady=8); e.focus()
e.bind('<Return>', lambda ev: r.destroy())
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def recovery_click_specific_position() -> str:
    """Tkinter popup where only the bottom-right corner has a small close
    button at a specific, non-standard position. Cost 1."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.title('Small control')
W,H=380,220
r.geometry(f'{W}x{H}+{random.randint(500,1100)}+{random.randint(250,500)}')
tk.Label(r, text='Look for the small × in the corner', font=('Sans',11), pady=40).pack()
# Tiny close button at bottom-right
f=tk.Frame(r); f.pack(fill='x', padx=8, pady=4)
tk.Label(f, text='  ', width=40).pack(side='left')
tk.Button(f, text='×', width=2, height=1, command=r.destroy, relief='flat').pack(side='right')
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def recovery_drag_to_corner() -> str:
    """Noise-owned window starts in the screen center and must be dragged to
    an edge to reveal the close button (invisible in center). Cost 2
    (approximated via wmctrl manipulation; uses tkinter for variety)."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.title('Floating widget')
r.geometry(f'320x200+{random.randint(300,900)}+{random.randint(250,500)}')
tk.Label(r, text='Drag to the edge of the screen, then double-click to close.', wraplength=280, pady=40).pack()
r.bind('<Double-Button-1>', lambda e: r.destroy())
r.mainloop()
" >/dev/null 2>&1 & disown
""")


def recovery_minimize_instead() -> str:
    """Noise-owned window where the 'correct' recovery is to minimize it
    (press the minimize button or Alt-Space → N) — it won't close, only
    hide. Cost 1."""
    return _c(r"""
gedit --new-window /tmp/minreq_$RANDOM.txt >/dev/null 2>&1 & disown;
sleep 0.9;
wmctrl -a 'gedit' 2>/dev/null;
sleep 0.2;
# Make it nominally closable but lingering
wmctrl -r 'gedit' -e "0,$((RANDOM % 600 + 200)),$((RANDOM % 400 + 100)),500,300" 2>/dev/null
""")


def recovery_ignore_decoy() -> str:
    """A visible but non-modal decoy element (small always-on-top label in
    a corner) that the agent should learn to ignore. Auto-dismisses after
    5 seconds. Cost 0 (correct action: do nothing)."""
    return _c(r"""
DISPLAY=:0 python3 -c "
import tkinter as tk, random
r=tk.Tk(); r.overrideredirect(True); r.attributes('-topmost', True)
W,H=180,40
corner=random.choice(['tl','tr','bl','br'])
if corner=='tl': X,Y=30,30
elif corner=='tr': X,Y=1720,30
elif corner=='bl': X,Y=30,1010
else: X,Y=1720,1010
r.geometry(f'{W}x{H}+{X}+{Y}')
r.configure(bg='#fefce8')
tk.Label(r, text=random.choice(['Autosaved','Background','Synced','Ready','Active']), bg='#fefce8', font=('Sans',10)).pack(pady=10)
r.after(5000, r.destroy); r.mainloop()
" >/dev/null 2>&1 & disown
""")
