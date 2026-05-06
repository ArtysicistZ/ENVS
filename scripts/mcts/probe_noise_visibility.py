#!/usr/bin/env python3
"""Diagnostic: verify noise actually produces visible screen changes.
Resets env with forced noise at step 1, takes a dummy action, saves the
next screenshot. Then we inspect whether the noise artifact is visible."""
import os, sys, json, base64
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "OSWorld"))

import ray
from verl.mcts.env_client import MCTSEnvClient

# Multiple foreign-noise elements to test
NOISE_CANDIDATES = {
    "fullscreen_tkinter": {
        # Guaranteed-visible: tkinter fullscreen opaque window with big text.
        "category": "overlay_fullscreen",
        "recovery_cost": 1,
        "command": r"""
DISPLAY=:0 python3 -c "
import tkinter as tk
r = tk.Tk()
r.attributes('-fullscreen', True)
r.attributes('-topmost', True)
r.configure(bg='#2e86ab')
lbl = tk.Label(r,
  text='SYSTEM MAINTENANCE\n\nAnother user is running diagnostics.\nPlease wait.\n\n(Close this window to continue)',
  font=('Arial', 48, 'bold'), fg='white', bg='#2e86ab')
lbl.pack(expand=True)
btn = tk.Button(r, text='Close', font=('Arial', 24),
  command=r.destroy, padx=30, pady=15)
btn.pack(pady=50)
r.mainloop()
" >/dev/null 2>&1 & disown
sleep 1.2
""",
    },
    "gedit_full": {
        "category": "human_writing_session",
        "recovery_cost": 1,
        "command": r"""
gedit --new-window /tmp/notes_test.txt >/dev/null 2>&1 & disown;
sleep 1.0; wmctrl -a 'gedit' 2>/dev/null; sleep 0.3;
""",
    },
    "zenity_modal": {
        "category": "modal",
        "recovery_cost": 1,
        "command": r"""
DISPLAY=:0 zenity --info --title='System Alert' --text='An unexpected event occurred.' &
sleep 0.5;
""",
    },
}


def build_meta(key: str, fire_step: int = 1):
    el = NOISE_CANDIDATES[key]
    return {
        "trigger_mode": "deterministic_schedule_v4_forced",
        "success_rate_used": 0.5,
        "observed_sr_used": 0.5,
        "fires_count": 1,
        "fire_steps": [fire_step],
        "total_recovery_cost": el["recovery_cost"],
        "elements": [{
            "id": f"probe_{key}",
            "category": el["category"],
            "recovery_cost": el["recovery_cost"],
            "once": True,
            "command": el["command"],
            "fire_step": fire_step,
        }],
    }


def load_task():
    tid = "2ad9387a-65d8-4e33-ad5b-7580065a27ca"
    p = os.path.join(PROJ_ROOT, "OSWorld", "evaluation_examples", "examples", "chrome", f"{tid}.json")
    tc = json.load(open(p))
    tc["id"] = tid
    tc["domain"] = "chrome"
    return tc


def main():
    ray.init(ignore_reinit_error=True)
    out_dir = "/tmp/noise_probe"
    os.makedirs(out_dir, exist_ok=True)

    for key in list(NOISE_CANDIDATES.keys()):
        print(f"\n\n===== TESTING: {key} =====")
        tc = load_task()
        tc["enable_noise"] = True
        tc["noise_mode"] = "runtime_library"
        tc["noise_meta"] = build_meta(key, fire_step=1)

        env = ray.remote(num_cpus=0, num_gpus=0)(MCTSEnvClient).remote(
            worker_idx=0,
            remote_server_url="http://10.100.4.6:15001",
            slot_id=0,
        )

        print("Resetting...")
        try:
            ray.get(env.reset.remote(tc), timeout=300)
        except Exception as e:
            print(f"RESET FAILED: {e}")
            continue

        # Save initial screenshot
        init_ss = ray.get(env.get_obs_screenshot.remote())
        if init_ss:
            with open(f"{out_dir}/{key}_00_initial.png", "wb") as f:
                f.write(base64.b64decode(init_ss))
            print(f"Saved initial screenshot ({len(init_ss)} chars b64)")

        # Step 0: dummy action (just wait)
        print("Step 0: wait()")
        result = ray.get(env.step.remote("Thought: wait\nAction: wait()"))
        nb = result.get("noise_burden", {}) or {}
        print(f"  noise_burden: fired={nb.get('events_fired_this_step', 0)}, cost={nb.get('step_recovery_cost', 0)}")
        ss1 = ray.get(env.get_obs_screenshot.remote())
        if ss1:
            with open(f"{out_dir}/{key}_01_afterstep0.png", "wb") as f:
                f.write(base64.b64decode(ss1))

        # Step 1: noise fires during this step
        print("Step 1: wait() (noise should fire)")
        result = ray.get(env.step.remote("Thought: wait\nAction: wait()"))
        nb = result.get("noise_burden", {}) or {}
        print(f"  noise_burden: fired={nb.get('events_fired_this_step', 0)}, cost={nb.get('step_recovery_cost', 0)}")
        ss2 = ray.get(env.get_obs_screenshot.remote())
        if ss2:
            with open(f"{out_dir}/{key}_02_afterstep1_NOISE.png", "wb") as f:
                f.write(base64.b64decode(ss2))

        # One more step to confirm persistence
        print("Step 2: wait() (noise should still be visible)")
        result = ray.get(env.step.remote("Thought: wait\nAction: wait()"))
        ss3 = ray.get(env.get_obs_screenshot.remote())
        if ss3:
            with open(f"{out_dir}/{key}_03_afterstep2.png", "wb") as f:
                f.write(base64.b64decode(ss3))

        print(f"Screenshots saved in {out_dir}/{key}_*.png")

    print("\n\nDone. Inspect the *_02_afterstep1_NOISE.png files to see if noise was visible.")


if __name__ == "__main__":
    main()
