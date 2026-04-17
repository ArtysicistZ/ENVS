# Noise Analysis: 8ba5ae7a-5ae5-4eab-9fcc-5dd4fe3abf89

## Task Context
- **Domain**: vlc
- **Instruction**: Help me modify the folder used to store my recordings to Desktop
- **MCTS Success Rate**: 0.1518 (17/112 rollouts)
- **Difficulty Tier**: medium
- **Noise Budget**: MODERATE noise budget: add 4-5 noise elements across 2-3 categories. Mix passive noise (notifications, background apps, filesystem clutter) with 1-2 active disruptions from: window_geometry, panel_toggle, or view_mode_change. Active disruptions should be moderate — e.g., a sidebar opens or zoom changes to 125%. The task MUST remain tractable.

## Original Config Summary
The clean config launches VLC in headless mode (no audio, no video title) and then clicks the center of the screen to ensure the VLC window is focused. The agent must then navigate VLC's preferences to change the recordings folder to ~/Desktop.

## Evaluator Analysis
- **Function(s)**: is_vlc_recordings_folder
- **Protected paths / URLs / windows**: `/home/user/Desktop` (the expected recording_file_path value in evaluator.expected); `vlcrc` (the VLC config file read by evaluator.result)
- **Failure modes to avoid**: Any noise that modifies `vlcrc` or writes to `/home/user/Desktop` in a way that could confuse the evaluator; any noise that prevents VLC from launching or becomes unresponsive; noise that opens VLC's preferences dialog prematurely (would confuse the agent); noise that changes the VLC window title in a way that breaks wmctrl partial matching.

## Proposed Noise Additions

### 1. background_apps: Launch gedit to clutter the desktop
- **Description**: Opens gedit text editor at startup so an extra application window exists on the desktop, adding visual clutter and potentially stealing focus later.
- **Config step (JSON)**:
  ```json
  {"type": "launch", "parameters": {"command": ["gedit"], "shell": false}}
  ```
- **Where to insert**: Appended after original step index 1 (after the pyautogui click step)
- **Rationale**: gedit is a realistic background application on Ubuntu desktops. Having it open adds a plausible distraction and enables the later focus-stealing noise step. It does not touch VLC config or evaluator paths.
- **Evaluator interference risk**: low — gedit does not interact with vlcrc or /home/user/Desktop in any evaluator-relevant way.

### 2. random_notifications: Burst notification (email + calendar)
- **Description**: Fires a correlated burst of two desktop notifications at random intervals — first an email alert, then a calendar reminder 5-15 seconds later — to distract the agent mid-task.
- **Config step (JSON)**:
  ```json
  {"type": "execute", "parameters": {"command": ["bash", "-c", "(sleep $((RANDOM % 60 + 15)) && notify-send 'Email' 'New message: Project update from Alex' && sleep $((RANDOM % 10 + 5)) && notify-send 'Calendar' 'Meeting starts in 10 minutes') &"]}}
  ```
- **Where to insert**: Appended after the gedit launch step (index 3 in noisy config)
- **Rationale**: Notification bursts are a realistic distraction on a typical Ubuntu desktop during a task. They do not interfere with VLC or the evaluator.
- **Evaluator interference risk**: low — notify-send only posts UI notifications, no file system changes to protected paths.

### 3. filesystem_clutter: Create decoy files in ~/Downloads
- **Description**: Progressively creates temporary files in ~/Downloads via a background daemon, making the filesystem appear busier and adding mild background activity.
- **Config step (JSON)**:
  ```json
  {"type": "execute", "parameters": {"command": ["bash", "-c", "(while true; do sleep $((RANDOM % 90 + 30)); touch /home/user/Downloads/download_$RANDOM.tmp; done) &"]}}
  ```
- **Where to insert**: Appended after the notifications step (index 4 in noisy config)
- **Rationale**: Progressive filesystem clutter is a realistic property of an active desktop. Files accumulate in ~/Downloads as background downloads complete. This does not affect ~/Desktop or vlcrc.
- **Evaluator interference risk**: low — files are created in ~/Downloads, not ~/Desktop or any evaluator-protected path.

### 4. window_geometry: Move VLC window partially off-screen
- **Description**: After a short random delay, moves the VLC window 180px off the left edge of the screen so some UI elements (left-side buttons) are clipped. VLC remains usable but the layout differs from the expected default.
- **Config step (JSON)**:
  ```json
  {"type": "execute", "parameters": {"command": ["bash", "-c", "(sleep $((RANDOM % 20 + 8)) && wmctrl -r 'VLC' -e 0,-180,60,1680,950 2>/dev/null) &"]}}
  ```
- **Where to insert**: Appended after the filesystem clutter step (index 5 in noisy config)
- **Rationale**: Moving the VLC window off-screen is the quintessential "messy desk" disruption for this domain. The agent must reposition or scroll the window to access menus. VLC is still functional; the Preferences dialog can still be opened. This directly targets VLC per the domain-specific hint.
- **Evaluator interference risk**: low — window geometry changes do not affect vlcrc or the recorded folder path.

### 5. panel_toggle: Toggle VLC Advanced Controls toolbar
- **Description**: After a random delay, focuses VLC and toggles the Advanced Controls toolbar on/off using the View menu shortcut. This changes VLC's toolbar layout, making the interface look slightly different.
- **Config step (JSON)**:
  ```json
  {"type": "execute", "parameters": {"command": ["bash", "-c", "(sleep $((RANDOM % 35 + 12)) && wmctrl -a 'VLC' 2>/dev/null && DISPLAY=:0 python3 -c \"import pyautogui, time; time.sleep(0.5); pyautogui.hotkey('alt', 'v'); time.sleep(0.3); pyautogui.press('a')\") &"]}}
  ```
- **Where to insert**: Appended after the window_geometry step (index 6 in noisy config)
- **Rationale**: The VLC domain-specific hint explicitly mentions toggling the Advanced Controls toolbar as valid noise for this domain. It changes VLC's visual layout moderately without closing any dialog or corrupting any config value.
- **Evaluator interference risk**: low — toggling the toolbar only affects VLC's UI display, not vlcrc's recording folder setting.

## Temporal Randomness Strategy
- **Which steps use randomized timing?**
  - Notification burst (step index 3): `sleep $((RANDOM % 60 + 15))` → fires 15-75s after config executes; inner gap `sleep $((RANDOM % 10 + 5))` → 5-15s between email and calendar alerts.
  - Filesystem clutter daemon (step index 4): `sleep $((RANDOM % 90 + 30))` → each file appears 30-120s apart, running continuously.
  - Window geometry move (step index 5): `sleep $((RANDOM % 20 + 8))` → fires 8-28s after config; this is timed to hit after VLC has loaded but potentially while the agent is navigating menus.
  - Panel toggle (step index 6): `sleep $((RANDOM % 35 + 12))` → fires 12-47s after config; overlaps with the agent's likely navigation time.

- **Are there any ongoing background daemons?** Yes — the filesystem clutter step spawns a `while true` daemon that continuously drops `.tmp` files into ~/Downloads every 30-120 seconds.

- **Are there correlated event bursts?** Yes — the notification burst step chains an email alert immediately followed by a calendar reminder 5-15s later, simulating a realistic flurry of distractions.

- **Does the environment get progressively messier over time?** Yes — the filesystem daemon accumulates files over time; the window geometry and panel toggle fire at staggered random delays (8-28s and 12-47s), so early on VLC looks normal but it becomes harder to navigate as time passes.

- **Any focus-stealing events?** No explicit focus-stealing step was added (budget is spent on 5 elements across 3 categories already), but gedit in the background combined with notifications can naturally pull focus on some VM configurations.

## Safety Checklist
- [x] All original config steps preserved in same relative order
- [x] All top-level fields preserved byte-identical (id, snapshot, instruction, source, trajectory, related_apps, evaluator, proxy, fixed_ip, possibility_of_env_change)
- [x] `instruction` unchanged
- [x] `evaluator.func` unchanged
- [x] `evaluator.expected` unchanged
- [x] No noise step touches any protected path listed above
- [x] No credentials, network disruption, or unsafe commands
- [x] Noise element count within budget (5 elements, within 4-5 range)
- [x] All timed noise steps use randomized delays (`$((RANDOM % N + M))`), NOT fixed `sleep` values

## Notes for Human Reviewer
- The VLC window geometry move uses `-r 'VLC'` as a partial title match, which should match "VLC media player" on typical installations. If VLC is not yet fully loaded when the geometry command fires (unlikely given the 8-28s delay), the wmctrl call will silently fail due to `2>/dev/null` — this is intentional and safe.
- The panel_toggle for VLC Advanced Controls uses `alt+v` then `a` to navigate View > Advanced Controls. This is the standard VLC keyboard shortcut path. If the menu has already been opened by the agent, the `alt+v` press will close it harmlessly, and the `a` press will type into whatever field is active — this is a very low-probability edge case and the timing randomness reduces the chance of collision.
- The gedit background app could theoretically steal focus, but since no explicit focus-stealing step was added, this is a passive risk that falls within acceptable bounds for the medium tier.
- The filesystem clutter daemon writes to ~/Downloads (never ~/Desktop), so it cannot create false positives in the evaluator which checks for /home/user/Desktop as the recording path.
