# Noise Addition Documentation

## Noise Categories

| Type | Implementation | Purpose |
|------|----------------|---------|
| Web Security/Phishing | Extra browser tabs | Test distractor filtering |
| Advertisement | Additional URLs in tabs | Test tab navigation |
| VPN Notifications | `notify-send` VPN messages | Test notification dismissal |
| System Notifications | `notify-send` alerts | Test focus maintenance |
| Environmental Noise | Background apps (gedit, calculator, nautilus) | Test multi-window focus |
| File Clutter | Extra similar files | Test target identification |

---

## Example 1: Chrome Task (0d8b7de3-e8de-4d86-b9fd-dd2dce58a217.json)

**Task**: Browse the natural products database

**Noises Added**:
- 3 background apps: `gedit`, `gnome-calculator`, `nautilus`
- 2 notifications: VPN connection, software update
- 2 extra browser tabs: example.com, wikipedia.org

**Agent Must**:
- Ignore 3 background windows
- Ignore 2 notifications
- Navigate to correct tab (drugs.com) among 3 tabs
- Complete task: reach drugs.com/npc/ or drugs.com/npp/

---

## Example 2: GIMP Task (2e6f678f-472d-4c55-99cc-8e7c5c402a71.json)

**Task**: Batch process 3 images by lifting brightness to 50

**Noises Added**:
- 2 distractor images: distractor1.jpeg, distractor2.jpeg
- 3 background apps: `nautilus`, `eog`, `gnome-system-monitor`
- 2 notifications: storage warning, software update

**Target Files**: squirrel.jpeg, panda.jpeg, heron.jpeg  
**Distractor Files**: distractor1.jpeg, distractor2.jpeg

**Agent Must**:
- Identify correct 3 images among 5 total
- Ignore file manager, image viewer, system monitor
- Ignore 2 notifications
- Process ONLY the 3 target images

---

## Noise Levels

| Level | Count | Components |
|-------|-------|------------|
| Low | 1-2 | Notifications only |
| Medium | 3-5 | 2-3 apps + 1-2 notifications + 1-2 tabs/files |
| High | 6+ | 3+ apps + 2+ notifications + multiple tabs/files |

---

## Ubuntu Tools

**Applications**: `gedit`, `gnome-calculator`, `nautilus`, `eog`, `evince`, `gnome-system-monitor`, `firefox`

**Commands**: `notify-send "Title" "Message"`, `nautilus /path`, `eog /path/to/image.jpg`