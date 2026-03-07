# VMware VM Configuration Options

## Supported Configuration Types

### 1. Operating System Type
- **`os_type`**: `"Ubuntu"` (default) or `"Windows"`
- Ubuntu is the standard OSWorld setup
- Windows support available but less common

### 2. Screen Resolution
- **`screen_width`**: Integer (default: `1920`)
- **`screen_height`**: Integer (default: `1080`)
- **`screen_size`**: Tuple `(width, height)` (default: `(1920, 1080)`)
- Common resolutions: 1920x1080, 1280x720, 2560x1440

### 3. Display Mode
- **`headless`**: Boolean (default: `False`)
  - `False`: VM window visible (useful for debugging)
  - `True`: Run VM in background (faster, no GUI)

### 4. Action Space
- **`action_space`**: `"pyautogui"` or `"computer_13"`
  - `"pyautogui"`: Direct mouse/keyboard actions (simpler)
  - `"computer_13"`: Structured action format (more complex)

### 5. Observation Type
- **`observation_type`**: One of:
  - `"screenshot"`: Visual screenshots only
  - `"a11y_tree"`: Accessibility tree only
  - `"screenshot_a11y_tree"`: Both screenshot + accessibility tree
  - `"som"`: Screen Object Model

### 6. VM Path & Snapshots
- **`path_to_vm`**: Path to `.vmx` file (optional, auto-detected if not provided)
- **`snapshot_name`**: Snapshot to revert to (default: `"init_state"`)

### 7. Provider Type
- **`provider_name`**: `"vmware"` (default), `"docker"`, `"virtualbox"`, `"aws"`, `"azure"`
- VMware Fusion is used on macOS

### 8. Other Options
- **`cache_dir`**: Directory for caching task files (default: `"cache"`)
- **`require_a11y_tree`**: Boolean (default: `True`)
- **`require_terminal`**: Boolean (default: `False`)
- **`region`**: String (for cloud providers, default: `"us-east-1"`)

---

## Example Configurations

### Standard Configuration (Current Setup)
```python
env = DesktopEnv(
    provider_name="vmware",
    os_type="Ubuntu",
    screen_size=(1920, 1080),
    headless=True,
    action_space="pyautogui",
    observation_type="screenshot"
)
```

### Windows VM
```python
env = DesktopEnv(
    provider_name="vmware",
    os_type="Windows",
    screen_size=(1920, 1080),
    headless=False,
    action_space="pyautogui",
    observation_type="screenshot"
)
```

### High Resolution
```python
env = DesktopEnv(
    provider_name="vmware",
    os_type="Ubuntu",
    screen_size=(2560, 1440),
    headless=True,
    action_space="pyautogui",
    observation_type="screenshot"
)
```

### With Accessibility Tree
```python
env = DesktopEnv(
    provider_name="vmware",
    os_type="Ubuntu",
    screen_size=(1920, 1080),
    headless=True,
    action_space="pyautogui",
    observation_type="screenshot_a11y_tree",
    require_a11y_tree=True
)
```

---

## Command Line Arguments

When using `run_uitars.py`:

```bash
python run_uitars.py \
    --headless \
    --screen_width 1920 \
    --screen_height 1080 \
    --action_space pyautogui \
    --observation_type screenshot \
    --os_type Ubuntu
```

---

## Notes

- **Default**: Ubuntu, 1920x1080, headless=False, pyautogui, screenshot
- **Most common**: Ubuntu, 1920x1080, headless=True (for evaluation)
- **Windows**: Requires separate VM image download (~50GB)
- **Screen size**: Must match VM's actual resolution for accurate actions
