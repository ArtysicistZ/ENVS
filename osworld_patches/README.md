# OSWorld Modified Files

These are the modified OSWorld files for our setup. Copy these over the original OSWorld files after cloning.

## Files Included

1. **`uitars_agent.py`** → `OSWorld/mm_agents/uitars_agent.py`
   - **Line 562**: Changed `base_url` from localhost to Colab GPU server URL
   - **Line 596**: Added `timeout=1800.0` (30 minutes for slow inference)
   
2. **`run_multienv_uitars.py`** → `OSWorld/run_multienv_uitars.py`
   - **Line 252**: Changed `provider_name="docker"` to `"vmware"` (for macOS)
   
3. **`run_uitars.py`** → `OSWorld/run_uitars.py`
   - **Line 165**: Changed `provider_name="docker"` to `"vmware"` (for macOS)
   - **Lines 143-156**: Optimized `runtime_conf` (reduced history, smaller images, token caps)
   - Fixed parameter passing to UITARSAgent

## How to Apply

After cloning the repository:

```bash
# Navigate to repository
cd arpo_replica

# Initialize OSWorld submodule
git submodule update --init --recursive

# Copy modified files
cp osworld_patches/uitars_agent.py OSWorld/mm_agents/
cp osworld_patches/run_multienv_uitars.py OSWorld/
cp osworld_patches/run_uitars.py OSWorld/

echo "✅ Applied OSWorld modifications"
```

## Important Notes

1. **Update ngrok URL**: Before running, edit `uitars_agent.py` line 562 with your actual Colab server URL
2. **VM Location**: VM will auto-download to `OSWorld/vmware_vm_data/` (relative to OSWorld directory)
3. **VMware Required**: These files assume VMware Fusion is installed on macOS

## Changes Summary

- **VMware provider** instead of Docker (for macOS)
- **Configurable base_url** for remote GPU servers
- **Extended timeout** (30 min) for slow inference
- **CPU optimizations** (reduced history, smaller images, token caps)
