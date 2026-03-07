# ARPO Files Overview

## 📁 Essential Files (13 files)

### 🚀 Training & Execution (5 files)

1. **`uitars_2b_server.py`** ⭐
   - Flask server for UI-TARS-2B inference
   - Optimized for CPU (image resizing, token limits)
   - Timeout: 30 minutes
   - Usage: `python uitars_2b_server.py`

2. **`config_uitars_2b_mac.yaml`** ⭐
   - Complete VERL training configuration
   - ARPO settings (replay_enabled, GRPO)
   - Mac CPU optimized (1 VM, 8 tasks, 5 epochs)
   
3. **`train_uitars_2b_arpo.sh`** ⭐
   - Full ARPO training script
   - Checks server, starts Ray, runs training
   - Usage: `bash train_uitars_2b_arpo.sh`

4. **`test_server.sh`**
   - Test UI-TARS-2B server
   - Verifies API endpoints work
   - Usage: `bash test_server.sh`

5. **`test_osworld_uitars.sh`**
   - Test OSWorld + UI-TARS-2B integration
   - End-to-end pipeline verification
   - Usage: `bash test_osworld_uitars.sh`

---

### 📚 Documentation (4 files)

6. **`START_HERE.md`** ⭐ **READ FIRST**
   - Main entry point
   - Quick 3-step training guide
   - Prerequisites checklist

7. **`TRAINING_GUIDE.md`**
   - Complete training workflow
   - Monitoring and evaluation
   - Expected results and metrics
   
8. **`PAPER_SUMMARY.md`**
   - ARPO paper deep dive
   - GRPO algorithm explained
   - Experience replay mechanism

9. **`TROUBLESHOOTING.md`**
   - Common issues and solutions
   - Debugging checklist
   - Error resolution

---

### 📓 Interactive (1 file)

10. **`arpo_training_notebook.ipynb`**
    - Interactive Jupyter notebook
    - Step-by-step guide with explanations
    - Cells for testing and configuration

---

### 📝 Configuration (3 files)

11. **`README.md`**
    - Project overview
    - Points to START_HERE.md

12. **`requirements.txt`**
    - Python dependencies
    - Usage: `pip install -r requirements.txt`

13. **`LICENSE`**
    - Apache 2.0 license

---

## 📂 Directories (Not Files)

### Data & Code
- `OSWorld/` - OSWorld benchmark (submodule, 38GB with VM)
- `verl/` - VERL training framework
- `examples/` - Example training scripts
- `scripts/` - Helper scripts
- `evaluation_examples` → `OSWorld/evaluation_examples` (symlink)
- `cache_dirs/` - Cache symlinks
- `vmware_vm_data` → `OSWorld/vmware_vm_data` (symlink)

### Output (Created During Use)
- `checkpoints_2b/` - Model checkpoints
- `results_2b/` - Training results
- `results_test_2b/` - Test results
- `logs/` - Log files

---

## 🎯 Which Files to Use

### To Start Training:
1. Read: `START_HERE.md`
2. Run: `test_server.sh`
3. Run: `test_osworld_uitars.sh`
4. Run: `train_uitars_2b_arpo.sh`

### To Understand ARPO:
- Read: `PAPER_SUMMARY.md`
- Explore: `arpo_training_notebook.ipynb`

### If You Have Problems:
- Check: `TROUBLESHOOTING.md`
- See: `TRAINING_GUIDE.md`

---

## 🗑️ Removed (11 files cleaned up)

Setup scripts (no longer needed):
- setup.sh, finish_mac_setup.sh, install_dependencies.sh
- fix_transformers.sh, start_server.sh
- setup.py, pyproject.toml

Redundant docs (consolidated):
- README_SETUP.md, QUICKSTART_2B.md, FINAL_SUMMARY.md
- STATUS.md, ENVIRONMENT_SETUP.md, SETUP_MACOS.md
- QUICK_START.txt, requirements_cpu.txt

Old scripts:
- train_cpu_subset32.sh, train_uitars_2b.sh

---

## 📊 Disk Usage

- **OSWorld VM**: ~38GB (essential)
- **UI-TARS-2B**: ~10GB (cached model)
- **Documentation**: ~2MB
- **Scripts**: <1MB
- **Total**: ~48GB

---

**Everything is minimal and organized. Start with `START_HERE.md`!** 🚀
