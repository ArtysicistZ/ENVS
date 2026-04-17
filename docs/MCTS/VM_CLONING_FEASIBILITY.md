# VM Cloning Feasibility for MCTS Branching

**Date:** 2026-03-20
**Goal:** Clone a running Docker container (Xorg + GNOME + Chrome + LibreOffice) mid-episode to explore different MCTS branches.

## Host Environment

- Ubuntu 24.04.3 LTS, kernel 6.17.0-1008-azure, AMD EPYC 7V12 (96 cores), 1.7 TiB RAM
- Docker v29.1.4, overlay2 on ext4, experimental=OFF
- 64 running containers, each ~627 MiB idle / ~950 MiB with apps, ~82 processes
- Docker overlay upper dir: **~2.2 MB** (tiny — mostly config changes)
- **No `/dev/kvm`** (Azure VM without nested virtualization)
- **No CRIU** installed (removed from Ubuntu 24.04 repos)

## Results Summary

| Approach | Clone Time | Full State? | Viable? | Blocker |
|---|:-:|:-:|:-:|---|
| CRIU + Docker checkpoint | 5-15s | Yes (theory) | **NO** | X11 + Chrome + D-Bus = CRIU crashes |
| CRIU `--leave-running` | Same | Same | **NO** | Same + CoW memory explosion |
| Podman + CRIU | Same | Same | **NO** | Same CRIU underneath |
| **QEMU/KVM savevm/loadvm** | **3-8s** | **YES, perfect** | **NO (here)** | No `/dev/kvm` on Azure |
| Firecracker microVM | <2s | Yes | **NO** | No GUI/display support |
| **OverlayFS copy + warm pool** | **6-16s** | Disk only | **YES** | Apps need re-launch |
| **ZFS snapshot + warm pool** | **1-3s (fs) + 5-15s restart** | Disk only | **YES** | Need to install ZFS |
| Namespace cloning | N/A | Nothing | **NO** | Can't fork process state |
| Action replay (current) | 20-48s | Re-created | **YES** | Slowest option |

## Why CRIU Fails for Our Stack

1. **Xorg**: CRIU cannot checkpoint X11 unix sockets (`/tmp/.X11-unix/X0`). All X clients (Chrome, GIMP, GNOME) crash on restore.
2. **Chrome**: Multi-process architecture with >1TB virtual memory areas exceeds CRIU's `MAX_RW_COUNT`. Chrome's seccomp sandbox adds further complications.
3. **D-Bus**: 30+ D-Bus clients in GNOME session; socket FD matching on restore is fragile.
4. **systemd as PID 1**: CRIU wiki lists systemd as "not fully supported."
5. **Zero published examples** of CRIU with Xorg + GNOME + Chrome. No one has done this.

## The Two Viable Paths

### Path A: OverlayFS/ZFS Copy + Warm Pool (Works Now)

1. Pre-warm spare containers (booted, GNOME running, no task state)
2. Clone: copy overlay upper dir to spare container (<1s with ZFS, 1-2s with cp)
3. The clone has correct filesystem but fresh processes
4. Re-launch task-specific apps OR replay only the app-opening actions (~5-15s)
5. **Total: 6-16s per clone**

### Path B: QEMU/KVM (The Proper Solution, Needs Infra Change)

1. Switch from Docker to QEMU/KVM VMs (or enable nested virt on Azure)
2. `savevm` = full snapshot of CPU + memory + display + all processes in 2-5s
3. `loadvm` = restore any snapshot in 2-4s
4. Fork: snapshot → start N new VMs from it = **true cloning in 3-8s**
5. The `VMwareProvider` pattern already exists in the codebase (`save_state` / `revert_to_snapshot`)
6. **Requires**: bare metal servers OR Azure nested virt (Standard_D*s_v3 + `--enable-nesting`)

## Research Context

- **No published work** combines MCTS tree search with true VM cloning for desktop GUI agents
- ExACT (ICLR 2025): MCTS for web agents, uses sequential replay, no VM snapshots
- CodeSandbox: <2s VM forking using Firecracker microVMs — but no GUI support
- OSWorld community: uses QEMU inside Docker for parallelization, standard savevm/loadvm, no instant cloning

## Recommendation

**Short term:** OverlayFS copy + warm container pool (6-16s clones, no infra changes)
**Medium term:** Investigate Azure nested virt or bare-metal servers for QEMU/KVM (3-8s perfect clones)
**Long term:** CRIU may eventually support X11/Chrome — monitor upstream progress
