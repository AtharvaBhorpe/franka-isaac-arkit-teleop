# Franka × iPhone-ARKit Teleop (Isaac Sim 6.0 · ROS2 PoC · Ubuntu)

Teleoperate a **simulated Franka Emika Panda** in **NVIDIA Isaac Sim 6.0** by
waving your **iPhone** — using **ARKit** for pose and **ROS2 + Pinocchio** for
control. The task scene is a small **cube**, a small **bin**, and a
**wrist-mounted camera**; you pick the cube and drop it in the bin from the
phone. It reuses the Pinocchio servo-IK *technique* from
[SpesRobotics/teleop](https://github.com/SpesRobotics/teleop) — not the package;
the ARKit input path is our own.

The **teleop PoC (Phases 0–6) works** end-to-end. The project is now in **Phase 7 —
imitation learning over Rerun `.rrd`**: record demos (manually, or hands-off via the
scripted expert) → train a policy (ACT / Diffusion) → let it drive the arm closed-loop.
A simulated **gripper tactile sensor** (`--tactile`) adds a force-field modality, with a
success-rate harness to test whether it helps. (No LeRobot — the pipeline is in-house.)

- **New here? Follow [docs/HOWTO.md](docs/HOWTO.md)** — clone→install→run, step by
  step (Phases 0–6 + the Phase-7 record/train/infer + ablation runbook).
- **Lost in the files?** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — what each file does, how
  they connect (diagrams), and an end-to-end runbook (teleop → record → train → infer).
- Plan, decisions, and rationale: [PROJECT.md](PROJECT.md).

## Platform
Everything runs on **one native Ubuntu 26.04 machine** (dual-booted with
Windows): Isaac Sim, ROS2 (RoboStack), teleop, and the ARKit receiver. The
iPhone connects over WiFi. (We moved off Windows because its bleeding-edge GPU
driver crashed Isaac Sim's RTX renderer — see PROJECT.md §2.)

## Run model — the Isaac Sim 6.0 standalone binary
We run Isaac Sim from the **downloaded 6.0 standalone binary** (not the
`isaacsim` pip package). The install lives **outside the repo** and is surfaced
as a gitignored symlink at the project root:

```
franka-isaac-arkit-teleop/.isaac-sim  ->  ~/isaac-sim/6.0.0
```

Everything launches through `scripts/run_isaac.sh`, which uses that symlink and
the binary's bundled Python. **pixi** hosts the ROS2 env (RoboStack Jazzy +
Pinocchio + our own `teleop_arkit` package).

If you move machines, just re-point the symlink at that machine's install:
```bash
ln -sfn ~/isaac-sim/6.0.0 .isaac-sim
```

## Quickstart — Phase 1 (Franka pick-and-place scene)

```bash
# 0. (once) Install pixi if needed, then open a new shell:
#    curl -fsSL https://pixi.sh/install.sh | bash

# 1. Sanity-check the driver and the .isaac-sim symlink:
chmod +x scripts/setup_ubuntu.sh scripts/run_isaac.sh
./scripts/setup_ubuntu.sh

# 2. Run the Franka pick-and-place scene (GUI):
pixi run franka
#    or directly, bypassing pixi:
#    ./scripts/run_isaac.sh isaac/load_franka_pickplace.py
#    Hybrid-GPU laptop? Force the NVIDIA GPU:
#    __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia pixi run franka
```

You should see the Franka pick the cube, drop it into the bin, and per-step
joint positions + wrist-camera frame shapes printed. That confirms Phase 1.

To open the full Isaac Sim GUI app instead: `pixi run isaac-sim`.

## Requirements
- Ubuntu 26.04 (Isaac Sim 6.0 officially validates 22.04/24.04 — see PROJECT.md §2).
- NVIDIA GPU with driver **≥ 580.95.05**.
- The Isaac Sim 6.0 standalone binary, symlinked as `.isaac-sim` (see above).
- `pixi` (for the convenience tasks now, the ROS2 env later).

## Status
**The full PoC works (Phases 0–6):** you teleop a complete cube→bin
pick-and-place from the iPhone, and the §6 end-to-end data check is confirmed.
See the phase checklist in
[PROJECT.md](PROJECT.md#5-phased-step-plan).
