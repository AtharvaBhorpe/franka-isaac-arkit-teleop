# How-To Guide — set up & run from a fresh machine

Step-by-step to clone, install, and run this project on a new Ubuntu PC. Follow
top to bottom. Each phase ends with a **✓ Check** so you know it worked.

> **Covered: the full PoC (Phases 0–6)** — Isaac Sim scene + cameras + ROS2 bridge
> + RoboStack interop + Pinocchio IK + **6-DoF iPhone/ARKit teleoperation**.
>
> Background & rationale for every decision live in [PROJECT.md](../PROJECT.md).

---

## 0. What you're setting up

A simulated **Franka Panda** in **Isaac Sim 6.0** doing a cube→bin pick-and-place
scene with a **wrist camera** and a **scene camera**, all telemetry published over
**ROS2** (joint states, TF, clock, both camera streams), verified from a separate
**RoboStack ROS2 Jazzy** environment. Two ROS2 stacks (the Isaac binary's and
RoboStack's) talk over **localhost DDS**.

The end state: **you wave an iPhone (ARKit, via ZIG SIM PRO) and the simulated
Franka mirrors it in 6-DoF** — moving the arm, tilting the wrist, and opening/
closing the gripper to pick the cube and drop it in the bin, all over ROS2 with
Pinocchio inverse kinematics.

---

## 1. Prerequisites

**Hardware / OS**
- An **NVIDIA RTX GPU** (this was built on an RTX 5060 Laptop GPU).
- **NVIDIA driver ≥ 580.95.05** (check with `nvidia-smi`). Older drivers can crash
  the RTX renderer.
- **Ubuntu** — Isaac Sim 6.0 officially validates **22.04 / 24.04**. This project
  runs on **26.04** by choice (works, but unsupported by NVIDIA).
- On a **hybrid-GPU laptop** (NVIDIA dGPU + AMD/Intel iGPU) you may need to force
  the NVIDIA GPU — see [Troubleshooting](#8-troubleshooting).

**Tools**
- **pixi** (package/env manager). Install and open a new shell:
  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```
- **git**, **unzip**, and ~30 GB free disk (the Isaac Sim binary is ~13 GB zipped
  and similar unzipped).

✓ **Check:** `nvidia-smi` prints your GPU + a driver ≥ 580.95.05, and
`pixi --version` works in a fresh shell.

---

## 2. Get the project

```bash
git clone <YOUR_REPO_URL> franka-arktit-teleop
cd franka-arktit-teleop
```

Work from the **Linux filesystem** (e.g. `~/franka-arktit-teleop`), not a mounted
Windows/NTFS partition.

---

## 3. Install Isaac Sim 6.0 (standalone binary)

We run Isaac Sim from NVIDIA's **standalone binary**, *not* the `isaacsim` pip
package. The install lives **outside** the repo and is linked in via a gitignored
symlink named `.isaac-sim`.

1. Download **Isaac Sim 6.0 — Linux** from the NVIDIA Isaac Sim downloads page.
   You want the standalone zip: `isaac-sim-standalone-6.0.0-linux-x86_64.zip`.
2. Unzip it to a stable location (not `~/Downloads`):
   ```bash
   mkdir -p ~/isaac-sim
   unzip ~/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64.zip -d ~/isaac-sim/6.0.0
   ```
   You should end up with `~/isaac-sim/6.0.0/isaac-sim.sh`, `python.sh`, `exts/`, …
3. Create the symlink the repo expects:
   ```bash
   ln -sfn ~/isaac-sim/6.0.0 .isaac-sim
   ```
   (Per machine; `.isaac-sim` is gitignored. On another PC, just re-point it.)

4. Run the bootstrap check (driver + symlink + pixi):
   ```bash
   chmod +x scripts/setup_ubuntu.sh scripts/run_isaac.sh
   ./scripts/setup_ubuntu.sh
   ```

✓ **Check:** `ls .isaac-sim/python.sh` resolves, and `setup_ubuntu.sh` reports the
driver and Isaac Sim path without errors.

---

## 4. Phase 1 — run the pick-and-place scene

```bash
pixi run franka
# equivalent, and avoids a harmless "running in conda env" warning:
# ./scripts/run_isaac.sh isaac/load_franka_pickplace.py
```

First launch streams the Isaac asset library from the cloud — **it can take
10+ minutes** the first time. Subsequent runs are fast.

What it does: loads the Franka + a 5 cm cube + a scaled-down KLT bin + a wrist
camera + a scene camera, then runs 3 autonomous pick-and-place cycles (this
autonomous motion is a temporary self-test), printing joint values each step and
saving camera frames to `outputs/`.

✓ **Check (in the log):**
- `RTX renderer` initializes (your NVIDIA GPU listed `Active: Yes`, the iGPU
  "Skipping unsupported non-NVIDIA GPU").
- `cycle 1/3 complete (cube placed).` … `cycle 3/3`.
- `wrote TCP offset -> .../config/tcp_offset.yaml`.
- `outputs/` contains `wrist_cam_step*.png` and `scene_cam_step*.png`
  (open them — you should see the gripper/cube and the whole rig).

Useful flags: `--headless` (no viewport), `--cycles N`, `--device {cpu,cuda}`.

---

## 5. Phases 2 & 3 — ROS2 bridge + RoboStack verification

### 5a. Run Isaac with the ROS2 bridge

```bash
pixi run franka-ros
# = ./scripts/run_isaac.sh isaac/load_franka_pickplace.py --ros
# add --cycles 60 to keep it running long enough to inspect from another terminal
```

`run_isaac.sh` sets the env the bridge needs on Ubuntu 26.04 before Isaac boots
(`ROS_DISTRO=jazzy`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, internal-libs
`LD_LIBRARY_PATH`). It builds a `/ROS2Graph` action graph publishing `/clock`,
`/joint_states`, `/tf`, and both cameras.

✓ **Check (in the log):** `rclpy loaded`, `isaacsim.ros2.bridge … startup` (and it
stays up, no immediate shutdown), and
`[franka] ROS2 bridge graph built (/clock, /joint_states, /tf, cameras).`

### 5b. Install the RoboStack ROS2 env (one-time)

```bash
pixi install -e ros
```
Pulls RoboStack ROS2 Jazzy (`ros-base` + `rmw-fastrtps-cpp` + `rqt-image-view` +
`rviz2`), isolated from the default env. A few hundred MB; the solve can take a
few minutes.

### 5c. Verify the topics flow (two terminals)

**Terminal A** — Isaac publishing:
```bash
pixi run franka-ros          # or: ./scripts/run_isaac.sh isaac/load_franka_pickplace.py --ros --cycles 60
```

**Terminal B** — the RoboStack side (same `ROS_DOMAIN_ID=0`, same FastDDS):
```bash
pixi run -e ros ros2 topic list
pixi run -e ros ros2 topic echo /joint_states
pixi run -e ros ros2 topic hz /wrist_cam/image_raw
pixi run -e ros ros2 run rqt_image_view rqt_image_view   # GUI viewer (optional)
```

✓ **Check:** `ros2 topic list` shows:
```
/clock  /joint_states  /tf
/wrist_cam/image_raw   /wrist_cam/camera_info
/scene_cam/image_raw   /scene_cam/camera_info
```
and `echo /joint_states` prints 9 joint values that change as the arm moves. That
confirms the Isaac binary ↔ RoboStack DDS interop.

### 5d. Drive the arm from ROS (control path)

So far the arm moves itself (autonomous self-test). To make it **ROS-driven**,
run with `--control ros` — now it ignores the self-test and follows
`/joint_command` (`sensor_msgs/JointState`):

**Terminal A:**
```bash
./scripts/run_isaac.sh isaac/load_franka_pickplace.py --control ros
```

**Terminal B** — first read the exact joint names, then command the 7 arm joints:
```bash
pixi run -e ros ros2 topic echo --once /joint_states          # note the 'name:' list
pixi run -e ros ros2 topic pub -1 /joint_command sensor_msgs/msg/JointState \
  "{name: ['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7'], \
    position: [0.5, -0.5, 0.0, -2.0, 0.0, 2.0, 0.8]}"
```

✓ **Check:** the arm moves to the commanded pose, and `ros2 topic echo /joint_states`
reflects the new positions — a full closed loop (command in → motion → state out).

---

## 6. Phases 4 & 5 — iPhone teleoperation (Pinocchio IK + ARKit)

The payoff: drive the Franka in real time from an iPhone.

```
ZIG SIM (iPhone: ARKit + touch) ──UDP/JSON──▶ arkit_receiver
   ──/target_frame, /gripper_command──▶ joint_command_node (Pinocchio servo-IK)
   ──/joint_command──▶ Isaac ROS2 bridge ──▶ Franka
```
All of this is our own `teleop_arkit/` package; it reuses only the Pinocchio
servo-IK *technique* from SpesRobotics/teleop, not the package.

### 6a. Deps (one-time)
The `ros` env already lists `pinocchio` + `example-robot-data` (a Pinocchio-ready
Panda URDF + meshes). If you installed the env before these were added, re-sync:
```bash
pixi install -e ros
```
✓ **Check** — the IK solver works standalone (no ROS/Isaac needed):
```bash
pixi run -e ros python -m teleop_arkit.ik     # prints "reached in N steps", sub-mm error
```

### 6b. iPhone app — ZIG SIM PRO
You need **ZIG SIM PRO** (ARKit 6-DoF is a PRO feature; the free app is
orientation-only and won't give world position).
- iPhone on the **same WiFi** as the PC; open the port once: `sudo ufw allow 50000/udp`.
- In ZIG SIM: enable data items **ARKit** *and* **touch**; protocol **UDP**, format **JSON**.
- Destination = this PC's IP (`hostname -I` → pick the LAN `192.168.x.x`), port **50000**.
- Confirm packets arrive: `pixi run -e ros sniff` (then Start in ZIG SIM).

### 6c. Launch (3 terminals)
Free GPU VRAM first (Isaac needs a few GB — see Troubleshooting).
```bash
# A — Isaac, ROS-driven (no autonomous motion)
./scripts/run_isaac.sh isaac/load_franka_pickplace.py --control ros
# B — Pinocchio servo-IK:  /target_frame + /gripper_command -> /joint_command
pixi run -e ros ik-topic
# C — ARKit receiver:  iPhone -> /target_frame + /gripper_command
pixi run -e ros arkit --scale 1.5
```
Then hit **Start** in ZIG SIM. Terminal C should log `robot start EE = …` and,
when you touch the screen, `move engaged`.

### 6d. Control scheme
| Gesture | Action |
|---|---|
| **1 finger** held | **Move** (clutch): EE follows phone **position + rotation**. Re-zeros on each press (no jump). |
| **0 fingers** | **Freeze** — reposition the phone, or carry without moving. |
| **2-finger tap** | **Toggle gripper** open↔closed (latched). |

Full pick-and-place: 1-finger move over the cube → lower → 2-finger tap to grip →
carry to the bin → 2-finger tap to release.

✓ **Check:** phone up→arm up, forward→forward, left→left; yaw/pitch/roll track; the
cube is gripped, carried without slipping, and dropped in the bin.

### 6e. Tuning knobs
- **Translation gain:** `arkit --scale 1.5`.
- **Latency / snappiness** (lag ≈ `1/kp_lin`): run the IK node with overrides,
  e.g. `pixi run -e ros python -m teleop_arkit.joint_command_node --source topic --rate 120 --kp-lin 6`.
- **Orientation:** `arkit --no-orient` (downward-only) or `--quat-order wxyz` if
  rotations look mirrored.
- **Axis flipped?** edit `ARKIT_TO_ROS` in `teleop_arkit/arkit_receiver.py` (flip a row's sign).
- **Grasp slips?** raise friction in `apply_grasp_friction` (`isaac/franka_scene.py`, μs/μd).

---

## 7. Repo map (where things are)

| Path | What |
|------|------|
| `PROJECT.md` | The plan: phases, decisions, risks. |
| `docs/HOWTO.md` | This guide. |
| `isaac/franka_scene.py` | Sim library: constants, helpers, scene builders, ROS2 graph, grasp friction. |
| `isaac/load_franka_pickplace.py` | Sim app entry: arg parsing, run loops, `main` (imports `franka_scene`). |
| `teleop_arkit/ik.py` | Compact Pinocchio Cartesian servo-IK (our code). |
| `teleop_arkit/joint_command_node.py` | ROS node: target → IK → `/joint_command` (+ gripper). |
| `teleop_arkit/arkit_receiver.py` | ROS node: ZIG SIM ARKit+touch → `/target_frame` + `/gripper_command`. |
| `teleop_arkit/sniff_stream.py` | UDP/TCP sniffer to inspect phone packets. |
| `scripts/run_isaac.sh` | Launches the Isaac binary (sets EULA + ROS2 env). |
| `scripts/setup_ubuntu.sh` | Driver / symlink / pixi sanity checks. |
| `pixi.toml` | Default env (binary tasks) + `ros` env (RoboStack). |
| `.isaac-sim` | Gitignored symlink → your Isaac Sim 6.0 install. |
| `config/tcp_offset.yaml` | Measured panda_hand→grasp-TCP offset (for IK). |
| `outputs/` | Camera frames written by runs (gitignored). |

---

## 8. Troubleshooting

- **Hybrid-GPU laptop / renderer crash on first frame** — force the NVIDIA GPU:
  ```bash
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia pixi run franka
  ```
  The loader also passes `multi_gpu:False` + `active_gpu:0`.
- **`running in conda env, please deactivate`** — harmless; comes from `pixi run`
  wrapping the binary's own Python. Use `./scripts/run_isaac.sh …` directly to
  avoid it.
- **First run hangs "resolving assets" / is very slow** — it's streaming the asset
  library from NVIDIA's cloud; first time only, give it 10+ minutes.
- **`ros2 topic list` is empty** (Terminal B sees nothing) — make sure both sides
  use the same `ROS_DOMAIN_ID` and RMW. The `ros` env sets
  `ROS_DOMAIN_ID=0` + `rmw_fastrtps_cpp`; `run_isaac.sh` sets the same. If still
  empty, try forcing localhost discovery on both:
  `export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`.
- **Crash at startup with `Out of GPU memory` / `ERROR_OUT_OF_DEVICE_MEMORY` /
  `gpuOutOfMemory='oom'`** — the RTX renderer (viewport + camera render products)
  ran out of VRAM. On an 8 GB laptop GPU, free memory first: stop other GPU hogs
  (`ollama stop`/`pkill -f ollama`, close Brave/Electron/etc.), check with
  `nvidia-smi` (want ~4 GB+ free), and/or run Isaac `--headless`.
- **`ROS2 Bridge startup failed` / `RMW was not loaded`** — the bridge env wasn't
  set before boot. Launch via `scripts/run_isaac.sh` (it exports `ROS_DISTRO`,
  `RMW_IMPLEMENTATION`, `LD_LIBRARY_PATH`); don't call `python.sh` directly.
- **(Teleop) no phone packets** — same WiFi? Targeting the **LAN** IP (not a
  Tailscale `100.x`/Docker `172.x` one)? `sudo ufw allow 50000/udp` run? Confirm
  with `pixi run -e ros sniff`.
- **(Teleop) laggy/trailing arm** — it's the servo time-constant (≈ `1/kp_lin`) +
  render, **not** the IK solve. Raise `--kp-lin` / `--rate` on the IK node; the
  receiver is already event-driven. There's a ~30–50 ms floor (PhysX + render).
- **(Teleop) rotations mirrored / wrong axis** — try `arkit --quat-order wxyz`,
  or flip a row sign in `ARKIT_TO_ROS`; or `--no-orient` to fall back to downward.
- **(Teleop) cube slips from the gripper** — raise friction in `apply_grasp_friction`
  (`isaac/franka_scene.py`); if still slipping, the finger grip force needs raising.

---

## 9. Status

**The full PoC works (Phases 0–6):** Isaac Sim scene + cameras, ROS2 bridge,
RoboStack interop, Pinocchio servo-IK, and **6-DoF iPhone/ARKit teleoperation** —
you teleop a complete cube→bin pick-and-place from the phone.

**Optional polish not yet done:** rviz2 `RobotModel` (Phase 4.1, needs
`package://` mesh resolution); finer grip-force tuning. See
[PROJECT.md](../PROJECT.md) for the full phase log and decisions.
