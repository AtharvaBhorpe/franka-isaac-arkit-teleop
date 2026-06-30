# isaac — Isaac Sim 6.0 simulation side (separate Python runtime)

## Purpose
The simulated Franka pick-and-place scene + app that publishes the ROS2 streams teleop and the IL
pipeline consume. Runs inside the **Isaac Sim binary's own bundled Python** (isaacsim/omni/pxr/usdrt
+ rclpy), launched via `scripts/run_isaac.sh` — NOT the pixi `ros` env.

## Ownership
- `franka_scene.py` — sim **library**: scene/camera/ROS2-graph builders, `CAMERAS`,
  `WRIST_CAM_RES`/`SCENE_CAM_RES`, `CUBE_SPAWN_REGION`, `randomize_cube_pose()`,
  `bin_world_aabb()` + `cube_in_bin()` (success label), `get_rgba`/`save_rgba`. Imports only
  numpy/os at top so it stays import-safe.
- `load_franka_pickplace.py` — sim **app**: arg parsing, run loops, `main`, the rclpy
  `ResetListener` (`/episode/reset` → `pick_place.reset()` + randomize cube), `--control ros|auto`,
  `--headless`. `--control auto --record` (task `franka-auto-record`) = hands-off data collection:
  each cycle randomizes the cube, drives the `record` node via `/record/command` (s/e/f), mirrors
  the expert's joint targets onto `/joint_command`, and auto-labels by `cube_in_bin`. Pair with the
  ros-env `record-auto` task (`record --settle-secs 0`). `--tactile` (both control modes) publishes
  the gripper force grid (see `tactile.py`). `--control ros --eval` (task `franka-eval`) = closed-loop
  policy **success-rate** harness: N trials, the ros-env `infer` policy drives `/joint_command`, score
  `cube_in_bin` per trial, print the rate — run two policies (± the tactile camera) through it to
  ablate a modality.
- `tactile.py` — `TactileGrid`: per-fingerpad PhysX contact → a **per-pad 32×12 normal+shear force
  field** (`get_contact_force_data` normal + `get_friction_data` tangential), the **TacSL/FlexiTac
  representation** ported natively (no IsaacLab/GelSight; see PROJECT.md §9 2026-06-30). Sparse rigid
  contacts → bbox-filled + Gaussian-blurred into a smooth blob. Renders the two pads **separately**,
  side by side on `/tactile/image_raw` — `--tactile-mode normal` (per-pad **jet**-colormap force blob,
  FlexiTac-style, default) or `shear` (R,G=shear, B=normal). Rides the **camera** abstraction (a force image is an image), so the
  recorder logs it as a modality with no schema change — add `tactile=/tactile/image_raw` to
  `--cameras` (the `record-tac` task does this). `PAD_ORIGIN`/extents/`PATCH`/`FORCE_MAX_N` are
  **calibration constants** (tuned 2026-06-30 from the auto-grasp contact cloud).

## Local Contracts
- **No `isaacsim`/`omni`/`pxr`/`usdrt` imports at module top** — Kit must boot (`SimulationApp`)
  first; import inside post-boot functions.
- **Runtime boundary:** this directory CANNOT import `teleop_arkit` (different Python). The only
  coupling is ROS2 topics, over DDS localhost, `ROS_DOMAIN_ID=0`, `rmw_fastrtps_cpp`.
- Publishes `/joint_states /wrist_cam/image_raw /scene_cam/image_raw /tf /clock`; subscribes
  `/joint_command` (only when `--control ros`) + `/episode/reset`. Under `--record` the app instead
  *publishes* `/joint_command` (expert targets, for the recorder) + `/record/command`. Under
  `--tactile` it also publishes `/tactile/image_raw` (32×12 RGB normal+shear force field).

## Work Guidance
- Cameras are GPU-render-bound (~18 Hz idle, ~5–7 Hz under teleop load) despite
  `Camera(frequency=20)`; the `.rrd` logs the native rate (the load-time grid resamples).
- Launch via the pixi `default`-env tasks, never `python isaac/...` directly.

## Verification
- `pixi run franka` (scene) · `franka-teleop` (`--control ros` — the recording/inference launch) ·
  `franka-headless`.

## Child DOX Index
None (leaf).
