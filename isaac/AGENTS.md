# isaac — Isaac Sim 6.0 simulation side (separate Python runtime)

## Purpose
The simulated Franka pick-and-place scene + app that publishes the ROS2 streams teleop and the IL
pipeline consume. Runs inside the **Isaac Sim binary's own bundled Python** (isaacsim/omni/pxr/usdrt
+ rclpy), launched via `scripts/run_isaac.sh` — NOT the pixi `ros` env.

## Ownership
- `franka_scene.py` — sim **library**: scene/camera/ROS2-graph builders, `CAMERAS`,
  `WRIST_CAM_RES`/`SCENE_CAM_RES`, `CUBE_SPAWN_REGION`, `randomize_cube_pose()`,
  `get_rgba`/`save_rgba`. Imports only numpy/os at top so it stays import-safe.
- `load_franka_pickplace.py` — sim **app**: arg parsing, run loops, `main`, the rclpy
  `ResetListener` (`/episode/reset` → `pick_place.reset()` + randomize cube), `--control ros|auto`,
  `--headless`.

## Local Contracts
- **No `isaacsim`/`omni`/`pxr`/`usdrt` imports at module top** — Kit must boot (`SimulationApp`)
  first; import inside post-boot functions.
- **Runtime boundary:** this directory CANNOT import `teleop_arkit` (different Python). The only
  coupling is ROS2 topics, over DDS localhost, `ROS_DOMAIN_ID=0`, `rmw_fastrtps_cpp`.
- Publishes `/joint_states /wrist_cam/image_raw /scene_cam/image_raw /tf /clock`; subscribes
  `/joint_command` + `/episode/reset`.

## Work Guidance
- Cameras are GPU-render-bound (~18 Hz idle, ~5–7 Hz under teleop load) despite
  `Camera(frequency=20)`; the `.rrd` logs the native rate (the load-time grid resamples).
- Launch via the pixi `default`-env tasks, never `python isaac/...` directly.

## Verification
- `pixi run franka` (scene) · `franka-teleop` (`--control ros` — the recording/inference launch) ·
  `franka-headless`.

## Child DOX Index
None (leaf).
