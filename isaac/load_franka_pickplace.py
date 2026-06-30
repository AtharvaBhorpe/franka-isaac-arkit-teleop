"""Runnable entry — Franka pick-and-place scene in Isaac Sim 6.0 (standalone binary).

A stock **Franka Panda** + a 5 cm **cube** + a scaled-down **bin** + a **wrist
camera** + a fixed **scene camera**, with an optional **ROS2 bridge**.

Run from the repo root via the binary wrapper (NOT a pip isaacsim):
    pixi run franka                  # autonomous self-test, GUI
    pixi run franka-headless         # autonomous self-test, no viewport
    pixi run franka-ros              # + ROS2 bridge (publish telemetry)
    ./scripts/run_isaac.sh isaac/load_franka_pickplace.py --control ros   # ROS-driven

Two control modes (--control):
  * auto (default) — run NVIDIA's validated FrankaPickPlace self-test: grasp the
    cube, drop it in the bin, N cycles. Also measures the panda_hand->grasp-TCP
    offset and writes config/tcp_offset.yaml.
  * ros — no autonomous motion; the arm follows /joint_command via the bridge.

Structure: scene construction (constants, helpers, builders, ROS2 graph) lives in
`franka_scene.py` (the library). THIS file is the app: argument parsing, the two
run loops, and `main` (boot Kit -> assemble -> dispatch). Start reading at main.

Hard rule (see franka_scene.py): nothing from isaacsim.* / omni.* / pxr / usdrt
may be imported before SimulationApp is constructed, so those imports live inside
post-boot functions. `franka_scene` only pulls numpy/os at module level, so it is
safe to import here at the top.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# Accept the Omniverse EULA non-interactively (also set by run_isaac.sh). Must be
# set before importing SimulationApp.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import franka_scene as fs  # noqa: E402  (sibling module; safe — no isaacsim at import)


# ============================================================================
# run loops (post-boot)
# ============================================================================

def _make_reset_listener():
    """Tiny rclpy node that flags when std_msgs/Empty arrives on /episode/reset.

    Isaac's bundled Python ships rclpy (cf. standalone_examples/.../ros2.bridge/subscriber.py),
    so we can react to a ROS topic with custom Python here, alongside the OmniGraph bridge.
    Returns (rclpy_module, node), or (None, None) if rclpy can't be set up — in which case the
    sim still runs and only /episode/reset is disabled.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Empty

        if not rclpy.ok():
            rclpy.init()

        class _ResetListener(Node):
            def __init__(self):
                super().__init__("episode_reset_listener")
                self._flag = False
                self.create_subscription(Empty, "/episode/reset", self._on, 1)

            def _on(self, _msg):
                self._flag = True

            def pop(self) -> bool:
                f, self._flag = self._flag, False
                return f

        return rclpy, _ResetListener()
    except Exception as exc:  # never let reset wiring crash the sim
        print(f"[franka] WARN: /episode/reset disabled (rclpy unavailable): {exc}")
        return None, None


def _make_record_node():
    """rclpy node that drives the `record` recorder for hands-off auto data collection.

    Publishes episode commands on /record/command (String: s/e/f) and mirrors the
    expert's joint targets onto /joint_command (JointState) so the recorder captures
    the action stream exactly as it does for human teleop. Returns (rclpy, node) or
    (None, None) if rclpy is unavailable (recording then disabled; sim still runs).
    """
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init()

        class _RecordBridge(Node):
            def __init__(self):
                super().__init__("auto_record_bridge")
                self._cmd = self.create_publisher(String, "/record/command", 10)
                self._jc = self.create_publisher(JointState, "/joint_command", 10)

            def _send(self, s: str):
                m = String()
                m.data = s
                self._cmd.publish(m)

            def start(self):
                self._send("s")

            def end(self, success: bool):
                self._send("e" if success else "f")

            def joint_command(self, names, positions):
                # ponytail: achieved dof positions stand in for the commanded target — the
                # experimental Articulation doesn't expose its applied action, and the dataset
                # samples the next-window position as the action, so the trajectory is the signal.
                js = JointState()
                js.name = list(names)
                js.position = [float(v) for v in positions]
                self._jc.publish(js)

        return rclpy, _RecordBridge()
    except Exception as exc:
        print(f"[franka] WARN: --record disabled (rclpy unavailable): {exc}")
        return None, None


def _make_tactile(robot_path, filter_path, mode="normal", debug=False):
    """Build the tactile grid + a /tactile/image_raw publisher. Returns a bundle
    (grid, rclpy, pub_node) consumed by _tick_tactile / _close_tactile, or None if
    setup fails (sim still runs; tactile just disabled)."""
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        import tactile

        if not rclpy.ok():
            rclpy.init()
        grid = tactile.TactileGrid(robot_path, filter_path, mode=mode, debug=debug)

        class _TactilePub(Node):
            def __init__(self):
                super().__init__("tactile_pub")
                self._pub = self.create_publisher(Image, "/tactile/image_raw", 1)

            def publish(self, rgb):
                m = Image()
                m.height, m.width = int(rgb.shape[0]), int(rgb.shape[1])
                m.encoding = "rgb8"
                m.is_bigendian = 0
                m.step = int(rgb.shape[1]) * 3
                m.data = rgb.tobytes()
                self._pub.publish(m)

        print(f"[franka] tactile ON: {tactile.GRID_ROWS}x{tactile.GRID_COLS} force field -> "
              f"/tactile/image_raw (add `tactile=/tactile/image_raw` to the recorder --cameras).")
        return (grid, rclpy, _TactilePub())
    except Exception as exc:
        print(f"[franka] WARN: --tactile disabled: {exc}")
        return None


def _tick_tactile(bundle):
    if bundle is None:
        return
    grid, _, pub = bundle
    if grid.failed:
        return
    grid.frame += 1
    if grid.frame % grid.every:                     # throttle the GPU contact read (~20 Hz)
        return
    try:
        pub.publish(grid.image())
    except Exception as exc:                         # never let a contact-API error freeze/kill the sim
        grid.failed = True
        print(f"[tactile] disabled after read error: {exc}", flush=True)
        return
    if grid.debug:                                   # print the pad-local contact coords ONCE per grasp
        has = grid.last[0] > 0
        if has and not grid._had_contact:
            n, mn, ms = grid.last
            print(f"[tactile] GRASP: contacts={n}  max_normal={mn:.2f}N  max_shear={ms:.2f}N",
                  flush=True)
            for pad_i, loc in grid._dbg_locals[:6]:  # calibration data: where contacts land in pad-local frame
                print(f"[tactile]   pad{pad_i} local(x,y,z)={[round(v, 4) for v in loc]}", flush=True)
        grid._had_contact = has


def _close_tactile(bundle):
    if bundle is not None:
        bundle[2].destroy_node()


def run_ros_controlled(sim_app, pick_place, tactile=None) -> int:
    """--control ros: the arm follows /joint_command; we just step + report.

    Also listens on /episode/reset (std_msgs/Empty): each request homes the arm and
    respawns the cube at a random spot (fs.randomize_cube_pose) for episode-by-episode
    data collection. The IK node re-seeds off the same topic so it won't fight the reset.
    """
    from isaacsim.core.simulation_manager import SimulationManager

    fs.randomize_cube_pose(pick_place)  # clean, randomized start
    rclpy, reset_node = _make_reset_listener()
    print("[franka] ROS-controlled: arm follows /joint_command (sensor_msgs/JointState). "
          "Publish std_msgs/Empty on /episode/reset to home+respawn. Ctrl-C / close to stop.")
    step = 0
    while sim_app.is_running():
        if reset_node is not None:
            rclpy.spin_once(reset_node, timeout_sec=0.0)
            if reset_node.pop():
                fs.randomize_cube_pose(pick_place)
                for _ in range(10):  # let physics settle before logging resumes
                    sim_app.update()
        if SimulationManager.is_simulating():
            _tick_tactile(tactile)
            if step % 120 == 0:
                q = pick_place.robot.get_dof_positions().numpy()[0]
                print(f"[franka] (ros) step {step:6d}  q={np.round(q, 3).tolist()}")
            step += 1
        sim_app.update()
    if reset_node is not None:
        reset_node.destroy_node()
    _close_tactile(tactile)
    return 0


def run_eval(sim_app, pick_place, bin_prim, args, tactile=None) -> int:
    """--control ros --eval N: measure CLOSED-LOOP policy success rate over N trials.

    The trained policy (the ros-env `infer` node) drives /joint_command; here we just
    randomize the cube, give the policy a fixed time budget, then score cube-in-bin —
    the same auto-label as data collection. Run two policies (e.g. with vs without the
    tactile camera) through this to compare success rates: the modality ablation.
    """
    from isaacsim.core.simulation_manager import SimulationManager

    bmin, bmax = fs.bin_world_aabb(bin_prim)
    print(f"[franka] EVAL: {args.eval_trials} trials × {args.eval_secs:.0f}s — the `infer` policy "
          f"drives /joint_command; scoring cube-in-bin. Start `infer` in the ros env now.")
    successes, trial = 0, 0
    while sim_app.is_running() and trial < args.eval_trials:
        fs.randomize_cube_pose(pick_place)                  # homes arm + randomizes cube
        for _ in range(args.settle_steps):                  # settle before the policy acts
            sim_app.update()
        for _ in range(int(args.eval_secs * 60)):           # policy's time budget (~60 Hz physics)
            if not sim_app.is_running():
                break
            if SimulationManager.is_simulating():
                _tick_tactile(tactile)
            sim_app.update()
        trial += 1
        cube_p = pick_place.cube.get_world_poses()[0].numpy()[0]
        ok = fs.cube_in_bin(cube_p, bmin, bmax)
        successes += int(ok)
        print(f"[franka] eval {trial}/{args.eval_trials}: {'SUCCESS' if ok else 'fail   '} "
              f"-> {successes}/{trial} ({successes / trial:.0%})  cube={np.round(cube_p, 3).tolist()}")
    _close_tactile(tactile)
    rate = successes / max(1, args.eval_trials)
    print(f"[franka] ===== EVAL DONE: {successes}/{args.eval_trials} = {rate:.1%} success =====")
    return 0


def run_autonomous(sim_app, pick_place, cams, args, save_dir, config_dir, bin_prim=None,
                   tactile=None) -> int:
    """--control auto: run the FrankaPickPlace self-test for N cycles.

    Streams joint + camera data, saves a few first-cycle frames, and measures the
    panda_hand->grasp-TCP offset (cube center while grasped) -> tcp_offset.yaml.
    `cams` maps ros_name -> Camera object.

    With --record: each cycle becomes a recorded episode — randomize the cube, signal
    the `record` node to start (s), drive the expert while mirroring its joint targets
    onto /joint_command, then label by whether the cube landed in the bin (e/f). Run the
    recorder with `--settle-secs 0` (this loop owns settle). Hands-off dataset generation.
    """
    from isaacsim.core.simulation_manager import SimulationManager

    rclpy, rec = (None, None)
    bmin = bmax = None
    if args.record:
        rclpy, rec = _make_record_node()
        bmin, bmax = fs.bin_world_aabb(bin_prim)
        print(f"[franka] auto-record ON: /joint_command + /record/command; "
              f"bin footprint x={bmin[0]:.2f}..{bmax[0]:.2f} y={bmin[1]:.2f}..{bmax[1]:.2f} "
              f"rim_z={bmax[2]:.2f}. Run `record --settle-secs 0` in the ros env.")

    def begin_episode():
        fs.randomize_cube_pose(pick_place)         # randomized cube + home arm (resets the state machine)
        if rec:
            rec.start()                            # "s" — recorder begins logging immediately
            for _ in range(args.settle_steps):     # homing + physics settle, logged as a short static prefix
                sim_app.update()

    cycle = 0
    step = 0
    tcp_offset = None
    begin_episode() if args.record else pick_place.reset()
    while sim_app.is_running() and cycle < args.cycles:
        if SimulationManager.is_simulating():
            pick_place.forward(args.ik_method)
            _tick_tactile(tactile)
            if rec:                                # mirror the expert's joint targets -> /joint_command
                rec.joint_command(pick_place.robot.dof_names,
                                  pick_place.robot.get_dof_positions().numpy()[0])
            if pick_place.is_done():
                cycle += 1
                if rec:
                    cube_p = pick_place.cube.get_world_poses()[0].numpy()[0]
                    ok = fs.cube_in_bin(cube_p, bmin, bmax)
                    rec.end(ok)                    # "e" (in bin) / "f" (missed)
                    print(f"[franka] cycle {cycle}/{args.cycles}: "
                          f"{'SUCCESS' if ok else 'FAILURE'}  cube={np.round(cube_p, 3).tolist()}")
                    if cycle < args.cycles:
                        for _ in range(args.gap_steps):  # let the recorder finalize the .rrd
                            sim_app.update()
                        begin_episode()
                else:
                    print(f"[franka] cycle {cycle}/{args.cycles} complete (cube placed).")
                    if cycle < args.cycles:
                        pick_place.reset()

            if step % 60 == 0:
                q = pick_place.robot.get_dof_positions().numpy()[0]
                hand_pose = pick_place.robot.end_effector_link.get_world_poses()
                hand_p = hand_pose[0].numpy()[0]
                hand_q = hand_pose[1].numpy()[0]  # (w, x, y, z)
                cube_p = pick_place.cube.get_world_poses()[0].numpy()[0]
                print(f"[franka] step {step:5d}  q={np.round(q, 3).tolist()}")
                print(f"[franka]   hand_pos={np.round(hand_p, 3).tolist()}  "
                      f"cube_pos={np.round(cube_p, 3).tolist()}")

                # Save a few first-cycle frames (approach/grasp/lift) for inspection.
                if step in (60, 120, 180):
                    for name, cam in cams.items():
                        frame = cam.get_rgba()
                        if frame is not None:
                            out = os.path.join(save_dir, f"{name}_step{step:03d}.png")
                            if fs.save_rgba(frame, out):
                                print(f"[franka] saved {name} frame -> {out}")

                # TCP offset: when the gripper is closed on the cube and the cube
                # is lifted, the cube center == the grasp TCP. Express it in the
                # panda_hand local frame.
                if tcp_offset is None and q[7] < 0.035 and cube_p[2] > 0.15:
                    p_local = fs.quat_to_rotmat(hand_q).T @ (cube_p - hand_p)
                    tcp_offset = {
                        "translation": [round(float(v), 5) for v in p_local],
                        "hand_world_pos": [round(float(v), 5) for v in hand_p],
                        "hand_world_quat_wxyz": [round(float(v), 5) for v in hand_q],
                        "cube_world_pos": [round(float(v), 5) for v in cube_p],
                    }
                    print("[franka] measured panda_hand->TCP offset (local) = "
                          f"{tcp_offset['translation']}")
            step += 1
        sim_app.update()
        if rec is not None:
            rclpy.spin_once(rec, timeout_sec=0.0)   # pump the publishers

    if rec is not None:
        rec.destroy_node()
    _close_tactile(tactile)
    if not args.record:  # the committed TCP calibration is for the self-test, not data runs
        if tcp_offset is not None:
            fs.write_tcp_offset(os.path.join(config_dir, "tcp_offset.yaml"), tcp_offset)
        else:
            print("[franka] WARN: never captured a grasp; tcp_offset.yaml not written.")
    print(f"[franka] OK — auto pick-and-place ran {cycle} cycle(s)"
          f"{' (recorded)' if args.record else ''}.")
    return 0


# ============================================================================
# main — boot Kit, assemble the scene, dispatch
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--headless", action="store_true", help="Run without the GUI viewport.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="Physics device. Default cpu matches NVIDIA's validated example.")
    p.add_argument("--ik-method", default="damped-least-squares",
                   choices=["singular-value-decomposition", "pseudoinverse",
                            "transpose", "damped-least-squares"],
                   help="Differential-IK method for the autonomous self-test.")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index for rendering (the RTX dGPU is usually 0).")
    p.add_argument("--cycles", type=int, default=3,
                   help="Autonomous pick-and-place cycles to run (--control auto).")
    p.add_argument("--ros", action="store_true",
                   help="Enable the ROS2 bridge (publish /clock /joint_states /tf + cameras).")
    p.add_argument("--ros-domain", type=int, default=0,
                   help="ROS_DOMAIN_ID for the bridge (match the RoboStack side).")
    p.add_argument("--control", choices=["auto", "ros"], default="auto",
                   help="auto: FrankaPickPlace self-test. ros: drive the arm from "
                        "/joint_command (implies --ros; no autonomous motion).")
    p.add_argument("--record", action="store_true",
                   help="(auto only) Hands-off data collection: randomize the cube each cycle, "
                        "drive the `record` node via /record/command, mirror joint targets onto "
                        "/joint_command, auto-label by cube-in-bin. Implies --ros.")
    p.add_argument("--settle-steps", type=int, default=90,
                   help="(--record) Sim steps to settle after reset before the pick (logged prefix).")
    p.add_argument("--gap-steps", type=int, default=120,
                   help="(--record) Sim steps between episodes for the recorder to finalize the .rrd.")
    p.add_argument("--tactile", action="store_true",
                   help="Publish a 32x12 gripper contact-force field (TacSL-style) on /tactile/image_raw. "
                        "Add it to the recorder --cameras to log it as a modality. Implies --ros.")
    p.add_argument("--tactile-mode", choices=["shear", "normal"], default="normal",
                   help="normal: per-pad jet-colormap force blob, FlexiTac-style (default). "
                        "shear: TacSL shear+normal image (R,G=shear, B=normal).")
    p.add_argument("--tactile-debug", action="store_true",
                   help="Print ~1 Hz contact count + max normal/shear force (first-run diagnostic).")
    p.add_argument("--eval", action="store_true",
                   help="Closed-loop policy success-rate eval: N trials, the `infer` policy drives, "
                        "score cube-in-bin. Forces --control ros. Run two policies to ablate a modality.")
    p.add_argument("--eval-trials", type=int, default=20, help="(--eval) number of trials.")
    p.add_argument("--eval-secs", type=float, default=12.0, help="(--eval) seconds the policy gets per trial.")
    args = p.parse_args()
    if args.eval:
        args.control = "ros"  # the policy drives /joint_command -> needs the subscribe graph
    if args.control == "ros" or args.record or args.tactile:
        args.ros = True  # ROS-driven control / recording / tactile publishing need the bridge up
    return args


def main() -> int:
    args = parse_args()

    # Repo-relative outputs: cam PNGs (gitignored) + tcp_offset.yaml (committed).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(repo_root, "outputs")
    config_dir = os.path.join(repo_root, "config")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Boot Kit FIRST. Force single-GPU rendering on the NVIDIA card — hybrid
    # laptops (RTX dGPU + AMD iGPU) crash the RTX renderer on its first frame when
    # Kit's multi-GPU path tries to span the unsupported iGPU.
    from isaacsim import SimulationApp

    sim_app = SimulationApp({
        "headless": args.headless,
        "multi_gpu": False,
        "active_gpu": args.gpu,
        "physics_gpu": args.gpu,
    })

    try:
        import omni.kit.app
        import omni.timeline
        from isaacsim.core.simulation_manager import SimulationManager

        # Enable extensions: the Franka example classes, and (if requested) the
        # ROS2 bridge — both must be enabled before importing/using their nodes.
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_mgr.set_extension_enabled_immediate(
            "isaacsim.robot.experimental.manipulators.examples", True)
        if args.ros:
            os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain)
            ext_mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
            sim_app.update()

        SimulationManager.set_physics_sim_device(args.device)
        sim_app.update()

        # Assemble the scene (library calls).
        pick_place, bin_prim = fs.build_scene()
        wrist_cam, wrist_prim = fs.add_camera(
            fs.WRIST_CAM_PATH, fs.WRIST_CAM_EYE, fs.WRIST_CAM_TARGET,
            fs.WRIST_CAM_RES, focal=12.0)
        scene_cam, scene_prim = fs.add_camera(
            fs.SCENE_CAM_PATH, fs.SCENE_CAM_EYE, fs.SCENE_CAM_TARGET,
            fs.SCENE_CAM_RES, focal=24.0)
        if args.ros:
            fs.build_ros2_graph(fs.ROBOT_PATH, fs.CAMERAS, args.ros_domain,
                                subscribe=(args.control == "ros"))
        if args.tactile:                       # enable PhysX contact reporting BEFORE physics init
            import tactile
            tactile.enable_contact_reporting(fs.ROBOT_PATH, fs.CUBE_PATH)

        # Start physics, initialize cameras, then finalize scene-dependent setup.
        omni.timeline.get_timeline_interface().play()
        sim_app.update()
        for cam in (wrist_cam, scene_cam):
            cam.initialize()
            cam.add_motion_vectors_to_frame()
        fs.retarget_release_to_bin(sim_app, pick_place, bin_prim)
        fs.log_camera_aim([("wrist", wrist_prim), ("scene", scene_prim)])
        tactile = (_make_tactile(fs.ROBOT_PATH, fs.CUBE_PATH, args.tactile_mode, args.tactile_debug)
                   if args.tactile else None)
        print("[franka] scene ready: Franka + cube + bin + wrist cam + scene cam"
              f"{' + tactile' if tactile else ''}.")

        # Dispatch on control mode.
        if args.control == "ros":
            if args.eval:
                return run_eval(sim_app, pick_place, bin_prim, args, tactile=tactile)
            return run_ros_controlled(sim_app, pick_place, tactile=tactile)
        return run_autonomous(
            sim_app, pick_place,
            cams={"wrist_cam": wrist_cam, "scene_cam": scene_cam},
            args=args, save_dir=save_dir, config_dir=config_dir, bin_prim=bin_prim,
            tactile=tactile)
    finally:
        sim_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
