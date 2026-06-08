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

def run_ros_controlled(sim_app, pick_place) -> int:
    """--control ros: the arm follows /joint_command; we just step + report."""
    from isaacsim.core.simulation_manager import SimulationManager

    pick_place.reset()  # sane initial pose, then ROS takes over
    print("[franka] ROS-controlled: arm follows /joint_command "
          "(publish sensor_msgs/JointState). Ctrl-C / close to stop.")
    step = 0
    while sim_app.is_running():
        if SimulationManager.is_simulating():
            if step % 120 == 0:
                q = pick_place.robot.get_dof_positions().numpy()[0]
                print(f"[franka] (ros) step {step:6d}  q={np.round(q, 3).tolist()}")
            step += 1
        sim_app.update()
    return 0


def run_autonomous(sim_app, pick_place, cams, args, save_dir, config_dir) -> int:
    """--control auto: run the FrankaPickPlace self-test for N cycles.

    Streams joint + camera data, saves a few first-cycle frames, and measures the
    panda_hand->grasp-TCP offset (cube center while grasped) -> tcp_offset.yaml.
    `cams` maps ros_name -> Camera object.
    """
    from isaacsim.core.simulation_manager import SimulationManager

    cycle = 0
    step = 0
    tcp_offset = None
    pick_place.reset()
    while sim_app.is_running() and cycle < args.cycles:
        if SimulationManager.is_simulating():
            pick_place.forward(args.ik_method)
            if pick_place.is_done():
                cycle += 1
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

    if tcp_offset is not None:
        fs.write_tcp_offset(os.path.join(config_dir, "tcp_offset.yaml"), tcp_offset)
    else:
        print("[franka] WARN: never captured a grasp; tcp_offset.yaml not written.")
    print("[franka] OK — pick-and-place ran, cube placed in bin, cameras streamed.")
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
    args = p.parse_args()
    if args.control == "ros":
        args.ros = True  # ROS-driven control needs the bridge up
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

        # Start physics, initialize cameras, then finalize scene-dependent setup.
        omni.timeline.get_timeline_interface().play()
        sim_app.update()
        for cam in (wrist_cam, scene_cam):
            cam.initialize()
            cam.add_motion_vectors_to_frame()
        fs.retarget_release_to_bin(sim_app, pick_place, bin_prim)
        fs.log_camera_aim([("wrist", wrist_prim), ("scene", scene_prim)])
        print("[franka] scene ready: Franka + cube + bin + wrist cam + scene cam.")

        # Dispatch on control mode.
        if args.control == "ros":
            return run_ros_controlled(sim_app, pick_place)
        return run_autonomous(
            sim_app, pick_place,
            cams={"wrist_cam": wrist_cam, "scene_cam": scene_cam},
            args=args, save_dir=save_dir, config_dir=config_dir)
    finally:
        sim_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
