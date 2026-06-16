"""Reusable scene construction for the Franka pick-and-place demo (Isaac Sim 6.0).

This is the **library** half — constants, pure helpers, scene builders, and the
ROS2 bridge graph. The runnable entry point is `load_franka_pickplace.py`, which
imports from here.

IMPORT RULE: nothing from isaacsim.* / omni.* / pxr / usdrt may be imported at
module top — Kit must boot (SimulationApp) before those packages exist. So every
such import lives INSIDE the function that runs it (post-boot). Only `os` / numpy
are top-level here, which makes this module safe to `import` before the app boots.
"""

from __future__ import annotations

import os

import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

# Cube geometry mirrors NVIDIA's FrankaPickPlace defaults (proven graspable):
# 51.5 mm cube vs an 80 mm gripper opening -> ~28 mm clearance.
CUBE_SIZE_M = 0.0515

# Bin: the KLT prop scaled DOWN so a 5 cm cube isn't lost in a 30 cm crate and the
# gripper can reach the interior. small_KLT ~0.30x0.20x0.147 m -> ~0.15x0.10x0.073
# at 0.5x. 6.0 reorganized asset paths, so we probe a couple of candidates.
BIN_SCALE = 0.5
BIN_XY = (0.0, 0.5)  # ground-plane position (matches FrankaPickPlace's target xy)
BIN_USD_CANDIDATES = [
    "/Isaac/Props/KLT_Bin/small_KLT.usd",
    "/Isaac/Props/KLT_Bin/small_KLT_visual.usd",
]

# Prim paths on the stage.
ROBOT_PATH = "/World/robot"
CUBE_PATH = "/World/Cube"            # FrankaPickPlace's default cube prim
BIN_PATH = "/World/Bin"
WRIST_CAM_PATH = ROBOT_PATH + "/panda_hand/wrist_cam"  # child of the hand -> tracks it
SCENE_CAM_PATH = "/World/scene_cam"

# Episode-reset cube spawn randomization (Phase 7): a small table box around the
# FrankaPickPlace default cube spawn (~0.5, 0.0), kept clear of the bin at (0, 0.5).
# Bin position stays fixed until after ACT validation.
CUBE_SPAWN_REGION = {"x": (0.42, 0.58), "y": (-0.12, 0.12)}

# Grasp stability: bind a high-friction physics material to the cube + fingertips
# so a pinch grip actually holds (the default material is too slippery), and raise
# the finger drive's force ceiling so it squeezes firmly.
GRASP_STATIC_FRICTION = 1.4
GRASP_DYNAMIC_FRICTION = 1.2
GRIPPER_MAX_FORCE = 200.0  # N — finger drive force ceiling (real Panda ~70 N)

# Wrist cam (offsets in the panda_hand LOCAL frame): the approach/grasp axis is
# local +Z (the hand points down through every phase), so the cam looks down +Z,
# offset to the side/back to clear the hand mesh.
WRIST_CAM_EYE = (0.10, 0.0, -0.04)
WRIST_CAM_TARGET = (0.0, 0.0, 0.10)  # the grasp TCP
WRIST_CAM_RES = (640, 480)

# Scene cam (world frame): a fixed front-right 3/4 view of the whole rig.
SCENE_CAM_EYE = (1.45, -1.05, 1.15)
SCENE_CAM_TARGET = (0.25, 0.25, 0.2)  # workspace centroid (between cube + bin)
SCENE_CAM_RES = (1280, 720)

# Cameras as (prim_path, ros_name, width, height) — used for both creation and the
# ROS2 graph so the two stay in sync.
CAMERAS = [
    (WRIST_CAM_PATH, "wrist_cam", WRIST_CAM_RES[0], WRIST_CAM_RES[1]),
    (SCENE_CAM_PATH, "scene_cam", SCENE_CAM_RES[0], SCENE_CAM_RES[1]),
]


# ============================================================================
# pure helpers (math / IO)
# ============================================================================

def url_exists(url: str) -> bool:
    """Return True if a USD URL resolves (local cache or cloud asset root)."""
    import omni.client  # available once Kit has booted

    result, _entry = omni.client.stat(url)
    return result == omni.client.Result.OK


def look_at_basis(eye, target, up_hint=(0.0, 0.0, 1.0)):
    """Orthonormal basis (x, y, z) for a USD camera at `eye` looking at `target`.

    USD cameras look down their local -Z with +Y up, so z = (eye - target) is the
    camera's +Z (pointing away from the target) and x/y complete a right-handed
    frame. eye/target are in the camera prim's PARENT frame, so the basis can be
    baked directly as a LOCAL transform.
    """
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up_hint))
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:  # up parallel to view dir; pick another up
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def quat_to_rotmat(q):
    """3x3 rotation matrix from a scalar-first quaternion (w, x, y, z)."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def save_rgba(rgba, path) -> bool:
    """Best-effort save of an HxWx(3/4) array to PNG, for offline inspection."""
    try:
        from PIL import Image

        Image.fromarray(np.asarray(rgba)[:, :, :3].astype("uint8")).save(path)
        return True
    except Exception as exc:  # diagnostic only — never fail the run over this
        print(f"[franka] WARN: could not save {path}: {exc}")
        return False


def write_tcp_offset(path, measured) -> None:
    """Write the panda_hand->grasp-TCP offset to YAML (no pyyaml dependency)."""
    t = measured["translation"]
    text = f"""# config/tcp_offset.yaml  (generated by isaac/load_franka_pickplace.py)
#
# panda_hand -> grasp-TCP offset, MEASURED from the scene: the cube center while
# grasped == the grasp point between the fingers. Phase 4 uses this as the
# end-effector for Pinocchio IK (set IK EE = panda_hand, apply this offset so the
# controlled point is the grasp TCP, not the flange).
#
# Convention: translation in the panda_hand LOCAL frame (meters); rotation as a
# quaternion (w, x, y, z) relative to panda_hand (identity = TCP shares the hand's
# orientation, the standard panda_hand_tcp convention).
parent_frame: panda_hand
tcp_offset:
  translation: [{t[0]}, {t[1]}, {t[2]}]
  rotation_wxyz: [1.0, 0.0, 0.0, 0.0]
measured:
  source: cube-center-at-grasp
  hand_world_pos: {measured['hand_world_pos']}
  hand_world_quat_wxyz: {measured['hand_world_quat_wxyz']}
  cube_world_pos: {measured['cube_world_pos']}
reference:
  # Canonical Franka panda_hand -> panda_hand_tcp is ~0.1034 m along +Z; the
  # measured translation z above should land near this as a sanity check.
  franka_panda_hand_tcp_z_m: 0.1034
"""
    with open(path, "w") as f:
        f.write(text)
    print(f"[franka] wrote TCP offset -> {path}")


# ============================================================================
# ROS2 bridge action graph
# ============================================================================

def build_ros2_graph(robot_path, cameras, domain_id, subscribe=False):
    """Build the isaacsim.ros2.bridge OmniGraph action graph (one /ROS2Graph).

    Publishes /clock, /joint_states, /tf for the articulation, and an RGB +
    camera_info stream per camera. Node types/wiring follow the binary's own
    examples (standalone_examples/api/isaacsim.ros2.bridge/{moveit,camera_periodic}.py).

    When `subscribe` is True, also subscribe to /joint_command and drive the
    articulation from it (--control ros). Otherwise the graph only publishes and
    the joints are moved by the autonomous self-test.
    """
    import omni.graph.core as og
    import usdrt.Sdf

    keys = og.Controller.Keys

    # --- always-on: clock + joint states + TF -----------------------------
    nodes = [
        ("OnTick", "omni.graph.action.OnPlaybackTick"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("ReadJointState", "isaacsim.sensors.physics.IsaacReadJointState"),
        ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
    ]
    connect = [
        ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
        ("OnTick.outputs:tick", "ReadJointState.inputs:execIn"),
        ("OnTick.outputs:tick", "PublishJointState.inputs:execIn"),
        ("OnTick.outputs:tick", "PublishTF.inputs:execIn"),
        ("Context.outputs:context", "PublishClock.inputs:context"),
        ("Context.outputs:context", "PublishJointState.inputs:context"),
        ("Context.outputs:context", "PublishTF.inputs:context"),
        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
        ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
        ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
        ("ReadJointState.outputs:execOut", "PublishJointState.inputs:execIn"),
        ("ReadJointState.outputs:jointNames", "PublishJointState.inputs:jointNames"),
        ("ReadJointState.outputs:jointPositions", "PublishJointState.inputs:jointPositions"),
        ("ReadJointState.outputs:jointVelocities", "PublishJointState.inputs:jointVelocities"),
        ("ReadJointState.outputs:jointEfforts", "PublishJointState.inputs:jointEfforts"),
        ("ReadJointState.outputs:jointDofTypes", "PublishJointState.inputs:jointDofTypes"),
        ("ReadJointState.outputs:stageMetersPerUnit", "PublishJointState.inputs:stageMetersPerUnit"),
        ("ReadJointState.outputs:sensorTime", "PublishJointState.inputs:sensorTime"),
    ]
    values = [
        ("Context.inputs:domain_id", int(domain_id)),
        ("ReadJointState.inputs:prim", [usdrt.Sdf.Path(robot_path)]),
        ("PublishJointState.inputs:topicName", "joint_states"),
        ("PublishTF.inputs:topicName", "/tf"),
        ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(robot_path)]),
    ]

    # --- one render-product + RGB + camera_info publisher per camera -------
    for prim_path, ros_name, width, height in cameras:
        rp, rgb, info = f"RP_{ros_name}", f"CamRgb_{ros_name}", f"CamInfo_{ros_name}"
        nodes += [
            (rp, "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            (rgb, "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (info, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),  # NOT ROS2CameraHelper
        ]
        connect += [
            ("OnTick.outputs:tick", f"{rp}.inputs:execIn"),
            (f"{rp}.outputs:execOut", f"{rgb}.inputs:execIn"),
            (f"{rp}.outputs:execOut", f"{info}.inputs:execIn"),
            (f"{rp}.outputs:renderProductPath", f"{rgb}.inputs:renderProductPath"),
            (f"{rp}.outputs:renderProductPath", f"{info}.inputs:renderProductPath"),
            ("Context.outputs:context", f"{rgb}.inputs:context"),
            ("Context.outputs:context", f"{info}.inputs:context"),
        ]
        values += [
            (f"{rp}.inputs:cameraPrim", [usdrt.Sdf.Path(prim_path)]),
            (f"{rp}.inputs:width", int(width)),
            (f"{rp}.inputs:height", int(height)),
            (f"{rgb}.inputs:frameId", ros_name),
            (f"{rgb}.inputs:topicName", f"{ros_name}/image_raw"),
            (f"{rgb}.inputs:type", "rgb"),
            (f"{info}.inputs:frameId", ros_name),
            (f"{info}.inputs:topicName", f"{ros_name}/camera_info"),
        ]

    # --- optional: subscribe /joint_command -> articulation ----------------
    if subscribe:
        nodes += [
            ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
        ]
        connect += [
            ("OnTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
            ("OnTick.outputs:tick", "ArticulationController.inputs:execIn"),
            ("Context.outputs:context", "SubscribeJointState.inputs:context"),
            ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
        ]
        values += [
            ("SubscribeJointState.inputs:topicName", "joint_command"),
            ("ArticulationController.inputs:robotPath", robot_path),
        ]

    og.Controller.edit(
        {"graph_path": "/ROS2Graph", "evaluator_name": "execution"},
        {keys.CREATE_NODES: nodes, keys.CONNECT: connect, keys.SET_VALUES: values},
    )
    extra = " + /joint_command->articulation" if subscribe else ""
    print(f"[franka] ROS2 bridge graph built (/clock, /joint_states, /tf, cameras{extra}).")


# ============================================================================
# scene builders (post-boot)
# ============================================================================

def build_scene():
    """Build the Franka + cube + bin scene; return (pick_place, bin_prim).

    Reuses NVIDIA's FrankaPickPlace (robot + cube + ground + dome light), then
    references our scaled-down KLT bin under the release point. Raises SystemExit
    if the Isaac asset library / bin USD can't be resolved.
    """
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.robot.experimental.manipulators.examples.franka import FrankaPickPlace
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, UsdGeom

    # Franka + cube (+ ground + dome light). Release above the bin so it drops in.
    pick_place = FrankaPickPlace()
    pick_place.setup_scene(
        target_position=np.array([BIN_XY[0], BIN_XY[1], 0.14]),
        cube_size=np.array([CUBE_SIZE_M] * 3),
        robot_path=ROBOT_PATH,
    )

    # Resolve + reference the bin, scaled and placed on the ground at BIN_XY.
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("[franka] ERROR: could not resolve the Isaac assets root "
              "(check connection / asset cache).")
        raise SystemExit(2)
    bin_usd = next((assets_root + rel for rel in BIN_USD_CANDIDATES
                    if url_exists(assets_root + rel)), None)
    if bin_usd is None:
        print("[franka] ERROR: none of the candidate KLT bin USD paths resolved:")
        for rel in BIN_USD_CANDIDATES:
            print(f"           {assets_root + rel}")
        raise SystemExit(3)
    print(f"[franka] referencing bin USD: {bin_usd}")
    bin_prim = stage_utils.add_reference_to_stage(usd_path=bin_usd, path=BIN_PATH)
    xform = UsdGeom.XformCommonAPI(bin_prim)
    xform.SetScale(Gf.Vec3f(BIN_SCALE, BIN_SCALE, BIN_SCALE))
    xform.SetTranslate(Gf.Vec3d(BIN_XY[0], BIN_XY[1], 0.0))

    apply_grasp_friction()
    boost_gripper_grip()
    return pick_place, bin_prim


def boost_gripper_grip(max_force=GRIPPER_MAX_FORCE):
    """Raise the Panda finger drive force ceiling so the grip squeezes firmly.

    Position-controlled fingers exert force ∝ stiffness·(target−actual), capped by
    the drive's maxForce. We raise maxForce (never lower it) and print the current
    stiffness/damping so we can tune further if a grasp still slips.
    """
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Usd, UsdPhysics

    stage = stage_utils.get_current_stage()
    found = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH)):
        if prim.GetName() in ("panda_finger_joint1", "panda_finger_joint2"):
            drive = UsdPhysics.DriveAPI.Get(prim, "linear")
            if not prim.HasAPI(UsdPhysics.DriveAPI, "linear"):
                drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
            cur_f = drive.GetMaxForceAttr().Get()
            cur_k = drive.GetStiffnessAttr().Get()
            cur_d = drive.GetDampingAttr().Get()
            new_f = max(float(max_force), float(cur_f) if cur_f else 0.0)
            drive.CreateMaxForceAttr().Set(new_f)
            found.append((prim.GetName(), cur_f, cur_k, cur_d, new_f))

    for name, f, k, d, nf in found:
        print(f"[franka] {name} drive: maxForce {f}->{nf}  (stiffness={k}, damping={d})")
    if not found:
        print("[franka] WARN: finger joints not found for grip-force boost")


def apply_grasp_friction(static_f=GRASP_STATIC_FRICTION, dynamic_f=GRASP_DYNAMIC_FRICTION):
    """Bind a high-friction physics material to the cube + gripper fingertips.

    The default contact material is too slippery for a stable pinch grasp, so a
    light cube slides out. We define one PhysicsMaterial and bind it (physics
    purpose) to the cube and both Panda finger links; binding is inherited by
    their colliders. Finger prim paths vary by USD, so we find them by name.
    """
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Usd, UsdPhysics, UsdShade

    stage = stage_utils.get_current_stage()
    mat = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/grip")
    UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi = UsdPhysics.MaterialAPI(mat.GetPrim())
    mapi.CreateStaticFrictionAttr().Set(float(static_f))
    mapi.CreateDynamicFrictionAttr().Set(float(dynamic_f))
    mapi.CreateRestitutionAttr().Set(0.0)

    targets = [stage.GetPrimAtPath(CUBE_PATH)]
    robot_root = stage.GetPrimAtPath(ROBOT_PATH)
    for prim in Usd.PrimRange(robot_root):
        if prim.GetName() in ("panda_leftfinger", "panda_rightfinger"):
            targets.append(prim)

    for prim in targets:
        if not prim or not prim.IsValid():
            continue
        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(
            mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics")
    names = [p.GetName() for p in targets if p and p.IsValid()]
    print(f"[franka] grasp friction (μs={static_f}, μd={dynamic_f}) bound to: {names}")


def add_camera(prim_path, eye, target, resolution, focal,
               near=0.01, far=1.0e6, up_hint=(0.0, 0.0, 1.0)):
    """Create a Camera and bake its transform onto the prim; return (cam, prim).

    We DON'T use the Camera ctor's position/orientation — those are applied in
    WORLD space, so a child (wrist) cam wouldn't track its parent. Instead we bake
    the transform relative to the prim's PARENT, and fix two USD defaults that
    otherwise blank the view: the 1.0 m near-clip (a close-up cam needs ~0.01 m)
    and a too-narrow lens.
    """
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.sensors.camera import Camera
    from pxr import Gf, UsdGeom

    cam = Camera(prim_path=prim_path, frequency=20, resolution=resolution)
    eye = np.asarray(eye, float)
    bx, by, bz = look_at_basis(eye, np.asarray(target, float), up_hint)

    prim = stage_utils.get_current_stage().GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()  # drop the ctor's ops; set our own
    xf.AddTransformOp().Set(Gf.Matrix4d(  # USD row-vector convention: basis as rows
        float(bx[0]), float(bx[1]), float(bx[2]), 0.0,
        float(by[0]), float(by[1]), float(by[2]), 0.0,
        float(bz[0]), float(bz[1]), float(bz[2]), 0.0,
        float(eye[0]), float(eye[1]), float(eye[2]), 1.0,
    ))
    ucam = UsdGeom.Camera(prim)
    ucam.GetClippingRangeAttr().Set(Gf.Vec2f(near, far))
    ucam.GetFocalLengthAttr().Set(focal)
    return cam, prim


def retarget_release_to_bin(sim_app, pick_place, bin_prim, warmup=20):
    """Aim FrankaPickPlace's release at the bin's TRUE world-bbox center.

    The KLT asset pivot is offset, so placing the prim at BIN_XY does NOT center it
    there. We warm up a few frames so the bin payload streams in + renders, read
    its world bounding box, and release ABOVE the rim so the cube clears the walls
    and drops in rather than catching the edge.
    """
    from pxr import Usd, UsdGeom

    for _ in range(warmup):
        sim_app.update()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bin_range = bbox_cache.ComputeWorldBound(bin_prim).ComputeAlignedRange()
    bmin, bmax = bin_range.GetMin(), bin_range.GetMax()
    pick_place.target_position = np.array([
        (bmin[0] + bmax[0]) / 2.0,   # bin center x
        (bmin[1] + bmax[1]) / 2.0,   # bin center y
        bmax[2] + 0.14,              # rim top + clearance so the cube drops in
    ])
    print(f"[franka] bin world bbox min={tuple(round(v, 3) for v in bmin)} "
          f"max={tuple(round(v, 3) for v in bmax)}")
    print(f"[franka] release target -> {np.round(pick_place.target_position, 3).tolist()}")


def log_camera_aim(named_prims):
    """Print each camera's world position + optical axis (a USD cam looks down -Z).

    `named_prims` is a list of (label, prim). An optical axis near (0,0,-1) means
    the camera is looking straight down at the table.
    """
    from pxr import Gf, UsdGeom

    xc = UsdGeom.XformCache()
    for label, prim in named_prims:
        l2w = xc.GetLocalToWorldTransform(prim)
        optical = l2w.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
        pos = l2w.ExtractTranslation()
        print(f"[franka] {label} cam world pos={tuple(round(v, 3) for v in pos)} "
              f"optical_axis={tuple(round(v, 2) for v in optical)}")


def randomize_cube_pose(pick_place, region=CUBE_SPAWN_REGION):
    """Episode reset: home the arm (+ open gripper) and respawn the cube at a random
    (x, y) in `region` (z = its initial height), with zero velocity. Bin stays fixed.

    Reuses NVIDIA's FrankaPickPlace.reset(cube_position=...) (homes the robot via
    reset_to_default_pose + places the cube), then zeros the cube's velocity so it can't
    carry momentum from being mid-carry or resting in the bin. Returns the chosen (x, y, z).
    """
    z = float(pick_place.cube_initial_position[2])
    x = float(np.random.uniform(*region["x"]))
    y = float(np.random.uniform(*region["y"]))
    pos = np.array([x, y, z], dtype=float)
    pick_place.reset(cube_position=pos)  # home arm + open gripper + place cube
    try:
        pick_place.cube.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
    except Exception as exc:  # backend may not support it this frame — non-fatal
        print(f"[franka] WARN: could not zero cube velocity: {exc}")
    print(f"[franka] episode reset: cube spawned at ({x:.3f}, {y:.3f}), arm homed")
    return pos
