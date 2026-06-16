"""teleop_arkit.core.robot — the Franka Panda robot spec (single source of truth).

Joint names, gripper convention, and the home pose. This is the one "RobotSpec" the whole
pipeline imports; retarget to another arm by swapping these (robot-agnostic by construction).
Previously these lived in joint_command_node.py (consts) + record_rrd.py (GRIPPER_JOINT).
"""
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
GRIPPER_JOINT = FINGER_JOINTS[0]          # the two fingers mirror; track one as the gripper DOF
GRIPPER_OPEN, GRIPPER_CLOSED = 0.04, 0.0
# Franka "ready" home for the 7 arm joints — matches the sim's reset_to_default_pose, so an
# episode reset commands the SAME home Isaac teleports to (no controller fight).
HOME_ARM_Q = [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741]


def default_panda_urdf() -> str:
    """Path to the example-robot-data Panda URDF inside the active conda env."""
    import os

    return os.path.join(
        os.environ["CONDA_PREFIX"],
        "share/example-robot-data/robots/panda_description/urdf/panda.urdf",
    )

