"""teleop_arkit.core.schema — the `.rrd` obs/action SCHEMA (single source of truth).

The data CONTRACT shared by the recorder (writer), the dataset (reader), and inference
(consumer): which Rerun entities carry which streams, and how the state/action vectors are
built from a sensor_msgs/JointState. Previously the entity paths were literals in the writer
(record_rrd) AND constants in the reader (rrd_dataset) — a drift risk; the state/action
builders were duplicated in record_rrd + infer_node. Now there is one place. Add a modality =
add an entity here; nothing else moves.
"""
from teleop_arkit.core.robot import ARM_JOINTS, GRIPPER_JOINT, GRIPPER_OPEN

# Rerun entity paths (reader side; rr.log normalizes a leading "/").
STATE_ENTITY = "/observation/state"
ACTION_ENTITY = "/action"
GRIPPER_ENTITY = "/action/gripper_command"
TARGET_POSE_ENTITY = "/action/target_pose"
IMAGE_PREFIX = "/observation/images/"

# Writer side (the strings passed to rr.log) — the same paths without the leading slash.
STATE_LOG = STATE_ENTITY[1:]
ACTION_LOG = ACTION_ENTITY[1:]
GRIPPER_LOG = GRIPPER_ENTITY[1:]
TARGET_POSE_LOG = TARGET_POSE_ENTITY[1:]

# stats.json / normalization keys.
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def image_log(name: str) -> str:
    """rr.log entity for a camera frame: 'observation/images/<name>'."""
    return IMAGE_PREFIX[1:] + name


def by_name(msg, names):
    """sensor_msgs/JointState + ordered joint names -> [positions], or None if any missing."""
    d = dict(zip(msg.name, msg.position))
    return [float(d[n]) for n in names] if all(n in d for n in names) else None


def state_vec(msg, include_vel: bool = False):
    """JointState -> observation.state: 7 arm positions + gripper (+ 8 velocities if include_vel)."""
    arm = by_name(msg, ARM_JOINTS)
    if arm is None:
        return None
    d = dict(zip(msg.name, msg.position))
    state = arm + [float(d.get(GRIPPER_JOINT, GRIPPER_OPEN))]
    if include_vel and msg.velocity and len(msg.velocity) >= len(msg.name):
        dv = dict(zip(msg.name, msg.velocity))
        state += [float(dv.get(n, 0.0)) for n in ARM_JOINTS] + [float(dv.get(GRIPPER_JOINT, 0.0))]
    return state


def action_vec(msg):
    """JointState (/joint_command) -> action: 7 arm positions + gripper."""
    arm = by_name(msg, ARM_JOINTS)
    if arm is None:
        return None
    d = dict(zip(msg.name, msg.position))
    return arm + [float(d.get(GRIPPER_JOINT, GRIPPER_OPEN))]
