"""teleop_arkit.core — the shared CONTRACT (robot spec + .rrd schema + camera parsing).

One source of truth imported by the teleop nodes, the data pipeline, and inference — so the
recorder (writer), dataset (reader), and inference (consumer) can't drift. Research code: the
interface may change; not a stability-guaranteed public API.

    from teleop_arkit.core.robot import ARM_JOINTS, GRIPPER_OPEN, HOME_ARM_Q
    from teleop_arkit.core import schema          # entity paths + state_vec/action_vec/by_name
    from teleop_arkit.core.cameras import parse_cameras
"""
