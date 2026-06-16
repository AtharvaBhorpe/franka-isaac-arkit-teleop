"""teleop_arkit.core.rosutil — shared ROS2 spin/shutdown helper.

`rclpy.spin` exits differently per signal: Ctrl-C (SIGINT) raises KeyboardInterrupt /
ExternalShutdownException, but SIGTERM (kill, `timeout`, launch teardown) can invalidate the
context *while the executor is mid-wait* — surfacing a low-level `RCLError: context is not
valid` from the WaitSet. All of these are normal shutdown. We swallow them, but RE-RAISE any
exception that fires while the context is still valid, so genuine errors aren't hidden.
"""
import rclpy
from rclpy.executors import ExternalShutdownException


def spin(node):
    """rclpy.spin tolerant of SIGINT/SIGTERM shutdown races."""
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():        # context still valid -> a real error, not the shutdown race
            raise


def run(node):
    """spin(node) + standard clean teardown (destroy_node + idempotent try_shutdown)."""
    try:
        spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
