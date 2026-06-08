"""ROS2 node: target EE pose -> Pinocchio servo-IK -> /joint_command.

Closes the teleop control loop on the ROS side. It:
  * seeds joint state once from /joint_states (published by the Isaac bridge),
  * servos the arm toward a target pose with CartesianServoIK,
  * publishes /joint_command (sensor_msgs/JointState: 7 arm + 2 finger),
    which the Isaac bridge's ArticulationController executes.

Target source (--source):
  * demo  (default) — scripted poses + gripper toggle, for PHONE-FREE validation
    of the whole IK -> /joint_command -> sim arm loop.
  * topic — follow /target_frame (geometry_msgs/PoseStamped); this is the seam
    the ARKit receiver feeds in Phase 5.

Run (with Isaac up in --control ros, in another terminal):
    pixi run -e ros python -m teleop_arkit.joint_command_node            # demo
    pixi run -e ros python -m teleop_arkit.joint_command_node --source topic
"""

from __future__ import annotations

import argparse

import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from teleop_arkit.ik import CartesianServoIK, _default_panda_urdf

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
GRIPPER_OPEN, GRIPPER_CLOSED = 0.04, 0.0


class JointCommandNode(Node):
    def __init__(self, args):
        super().__init__("teleop_arkit_ik")
        self.ik = CartesianServoIK(
            args.urdf, ee_frame=args.ee_frame,
            kp_lin=args.kp_lin, kp_ang=args.kp_ang,
            max_lin_vel=args.max_lin_vel, max_ang_vel=args.max_ang_vel)
        self.dt = 1.0 / args.rate
        self.source = args.source

        # Map each commanded joint name -> its index in the Pinocchio q vector,
        # so we can seed q from /joint_states regardless of message ordering.
        self.qidx = {
            n: self.ik.model.joints[self.ik.model.getJointId(n)].idx_q
            for n in ARM_JOINTS + FINGER_JOINTS
        }
        self.seeded = False
        self.target: pin.SE3 | None = None
        self.gripper = GRIPPER_OPEN

        # Demo state.
        self._demo_targets: list[pin.SE3] = []
        self._demo_grips: list[float] = []
        self._demo_i = 0

        self.create_subscription(JointState, args.joint_state_topic, self._on_joint_states, 1)
        if self.source == "topic":
            self.create_subscription(PoseStamped, args.target_topic, self._on_target, 1)
            self.create_subscription(Float64, args.gripper_topic, self._on_gripper, 1)
        self.cmd_pub = self.create_publisher(JointState, args.command_topic, 1)
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"teleop_arkit IK node up: ee={args.ee_frame} rate={args.rate}Hz source={self.source}")

    # -- inputs ------------------------------------------------------------
    def _on_joint_states(self, msg: JointState):
        if self.seeded:
            return
        q = self.ik.q.copy()
        for name, pos in zip(msg.name, msg.position):
            if name in self.qidx:
                q[self.qidx[name]] = pos
        self.ik.set_q(q)
        self.seeded = True
        start = self.ik.ee_pose()
        self.target = start
        self._build_demo(start)
        self.get_logger().info(f"seeded from /joint_states; start EE = {np.round(start.translation, 3)}")

    def _on_target(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        rot = pin.Quaternion(o.w, o.x, o.y, o.z).matrix()
        self.target = pin.SE3(rot, np.array([p.x, p.y, p.z]))

    def _on_gripper(self, msg: Float64):
        self.gripper = float(msg.data)

    # -- demo target generation -------------------------------------------
    def _build_demo(self, start: pin.SE3):
        # Keep the (downward) orientation; sweep position + toggle the gripper:
        # over the cube area -> down (close) -> lift (hold) -> back (open).
        offsets = [(0.0, 0.0, 0.0), (0.0, 0.0, -0.12), (0.0, 0.0, 0.05), (-0.1, 0.2, 0.05)]
        grips = [GRIPPER_OPEN, GRIPPER_CLOSED, GRIPPER_CLOSED, GRIPPER_OPEN]
        self._demo_targets = [pin.SE3(start.rotation, start.translation + np.array(o)) for o in offsets]
        self._demo_grips = grips
        self._demo_i = 0

    # -- control tick ------------------------------------------------------
    def _tick(self):
        if not self.seeded or self.target is None:
            return

        if self.source == "demo":
            self.target = self._demo_targets[self._demo_i]
            self.gripper = self._demo_grips[self._demo_i]

        reached = self.ik.servo(self.target, self.dt)

        if self.source == "demo" and reached:
            self._demo_i = (self._demo_i + 1) % len(self._demo_targets)
            self.get_logger().info(f"reached demo pose; -> next [{self._demo_i}]")

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ARM_JOINTS + FINGER_JOINTS
        js.position = list(self.ik.arm_q()) + [self.gripper, self.gripper]
        self.cmd_pub.publish(js)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", default=None, help="Robot URDF (default: example-robot-data Panda).")
    p.add_argument("--ee-frame", default="panda_hand_tcp")
    p.add_argument("--rate", type=float, default=100.0, help="Control/servo rate (Hz). Higher = lower lag.")
    p.add_argument("--kp-lin", type=float, default=4.0, help="EE linear tracking gain (higher = snappier; ~1/kp s lag).")
    p.add_argument("--kp-ang", type=float, default=4.0, help="EE angular tracking gain.")
    p.add_argument("--max-lin-vel", type=float, default=0.6, help="EE linear speed cap (m/s).")
    p.add_argument("--max-ang-vel", type=float, default=1.5, help="EE angular speed cap (rad/s).")
    p.add_argument("--source", choices=["demo", "topic"], default="demo")
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--command-topic", default="/joint_command")
    p.add_argument("--target-topic", default="/target_frame")
    p.add_argument("--gripper-topic", default="/gripper_command")
    args, _ = p.parse_known_args()
    if args.urdf is None:
        args.urdf = _default_panda_urdf()

    rclpy.init()
    node = JointCommandNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
