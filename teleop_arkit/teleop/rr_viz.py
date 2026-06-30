"""Live Rerun monitor: Isaac camera feeds (over ROS2) + one all-joint-angles plot.

A read-only dashboard for teleop — no recording. Subscribes to the camera image
topics and /joint_states, logs them to a spawned Rerun viewer on its auto log_time
timeline. Cameras share a top row; every joint angle goes into ONE time-series plot.

Run (with Isaac up + teleop):
    pixi run -e ros rr-viz
    pixi run -e ros rr-viz --cameras "wrist=/wrist_cam/image_raw scene=/scene_cam/image_raw"
"""
from __future__ import annotations

import argparse

import cv2
import rclpy
import rerun as rr
import rerun.blueprint as rrb
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from teleop_arkit.core import schema
from teleop_arkit.core.cameras import parse_cameras
from teleop_arkit.core.rosutil import run


class RerunViz(Node):
    def __init__(self, args):
        super().__init__("rerun_viz")
        self.bridge = CvBridge()
        cams = [c for c in parse_cameras(args.cameras) if c[1] == "ros"]  # ponytail: sim feeds are ROS topics

        self.rec = rr.RecordingStream("franka_viz")
        bp = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(*[rrb.Spatial2DView(origin=schema.image_log(n), name=n)
                                 for n, _, _ in cams]),
                rrb.TimeSeriesView(origin="/joints", name="joint angles"),
            ),
            collapse_panels=True,
        )
        self.rec.spawn()
        self.rec.send_blueprint(bp)        # force: override any blueprint the viewer cached from a prior run

        for name, _, source in cams:
            self.create_subscription(Image, source, lambda m, n=name: self._on_image(n, m), 1)
        self.create_subscription(JointState, args.joint_state_topic, self._on_joints, 10)
        self.get_logger().info(f"viz spawned — cameras: {', '.join(n for n, _, _ in cams)} + /joints")

    def _on_image(self, name: str, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.rec.log(schema.image_log(name), rr.Image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    def _on_joints(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.rec.log(f"joints/{name}", rr.Scalars(float(pos)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cameras", default="wrist=/wrist_cam/image_raw scene=/scene_cam/image_raw",
                   help="Space-separated name=topic (ROS topics only; non-ROS sources are ignored).")
    p.add_argument("--joint-state-topic", default="/joint_states")
    args, _ = p.parse_known_args()

    rclpy.init()
    run(RerunViz(args))


if __name__ == "__main__":
    main()
