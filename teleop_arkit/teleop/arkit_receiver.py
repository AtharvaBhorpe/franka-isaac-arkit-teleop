"""ARKit receiver — ZIG SIM PRO iPhone pose + touch -> teleop (Phase 5).

Listens for ZIG SIM PRO's ARKit + touch JSON over UDP or TCP (`--proto`) and drives the robot:
  * /target_frame   (geometry_msgs/PoseStamped) — desired EE pose
  * /gripper_command (std_msgs/Float64)         — finger target (open/closed)
consumed by the IK node (joint_command_node --source topic).

ZIG SIM PRO payload (UDP/JSON):
    sensordata.arkit.position : [x, y, z]      world position (m)
    sensordata.arkit.rotation : [x, y, z, w]   orientation quaternion (unused in v1)
    sensordata.touch          : [ {x,y,...}, ... ]   array; length = # fingers down

Control scheme (finger count):
    1 finger  -> MOVE (clutch): EE follows phone translation. Re-zeros on press,
                 so the robot never jumps when you (re-)engage.
    0 fingers -> FROZEN: lift to reposition the phone, or carry without moving.
    2-finger tap -> toggle gripper open<->closed (latched).

v1 = POSITION-ONLY: EE keeps its (downward) orientation; full 6-DoF orientation
is a later refinement. Motion is relative to the robot's live EE + the phone pose
captured at each clutch engage.

Run (with Isaac --control ros and `ik-topic` both up):
    pixi run -e ros python -m teleop_arkit.teleop.arkit_receiver --scale 1.5
"""

from __future__ import annotations

import argparse
import json
import socket
import threading

import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from teleop_arkit.core.robot import (
    ARM_JOINTS, FINGER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSED, default_panda_urdf
)
from teleop_arkit.teleop.ik import CartesianServoIK

# ARKit world (RH, +Y up, camera looks -Z) -> robot base (+Z up, +X fwd, +Y left):
# phone up -> +Z, phone forward(-Z) -> +X, phone left(-X) -> +Y. Flip a row's sign
# if an axis comes out reversed during tuning.
ARKIT_TO_ROS = np.array([
    [0.0, 0.0, -1.0],   # robot_x = -arkit_z
    [-1.0, 0.0, 0.0],   # robot_y = -arkit_x
    [0.0, 1.0, 0.0],    # robot_z = +arkit_y
])


class ARKitReceiver(Node):
    def __init__(self, args):
        super().__init__("arkit_receiver")
        self.scale = args.scale
        self.frame_id = args.frame_id

        # Pinocchio FK to read the robot's live EE pose (the clutch re-zero ref).
        self.fk = CartesianServoIK(args.urdf, ee_frame=args.ee_frame)
        self.qidx = {
            n: self.fk.model.joints[self.fk.model.getJointId(n)].idx_q
            for n in ARM_JOINTS + FINGER_JOINTS
        }
        self._latest_q = None        # latest joint vector (set in spin thread)
        self.robot_ref = None        # EE pose captured at clutch engage
        self.phone_ref = None        # phone pos captured at clutch engage
        self.phone_rot_ref = None    # phone orientation (3x3) captured at engage
        self.moving = False          # clutch state (1 finger)
        self.gripper_closed = False  # latched gripper state
        self.prev_n = 0              # previous finger count
        self.orient = args.orient    # 6-DoF orientation on/off
        self.quat_order = args.quat_order  # ZIG SIM quaternion order
        self.C = ARKIT_TO_ROS        # ARKit->ROS basis change (for rotation remap)

        self.create_subscription(JointState, args.joint_state_topic, self._on_joint_states, 1)
        self.pose_pub = self.create_publisher(PoseStamped, args.target_topic, 1)
        self.grip_pub = self.create_publisher(Float64, args.gripper_topic, 1)

        self.proto = args.proto
        self.host, self.port = args.host, args.port
        self._rx_arrived = self._rx_handled = 0        # receive-rate diagnostic counters
        threading.Thread(target=self._recv_loop, daemon=True).start()
        self.create_timer(2.0, self._log_rx)
        self.get_logger().info(
            f"arkit_receiver up: {self.proto.upper()} :{self.port}  scale={self.scale}  ee={args.ee_frame}\n"
            f"   1 finger = move, 0 = freeze, 2-finger tap = toggle gripper")

    # -- robot joints (store latest; FK is done in the udp thread only) -----
    def _on_joint_states(self, msg: JointState):
        q = self.fk.q.copy()
        for name, pos in zip(msg.name, msg.position):
            if name in self.qidx:
                q[self.qidx[name]] = pos
        self._latest_q = q  # plain assignment; read by the udp thread

    def _quat_to_R(self, q):
        """ZIG SIM quaternion (list) -> 3x3 rotation matrix, honoring --quat-order."""
        if self.quat_order == "xyzw":
            x, y, z, w = q
        else:  # wxyz
            w, x, y, z = q
        return pin.Quaternion(float(w), float(x), float(y), float(z)).normalized().matrix()

    # -- phone stream (latest-only: always act on the FRESHEST frame, drop backlog) --------
    def _recv_loop(self):
        (self._udp_serve if self.proto == "udp" else self._tcp_serve)()

    def _udp_serve(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)   # absorb bursts
        except OSError:
            pass
        sock.bind((self.host, self.port))
        while rclpy.ok():
            data, _ = sock.recvfrom(65535)             # block for at least one datagram
            self._rx_arrived += 1
            sock.setblocking(False)
            while True:                                # drain the backlog -> keep only the newest
                try:
                    data, _ = sock.recvfrom(65535)
                    self._rx_arrived += 1
                except BlockingIOError:
                    break
            sock.setblocking(True)
            self._feed(data)

    def _tcp_serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        dec = json.JSONDecoder()
        while rclpy.ok():
            self.get_logger().info(f"TCP: waiting for ZIG SIM to connect on :{self.port} …")
            conn, src = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)      # disable Nagle (our side)
            self.get_logger().info(f"TCP: connected {src}")
            buf = ""
            try:
                while rclpy.ok():
                    chunk = conn.recv(65535)
                    if not chunk:
                        break                          # peer closed -> re-accept
                    buf += chunk.decode("utf-8", "ignore")
                    if len(buf) > (1 << 20):           # runaway guard (never a full object)
                        buf = ""
                    objs, buf = self._extract_objects(buf, dec)
                    self._rx_arrived += len(objs)
                    if objs:
                        self._feed_obj(objs[-1])       # newest complete frame only
            except OSError:
                pass
            finally:
                conn.close()

    @staticmethod
    def _extract_objects(buf, dec):
        """Pull every complete top-level JSON object out of a TCP byte-stream buffer. Framing-agnostic
        (works whether ZIG SIM newline-delimits or concatenates). Returns (objects, remaining_buf)."""
        objs, i, n = [], 0, len(buf)
        while i < n:
            while i < n and buf[i] in " \r\n\t":       # skip any delimiter whitespace
                i += 1
            if i >= n:
                break
            try:
                obj, end = dec.raw_decode(buf, i)
            except json.JSONDecodeError:
                break                                  # trailing partial object -> wait for more bytes
            objs.append(obj)
            i = end
        return objs, buf[i:]

    # -- decode one frame -> drive the robot --
    def _feed(self, data: bytes):
        try:
            self._feed_obj(json.loads(data.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            pass

    def _feed_obj(self, obj):
        try:
            sensors = obj["sensordata"]
            arkit = sensors["arkit"]
            pos = np.asarray(arkit["position"], dtype=float)
            rot = arkit["rotation"]
        except (KeyError, ValueError, TypeError):
            return
        self._rx_handled += 1
        self._process(pos, rot, len(sensors.get("touch") or []))

    def _log_rx(self):
        if self._rx_arrived or self._rx_handled:
            self.get_logger().info(
                f"rx: {self._rx_arrived/2:.0f}/s arrived, {self._rx_handled/2:.0f}/s handled (latest-only)")
        self._rx_arrived = self._rx_handled = 0

    def _process(self, pos: np.ndarray, rot, n: int):
        # Gripper: toggle on a rising edge into >=2 fingers; publish latched state.
        if n >= 2 and self.prev_n < 2:
            self.gripper_closed = not self.gripper_closed
            self.get_logger().info(f"gripper -> {'CLOSED' if self.gripper_closed else 'OPEN'}")
        self.grip_pub.publish(Float64(data=GRIPPER_CLOSED if self.gripper_closed else GRIPPER_OPEN))

        # Clutch: exactly 1 finger = moving. Re-zero on engage so no jump.
        moving = (n == 1)
        if moving and not self.moving:
            if self._latest_q is not None:
                self.fk.set_q(self._latest_q)
                self.robot_ref = self.fk.ee_pose()
                self.phone_ref = pos
                self.phone_rot_ref = self._quat_to_R(rot)
                self.moving = True
                self.get_logger().info("move engaged")
        elif not moving and self.moving:
            self.moving = False
            self.get_logger().info("move released")

        if self.moving and self.robot_ref is not None:
            delta = ARKIT_TO_ROS @ (pos - self.phone_ref) * self.scale
            target_pos = self.robot_ref.translation + delta

            if self.orient:
                # World-frame phone rotation since engage, mapped into the robot
                # base frame (similarity transform by C), applied to the start EE:
                # dR_robot = C (R_now R_ref^T) C^T ;  target = dR_robot @ robot_ref.
                d_arkit = self._quat_to_R(rot) @ self.phone_rot_ref.T
                d_robot = self.C @ d_arkit @ self.C.T
                target_R = d_robot @ self.robot_ref.rotation
            else:
                target_R = self.robot_ref.rotation
            quat = pin.Quaternion(target_R)

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, target_pos)
            msg.pose.orientation.w = float(quat.w)
            msg.pose.orientation.x = float(quat.x)
            msg.pose.orientation.y = float(quat.y)
            msg.pose.orientation.z = float(quat.z)
            self.pose_pub.publish(msg)

        self.prev_n = n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=50000)
    p.add_argument("--proto", choices=["udp", "tcp"], default="udp",
                   help="ZIG SIM transport. udp (default) = lowest latency; tcp = reliable, can add lag.")
    p.add_argument("--scale", type=float, default=1.5, help="Phone->robot translation gain.")
    p.add_argument("--no-orient", dest="orient", action="store_false",
                   help="Disable 6-DoF wrist orientation (EE stays downward).")
    p.set_defaults(orient=True)
    p.add_argument("--quat-order", choices=["xyzw", "wxyz"], default="xyzw",
                   help="ZIG SIM arkit.rotation component order.")
    p.add_argument("--urdf", default=None)
    p.add_argument("--ee-frame", default="panda_hand_tcp")
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--target-topic", default="/target_frame")
    p.add_argument("--gripper-topic", default="/gripper_command")
    p.add_argument("--frame-id", default="world")
    args, _ = p.parse_known_args()
    if args.urdf is None:
        args.urdf = default_panda_urdf()

    from teleop_arkit.core.rosutil import run
    rclpy.init()
    run(ARKitReceiver(args))                                   # spin + clean teardown (Ctrl-C / SIGTERM)


if __name__ == "__main__":
    main()
