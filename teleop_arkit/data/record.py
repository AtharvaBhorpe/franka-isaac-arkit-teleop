"""Phase 7 · Stage 1 — record teleop demonstrations to Rerun `.rrd`.

Subscribes to the live teleop topics and logs each stream on its OWN timeline at its
native rate (NO record-time fps/sync — alignment happens at LOAD time via the Rerun
query API; see rrd_dataset.py). One `.rrd` file per episode + a sidecar meta.json.

Operator control (keys typed in this terminal):
    s  start an episode  (first publishes /episode/reset → arm homes + cube randomizes,
                          waits --settle-secs, then begins logging)
    e  end episode, mark SUCCESS
    f  end episode, mark FAILURE
    d  discard the in-progress episode (delete it)
    q  quit

Schema (per PROJECT.md / the Phase-7 plan):
    observation/state          Scalars  [7 arm pos (+ gripper) (+ vel with --state-include-vel)]
    observation/images/<name>  EncodedImage (JPEG)         — one per --cameras entry
    action                     Scalars  [7 arm + gripper]  from /joint_command
    action/target_pose         Transform3D                 — /target_frame (auxiliary)

Cameras are configurable + robot-agnostic (decision #4):
    --cameras "wrist=/wrist_cam/image_raw scene=/scene_cam/image_raw"   (ROS topics, sim)
    --cameras "wrist=usb:0 side=usb:2"                                  (USB/V4L2, real robot)

Run (with Isaac up in --control ros + ik-topic + arkit):
    pixi run -e ros record --task "pick the cube into the bin"
    pixi run -e ros record --view            # also open a live Rerun viewer (camera framing)
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time

import numpy as np
import rclpy
import rerun as rr
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Empty, Float64, String

from teleop_arkit.core import schema
from teleop_arkit.core.cameras import parse_cameras

# All entities are logged on `self.sim_t` (latest /clock = sim time), NOT message header
# stamps: our ros-env nodes (joint_command_node, arkit_receiver) stamp with WALL time while
# Isaac stamps with SIM time, so headers are mixed-axis. One sim clock = one consistent
# timeline. (Rerun's auto `log_time` is also consistent across entities — rrd_dataset aligns
# on that; sim_time here keeps the Rerun viewer's sim timeline correct.)


class EpisodeRecorder(Node):
    def __init__(self, args):
        super().__init__("episode_recorder")
        self.args = args
        self.bridge = CvBridge()
        self.cams = parse_cameras(args.cameras)
        self.jpeg_quality = int(args.jpeg_quality)
        self.image_max_width = int(args.image_max_width)
        self.include_vel = bool(args.state_include_vel)

        self.out = os.path.expanduser(args.out)
        os.makedirs(self.out, exist_ok=True)
        self.ep_index = self._next_index()

        # --- live state (plain assignment; GIL-atomic, mirrors arkit_receiver) -----
        self.sim_t = 0.0                 # latest /clock time (s), fallback for header-less msgs
        self._js = None                  # latest JointState
        self._frames = {}                # cam name -> latest BGR np.uint8 (for usb logging)

        # --- recording state (guarded by _lock) ----------------------------------
        self._lock = threading.Lock()
        self.recording = False
        self._cmd_q = queue.Queue()      # episode commands from keyboard + /record/command
        self.ep = None                   # active per-episode RecordingStream
        self.ep_dir = None
        self.ep_meta = None
        self.ep_frames = 0
        self.ep_real_frames = {c[0]: 0 for c in self.cams}

        # --- optional live viewer (decision #2: camera framing while teleoping) ----
        self.view = None
        if args.view:
            import rerun.blueprint as rrb
            self.view = rr.RecordingStream("franka_teleop_live")
            # All cameras side-by-side, side panels collapsed -> maximum framing space for teleop
            # (use this instead of the Isaac GUI: `franka-teleop-headless` + `record --view`).
            bp = rrb.Blueprint(
                rrb.Horizontal(*[rrb.Spatial2DView(origin=schema.image_log(n), name=n)
                                 for n, _, _ in self.cams]),
                collapse_panels=True)
            self.view.spawn(default_blueprint=bp)
            self.get_logger().info(f"live Rerun viewer spawned — {len(self.cams)} cameras side-by-side")

        # --- ROS plumbing ---------------------------------------------------------
        self.reset_pub = self.create_publisher(Empty, args.reset_topic, 10)
        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(JointState, args.joint_state_topic, self._on_joint_states, 10)
        self.create_subscription(JointState, args.command_topic, self._on_command, 10)
        self.create_subscription(PoseStamped, args.target_topic, self._on_target, 10)
        self.create_subscription(Float64, args.gripper_topic, self._on_gripper, 10)
        for name, kind, source in self.cams:
            if kind == "ros":
                self.create_subscription(
                    Image, source, lambda m, n=name: self._on_ros_image(n, m), 1)
            else:
                threading.Thread(target=self._usb_loop, args=(name, source), daemon=True).start()

        # Episode control: keyboard (this terminal) + a /record/command topic (automation),
        # both feeding one queue drained by a worker thread, so the settle-sleep in
        # start_episode never blocks the ROS executor.
        self.create_subscription(String, "/record/command",
                                 lambda m: self._cmd_q.put(m.data.strip().lower()), 10)
        threading.Thread(target=self._keyboard_loop, daemon=True).start()
        threading.Thread(target=self._command_loop, daemon=True).start()
        self._print_banner()

    # ----- indexing -----------------------------------------------------------
    def _next_index(self) -> int:
        existing = [f for f in os.listdir(self.out) if f.startswith("episode_") and f.endswith(".rrd")]
        idx = [int(f[len("episode_"):-len(".rrd")]) for f in existing if f[len("episode_"):-len(".rrd")].isdigit()]
        return (max(idx) + 1) if idx else 0

    # ----- subscription callbacks (executor thread) ---------------------------
    def _on_clock(self, msg: Clock):
        self.sim_t = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _on_joint_states(self, msg: JointState):
        self._js = msg
        state = schema.state_vec(msg, self.include_vel)
        if state is None:
            return
        with self._lock:
            if self.recording and self.ep is not None:
                self.ep.set_time("sim_time", duration=self.sim_t)
                self.ep.log(schema.STATE_LOG, rr.Scalars(state))

    def _on_command(self, msg: JointState):
        action = schema.action_vec(msg)
        if action is None:
            return
        with self._lock:
            if self.recording and self.ep is not None:
                self.ep.set_time("sim_time", duration=self.sim_t)
                self.ep.log(schema.ACTION_LOG, rr.Scalars(action))

    def _on_target(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        with self._lock:
            if self.recording and self.ep is not None:
                self.ep.set_time("sim_time", duration=self.sim_t)
                self.ep.log(schema.TARGET_POSE_LOG, rr.Transform3D(
                    translation=[p.x, p.y, p.z], quaternion=[o.x, o.y, o.z, o.w]))

    def _on_gripper(self, msg: Float64):
        with self._lock:
            if self.recording and self.ep is not None:
                self.ep.set_time("sim_time", duration=self.sim_t)
                self.ep.log(schema.GRIPPER_LOG, rr.Scalars([float(msg.data)]))

    def _on_ros_image(self, name: str, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._log_frame(name, bgr, self.sim_t)

    def _usb_loop(self, name: str, source: str):
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            self.get_logger().error(f"camera '{name}': cannot open USB source '{source}'")
            return
        while rclpy.ok():
            ok, bgr = cap.read()
            if ok:
                self._log_frame(name, bgr, self.sim_t)
            time.sleep(0.01)

    def _log_frame(self, name: str, bgr, t: float):
        if self.image_max_width and bgr.shape[1] > self.image_max_width:   # downscale before encode
            s = self.image_max_width / bgr.shape[1]
            bgr = cv2.resize(bgr, (self.image_max_width, max(1, round(bgr.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        jpeg = buf.tobytes()
        ent = schema.image_log(name)
        if self.view is not None:                       # live framing (always)
            self.view.set_time("sim_time", duration=t)
            self.view.log(ent, rr.EncodedImage(contents=jpeg, media_type="image/jpeg"))
        with self._lock:
            if self.recording and self.ep is not None:  # into the episode file
                self.ep.set_time("sim_time", duration=t)
                self.ep.log(ent, rr.EncodedImage(contents=jpeg, media_type="image/jpeg"))
                self.ep_real_frames[name] += 1

    # ----- operator UX --------------------------------------------------------
    def _print_banner(self):
        cams = ", ".join(n for n, _, _ in self.cams)
        bar = "=" * 70
        print(
            f"\n{bar}\n"
            f"  TELEOP EPISODE RECORDER      ->  {self.out}\n"
            f"  task: \"{self.args.task}\"\n"
            f"  cameras: {cams}             next: episode_{self.ep_index:06d}\n"
            f"{'-' * 70}\n"
            "  s : START a new episode   (homes the arm + randomizes the cube,\n"
            "                             settles, then begins recording)\n"
            "  e : END + save as SUCCESS     (you completed the task)\n"
            "  f : END + save as FAILURE     (saved + flagged; excluded from training)\n"
            "  d : DISCARD the current episode   (delete it -- a bad take)\n"
            "  q : QUIT the recorder\n"
            "  h : show this help again\n"
            f"{'-' * 70}\n"
            "  Flow:   press s   ->   teleop the pick-and-place   ->   press e\n"
            f"{bar}",
            flush=True,
        )

    def _prompt_ready(self):
        print(f"--> ready. Press 's' to start episode_{self.ep_index:06d}   (or 'q' to quit).",
              flush=True)

    # ----- episode lifecycle (keyboard thread) --------------------------------
    def _publish_reset(self, n=3):
        for _ in range(n):                              # 2-3x to beat DDS single-shot drop
            self.reset_pub.publish(Empty())
            time.sleep(0.1)

    def start_episode(self):
        if self.recording:
            print(f"!! already recording episode_{self.ep_index:06d} -- press 'e' (success), "
                  "'f' (fail) or 'd' (discard) first.", flush=True)
            return
        print(f"... episode_{self.ep_index:06d}: resetting (home arm + randomize cube), "
              f"settling {self.args.settle_secs:.1f}s ...", flush=True)
        self._publish_reset()
        time.sleep(self.args.settle_secs)               # wait out homing + physics settle
        path = os.path.join(self.out, f"episode_{self.ep_index:06d}.rrd")
        ep = rr.RecordingStream("franka_teleop_ep")
        ep.save(path)
        with self._lock:
            self.ep = ep
            self.ep_real_frames = {c[0]: 0 for c in self.cams}
            self.ep_meta = {
                "episode": self.ep_index, "task": self.args.task,
                "cameras": {n: s for n, _, s in self.cams},
                "state_dim": 16 if self.include_vel else 8, "action_dim": 8,
                "jpeg_quality": self.jpeg_quality, "image_max_width": self.image_max_width,
                "sim_t_start": self.sim_t, "rerun_sdk": rr.__version__,
            }
            self.recording = True
        print(f"\n>>> RECORDING episode_{self.ep_index:06d}  --  do the task, then press:  "
              "'e'=SUCCESS   'f'=FAILURE   'd'=DISCARD\n", flush=True)

    def end_episode(self, success: bool):
        with self._lock:
            if not self.recording or self.ep is None:
                print("!! not recording -- press 's' to start an episode.", flush=True)
                return
            ep, meta = self.ep, self.ep_meta
            self.recording, self.ep = False, None
        ep.flush(timeout_sec=5.0)                        # finalize the .rrd
        meta.update(success=success, sim_t_end=self.sim_t,
                    real_frames=self.ep_real_frames)
        with open(os.path.join(self.out, f"episode_{self.ep_index:06d}.meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        tag = "SUCCESS" if success else "FAILURE"
        print(f"[SAVED] episode_{self.ep_index:06d} as {tag}   frames={self.ep_real_frames}", flush=True)
        self.ep_index += 1
        self._prompt_ready()

    def discard_episode(self):
        with self._lock:
            if not self.recording or self.ep is None:
                print("!! not recording -- nothing to discard.", flush=True)
                return
            ep, self.recording, self.ep = self.ep, False, None
        ep.flush(timeout_sec=2.0)
        p = os.path.join(self.out, f"episode_{self.ep_index:06d}.rrd")
        if os.path.exists(p):
            os.remove(p)
        print(f"[DISCARDED] episode_{self.ep_index:06d} deleted.", flush=True)
        self._prompt_ready()

    def _keyboard_loop(self):
        while rclpy.ok():
            try:
                self._cmd_q.put(input().strip().lower())
            except (EOFError, KeyboardInterrupt):
                break

    def _command_loop(self):
        actions = {
            "s": self.start_episode, "start": self.start_episode,
            "e": lambda: self.end_episode(True), "success": lambda: self.end_episode(True),
            "f": lambda: self.end_episode(False), "fail": lambda: self.end_episode(False),
            "d": self.discard_episode, "discard": self.discard_episode,
            "h": self._print_banner, "help": self._print_banner,
            "?": self._print_banner,
        }
        while rclpy.ok():
            try:
                cmd = self._cmd_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if cmd in ("q", "quit"):
                rclpy.shutdown()
                break
            act = actions.get(cmd)
            if act:
                act()
            elif cmd:
                print("?? unknown key.  s=start  e=success  f=fail  d=discard  q=quit", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="~/rerun_episodes",
                   help="Directory for episode_NNNNNN.rrd + .meta.json (external by default).")
    p.add_argument("--task", default="pick the cube and place it in the bin")
    p.add_argument("--cameras", default="wrist=/wrist_cam/image_raw scene=/scene_cam/image_raw",
                   help="Space-separated name=source; source = ROS topic (/...) or usb:N / device.")
    p.add_argument("--view", action="store_true", help="Also open a live Rerun viewer (camera framing).")
    p.add_argument("--settle-secs", type=float, default=1.5, help="Wait after reset before logging.")
    p.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality (try ~80 for campaigns).")
    p.add_argument("--image-max-width", type=int, default=0,
                   help="Downscale frames so width <= this BEFORE JPEG (aspect preserved; 0 = native). "
                        "e.g. 640 shrinks the 1280x720 scene cam ~4x; the wrist cam (640) is untouched. "
                        "We train at 224x224, so native scene res is overkill — use this for campaigns.")
    p.add_argument("--state-include-vel", action="store_true", help="Append joint velocities to state.")
    p.add_argument("--reset-topic", default="/episode/reset")
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--command-topic", default="/joint_command")
    p.add_argument("--target-topic", default="/target_frame")
    p.add_argument("--gripper-topic", default="/gripper_command")
    args, _ = p.parse_known_args()

    from teleop_arkit.core.rosutil import spin
    rclpy.init()
    node = EpisodeRecorder(args)
    try:
        spin(node)
    finally:
        if node.recording:
            node.end_episode(False)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
