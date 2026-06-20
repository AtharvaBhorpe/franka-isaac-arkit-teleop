"""Phase 7 · Stage 2 — ACT inference node: obs -> /joint_command (closed-loop in sim).

Loads an `act_min.pt` checkpoint, subscribes to `/joint_states` + the camera topics, runs the
compact ACT (CVAE z=0 at inference), and publishes `/joint_command` in `joint_command_node`'s
schema — so the *trained policy* drives the SAME sim loop the human teleoped.

Preprocessing MIRRORS `rrd_dataset.py` exactly (train/infer parity is critical):
  * state  = `ARM_JOINTS` positions (7) + gripper (`panda_finger_joint1`)  -> z-scored via ckpt stats
  * images = cv_bridge `bgr8` -> resize(W,H) -> BGR2RGB -> CHW/255   (same as the dataset decode)
  * action = model output denormalized -> arm (7) + gripper (1); gripper mirrored to both fingers

Control: receding horizon by default (`--exec-horizon 1`: re-infer each tick, publish the first
action). `--exec-horizon K` executes K actions open-loop before re-planning. Runs in the `ros`
env (rclpy + torch together after the 2026-06-09 env merge).

  pixi run -e ros infer
  pixi run -e ros infer --ckpt ~/rerun_episodes/checkpoints/act_min.pt --rate 10 --exec-horizon 1
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from teleop_arkit.core import schema
from teleop_arkit.core.cameras import parse_cameras, preprocess_image
from teleop_arkit.core.config import DatasetStats
from teleop_arkit.core.robot import ARM_JOINTS, FINGER_JOINTS, GRIPPER_CLOSED, GRIPPER_OPEN
from teleop_arkit.policies.registry import build_model


class InferNode(Node):
    def __init__(self, args):
        super().__init__("act_infer")
        self.device = args.device
        ckpt = torch.load(os.path.expanduser(args.ckpt), map_location=self.device,
                          weights_only=False)              # our own ckpt: config+stats dicts
        self.model = build_model(ckpt["config"]).to(self.device)  # validate config + dispatch act/diffusion
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.cameras = list(self.model.cameras)
        self.img_hw = tuple(self.model.img_hw)
        # 0 -> execute the FULL chunk open-loop, then re-plan. Right default for an absolute-position
        # policy: action[0] ≈ the current state (the recorded command barely leads /joint_states per
        # step), so single-step receding (=1) stalls — the displacement is in the later chunk actions.
        # Smaller values (e.g. 8) re-plan more often for closer-loop correction.
        self.exec_horizon = args.exec_horizon if args.exec_horizon > 0 else int(self.model.chunk)

        self.stats = ckpt.get("stats")
        if self.stats:                                     # same normalization as rrd_dataset
            DatasetStats.model_validate(self.stats)        # raises ValidationError on train→infer schema drift
            self.s_mean = np.asarray(self.stats["observation.state"]["mean"], np.float32)
            self.s_std = np.asarray(self.stats["observation.state"]["std"], np.float32)
            self.a_mean = np.asarray(self.stats["action"]["mean"], np.float32)
            self.a_std = np.asarray(self.stats["action"]["std"], np.float32)

        cam_topic = {n: src for (n, _kind, src) in parse_cameras(args.cameras)}
        missing = [c for c in self.cameras if c not in cam_topic]
        if missing:
            raise ValueError(f"ckpt expects cameras {self.cameras} but --cameras lacks {missing}")

        self.bridge = CvBridge()
        self._state = None                                 # latest raw 8-D state
        self._frames = {c: None for c in self.cameras}     # latest BGR frame per camera
        self._queue: list[np.ndarray] = []                 # pending denormalized actions
        self._warned = False

        self.create_subscription(JointState, args.joint_state_topic, self._on_js, 1)
        for cam in self.cameras:
            self.create_subscription(Image, cam_topic[cam], lambda m, c=cam: self._on_img(c, m), 1)
        self.pub = self.create_publisher(JointState, args.command_topic, 1)

        rate = args.rate or float(ckpt.get("fps", 10.0))   # match the training fps grid
        self.create_timer(1.0 / rate, self._tick)
        name = ckpt.get("config", {}).get("name", "act")
        loss_val = ckpt.get("loss", ckpt.get("l1", float("nan")))
        self.get_logger().info(
            f"{name}_infer up: ckpt loss={loss_val:.3f} cameras={self.cameras} "
            f"rate={rate:.0f}Hz horizon={self.exec_horizon} device={self.device} "
            f"norm={'yes' if self.stats else 'NONE'}")

    # -- inputs (executor thread; plain assignment, single-threaded executor = no race) --
    def _on_js(self, msg: JointState):
        sv = schema.state_vec(msg)               # 7 arm pos + gripper (positions-only)
        if sv is not None:
            self._state = np.asarray(sv, np.float32)

    def _on_img(self, cam: str, msg: Image):
        self._frames[cam] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _img_tensor(self, bgr):
        return preprocess_image(bgr, self.img_hw)

    @torch.no_grad()
    def _infer(self) -> list[np.ndarray]:
        state = self._state
        if self.stats:
            state = (state - self.s_mean) / (self.s_std + 1e-6)
        st = torch.from_numpy(np.ascontiguousarray(state)).float().unsqueeze(0).to(self.device)
        imgs = {c: self._img_tensor(self._frames[c]).unsqueeze(0).to(self.device) for c in self.cameras}
        chunk = self.model.predict(st, imgs)[0].cpu().numpy()     # (chunk, action_dim), normalized
        if self.stats:
            chunk = chunk * (self.a_std + 1e-6) + self.a_mean
        return [chunk[i] for i in range(min(self.exec_horizon, len(chunk)))]

    # -- control tick ----------------------------------------------------------
    def _tick(self):
        if self._state is None or any(self._frames[c] is None for c in self.cameras):
            if not self._warned:
                self.get_logger().warn("waiting for /joint_states + all camera frames…")
                self._warned = True
            return
        if not self._queue:
            self._queue = self._infer()
        action = self._queue.pop(0)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ARM_JOINTS + FINGER_JOINTS
        grip = float(np.clip(action[7], GRIPPER_CLOSED, GRIPPER_OPEN))
        js.position = [float(x) for x in action[:7]] + [grip, grip]
        self.pub.publish(js)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="~/rerun_episodes/checkpoints/act_min.pt")
    p.add_argument("--cameras", default="wrist=/wrist_cam/image_raw scene=/scene_cam/image_raw")
    p.add_argument("--rate", type=float, default=0.0, help="control Hz (0 = ckpt fps).")
    p.add_argument("--exec-horizon", type=int, default=0,
                   help="actions per re-plan (0 = full chunk [default]; 8 = re-plan twice/chunk; "
                        "1 = single-step receding, which STALLS for an absolute-action ACT).")
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--command-topic", default="/joint_command")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from teleop_arkit.core.rosutil import run
    rclpy.init()
    run(InferNode(args))                                       # spin + clean teardown (Ctrl-C / SIGTERM)


if __name__ == "__main__":
    main()
