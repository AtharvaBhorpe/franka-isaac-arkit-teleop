"""Compact Pinocchio Cartesian servo-IK for the Franka arm.

Velocity-servo (resolved-rate) IK: given a target end-effector pose, compute a
joint-velocity step via the damped-least-squares pseudo-inverse of the frame
Jacobian, integrate it, and stream the resulting joint positions. Only the 7 arm
joints are servoed; the 2 finger joints are left for separate gripper control.

The damped-least-squares + singularity-adaptive damping is our own (compact)
implementation of the technique used in SpesRobotics/teleop's JacobiRobot
(Apache-2.0) — we reuse the approach, not the code.

Self-test (no ROS / no Isaac needed), run inside the `ros` env:
    pixi run -e ros python -m teleop_arkit.ik
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin


def _clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale `v` down so its norm <= max_norm (direction preserved)."""
    n = np.linalg.norm(v)
    return v * (max_norm / n) if n > max_norm else v


class CartesianServoIK:
    """Resolved-rate Cartesian servo for a fixed-base manipulator.

    Args:
        urdf_path: path to the robot URDF (kinematics only; meshes ignored).
        ee_frame: end-effector frame name to control (e.g. "panda_hand_tcp").
        n_arm: number of leading DoFs to servo (7 for the Panda arm).
        kp_lin / kp_ang: proportional gains on linear / angular pose error.
        max_lin_vel / max_ang_vel: EE twist clamps (m/s, rad/s).
        max_joint_vel: per-joint velocity clamp (rad/s).
        damping: base DLS damping (lambda^2).
        sing_threshold / max_sing_boost: raise damping up to xN as the
            manipulability measure drops below the threshold (singularity guard).
        lin_tol / ang_tol: convergence tolerances (m, rad).
    """

    def __init__(
        self,
        urdf_path: str,
        ee_frame: str = "panda_hand_tcp",
        n_arm: int = 7,
        kp_lin: float = 2.0,
        kp_ang: float = 2.0,
        max_lin_vel: float = 0.4,
        max_ang_vel: float = 0.9,
        max_joint_vel: float = 2.0,
        damping: float = 1e-3,
        sing_threshold: float = 1e-2,
        max_sing_boost: float = 10.0,
        lin_tol: float = 1e-3,
        ang_tol: float = 5e-3,
    ) -> None:
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        if not self.model.existFrame(ee_frame):
            raise ValueError(f"EE frame '{ee_frame}' not in URDF {urdf_path}")
        self.ee_id = self.model.getFrameId(ee_frame)
        self.n_arm = n_arm
        self.q = pin.neutral(self.model)
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit

        self.kp_lin, self.kp_ang = kp_lin, kp_ang
        self.max_lin_vel, self.max_ang_vel = max_lin_vel, max_ang_vel
        self.max_joint_vel = max_joint_vel
        self.damping = damping
        self.sing_threshold, self.max_sing_boost = sing_threshold, max_sing_boost
        self.lin_tol, self.ang_tol = lin_tol, ang_tol

    # -- state -------------------------------------------------------------
    def set_q(self, q) -> None:
        """Seed the full joint configuration (length model.nq)."""
        q = np.asarray(q, float)
        if q.shape[0] != self.model.nq:
            raise ValueError(f"expected nq={self.model.nq}, got {q.shape[0]}")
        self.q = q.copy()

    def arm_q(self) -> np.ndarray:
        """Current arm joint positions (first n_arm DoFs)."""
        return self.q[: self.n_arm].copy()

    def ee_pose(self) -> pin.SE3:
        """Current end-effector pose as a Pinocchio SE3."""
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacement(self.model, self.data, self.ee_id)
        return self.data.oMf[self.ee_id]

    # -- the servo step ----------------------------------------------------
    def servo(self, target: pin.SE3, dt: float) -> bool:
        """Step the arm one tick toward `target`. Returns True if within tol."""
        current = self.ee_pose()

        # Pose error as a body-frame twist (moves current -> target).
        err = pin.log6(current.actInv(target)).vector  # [vx vy vz wx wy wz]
        lin_err, ang_err = err[:3], err[3:]

        # Desired EE twist: P control, clamped.
        twist = np.concatenate([
            _clamp_norm(self.kp_lin * lin_err, self.max_lin_vel),
            _clamp_norm(self.kp_ang * ang_err, self.max_ang_vel),
        ])

        # Arm Jacobian in the EE LOCAL frame (matches the body-frame twist above).
        J = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_id, pin.LOCAL)
        Ja = J[:, : self.n_arm]  # finger columns are ~0 for the hand frame anyway

        # Singularity-adaptive damping: boost lambda as manipulability drops.
        manip = float(np.sqrt(max(np.linalg.det(Ja @ Ja.T), 0.0)))
        damp = self.damping
        if manip < self.sing_threshold:
            damp *= min(self.sing_threshold / (manip + 1e-9), self.max_sing_boost)

        # Damped least squares: dq = Ja^T (Ja Ja^T + damp I)^-1 twist.
        dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + damp * np.eye(6), twist)
        dq = np.clip(dq, -self.max_joint_vel, self.max_joint_vel)

        # Integrate the arm DoFs (respect joint limits); fingers untouched.
        self.q[: self.n_arm] = np.clip(
            self.q[: self.n_arm] + dq * dt,
            self.q_min[: self.n_arm], self.q_max[: self.n_arm],
        )
        return np.linalg.norm(lin_err) < self.lin_tol and np.linalg.norm(ang_err) < self.ang_tol


def _default_panda_urdf() -> str:
    """Path to the example-robot-data Panda URDF inside the active conda env."""
    import os

    return os.path.join(
        os.environ["CONDA_PREFIX"],
        "share/example-robot-data/robots/panda_description/urdf/panda.urdf",
    )


if __name__ == "__main__":
    # Standalone convergence test: seed a pose, command a +x/+z offset, servo.
    ik = CartesianServoIK(_default_panda_urdf())
    # A reasonable arm config (Isaac default) + open fingers.
    ik.set_q(np.array([0.012, -0.57, 0.0, -2.81, 0.0, 3.04, 0.74, 0.04, 0.04]))

    start = ik.ee_pose()
    target = start.copy()
    target.translation = start.translation + np.array([0.10, 0.0, 0.10])
    print("start EE :", np.round(start.translation, 4))
    print("target EE:", np.round(target.translation, 4))

    dt = 1.0 / 100.0
    for step in range(2000):
        if ik.servo(target, dt):
            print(f"reached in {step} steps")
            break
    final = ik.ee_pose()
    print("final EE :", np.round(final.translation, 4))
    print("pos error:", round(float(np.linalg.norm(final.translation - target.translation)), 5), "m")
    print("arm q    :", np.round(ik.arm_q(), 3).tolist())
