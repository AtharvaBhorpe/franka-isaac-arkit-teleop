# teleop_arkit/teleop — live teleoperation ROS2 nodes

## Purpose
Drive the simulated (or, later, real) Franka in real time: iPhone/ARKit pose → servo-IK →
`/joint_command`. Also the phone-free demo driver and the rviz/diagnostic helpers.

## Ownership
- `ik.py` — `CartesianServoIK`: compact Pinocchio damped-least-squares servo; EE frame
  `panda_hand_tcp`.
- `joint_command_node.py` — `/target_frame` (or scripted `--source demo`) → IK → `/joint_command`
  (7 arm + 2 finger). On `/episode/reset`: command `HOME_ARM_Q` ~1 s, then re-seed from
  `/joint_states`. Tasks: `ik-demo`, `ik-topic`.
- `arkit_receiver.py` — ZIG SIM PRO (ARKit pose + touch) → `/target_frame` + `/gripper_command`;
  `--proto {udp,tcp}`, **latest-only** (drops backlog), clutch + `ARKIT_TO_ROS` remap + optional
  6-DoF orient. Tasks: `arkit`, `arkit-tcp`.
- `robot_state_pub.py` — `/robot_description` + `/tf` for an rviz `RobotModel`. Task: `robot-model`.
- `sniff_stream.py` — raw ZIG SIM UDP/TCP printer (diagnostic; `sniff --proto tcp` shows TCP framing). Task: `sniff`.

## Local Contracts
- Control seam: `/target_frame` (PoseStamped) → IK → `/joint_command` (JointState); gripper via
  `/gripper_command` (Float64; open 0.04 / closed 0.0).
- **Teleop controls:** 1 finger = move (clutch, re-zeros on engage) · 0 = freeze · 2-finger tap =
  toggle gripper.
- **A single `/episode/reset` is DDS-dropped → publish 2–3×** (the `reset` task sends 5×).
- **ARKit transport** = `--proto udp` (default, lowest latency) or `tcp` (reliable; the node is the
  TCP server, Nagle off, framing-agnostic JSON parse). Both **act on the freshest frame only** (backlog
  dropped) so lag can't accumulate; the node logs `rx: N/s arrived, M/s handled`.
- Joint names / gripper convention from `core.robot`; URDF via `core.robot.default_panda_urdf()`.

## Work Guidance
- Teleop lag is the servo time-constant + timer quantization, not the IK math — raise
  `--kp-lin`/`--rate` and publish event-driven. Nodes spin via `core.rosutil.run`.

## Verification
- `pixi run -e ros ik-demo` (phone-free loop; needs Isaac `franka-teleop`) · `ik-topic` + `arkit`
  (full phone path) · `robot-model` (rviz).

## Child DOX Index
None (leaf).
