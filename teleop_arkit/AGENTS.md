# teleop_arkit — ROS2 teleop + imitation-learning package

## Purpose
The ros-env side of the project: live iPhone→Franka teleoperation and the Phase-7 IL pipeline
(record → read → train → infer). Runs in the pixi `ros` env (RoboStack Jazzy, Python 3.12, with
torch + rerun-sdk[catalog] layered on). Built model-, robot-, and modality-agnostic; the shared
contract lives in `core/`.

## Ownership
- The package layout and the pixi-task → module-path mapping (task NAMES are the stable UX; only
  module paths live here).
- Two **leaf** sub-packages owned directly here (no child doc of their own yet):
  - `training/` — `train.py`: the PyTorch train/overfit loop. The checkpoint bundles
    `{model, config (core.config.ModelConfig), stats}`. `--cameras <names…>` trains on a camera
    **subset** of the recorded set (default all) — the modality-ablation knob. Tasks: `train`,
    `smoke-act` (`--steps 50 --max-episodes 1`), `train-tac` / `train-notac` (± tactile, same data).
  - `inference/` — `infer_node.py`: loads a ckpt, subscribes obs, publishes `/joint_command`
    (closed-loop in sim). Mirrors the recorder's preprocessing for train/infer parity. Tasks: `infer`,
    `infer-tac` / `infer-notac` (the ablation policies; pair with the Isaac `franka-eval` harness).
- The indexed children below own their own subtree.

## Local Contracts
- Entry points are pixi tasks (`pixi run -e ros <task>`), NOT a `python -m teleop_arkit` dispatcher.
  Task module paths: `teleop.{joint_command_node,arkit_receiver,sniff_stream,robot_state_pub,rr_viz}`,
  `data.{record,dataset,stats,cache}`, `training.train`, `inference.infer_node`. Task names are stable.
- The obs/action contract is `core.schema`; the robot spec is `core.robot` — never re-derive them here.
- Action space = joint-space absolute (`/joint_command`, 7 arm + gripper); EE-pose (`/target_frame`)
  is auxiliary.

## Work Guidance
- Import shared constants/builders from `core/`; do not duplicate joint names, entity paths, or the
  state/action builders (killing that duplication is why `core/` exists).
- ROS nodes spin via `core.rosutil.run`/`spin` for clean SIGINT/SIGTERM teardown.

## Verification
- `pixi run -e ros eval-rrd` (dataset shapes/alignment) · `smoke-act` (model+data sanity) ·
  `infer` (closed-loop — needs Isaac up via `franka-teleop`).

## Child DOX Index
- `core/` — shared contract: robot spec, `.rrd` schema, camera parsing, pydantic artifact configs, ROS spin helper.
- `data/` — record → store → read `.rrd`; stats; frame cache.
- `teleop/` — live teleop ROS2 nodes (IK, ARKit receiver, demo driver, rviz model, UDP sniffer).
- `policies/` — policy models + (planned) build registry; the model-agnostic seam.
