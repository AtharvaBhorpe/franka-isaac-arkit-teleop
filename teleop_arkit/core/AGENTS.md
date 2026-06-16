# teleop_arkit/core — the shared contract (single source of truth)

## Purpose
One place the writer (recorder), reader (dataset), and consumer (inference) all import, so they
can't drift. This is the project's **machine contract** — change it here and the whole pipeline moves.

## Ownership
- `robot.py` — Franka RobotSpec: `ARM_JOINTS` (7), `FINGER_JOINTS` (2), `GRIPPER_JOINT`,
  `GRIPPER_OPEN`/`GRIPPER_CLOSED` (0.04/0.0), `HOME_ARM_Q`, `default_panda_urdf()`. Retarget to
  another arm by swapping these (robot-agnostic by construction).
- `schema.py` — `.rrd` entity paths (`STATE_ENTITY`/`ACTION_ENTITY`/`GRIPPER_ENTITY`/
  `TARGET_POSE_ENTITY`/`IMAGE_PREFIX` + the `*_LOG` writer strings), stats keys (`STATE_KEY`/
  `ACTION_KEY`), and the builders `by_name`/`state_vec`/`action_vec`/`image_log`. Add a modality =
  add an entity here; nothing else moves.
- `cameras.py` — `parse_cameras("name=source …")` → `[(name, kind, source)]` (kind `ros`|`usb`);
  `preprocess_image(bgr, hw)` → normalized CHW RGB tensor (the ONE resize/colour path → train/infer
  vision parity).
- `config.py` — pydantic v2 artifact schemas: `EpisodeMeta` (↔ `meta.json`), `ModelConfig` (↔ the
  ckpt `config`), `DatasetStats`/`StatEntry` (↔ `stats.json`). `extra="ignore"` keeps them
  forward-compatible; validating on read catches drift with a clear error.
- `rosutil.py` — `spin(node)`/`run(node)`: SIGINT/SIGTERM-tolerant spin + clean teardown.

## Local Contracts
- `observation.state` = 7 arm positions + gripper (8-D); `+8` velocities only if `include_vel`.
- `action` = 7 arm positions + gripper (8-D), from `/joint_command`.
- These vectors, entity paths, and the gripper convention ARE the contract the dataset, the model,
  and inference depend on — other docs cite this file, they do not restate the numbers.

## Work Guidance
- Changing a vector layout / entity path / gripper convention is a breaking change to the dataset,
  existing checkpoints, AND inference at once — change deliberately and re-run the full verify chain.

## Verification
- Exercised by `pixi run -e ros eval-rrd` (schema round-trip) and `smoke-act` (end-to-end).

## Child DOX Index
None (leaf).
