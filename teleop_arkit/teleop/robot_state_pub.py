"""Publish the Panda model for rviz2 (Phase 4.1) via robot_state_publisher.

example-robot-data's URDF references meshes with `package://example-robot-data/...`
paths. rviz can't resolve those (it's a conda package, not an ament package), so
the RobotModel shows nothing. We rewrite them to absolute `file://` paths under
$CONDA_PREFIX and hand the URDF to robot_state_publisher, which then publishes
`/robot_description` (latched) and `/tf` computed from `/joint_states`. rviz2's
RobotModel + TF then render the live Franka.

Run alongside Isaac (so `/joint_states` flows), e.g. with `--control ros`:
    pixi run -e ros robot-model
Then in rviz2: add **RobotModel** (Description Topic `/robot_description`) + **TF**,
and set **Fixed Frame = panda_link0**.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from teleop_arkit.core.robot import default_panda_urdf


def main():
    share = os.path.join(os.environ["CONDA_PREFIX"], "share")
    urdf = open(default_panda_urdf()).read().replace("package://", f"file://{share}/")

    # Pass the URDF via a YAML params file (block scalar handles the multi-line XML
    # safely — inline `-p robot_description:=<xml>` would be YAML-mangled).
    body = "\n".join("      " + line for line in urdf.splitlines())
    params = (
        "robot_state_publisher:\n"
        "  ros__parameters:\n"
        "    robot_description: |\n" + body + "\n"
    )
    pf = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    pf.write(params)
    pf.close()

    print(f"[robot-model] meshes -> file://{share}/example-robot-data/...")
    print(f"[robot-model] starting robot_state_publisher (params: {pf.name})")
    print("[robot-model] in rviz2: RobotModel(/robot_description) + TF, "
          "Fixed Frame = panda_link0")
    subprocess.run([
        "ros2", "run", "robot_state_publisher", "robot_state_publisher",
        "--ros-args", "--params-file", pf.name,
    ])


if __name__ == "__main__":
    main()
