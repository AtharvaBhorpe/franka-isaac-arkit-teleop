"""teleop_arkit — our ROS2 teleop adapter for the Franka in Isaac Sim.

Owns the input → IK → /joint_command path:
  * ik.py                 compact Pinocchio Cartesian servo-IK (our code)
  * joint_command_node.py ROS2 node: target pose -> IK -> /joint_command  (Phase 4)
  * arkit_receiver.py     iPhone/ARKit stream -> 4x4 pose                  (Phase 5)

We reuse only the servo-IK *technique* from SpesRobotics/teleop (Apache-2.0),
not the package itself. See PROJECT.md §1 / §4.
"""
