"""teleop_arkit — our ROS2 teleop adapter for the Franka in Isaac Sim.

Owns the input → IK → /joint_command path:
  * ik.py                 compact Pinocchio Cartesian servo-IK (our code)
  * joint_command_node.py ROS2 node: target pose -> IK -> /joint_command  (Phase 4)
  * arkit_receiver.py     iPhone/ARKit (ZIG SIM) -> /target_frame + /gripper_command  (Phase 5)
  * robot_state_pub.py    publishes /robot_description + /tf for rviz2 RobotModel  (Phase 4.1)
  * sniff_stream.py       UDP/TCP sniffer for the raw phone packets

We reuse only the servo-IK *technique* from SpesRobotics/teleop (Apache-2.0),
not the package itself. See PROJECT.md §1 / §4.
"""
