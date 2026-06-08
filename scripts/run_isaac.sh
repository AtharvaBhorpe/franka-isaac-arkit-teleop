#!/usr/bin/env bash
# scripts/run_isaac.sh
#
# Canonical entrypoint for the Isaac Sim 6.0 STANDALONE BINARY.
#
# We run the simulator from the downloaded binary (not the `isaacsim` pip
# package), using its bundled Python. The install lives outside the repo and is
# surfaced as the gitignored symlink `.isaac-sim` at the project root:
#
#     franka-isaac-arkit-teleop/.isaac-sim  ->  ~/isaac-sim/6.0.0
#
# This wrapper runs OUTSIDE pixi on purpose: the binary's python.sh sets up its
# own self-contained environment, so we don't want pixi's conda env layered on
# top of it. (pixi is reserved for the ROS2 env in Phase 3.)
#
# Usage:
#   ./scripts/run_isaac.sh isaac/load_franka_pickplace.py [--headless] [--device cuda]
#   ./scripts/run_isaac.sh --gui          # launch the Isaac Sim GUI app instead
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC="$REPO_ROOT/.isaac-sim"

if [ ! -e "$ISAAC/python.sh" ]; then
    echo "ERROR: Isaac Sim not found at $ISAAC" >&2
    echo "  Expected the gitignored symlink '.isaac-sim' -> your Isaac Sim 6.0 install." >&2
    echo "  Create it, e.g.:  ln -sfn ~/isaac-sim/6.0.0 \"$REPO_ROOT/.isaac-sim\"" >&2
    exit 1
fi

# Accept the Omniverse EULA non-interactively for standalone runs.
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

# ROS2 bridge (Phase 2): Ubuntu 26.04 isn't auto-detected, so the bridge can't
# pick a ROS distro and its RMW fails to load. Point it at the binary's INTERNAL
# ROS2 libs. These must be set in the shell BEFORE Isaac boots (the ros2 core ext
# initializes during SimulationApp startup, and the dynamic loader reads
# LD_LIBRARY_PATH at process start). Harmless for non-ROS runs. Override-able.
# We use Jazzy + FastDDS to match the RoboStack side.
export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
_ros_lib="$ISAAC/exts/isaacsim.ros2.core/$ROS_DISTRO/lib"
if [ -d "$_ros_lib" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$_ros_lib"
fi

if [ "${1:-}" = "--gui" ]; then
    exec "$ISAAC/isaac-sim.sh" "${@:2}"
fi

exec "$ISAAC/python.sh" "$@"
