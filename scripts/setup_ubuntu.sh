#!/usr/bin/env bash
# scripts/setup_ubuntu.sh
#
# Ubuntu bootstrap for the Isaac Sim 6.0 side of the project.
#   1. Sanity-check the NVIDIA driver (Isaac Sim 6.0 wants >= 580.95.05 on Linux).
#   2. Verify the `.isaac-sim` symlink points at a real Isaac Sim install.
#   3. Ensure `pixi` is present (used for convenience tasks now, ROS2 env later).
#   4. Print the next commands.
#
# NOTE: we run Isaac Sim from the downloaded STANDALONE BINARY, not a pip
# package — so there is no multi-GB `pixi install` of isaacsim here anymore.
#
# Usage:
#   chmod +x scripts/setup_ubuntu.sh
#   ./scripts/setup_ubuntu.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

cyan()  { printf '\033[36m==> %s\033[0m\n' "$1"; }
gray()  { printf '\033[90m--  %s\033[0m\n' "$1"; }
warn()  { printf '\033[33m!!  %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------------------
# 1. NVIDIA driver check
# ---------------------------------------------------------------------------
MIN_DRIVER="580.95.05"
if command -v nvidia-smi >/dev/null 2>&1; then
    DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
    GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
    gray "GPU: ${GPU}  |  driver: ${DRV}  (Isaac Sim 6.0 wants >= ${MIN_DRIVER})"
    if [ "$(printf '%s\n%s\n' "$MIN_DRIVER" "$DRV" | sort -V | head -n1)" != "$MIN_DRIVER" ]; then
        warn "Driver ${DRV} is older than the validated ${MIN_DRIVER}. The RTX renderer may crash."
        warn "Install a recent NVIDIA Production Branch driver and re-run."
    fi
else
    warn "nvidia-smi not found. Install the NVIDIA driver (>= ${MIN_DRIVER}) before running Isaac Sim."
fi

# ---------------------------------------------------------------------------
# 2. Isaac Sim binary symlink
# ---------------------------------------------------------------------------
if [ -e ".isaac-sim/python.sh" ]; then
    gray "Isaac Sim found: $(readlink -f .isaac-sim)"
else
    warn ".isaac-sim symlink missing or broken. Point it at your Isaac Sim 6.0 install:"
    echo  "    ln -sfn ~/isaac-sim/6.0.0 .isaac-sim"
    echo  "  (move the extracted isaac-sim-standalone-6.0.0-linux-x86_64 there first)."
fi

# ---------------------------------------------------------------------------
# 3. pixi
# ---------------------------------------------------------------------------
if ! command -v pixi >/dev/null 2>&1; then
    warn "pixi is not on PATH. Install it, then re-run this script:"
    echo  "    curl -fsSL https://pixi.sh/install.sh | bash"
    echo  "  then open a NEW shell so PATH updates."
else
    gray "pixi found: $(pixi --version)"
fi

chmod +x scripts/run_isaac.sh 2>/dev/null || true

echo
printf '\033[32mDone. Next:\033[0m\n'
echo "  pixi run franka            # Franka pick-and-place scene (GUI)"
echo "  pixi run franka-headless   # same, no viewport"
echo "  pixi run isaac-sim         # full Isaac Sim 6.0 GUI app"
echo
warn "First Isaac Sim run pulls registry extensions + streams the asset library; it may take 10+ minutes."
warn "On a hybrid-GPU laptop, force the NVIDIA GPU, e.g.:"
echo  "    __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia pixi run franka"
