"""Phase 1 · Step 2 — load the Franka Panda in Isaac Sim 5.1 and stream joint data.

Run (after `pixi install`):
    pixi run franka                 # with GUI viewport
    pixi run franka-headless        # no viewport

What it verifies:
  * Isaac Sim 5.1 boots from the pip package.
  * The Franka articulation references and loads from the Isaac asset library.
  * Joint names + joint positions are readable every physics step
    (the proprioception half of "getting the data correctly").

Notes:
  * Isaac Sim 5.x renamed the Python namespace from `omni.isaac.*` to
    `isaacsim.*`. Imports below target 5.1.
  * NOTHING from `isaacsim.*` / `omni.*` may be imported before SimulationApp
    is constructed — Kit must boot first.
"""

from __future__ import annotations

import argparse
import os

# Accept the Omniverse EULA non-interactively (required for standalone runs).
# Must be set BEFORE importing SimulationApp.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


# Candidate Franka USD sub-paths under the Isaac assets root. 5.x reorganized
# some asset locations, so we probe a few and use the first that exists.
FRANKA_USD_CANDIDATES = [
    "/Isaac/Robots/Franka/franka.usd",
    "/Isaac/Robots/Franka/franka_instanceable.usd",
    "/Isaac/Robots/FrankaEmika/franka/franka.usd",
    "/Isaac/Robots/Franka/franka_alt_fingers.usd",
]

PRIM_PATH = "/World/Franka"


def _url_exists(url: str) -> bool:
    """Return True if a USD URL resolves (works for local cache or Nucleus/cloud)."""
    import omni.client  # available once Kit has booted

    result, _entry = omni.client.stat(url)
    return result == omni.client.Result.OK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without the GUI viewport.")
    parser.add_argument("--steps", type=int, default=600, help="Physics steps to run (~10s @ 60Hz).")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index for rendering + physics (the RTX dGPU is usually 0).")
    args = parser.parse_args()

    # 1. Boot Kit FIRST.
    from isaacsim import SimulationApp

    # Hybrid-GPU laptops (here: RTX 5060 dGPU + AMD Radeon 860M iGPU) crash the
    # RTX renderer on its first frame (access violation in _wait_for_viewport)
    # when Kit's default multi-GPU path is on, because it tries to span the
    # unsupported AMD iGPU. Force single-GPU on the NVIDIA card.
    sim_config = {
        "headless": args.headless,
        "multi_gpu": False,
        "active_gpu": args.gpu,
        "physics_gpu": args.gpu,
    }
    sim_app = SimulationApp(sim_config)

    try:
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        # 2. Resolve the Isaac asset library root.
        assets_root = get_assets_root_path()
        if assets_root is None:
            print(
                "[franka] ERROR: could not resolve the Isaac assets root.\n"
                "         Check your internet connection / Nucleus, or set up the\n"
                "         local asset cache. See Isaac Sim docs > Assets."
            )
            return 2
        print(f"[franka] assets root: {assets_root}")

        # 3. Pick the first Franka USD that actually exists, then reference it once.
        franka_usd = next(
            (assets_root + rel for rel in FRANKA_USD_CANDIDATES if _url_exists(assets_root + rel)),
            None,
        )
        if franka_usd is None:
            print("[franka] ERROR: none of the candidate Franka USD paths resolved:")
            for rel in FRANKA_USD_CANDIDATES:
                print(f"           {assets_root + rel}")
            print("         Open the Isaac Sim asset browser to find the current path and\n"
                  "         add it to FRANKA_USD_CANDIDATES.")
            return 3
        print(f"[franka] referencing Franka USD: {franka_usd}")
        add_reference_to_stage(usd_path=franka_usd, prim_path=PRIM_PATH)

        # 4. Wrap as an articulation so we get joint APIs, register with the scene.
        franka = world.scene.add(Robot(prim_path=PRIM_PATH, name="franka"))

        # 5. Reset initializes physics + the articulation view.
        world.reset()

        print(f"[franka] DOF count: {franka.num_dof}")
        print(f"[franka] DOF names: {list(franka.dof_names)}")

        # 6. Step and stream joint positions.
        for i in range(args.steps):
            world.step(render=not args.headless)
            if i % 60 == 0:
                q = np.asarray(franka.get_joint_positions(), dtype=float)
                print(f"[franka] step {i:4d}  q = {np.round(q, 3).tolist()}")

        print("[franka] OK — Franka loaded and joint data streamed successfully.")
        return 0
    finally:
        sim_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
