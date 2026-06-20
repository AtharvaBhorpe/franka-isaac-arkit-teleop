"""Phase 7 · Stage 2 — normalization stats over the `.rrd` dataset -> stats.json.

Reads only the low-dim scalar streams (`images=False` -> no JPEG decode, fast + scalable),
computes per-dimension mean/std/min/max of `observation.state` and `action` across the
success episodes, and writes `<root>/stats.json` (consumed by rrd_dataset.RrdDataset).

    pixi run -e train stats              # over ~/rerun_episodes
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from teleop_arkit.core.schema import ACTION_ENTITY, STATE_ENTITY
from teleop_arkit.data.dataset import read_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="~/rerun_episodes")
    p.add_argument("--include-failures", action="store_true",
                   help="Include success=False episodes in the stats (default: skip).")
    args = p.parse_args()
    root = os.path.expanduser(args.root)

    buckets = {"observation.state": (STATE_ENTITY, []), "action": (ACTION_ENTITY, [])}
    n_eps = 0
    for rrd in sorted(glob.glob(os.path.join(root, "episode_*.rrd"))):
        meta_p = rrd[:-len(".rrd")] + ".meta.json"
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        if not args.include_failures and meta.get("success") is False:
            continue
        ep = read_episode(rrd, images=False)          # scalars only -> fast
        for key, (entity, arrs) in buckets.items():
            if entity in ep["scalars"]:
                arrs.append(ep["scalars"][entity][1])
        n_eps += 1

    stats = {}
    for key, (_entity, arrs) in buckets.items():
        if not arrs:
            continue
        a = np.concatenate(arrs, axis=0).astype(np.float64)
        stats[key] = {
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
            "min": a.min(0).tolist(), "max": a.max(0).tolist(), "count": int(a.shape[0]),
        }

    out = os.path.join(root, "stats.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {out}  ({n_eps} episodes)")
    for k, v in stats.items():
        print(f"  {k} [{v['count']} rows]  mean={np.round(v['mean'], 3).tolist()}")
        print(f"  {' ' * len(k)}            std={np.round(v['std'], 3).tolist()}")


if __name__ == "__main__":
    main()
