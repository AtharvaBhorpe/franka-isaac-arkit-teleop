"""Phase 7 · Stage 2 — build a derived, train-ready cache from episode `.rrd` files.

Decodes each camera frame once, resizes to a small square, re-encodes to JPEG, and packs
them into one blob per camera (+ an offsets index + timestamps); scalars go to `.npy`. The
`.rrd` stays the source of truth — this cache is regenerable and safe to delete.

Why: the eager "decode-all-JPEGs-into-RAM" path in rrd_dataset doesn't scale past ~5
episodes. The cache gives `RrdDataset` ~0 resident RAM (lazy blob reads via `BlobJpegList`),
fast random access, and `num_workers>0` safety. `RrdDataset` auto-uses a cache when present
(`<root>/.cache/v1/episode_*/`); delete that dir (or `--force`) to rebuild after re-recording.

    pixi run -e ros cache                      # cache all ~/rerun_episodes/episode_*.rrd
    pixi run -e ros cache --cache-res 224       # leaner (= train res); default 256
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from teleop_arkit.data.dataset import cache_dir_for, read_episode


def build_cache(rrd_path: str, cache_res=256, quality=85, force=False):
    """Build (or refresh) the cache dir for one episode. Returns (cache_dir, image_bytes)."""
    cdir = cache_dir_for(rrd_path)
    man_p = os.path.join(cdir, "manifest.json")
    if (not force and os.path.exists(man_p)
            and os.path.getmtime(man_p) >= os.path.getmtime(rrd_path)):
        return cdir, 0                                  # up to date

    ep = read_episode(rrd_path, images=True)            # the one expensive decode pass
    os.makedirs(os.path.join(cdir, "scalars"), exist_ok=True)
    os.makedirs(os.path.join(cdir, "images"), exist_ok=True)

    for ent, (t, v) in ep["scalars"].items():
        safe = ent.strip("/").replace("/", "_")
        np.save(os.path.join(cdir, "scalars", safe + ".npy"), v.astype(np.float32))
        np.save(os.path.join(cdir, "scalars", safe + ".t.npy"), t.astype(np.float64))

    img_bytes = 0
    for cam, (t, blobs) in ep["images"].items():
        offs = [0]
        with open(os.path.join(cdir, "images", cam + ".blob"), "wb") as f:
            for jpeg in blobs:
                bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    small = jpeg                        # keep original if undecodable
                else:
                    r = cv2.resize(bgr, (cache_res, cache_res), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", r, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    small = buf.tobytes() if ok else jpeg
                f.write(small)
                offs.append(offs[-1] + len(small))
        np.save(os.path.join(cdir, "images", cam + ".off.npy"), np.asarray(offs, np.int64))
        np.save(os.path.join(cdir, "images", cam + ".t.npy"), t.astype(np.float64))
        img_bytes += offs[-1]

    with open(man_p, "w") as f:
        json.dump({"version": 1, "source": os.path.basename(rrd_path),
                   "cache_res": cache_res, "quality": quality,
                   "scalars": list(ep["scalars"].keys()),
                   "cameras": list(ep["images"].keys()),
                   "t0": ep["t0"], "t1": ep["t1"]}, f, indent=2)
    return cdir, img_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="~/rerun_episodes")
    p.add_argument("--cache-res", type=int, default=256, help="square cache resolution (>= train img).")
    p.add_argument("--quality", type=int, default=85, help="re-encode JPEG quality for the cache.")
    p.add_argument("--force", action="store_true", help="rebuild even if up to date.")
    args = p.parse_args()
    root = os.path.expanduser(args.root)

    rrds = sorted(glob.glob(os.path.join(root, "episode_*.rrd")))
    print(f"caching {len(rrds)} episode(s) -> {root}/.cache/v1/  (res {args.cache_res}², q{args.quality})")
    total = 0.0
    for rrd in rrds:
        t = time.time()
        _cdir, nb = build_cache(rrd, args.cache_res, args.quality, args.force)
        total += nb
        msg = "up-to-date" if nb == 0 else f"{nb/1e6:5.0f} MB images in {time.time()-t:.0f}s"
        print(f"  {os.path.basename(rrd):24} {msg}")
    print(f"✓ cache built (~{total/1e6:.0f} MB images total). RrdDataset auto-uses it; "
          f"delete {root}/.cache/ (or --force) to rebuild after re-recording.")


if __name__ == "__main__":
    main()
