"""Phase 7 · Stage 2 — read Rerun `.rrd` episodes into a PyTorch Dataset.

Reads the per-episode `.rrd` written by `record_rrd.py` (entities on the `sim_time`
timeline: `observation/state`, `action`, `action/gripper_command`, optional
`action/target_pose`, and `observation/images/<cam>`), aligns the multi-rate streams
onto a uniform fps grid by **latest-at** (we do this ourselves — Rerun 0.33's local
read API is the chunk-processing API `RrdReader().store().stream()`, not a high-level
`fill_latest_at` query), decodes the JPEG `EncodedImage` frames, chunks the action, and
returns tensors. Normalizes low-dim via `stats.json` if present (see compute_stats.py).

`pixi run -e train eval-rrd`  — read ~/rerun_episodes back; verify shapes/sync/throughput.
"""
from __future__ import annotations

import glob
import json
import os
import time

import cv2
import numpy as np
import pyarrow as pa
import rerun.experimental as rre
import torch
from torch.utils.data import Dataset

from teleop_arkit.core.cameras import preprocess_image
from teleop_arkit.core.config import DatasetStats
from teleop_arkit.core.schema import ACTION_ENTITY, IMAGE_PREFIX, STATE_ENTITY


# ----------------------------------------------------------------------------
# low-level .rrd extraction (Rerun 0.33 chunk API)
# ----------------------------------------------------------------------------
def _log_time_ns(rb: "pa.RecordBatch") -> np.ndarray:
    """`log_time` (timestamp[ns]) column -> int64 nanoseconds.

    We align on **log_time** (the recorder's wall clock, set per `log()`, consistent across
    ALL entities) rather than `sim_time`: our ros-env nodes stamp `/joint_command` and
    `/target_frame` with WALL time while Isaac's joints/cameras carry SIM time — so `sim_time`
    is mixed-axis. log_time is one consistent axis (only sub-frame recorder-latency skew).
    """
    return rb.column("log_time").cast(pa.int64()).to_numpy(zero_copy_only=False)


def read_episode(rrd_path: str, images: bool = True) -> dict:
    """Read one episode `.rrd` -> dict of time-sorted streams.

    Returns: {"scalars": {entity: (t[s] (N,), values (N,D))},
              "images":  {cam:    (t[s] (N,), [jpeg_bytes])},
              "t0","t1": episode sim-time span}.
    Eagerly holds the JPEG bytes (fine for a handful of episodes; for 50+ add a frame
    cache — see the decode-throughput note in main()).
    """
    chunks = rre.RrdReader(rrd_path).store().stream().to_chunks()
    scal_t, scal_v, img_t, img_b = {}, {}, {}, {}
    for c in chunks:
        ep = c.entity_path
        if ep == "/__properties":
            continue
        rb = c.to_record_batch()
        names = rb.schema.names
        t = _log_time_ns(rb)
        if "Scalars:scalars" in names:
            vals = rb.column("Scalars:scalars").to_pylist()        # [[d,...], ...]
            scal_t.setdefault(ep, []).append(t)
            scal_v.setdefault(ep, []).append(np.asarray(vals, dtype=np.float32))
        elif images and "EncodedImage:blob" in names:
            cam = ep[len(IMAGE_PREFIX):]
            blobs = rb.column("EncodedImage:blob")
            for i in range(rb.num_rows):
                b = blobs[i].as_py()
                inner = b[0] if (b and isinstance(b[0], (list, bytes, bytearray))) else b
                img_b.setdefault(cam, []).append(bytes(inner))
            img_t.setdefault(cam, []).append(t)

    base = min(int(np.concatenate(v).min()) for v in list(scal_t.values()) + list(img_t.values()))

    def _sorted_secs(ts, vs=None):
        t = (np.concatenate(ts).astype(np.int64) - base) / 1e9     # 0-based seconds
        order = np.argsort(t, kind="stable")
        return (t[order], order) if vs is None else (t[order], np.concatenate(vs)[order])

    scalars = {ep: _sorted_secs(scal_t[ep], scal_v[ep]) for ep in scal_t}
    images = {}
    for cam in img_t:
        ts, order = _sorted_secs(img_t[cam])
        blist = img_b[cam]
        images[cam] = (ts, [blist[i] for i in order])

    allt = [v[0] for v in scalars.values()] + [v[0] for v in images.values()]
    t0 = max(float(t[0]) for t in allt)       # overlap window so latest-at always has data
    t1 = min(float(t[-1]) for t in allt)
    return {"scalars": scalars, "images": images, "t0": t0, "t1": t1}


# ----------------------------------------------------------------------------
# derived frame cache (built by cache_episodes.py) — lets the Dataset scale past a few
# episodes: ~0 resident RAM (lazy blob reads), fast random access, num_workers-safe.
# ----------------------------------------------------------------------------
def cache_dir_for(rrd_path: str) -> str:
    name = os.path.basename(rrd_path)[:-len(".rrd")]
    return os.path.join(os.path.dirname(rrd_path), ".cache", "v1", name)


class BlobJpegList:
    """Lazy random-access list of JPEG byte-strings packed in one blob file. Fork-safe:
    the file handle opens on first access (per worker process), not at construction —
    so DataLoader(num_workers>0) is fine (each worker gets its own handle)."""

    def __init__(self, blob_path: str, offsets: np.ndarray):
        self.blob_path, self.offsets, self._fh = blob_path, offsets, None

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        if self._fh is None:
            self._fh = open(self.blob_path, "rb")
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        self._fh.seek(a)
        return self._fh.read(b - a)


def load_cached_episode(cache_dir: str) -> dict:
    """Read a cache dir into the SAME shape read_episode returns (images = lazy BlobJpegList)."""
    man = json.load(open(os.path.join(cache_dir, "manifest.json")))
    scalars = {}
    for ent in man["scalars"]:
        safe = ent.strip("/").replace("/", "_")
        scalars[ent] = (np.load(os.path.join(cache_dir, "scalars", safe + ".t.npy")),
                        np.load(os.path.join(cache_dir, "scalars", safe + ".npy")))
    images = {}
    for cam in man["cameras"]:
        off = np.load(os.path.join(cache_dir, "images", cam + ".off.npy"))
        images[cam] = (np.load(os.path.join(cache_dir, "images", cam + ".t.npy")),
                       BlobJpegList(os.path.join(cache_dir, "images", cam + ".blob"), off))
    return {"scalars": scalars, "images": images, "t0": man["t0"], "t1": man["t1"]}


def _latest_at(ts: np.ndarray, t: float) -> int:
    """Index of the last sample with ts <= t (clamped to [0, len-1])."""
    return int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1))


# ----------------------------------------------------------------------------
# torch Dataset
# ----------------------------------------------------------------------------
class RrdDataset(Dataset):
    """Uniform-fps, latest-at-aligned samples over a directory of episode `.rrd`s.

    Each item: observation.state (state_dim,), observation.images.<cam> (3,H,W) float[0,1],
    action (chunk, action_dim). Skips episodes whose meta.json has success=False.
    """

    def __init__(self, root="~/rerun_episodes", fps=10.0, chunk=16, img_hw=(224, 224),
                 cameras=None, include_failures=False, stats_path=None, max_episodes=0,
                 use_cache=True):
        self.root = os.path.expanduser(root)
        self.fps, self.dt, self.chunk = fps, 1.0 / fps, chunk
        self.img_hw = tuple(img_hw)
        self.episodes, self.index = [], []     # index = [(ep_idx, grid_i), ...]
        self.cameras = cameras

        stats_path = stats_path or os.path.join(self.root, "stats.json")
        self.stats = (DatasetStats.model_validate(json.load(open(stats_path))).model_dump()
                      if os.path.exists(stats_path) else None)

        for rrd in sorted(glob.glob(os.path.join(self.root, "episode_*.rrd"))):
            meta_p = rrd[:-len(".rrd")] + ".meta.json"
            meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
            if not include_failures and meta.get("success") is False:
                continue
            cdir = cache_dir_for(rrd)
            if use_cache and os.path.exists(os.path.join(cdir, "manifest.json")):
                ep = load_cached_episode(cdir)        # fast, low-RAM, num_workers-safe
            else:
                ep = read_episode(rrd)                # decode straight from the .rrd
            ep["meta"], ep["path"] = meta, rrd
            if self.cameras is None:
                self.cameras = sorted(ep["images"].keys())
            grid = np.arange(ep["t0"], ep["t1"] - self.chunk * self.dt, self.dt)
            if len(grid) <= 0:
                continue
            ep["grid"] = grid
            ei = len(self.episodes)
            self.episodes.append(ep)
            self.index += [(ei, gi) for gi in range(len(grid))]
            if max_episodes and len(self.episodes) >= max_episodes:
                break       # load only N episodes (fast smoke / overfit)

    def __len__(self):
        return len(self.index)

    def _norm(self, key, v):
        if not self.stats or key not in self.stats:
            return v
        s = self.stats[key]
        return (v - np.asarray(s["mean"], np.float32)) / (np.asarray(s["std"], np.float32) + 1e-6)

    def _decode(self, cam, ep, t):
        ts, blobs = ep["images"][cam]
        jpeg = blobs[_latest_at(ts, t)]
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.zeros((self.img_hw[0], self.img_hw[1], 3), np.uint8)
        return preprocess_image(bgr, self.img_hw)

    def __getitem__(self, i):
        ei, gi = self.index[i]
        ep = self.episodes[ei]
        t = float(ep["grid"][gi])
        st_t, st_v = ep["scalars"][STATE_ENTITY]
        ac_t, ac_v = ep["scalars"][ACTION_ENTITY]
        state = self._norm("observation.state", st_v[_latest_at(st_t, t)])
        chunk = self._norm("action", np.stack(
            [ac_v[_latest_at(ac_t, t + k * self.dt)] for k in range(self.chunk)]))
        out = {
            "observation.state": torch.from_numpy(np.ascontiguousarray(state)).float(),
            "action": torch.from_numpy(np.ascontiguousarray(chunk)).float(),
        }
        for cam in self.cameras:
            out[f"observation.images.{cam}"] = self._decode(cam, ep, t)
        return out


# ----------------------------------------------------------------------------
# eval-rrd: read back + verify (shapes / sync / decode throughput / a batch)
# ----------------------------------------------------------------------------
def main():
    import argparse
    from torch.utils.data import DataLoader

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="~/rerun_episodes")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    t = time.time()
    ds = RrdDataset(args.root, fps=args.fps, chunk=args.chunk, img_hw=(args.img, args.img))
    print(f"built dataset in {time.time()-t:.1f}s: {len(ds.episodes)} episodes, "
          f"{len(ds)} samples, cameras={ds.cameras}, stats={'yes' if ds.stats else 'none'}")
    if len(ds) == 0:
        print("✗ no samples — check episodes / success flags"); return

    s = ds[0]
    print("sample[0] tensors:")
    for k, v in s.items():
        print(f"  {k:30} {tuple(v.shape)} {v.dtype}")

    n = min(64, len(ds))
    t = time.time()
    for i in np.random.permutation(len(ds))[:n]:
        _ = ds[int(i)]
    fps_decode = n / (time.time() - t)
    print(f"decode throughput: {fps_decode:.1f} samples/s over {n} random samples")

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    b = next(iter(dl))
    print("DataLoader batch:")
    for k, v in b.items():
        print(f"  {k:30} {tuple(v.shape)} {v.dtype}")
    print("✓ eval-rrd OK")


if __name__ == "__main__":
    main()
