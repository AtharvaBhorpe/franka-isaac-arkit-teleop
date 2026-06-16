# Rerun 0.33.0 — pinned reference (Phase 7 data layer)

This project pins **`rerun-sdk==0.33.0`** (record) / **`rerun-sdk[catalog]==0.33.0`** (read+query)
as the single record→store→visualize→train format (`.rrd`). See [PROJECT.md](../../../PROJECT.md)
and the Phase-7 plan for *why* `.rrd` over `.npz`+`.mp4` / HDF5 / LeRobotDataset.

## Where to look for Rerun API specifics (priority order)
1. **The installed package = ground truth.** In the `ros` env (`rerun-sdk[catalog]` lives there now):
   ```bash
   pixi run -e ros python -c "import rerun.experimental as rre, inspect; print([m for m in dir(rre.RrdReader) if not m.startswith('_')])"
   ```
   Rerun's Python API docs are generated from these docstrings, so the installed package *is* the reference.
2. **`api-cheatsheet.md`** (this folder, generated from the installed package — see below). Most authoritative offline artifact.
3. **Hosted versioned ref** — https://ref.rerun.io/docs/python/0.33/ . Main docs: https://rerun.io/docs .
4. Re-verify on any version bump with the `/deps-doc-check` skill.

## Verified facts for 0.33 (release notes + 0.32→0.33 migration + dataframe/video docs)
- **Only breaking change 0.32→0.33:** the query/catalog pip extra was renamed
  `[datafusion]`/`[dataplatform]` → **`[catalog]`** (old names still resolve, deprecated). The
  logging API (`rr.init`/`rr.save`/`rr.set_time`/`rr.log`, `EncodedImage`, `VideoStream`) and the
  `.rrd` format are **unchanged** from 0.32.
- **Read local episodes** with **`rerun.experimental.RrdReader`** (the old `rerun.recording`
  module was deprecated in 0.32). **VERIFIED (step 4):** the local read API is the **chunk-processing
  API**, NOT a high-level dataframe — `rerun.dataframe` does NOT exist in 0.33, and the catalog
  `reader(using_index_values=, fill_latest_at=, window=)` is a *server/catalog* surface, not local.
- **Actual local read path** (see `teleop_arkit/rrd_dataset.py` — the working reference):
  `RrdReader(path).store().stream().to_chunks()` → per `Chunk`: `.entity_path`, `.timeline_names`,
  `.to_record_batch()` (Arrow). Columns: timelines `sim_time`/`log_time` (duration/timestamp ns) +
  `Scalars:scalars` (list<double>) or `EncodedImage:blob` (JPEG bytes). **No high-level local
  `fill_latest_at` — we do latest-at alignment ourselves** (searchsorted on a chosen fps grid), and
  **on `log_time`** not `sim_time` (sim_time is mixed-axis: our ROS nodes wall-stamp
  `/joint_command`/`/target_frame`, Isaac sim-stamps joints/cams).
- **Cameras as `EncodedImage` (JPEG per frame)** → each frame its own row; `EncodedImage:blob` per
  grid point → `cv2.imdecode`. **No H.264 keyframe-seek.** Measured **379 decoded samples/s** — not a
  bottleneck. (`VideoStream`/H.264 is the compactness alternative but needs keyframe seeking.)
- **0.33 bonus:** headless `ViewerClient` + screenshots (automated QA); chunk `apply_selector`/`Lens`
  transforms; `IndexedReader` for indexed/streaming reads.

## Regenerate `api-cheatsheet.md`
Introspect the installed package for exact signatures and write them here:
```bash
pixi run -e ros python - <<'PY'
import inspect, rerun.experimental as rre
for name in ["RrdReader", "IndexedReader", "Selector"]:
    obj = getattr(rre, name, None)
    print("##", name, "->", obj)
    try:
        print("  __init__", inspect.signature(obj.__init__))
    except (TypeError, ValueError):
        pass
    print("  methods:", [m for m in dir(obj) if not m.startswith("_")])
PY
```

## Sources
- Release 0.33.0 — https://github.com/rerun-io/rerun/releases/tag/0.33.0
- Migration 0.32→0.33 — https://rerun.io/docs/reference/migration/migration-0-33
- Dataframe queries — https://rerun.io/docs/reference/dataframes
- Video (AssetVideo vs VideoStream) — https://rerun.io/docs/reference/video
- Python API ref — https://ref.rerun.io/docs/python/0.33/
