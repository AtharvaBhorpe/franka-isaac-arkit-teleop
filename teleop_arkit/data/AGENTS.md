# teleop_arkit/data — record · store · read · stats (the `.rrd` data layer)

## Purpose
Turn live teleop into training-ready data and read it back: one Rerun `.rrd` per episode → aligned,
chunked tensors. Multi-rate sync is done at LOAD time, so native rates are kept on disk.

## Ownership
- `record.py` — `EpisodeRecorder` ROS2 node → one `.rrd` + `meta.json` per episode. Keys
  `s/e/f/d/q/h` + `/record/command`; `s` publishes `/episode/reset` first; flags `--cameras
  name=source`, `--image-max-width`, `--jpeg-quality`, `--view` (live Rerun blueprint). Task: `record`.
- `dataset.py` — `RrdDataset` (torch `Dataset`). Reads via the Rerun 0.33 chunk API
  (`RrdReader().store().stream().to_chunks()`); decodes camera JPEG via `core.cameras.preprocess_image`;
  options `use_cache`, `max_episodes`. Task: `eval-rrd`.
- `cache.py` — `build_cache(...)` → on-disk pre-decoded frame cache (regenerable; unlocks
  `num_workers>0`, which the eager-bytes path can't fork-share). Task: `cache`.
- `stats.py` — mean/std/min/max over state+action → `stats.json` (`core.config.DatasetStats`). Task: `stats`.

## Local Contracts
- One `.rrd` per episode + a `meta.json` (`core.config.EpisodeMeta`); `success=False` is excluded
  from training by default.
- Cameras are logged as `EncodedImage` JPEG (per-frame random access; sidesteps video keyframe-seek).
- **Alignment = latest-at on `log_time`, NOT `sim_time`** — our ROS nodes wall-stamp
  `/joint_command` + `/target_frame` while Isaac sim-stamps the rest, so `sim_time` is mixed-axis.
  The action chunk is the next-CHUNK window after each grid point.
- Entity paths and the state/action vectors come from `core.schema` (cite, don't restate).

## Work Guidance
- **Never `ros2 bag record` raw image topics** — uncompressed 1280×720 wrote ~85 MiB/s and filled
  the disk. Record light topics; log cameras as JPEG.
- Eager JPEG load won't scale past ~5 episodes: build the frame `cache` and shrink at source
  (`--image-max-width 640`, `--jpeg-quality ~80`) before a 50-demo campaign. Datasets live external
  (`~/rerun_episodes/`) — regenerable, deletable under disk pressure.

## Verification
- `pixi run -e ros eval-rrd` (dims 8 / action-chunk shapes / frames decode / throughput) · `stats` ·
  a `record` dry-run with Isaac up (`franka-teleop`).

## Child DOX Index
None (leaf).
