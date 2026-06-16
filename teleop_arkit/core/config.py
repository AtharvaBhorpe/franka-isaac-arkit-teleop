"""teleop_arkit.core.config — typed schemas for the cross-stage artifacts (pydantic v2).

These are the contracts one stage writes and another reads. Validating them on read catches
schema drift (an old recording, a stale ckpt) with a clear error instead of a mystery
KeyError three calls deep. `extra="ignore"` keeps them forward-compatible — a newer writer's
extra field won't break an older reader.

  EpisodeMeta   <->  episode_NNNNNN.meta.json   (data.record writes · data.dataset reads `success`)
  ModelConfig   <->  the "config" dict in act_min.pt   (training.train writes · inference reads)
  DatasetStats  <->  stats.json                 (data.stats writes · data.dataset + inference read)
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, RootModel


class EpisodeMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    episode: int
    task: str
    cameras: dict[str, str]                       # name -> source (ROS topic | usb:N)
    state_dim: int
    action_dim: int
    jpeg_quality: int | None = None
    image_max_width: int | None = None
    sim_t_start: float | None = None
    sim_t_end: float | None = None
    success: bool | None = None                   # None until ended; False is excluded from training
    real_frames: dict[str, int] | None = None
    rerun_sdk: str | None = None


class ModelConfig(BaseModel):
    """The kwargs an ACTPolicy is built from — saved in the ckpt, replayed at inference."""
    model_config = ConfigDict(extra="ignore")
    state_dim: int = 8
    action_dim: int = 8
    chunk: int = 16
    cameras: tuple[str, ...]
    kl_weight: float = 10.0
    img_hw: tuple[int, int] = (224, 224)


class StatEntry(BaseModel):
    mean: list[float]
    std: list[float]
    min: list[float] | None = None
    max: list[float] | None = None
    count: int | None = None


class DatasetStats(RootModel[dict[str, StatEntry]]):
    """stats.json — {'observation.state': StatEntry, 'action': StatEntry}."""

    def entry(self, key: str) -> StatEntry:
        return self.root[key]
