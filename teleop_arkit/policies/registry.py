"""teleop_arkit.policies.registry — build a policy from a (validated) config dict.

The model-agnostic seam: `training.train` writes `config` (with a `name`) into the ckpt;
`inference.infer_node` rebuilds the right policy from it via `build_model`. Add a model = add a
branch here (+ a `core.config.ModelConfig` field if it needs new typed hyperparams). Every policy
duck-types the same interface (compute_loss/predict + .cameras/.chunk/.img_hw), so train/infer stay model-agnostic.
"""
from __future__ import annotations

from teleop_arkit.core.config import ModelConfig
from teleop_arkit.policies.act import ACTPolicy
from teleop_arkit.policies.diffusion import DiffusionPolicy


def build_model(config: dict):
    """Validated config dict → an (untrained) policy, dispatched on `config['name']`."""
    m = ModelConfig.model_validate(config)
    common = dict(state_dim=m.state_dim, action_dim=m.action_dim, chunk=m.chunk,
                  cameras=tuple(m.cameras), img_hw=tuple(m.img_hw))
    if m.name == "act":
        return ACTPolicy(**common, kl_weight=m.kl_weight)
    if m.name == "diffusion":
        return DiffusionPolicy(**common, num_train_timesteps=m.num_train_timesteps,
                               num_inference_steps=m.num_inference_steps)
    raise ValueError(f"unknown policy name {m.name!r} (expected 'act' or 'diffusion')")
