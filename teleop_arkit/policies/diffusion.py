"""Diffusion Policy (Chi et al. 2023) — a denoising-diffusion action head.

A second IL baseline alongside ACT (the pipeline is model-agnostic; see `policies/registry.py`).
This is the **transformer** DP variant: a per-camera CNN + state → obs memory tokens, and a
conditional transformer that denoises a Gaussian-noised action chunk back to the demonstrated
actions, cross-attending to the obs. Chosen over the CNN `ConditionalUnet1D` for an in-house impl:
no down/up-sampling or skip-connection bookkeeping to get wrong, it reuses the project's transformer
idioms (same blocks as ACT), and it suits the short (chunk=16) horizon.

Scheduler is **in-house** (no `diffusers` dep): cosine-β DDPM (ε-prediction) for training,
deterministic DDIM for fast inference. Actions are z-scored upstream (≈ unit variance), matching the
diffusion N(0,I) prior; DDIM's x0 estimate is clipped each step to a generous z-scored range
(`action_clip`, default ±5σ) for sampling stability — NOT to [-1, 1] (which would distort z-scored data).

Same interface as `ACTPolicy`: `compute_loss(state, images, actions) -> (loss, metrics)`,
`predict(state, images) -> (B, chunk, action_dim)`, and `.cameras` / `.chunk` / `.img_hw`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from teleop_arkit.policies.act import CNNEncoder


def _cosine_alpha_bars(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule → cumulative ᾱ[0..T-1] (float32)."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(1e-8, 0.999)
    return torch.cumprod(1.0 - betas, dim=0).float()          # (T,)


class _SinusoidalPosEmb(nn.Module):
    """Diffusion timestep → sinusoidal embedding. t: (B,) long → (B, dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        a = t[:, None].float() * freqs[None, :]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim=8, action_dim=8, chunk=16, cameras=("scene", "wrist"),
                 img_hw=(224, 224), hidden=256, n_layers=6, nheads=8, dim_ff=1024,
                 dropout=0.1, num_train_timesteps=100, num_inference_steps=16, action_clip=5.0):
        super().__init__()
        self.cameras = list(cameras)
        self.chunk, self.action_dim, self.img_hw = chunk, action_dim, tuple(img_hw)
        self.T, self.infer_steps, self.action_clip = num_train_timesteps, num_inference_steps, action_clip

        # --- obs → memory tokens (per-camera CNN + state token), like ACT's encoder ---
        self.backbones = nn.ModuleDict({c: CNNEncoder(hidden) for c in self.cameras})
        n_img_tok = len(self.cameras) * (img_hw[0] // 16) * (img_hw[1] // 16)
        self.state_token = nn.Linear(state_dim, hidden)
        self.obs_pos = nn.Parameter(torch.randn(1, 1 + n_img_tok, hidden) * 0.02)
        enc = nn.TransformerEncoderLayer(hidden, nheads, dim_ff, dropout, batch_first=True)
        self.obs_encoder = nn.TransformerEncoder(enc, 2)

        # --- denoiser: noisy action chunk (+ time + pos) cross-attends to obs memory → ε ---
        self.action_in = nn.Linear(action_dim, hidden)
        self.action_pos = nn.Parameter(torch.randn(1, chunk, hidden) * 0.02)
        self.time_emb = nn.Sequential(_SinusoidalPosEmb(hidden),
                                      nn.Linear(hidden, hidden), nn.Mish(),
                                      nn.Linear(hidden, hidden))
        dec = nn.TransformerDecoderLayer(hidden, nheads, dim_ff, dropout, batch_first=True)
        self.denoiser = nn.TransformerDecoder(dec, n_layers)
        self.eps_head = nn.Linear(hidden, action_dim)

        self.register_buffer("alpha_bar", _cosine_alpha_bars(num_train_timesteps))

    def _obs_memory(self, state, images):
        img = torch.cat([self.backbones[c](images[c]) for c in self.cameras], dim=1)
        tok = torch.cat([self.state_token(state).unsqueeze(1), img], dim=1) + self.obs_pos
        return self.obs_encoder(tok)

    def _eps(self, noisy, t, memory):
        """Predict the noise in `noisy` actions at diffusion step `t` (B,), given obs `memory`."""
        a = self.action_in(noisy) + self.action_pos + self.time_emb(t).unsqueeze(1)
        return self.eps_head(self.denoiser(a, memory))

    def compute_loss(self, state, images, actions):
        B = actions.shape[0]
        t = torch.randint(0, self.T, (B,), device=actions.device)
        ab = self.alpha_bar[t][:, None, None]                  # (B,1,1)
        noise = torch.randn_like(actions)
        noisy = ab.sqrt() * actions + (1 - ab).sqrt() * noise   # forward q(a_t | a_0)
        eps = self._eps(noisy, t, self._obs_memory(state, images))
        loss = F.mse_loss(eps, noise)
        return loss, {"mse": loss.item()}

    @torch.no_grad()
    def predict(self, state, images):
        """DDIM (deterministic, η=0) from Gaussian noise → action chunk (B, chunk, action_dim)."""
        B = state.shape[0]
        memory = self._obs_memory(state, images)
        x = torch.randn(B, self.chunk, self.action_dim, device=state.device)
        steps = torch.linspace(self.T - 1, 0, self.infer_steps, device=state.device).round().long()
        for i, t in enumerate(steps):
            ab_t = self.alpha_bar[t]
            ab_prev = (self.alpha_bar[steps[i + 1]] if i + 1 < len(steps)
                       else torch.ones((), device=state.device))
            eps = self._eps(x, t.expand(B), memory)
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()    # predicted clean actions
            x0 = x0.clamp(-self.action_clip, self.action_clip)  # DDIM stability: keep x0 on the (z-scored) manifold
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x
