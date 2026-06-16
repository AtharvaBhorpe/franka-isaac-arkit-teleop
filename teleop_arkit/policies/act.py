"""Compact ACT (Action Chunking Transformer) — the data-validation baseline.

A faithful-but-small ACT (Zhao et al. 2023, ALOHA): per-camera CNN -> image tokens, a
proprioception (state) token, a CVAE latent (style variable; z=0 at inference), a
transformer encoder over the obs tokens, and a transformer decoder with `chunk` learned
query tokens -> an action chunk. Loss = L1(action) + kl_weight · KL(latent).

Backbone is a small from-scratch CNN (no torchvision dep) — enough to *overfit a few
demos* (the litmus) and, crucially, the policy DEPENDS on the images, so a clean overfit
also validates the image decode/align/resize pipeline. Swap in a torchvision ResNet18
backbone (`CNNEncoder` -> ResNet) for full-scale training / generalization later.

Model-agnostic by construction: it consumes whatever camera set + state/action dims the
`RrdDataset` provides (see `cameras`, `state_dim`, `action_dim`), so the same training
code serves ACT now and the Gemma VLA later.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Small strided CNN: (B,3,H,W) -> (B, (H/16)·(W/16), hidden) tokens."""

    def __init__(self, hidden: int):
        super().__init__()
        ch = [3, 32, 64, 128, 256]
        layers = []
        for i in range(len(ch) - 1):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1),
                       nn.GroupNorm(8, ch[i + 1]), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)          # 4 stride-2 convs -> /16
        self.proj = nn.Conv2d(ch[-1], hidden, 1)

    def forward(self, x):
        f = self.proj(self.conv(x))                 # (B, hidden, h, w)
        return f.flatten(2).transpose(1, 2)         # (B, h·w, hidden)


def _reparam(mu, logvar):
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


class ACTPolicy(nn.Module):
    def __init__(self, state_dim=8, action_dim=8, chunk=16, cameras=("scene", "wrist"),
                 hidden=256, enc_layers=4, dec_layers=4, nheads=8, dim_ff=1024,
                 latent_dim=32, dropout=0.1, kl_weight=10.0, img_hw=(224, 224)):
        super().__init__()
        self.cameras = list(cameras)
        self.chunk, self.latent_dim, self.action_dim = chunk, latent_dim, action_dim
        self.kl_weight = kl_weight

        # --- vision + state -> encoder tokens ---
        self.backbones = nn.ModuleDict({c: CNNEncoder(hidden) for c in self.cameras})
        n_img_tok = len(self.cameras) * (img_hw[0] // 16) * (img_hw[1] // 16)
        self.state_token = nn.Linear(state_dim, hidden)
        self.latent_out = nn.Linear(latent_dim, hidden)
        self.enc_pos = nn.Parameter(torch.randn(1, 2 + n_img_tok, hidden) * 0.02)
        enc = nn.TransformerEncoderLayer(hidden, nheads, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)

        # --- decoder: `chunk` learned queries -> action chunk ---
        self.query = nn.Parameter(torch.randn(1, chunk, hidden) * 0.02)
        dec = nn.TransformerDecoderLayer(hidden, nheads, dim_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.action_head = nn.Linear(hidden, action_dim)

        # --- CVAE encoder (train only): [cls, state, action_seq] -> mu, logvar ---
        self.cls = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.cvae_state = nn.Linear(state_dim, hidden)
        self.cvae_action = nn.Linear(action_dim, hidden)
        self.cvae_pos = nn.Parameter(torch.randn(1, 2 + chunk, hidden) * 0.02)
        cvae = nn.TransformerEncoderLayer(hidden, nheads, dim_ff, dropout, batch_first=True)
        self.cvae_encoder = nn.TransformerEncoder(cvae, enc_layers)
        self.latent_proj = nn.Linear(hidden, 2 * latent_dim)

    def _img_tokens(self, images):
        return torch.cat([self.backbones[c](images[c]) for c in self.cameras], dim=1)

    def forward(self, state, images, actions=None):
        B = state.shape[0]
        if actions is not None:                              # CVAE posterior (train)
            tok = torch.cat([self.cls.expand(B, -1, -1),
                             self.cvae_state(state).unsqueeze(1),
                             self.cvae_action(actions)], dim=1) + self.cvae_pos
            h = self.cvae_encoder(tok)[:, 0]                 # [cls] summary
            mu, logvar = self.latent_proj(h).chunk(2, dim=-1)
            z = _reparam(mu, logvar)
        else:                                                # prior mean (inference)
            mu = logvar = None
            z = torch.zeros(B, self.latent_dim, device=state.device)

        src = torch.cat([self.latent_out(z).unsqueeze(1),
                         self.state_token(state).unsqueeze(1),
                         self._img_tokens(images)], dim=1) + self.enc_pos
        memory = self.encoder(src)
        hs = self.decoder(self.query.expand(B, -1, -1), memory)
        return self.action_head(hs), mu, logvar             # (B, chunk, action_dim)

    def compute_loss(self, state, images, actions):
        a_hat, mu, logvar = self.forward(state, images, actions)
        l1 = F.l1_loss(a_hat, actions)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
              if mu is not None else torch.zeros((), device=state.device))
        return l1 + self.kl_weight * kl, {"l1": l1.item(), "kl": kl.item()}

    @torch.no_grad()
    def predict(self, state, images):
        """Inference: action chunk (B, chunk, action_dim) with z=0 (prior mean)."""
        return self.forward(state, images, actions=None)[0]
