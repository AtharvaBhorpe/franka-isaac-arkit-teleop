"""Phase 7 · Stage 2 — train the compact ACT on `.rrd` demos.

The **overfit test is the data-validation litmus**: train on 1–few episodes and watch L1 ->
~0. If a known-good arch can't fit a handful of demos, the *data/pipeline* is broken (bad
action labels, broken sync, wrong normalization) — not the model.

    pixi run -e train smoke-act                 # 50 steps, 1 ep — finite loss + shapes (sanity)
    pixi run -e train train --max-episodes 1 --epochs 300   # overfit one demo
    pixi run -e train train --epochs 100        # all success episodes
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from teleop_arkit.policies.act import ACTPolicy
from teleop_arkit.data.dataset import RrdDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="~/rerun_episodes")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps", type=int, default=0, help=">0: cap total optim steps (smoke).")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--max-episodes", type=int, default=0, help=">0: load only first N episodes.")
    p.add_argument("--out", default="~/rerun_episodes/checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    t0 = time.time()
    ds = RrdDataset(args.root, fps=args.fps, chunk=args.chunk, img_hw=(args.img, args.img),
                    max_episodes=args.max_episodes)
    if len(ds) == 0:
        print("✗ no samples — check episodes / success flags"); return
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    print(f"dataset: {len(ds.episodes)} eps, {len(ds)} samples, cameras={ds.cameras}, "
          f"stats={'yes' if ds.stats else 'NONE (raw!)'}, device={args.device} "
          f"({time.time()-t0:.0f}s)")

    cfg = dict(state_dim=8, action_dim=8, chunk=args.chunk, cameras=tuple(ds.cameras),
               kl_weight=args.kl_weight, img_hw=(args.img, args.img))
    model = ACTPolicy(**cfg).to(args.device)
    print(f"ACT: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "act_min.pt")
    step, best, t0 = 0, float("inf"), time.time()
    model.train()
    for epoch in range(args.epochs):
        parts = []
        for batch in dl:
            state = batch["observation.state"].to(args.device)
            actions = batch["action"].to(args.device)
            images = {c: batch[f"observation.images.{c}"].to(args.device) for c in ds.cameras}
            loss, p_ = model.compute_loss(state, images, actions)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            parts.append(p_); step += 1
            if args.steps and step >= args.steps:
                break
        l1 = float(np.mean([x["l1"] for x in parts])); kl = float(np.mean([x["kl"] for x in parts]))
        print(f"epoch {epoch:3d} | step {step:5d} | L1 {l1:.4f} | KL {kl:.3f} | {time.time()-t0:.0f}s")
        if l1 < best:
            best = l1
            torch.save({"model": model.state_dict(), "config": cfg,
                        "stats": ds.stats, "fps": args.fps, "l1": l1}, ckpt)
        if args.steps and step >= args.steps:
            print(f"✓ smoke OK — {step} steps, final L1 {l1:.4f}, no shape/NaN errors"); return
    print(f"✓ done — best L1 {best:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
