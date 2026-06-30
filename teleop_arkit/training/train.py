"""Phase 7 · Stage 2 — train a policy on `.rrd` demos (ACT or Diffusion).

The **overfit test is the data-validation litmus**: train on 1–few episodes and watch the loss →
~0. If a known-good arch can't fit a handful of demos, the *data/pipeline* is broken (bad action
labels, broken sync, wrong normalization) — not the model.

Model is chosen with `--model {act,diffusion}` and built via `policies.registry.build_model`; the
ckpt stores its `config` so `inference.infer_node` rebuilds the right policy.

    pixi run -e ros smoke-act / smoke-dp           # 50 steps, 1 ep — finite loss + shapes (sanity)
    pixi run -e ros train --max-episodes 1         # overfit one demo (ACT)
    pixi run -e ros train-dp --max-episodes 1      # overfit one demo (Diffusion)
    pixi run -e ros train                          # all success episodes (ACT)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from teleop_arkit.data.dataset import RrdDataset
from teleop_arkit.policies.registry import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["act", "diffusion"], default="act")
    p.add_argument("--root", default="~/rerun_episodes")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps", type=int, default=0, help=">0: cap total optim steps (smoke).")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--kl-weight", type=float, default=10.0, help="ACT only.")
    p.add_argument("--diffusion-steps", type=int, default=100, help="Diffusion: DDPM train steps.")
    p.add_argument("--infer-steps", type=int, default=16, help="Diffusion: DDIM inference steps.")
    p.add_argument("--max-episodes", type=int, default=0, help=">0: load only first N episodes.")
    p.add_argument("--cameras", nargs="*", default=None,
                   help="Camera names to train on (subset of the recorded set). Default: all. "
                        "Modality ablation: omit 'tactile' to train vision+joints only.")
    p.add_argument("--out", default="~/rerun_episodes/checkpoints")
    p.add_argument("--ckpt-name", default="", help="Override ckpt filename (default: <model>.pt).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    t0 = time.time()
    ds = RrdDataset(args.root, fps=args.fps, chunk=args.chunk, img_hw=(args.img, args.img),
                    cameras=args.cameras, max_episodes=args.max_episodes)
    if len(ds) == 0:
        print("✗ no samples — check episodes / success flags"); return
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    print(f"dataset: {len(ds.episodes)} eps, {len(ds)} samples, cameras={ds.cameras}, "
          f"stats={'yes' if ds.stats else 'NONE (raw!)'}, device={args.device} ({time.time()-t0:.0f}s)")

    if args.model == "act":
        cfg = dict(name="act", state_dim=8, action_dim=8, chunk=args.chunk,
                   cameras=tuple(ds.cameras), kl_weight=args.kl_weight, img_hw=(args.img, args.img))
    else:
        cfg = dict(name="diffusion", state_dim=8, action_dim=8, chunk=args.chunk,
                   cameras=tuple(ds.cameras), img_hw=(args.img, args.img),
                   num_train_timesteps=args.diffusion_steps, num_inference_steps=args.infer_steps)
    model = build_model(cfg).to(args.device)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    # ACT keeps its legacy filename for back-compat; new models -> "<model>.pt".
    default_name = "act_min.pt" if args.model == "act" else f"{args.model}.pt"
    ckpt = os.path.join(out, args.ckpt_name or default_name)
    step, best, t0 = 0, float("inf"), time.time()
    model.train()
    for epoch in range(args.epochs):
        parts = []
        for batch in dl:
            state = batch["observation.state"].to(args.device)
            actions = batch["action"].to(args.device)
            images = {c: batch[f"observation.images.{c}"].to(args.device) for c in ds.cameras}
            loss, m = model.compute_loss(state, images, actions)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            m["loss"] = loss.item(); parts.append(m); step += 1
            if args.steps and step >= args.steps:
                break
        agg = {k: float(np.mean([x[k] for x in parts])) for k in parts[0]}
        print(f"epoch {epoch:3d} | step {step:5d} | "
              + " | ".join(f"{k} {agg[k]:.4f}" for k in agg) + f" | {time.time()-t0:.0f}s")
        if agg["loss"] < best:
            best = agg["loss"]
            torch.save({"model": model.state_dict(), "config": cfg, "stats": ds.stats,
                        "fps": args.fps, "loss": best, "metrics": agg}, ckpt)
        if args.steps and step >= args.steps:
            print(f"✓ smoke OK — {step} steps, final loss {agg['loss']:.4f}, no shape/NaN errors"); return
    print(f"✓ done — best loss {best:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
