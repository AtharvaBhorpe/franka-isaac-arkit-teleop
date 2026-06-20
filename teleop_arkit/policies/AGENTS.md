# teleop_arkit/policies — policy models (the model-agnostic seam)

## Purpose
Where learned policies live. ACT + Diffusion baselines now (ACT is also the data-validation litmus);
a custom Gemma-based VLA later. Kept model-agnostic — adding/swapping a model doesn't touch the data
or training plumbing (see `registry.py`).

## Ownership
- `registry.py` — `build_model(config)`: validates `core.config.ModelConfig`, dispatches on
  `config["name"]` to the right policy. The model-agnostic seam train/infer go through.
- `act.py` — `ACTPolicy`: compact CVAE ACT (~11.6 M); `predict` returns the action chunk with z=0.
- `diffusion.py` — `DiffusionPolicy`: transformer Diffusion Policy (Chi et al. 2023); **in-house**
  cosine-β DDPM (ε-prediction) train + DDIM inference (no `diffusers` dep); `compute_loss` → MSE(ε).

## Local Contracts
- Every policy duck-types: `compute_loss(state, images, actions) -> (loss, metrics)`,
  `predict(state, images) -> (B, chunk, action_dim)`, and attrs `.cameras`/`.chunk`/`.img_hw`.
- Built only via `registry.build_model(config)` (train/infer never construct a policy directly); the
  ckpt stores `{model, config, stats}` with `config["name"]` so inference rebuilds the right policy.
- Add a model = a new module here + a branch in `registry.py` (+ a `ModelConfig` field for any new
  typed hyperparams).

## Work Guidance
None yet (only standard: construct from `ModelConfig`, keep model-agnostic).

## Verification
- `pixi run -e ros smoke-act` / `smoke-dp` (build + 50-step train: finite loss, no shape errors).

## Child DOX Index
None (leaf).
