# teleop_arkit/policies — policy models (the model-agnostic seam)

## Purpose
Where learned policies live. A compact ACT baseline now (the data-validation litmus); a custom
Gemma-based VLA later. Kept model-agnostic so adding/swapping a model doesn't touch the data or
training plumbing.

## Ownership
- `act.py` — `ACTPolicy`: compact CVAE ACT (~11.6 M params). Built from `core.config.ModelConfig`
  kwargs (`state_dim`/`action_dim`/`chunk`/`cameras`/`kl_weight`/`img_hw`); `predict(state, images)`
  → action chunk `(chunk, action_dim)` (CVAE latent z=0 at inference).

## Local Contracts
- A policy is constructed from a `core.config.ModelConfig` and consumes a `core.schema` state vector
  plus per-camera images; the checkpoint stores `{model, config, stats}` so inference rebuilds it
  exactly.
- **Planned, not yet present:** a `base.py` Policy interface + a `registry.build_model` (the
  `__init__` docstring names the registry as intent). Add them when the second model lands; until
  then `act.py` is the only policy.

## Work Guidance
None yet (only standard: construct from `ModelConfig`, keep model-agnostic).

## Verification
- `pixi run -e ros smoke-act` (builds + runs a 50-step train: finite loss, no shape errors).

## Child DOX Index
None (leaf).
