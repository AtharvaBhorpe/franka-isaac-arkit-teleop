"""teleop_arkit.policies — policy models + the build_model registry.

ACT (`act.py`) and Diffusion (`diffusion.py`) now; a Gemma VLA later. `registry.build_model`
dispatches on `config['name']`; every model duck-types the same interface (see AGENTS.md).
"""
