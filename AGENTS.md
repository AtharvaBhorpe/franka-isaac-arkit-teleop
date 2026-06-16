# AGENTS.md — project guide for AI assistants

> The cross-tool guide for AI assistants (the open `AGENTS.md` standard). Claude Code auto-loads it
> each session via the one-line `@AGENTS.md` import in `CLAUDE.md`; other agents read it directly.
> Keep it current. The **canonical, detailed source of truth is
> [PROJECT.md](PROJECT.md)** (plan, phase checklist, and the §9 dated decisions/gotchas log).
> This file is the fast index: what the project is, where to look, and how to work on it.

## What this project is
Real-time **teleoperation of a simulated Franka Emika Panda in NVIDIA Isaac Sim 6.0** from an
**iPhone (ARKit, via ZIG SIM PRO)** over **ROS2 + Pinocchio** — wave the phone, the arm mirrors it
and picks a cube into a bin. It reuses only the Pinocchio servo-IK *technique* from
SpesRobotics/teleop (not the package). One native **Ubuntu 26.04** machine; iPhone over WiFi.

**Now (Phase 7):** record teleop demonstrations to **Rerun `.rrd`** and train imitation-learning
policies — a compact **ACT** baseline first (also the data validator), then a **custom Gemma-based
VLA** (LLM backbone, vision encoder replaced by an embedding layer). The pipeline is intentionally
**model-, policy-, robot-, and modality-agnostic** (tactile sensing comes later) and industry-minded.

## Status snapshot
- **Phases 0–6: DONE** — PoC works end-to-end; iPhone teleops a full cube→bin pick-and-place;
  the §6 data-verification pass is complete.
- **Phase 7: IN PROGRESS** (plan: `.claude/plans/now-lets-talk-about-moonlit-rivest.md`; all decisions locked).
  - **Steps 1–3 DONE** — reset + cube-randomize (`/episode/reset`) + IK re-seed; `data/record.py`; **3 success episodes** in `~/rerun_episodes/`.
  - **Step 4 DONE** — `data/dataset.py` (reads `.rrd` via the 0.33 chunk API `RrdReader().store().stream().to_chunks()`; aligns multi-rate by **latest-at on `log_time`**, NOT `sim_time`) + `data/stats.py`. Verified on a real demo (740 samples, decode 379/s).
  - **Step 5 DONE** — `policies/act.py` (compact ACT, 11.6M) + `training/train.py` + `inference/infer_node.py` built; **overfit one episode → L1 0.57→0.066** ⇒ data validated. **NEXT (Step 6): closed-loop validation in sim, then scale demos.**
- **Envs MERGED (2026-06-09):** the separate `train` env was folded into **`ros`** (RoboStack + `torch 2.10+cu128` + `rerun-sdk[catalog]==0.33.0`, py3.12) — **one env for record+read+train+infer**; numpy 2.4.x coexists (no ABI clobber), ~7 GB freed. Exact 0.33 API: `docs/vendor/rerun-0.33/`.
- **Phase-7 gotchas:** single `/episode/reset` is DDS-dropped → publish 2–3×; **train aligns on `log_time`** (ros nodes wall-stamp cmds, Isaac sim-stamps); episodes large (scene-cam 1280×720 ≈ 80% — **eager JPEG load + record-res shrink = scale-prep before 50 demos**); disk ~89%; iPhone↔laptop over **Tailscale** (ZIG SIM → `100.112.249.78:50000`).
- All Phase-7 code is **uncommitted** (on disk).

## Repo map — where to look for what
| Path | What |
|---|---|
| `PROJECT.md` | **Source of truth**: plan, phase checklist, §9 dated decisions/gotchas log. |
| `README.md` | Quickstart. |
| `docs/HOWTO.md` | Run-from-a-fresh-machine guide (envs, launch, teleop). |
| `docs/ARCHITECTURE.md` | **Code flow + file connections (Mermaid) + end-to-end runbook** (teleop → record → train → infer). Start here if lost in the files. |
| `docs/vendor/<lib>-<ver>/` | **Pinned library docs + verified API findings** (e.g. `rerun-0.33/`). |
| `isaac/franka_scene.py` | Sim **library**: constants, cameras, scene builders, ROS2 graph. |
| `isaac/load_franka_pickplace.py` | Sim **app**: arg parsing, run loops, `main`. |
| `teleop_arkit/` | ROS-env package (py3.12). Sub-packages: `core/` (shared contract), `teleop/` (IK + ARKit nodes), `data/` (record/read `.rrd` + stats + cache), `policies/` (ACT), `training/`, `inference/`. Each carries its own `AGENTS.md`. |
| `config/tcp_offset.yaml` | Measured panda_hand→grasp-TCP offset (cross-check; IK uses the URDF `panda_hand_tcp` frame). |
| `.claude/plans/` | Active implementation plans. |
| `.claude/skills/` | Project skills (e.g. `deps-doc-check`). |

## Environments & how to run (pixi)
- `default` — launches the Isaac Sim 6.0 **standalone binary** (`.isaac-sim` symlink) via `scripts/run_isaac.sh`. Tasks: `franka`, `franka-headless`, `franka-ros`, `isaac-sim`.
- `ros` — **RoboStack ROS2 Jazzy, Python 3.12** + `torch 2.10+cu128` (Blackwell/RTX-5060, cuda ✓) + `rerun-sdk[catalog]==0.33.0` (pypi, layered on the conda solve). **ONE env for ALL of Phase 7 — record + read + train + infer.** (The separate `train` env was **merged in 2026-06-09**: conda `numpy 2.4.x` satisfies torch+pyarrow → no ABI clobber, one torch, ~7 GB freed; `cv2`/`numpy`/`pinocchio` come from conda.) Tasks: `ros-topics`, `ros-joints`, `ik-demo`, `ik-topic`, `arkit`, `robot-model`, `sniff`, `record`, `cam-hz`, `eval-rrd`, `stats`, `smoke-act`, `train` (+ `infer` to come).
- Sim runs from the binary's own Python; the ROS env talks to it over **localhost FastDDS, `ROS_DOMAIN_ID=0`**.
- **Known gotcha:** pixi envs were built before a folder rename, so console scripts (`ros2`, `cv2`) had stale baked paths. Workaround in place: symlink `~/franka-arktit-teleop → <repo>`. Clean fix: `rm -rf .pixi && pixi install`.
- **Working rule for accurate code: pin a version → install it → introspect the installed package** (below).

## Where to find API/library info (to write accurate code) — priority order
1. **Installed package = ground truth.** `.pixi/envs/<env>/lib/python*/site-packages/<pkg>/`; `python -c "import x; help(x.Y)"`; `inspect.signature`. (Rerun's Python API docs are *generated from* these docstrings, so this is the reference.)
2. **`docs/vendor/<lib>-<ver>/`** — pinned, verified findings + cheat-sheets (start at its README).
3. **Hosted versioned ref** — `ref.rerun.io/docs/python/0.33/` (Python), `docs.rs/rerun` (Rust); Rerun main docs = markdown in the rerun repo `docs/content/`.
4. **Refresh via the `/deps-doc-check` skill** when bumping a pinned version.

Do **not** answer library-API questions from memory for fast-moving libs (Rerun, LeRobot, torch) — read the installed package or the pinned vendor doc.

## Where to document progress / changes / observations
**Documentation model (DOX).** `PROJECT.md` is the append-only **diary/history** — it keeps
superseded/abandoned approaches on purpose, so DOX's "no diary / delete history" rule does **NOT**
apply to it. The **`AGENTS.md` tree** states only the **current** scope/contracts per directory (no
dated entries; trim stale text). The obs/action **machine contract is code** (`teleop_arkit/core/`)
— docs cite it, never restate it. Precedence: local how-to-work → nearest `AGENTS.md`;
decisions/narrative/plan → `PROJECT.md`; machine contract → code.

- **`PROJECT.md` §9 dated log** — append an entry per meaningful decision, change, or observation (the project narrative).
- **`PROJECT.md` phase checklist** — flip boxes as work completes.
- **`docs/HOWTO.md`** — update when run/launch steps change.
- **`.claude/projects/-home-atharvab-franka-isaac-arkit-teleop/memory/`** — cross-session memory (facts/gotchas); index in `MEMORY.md`.
- **`.claude/plans/`** — active implementation plans.

## Conventions & standing gotchas (project-wide; scope-local detail lives in the child `AGENTS.md`)
- **No `isaacsim`/`omni`/`pxr`/`usdrt` imports at module top** (Kit must boot first) — detail in `isaac/AGENTS.md`.
- **Never `ros2 bag record` raw image topics** (uncompressed 1280×720 ≈ 85 MiB/s → fills the disk); log cameras as JPEG `EncodedImage` — detail in `teleop_arkit/data/AGENTS.md`.
- **Teleop controls:** 1 finger = move (clutch) · 0 = freeze · 2-finger tap = toggle gripper — detail in `teleop_arkit/teleop/AGENTS.md`.
- **Action = joint-space** (`/joint_command`, 7 arm + gripper); EE-pose (`/target_frame`) is auxiliary. The vectors + entity paths are code: `teleop_arkit/core/schema.py`.
- **Commit messages** end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. **Commit/push only when asked**; branch off `main` first.

## Key ROS2 topics
`/joint_states` (9 DOF, ~40–70 Hz) · `/joint_command` (action: 7 arm + 2 finger) · `/target_frame` (PoseStamped, EE pose cmd) · `/gripper_command` (Float64; 0.04 open / 0.0 closed) · `/wrist_cam/image_raw` (640×480) · `/scene_cam/image_raw` (1280×720) · `/tf` · `/clock`.

# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md.
- **Documentation model** — `PROJECT.md` = append-only diary/history (keeps superseded ideas); the
  `AGENTS.md` tree = current scope/contracts only (no diary); the machine contract is code in
  `teleop_arkit/core/`. (Full statement under "Where to document progress / changes / observations".)
- **Commit/push only when explicitly asked** — work stays uncommitted on disk by default; branch off
  `main` first; commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Push back, don't just agree** — challenge weak ideas and surface tradeoffs rather than complying by default.

## Child DOX Index

Root owns the project-wide rail plus `docs/`, `config/`, `scripts/`, and `outputs/` directly. Direct children:
- **`teleop_arkit/AGENTS.md`** — the ROS-env package (teleop + IL pipeline); itself indexes `core/`,
  `data/`, `teleop/`, `policies/` and owns `training/` + `inference/` directly.
- **`isaac/AGENTS.md`** — the Isaac Sim side (separate binary-Python runtime; publishes the ROS2 streams).
