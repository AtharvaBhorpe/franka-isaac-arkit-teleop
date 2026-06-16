---
name: deps-doc-check
description: Verify a pinned library/tool's real API against its installed version + official docs before writing code against it. Use when pinning or bumping a fast-moving dependency (rerun-sdk, lerobot, torch, etc.), or when unsure whether the project's intended API usage is valid at a given version. Produces a cited findings doc under docs/vendor/<lib>-<ver>/.
---

# deps-doc-check

Codifies a workflow this project repeats often: **pin a fast-moving library to a version → verify
its current API / migration / breaking changes → confirm the project's intended usage is valid →
produce cited findings with exact signatures → fold into `docs/vendor/` and `AGENTS.md`.**

## When to use
- Pinning a new dependency, or **bumping a pinned version**.
- Before writing non-trivial code against a fast-moving library (Rerun, LeRobot, torch, …).
- When a web answer about an API feels version-drifted, or you'd otherwise have to say
  "verify at implementation."

## Steps
1. **Installed package = ground truth.** If the lib is installed in a pixi env, introspect it:
   `pixi run -e <env> python -c "import <pkg>, inspect; help(<pkg>.<X>); print(inspect.signature(<pkg>.<X>))"`,
   and read `.pixi/envs/<env>/lib/python*/site-packages/<pkg>/` (including `.pyi` stubs). These
   exact signatures override anything the web says.
2. **Official docs for the pinned version.** Fetch the release notes, the migration guide for that
   exact bump, and the versioned API reference (e.g. `ref.rerun.io/docs/python/<ver>/`). Record
   every breaking change as old → new.
3. **Check intended usage.** Compare the project's planned calls (from the active plan / PROJECT.md
   / existing code) against the real API. Flag anything removed, renamed, or still experimental.
4. **Write findings.** Create/update `docs/vendor/<lib>-<ver>/findings.md`: pinned version, breaking
   changes, **exact signatures**, gotchas, a short pros/cons if a design choice depends on it, and
   **source URLs** for every claim. Generate an `api-cheatsheet.md` from introspection when possible.
5. **Update pointers.** Point `AGENTS.md`'s "Where to find API/library info" section at the new
   `docs/vendor/<lib>-<ver>/`, and add a dated note to `PROJECT.md` §9 if a decision changed.

## Output
A committed, reviewable `docs/vendor/<lib>-<ver>/findings.md` (+ `api-cheatsheet.md`) — deterministic
and re-checkable, instead of fuzzy web recall.

## Install
Project skill at `.claude/skills/deps-doc-check/SKILL.md` — picked up automatically for this repo
and invocable as `/deps-doc-check` (a new session may be needed for it to register).

## Use
`/deps-doc-check rerun-sdk==0.33.0` — or describe the lib + version + intended usage and invoke it.
Re-run whenever you bump the pin.
