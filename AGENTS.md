# GenerativeQC agent instructions

These repository-local rules complement user-level agent instructions. They apply
repo-wide unless a nested `AGENTS.md` adds more specific constraints.

## Repository-wide invariants

- Prefer scientific correctness, reproducibility, and complete-endpoint behavior
  over isolated kernel or microbenchmark wins.
- Production paths must not silently depend on CPU/PySCF/reference-oracle work
  unless that consumer is explicitly part of the production contract.
- Performance claims must include complete endpoint timing and semantic work
  counts; memory-bounded is not necessarily work-bounded. See
  `docs/maintainer/performance_engineering.md`.
- Numerical, precision, derivative, and response changes require an independent
  oracle/reference and explicit acceptance gates appropriate to the method.
- Keep explicit bounded fallbacks when a faster path depends on optional resident
  storage, identity/lifetime assumptions, or other resource preconditions.
- Preserve durable rationale for non-trivial architecture, numerics, performance,
  and compatibility decisions as Agent Notes under `.agents/notes/`.

## Compiler caching

- Use a compiler cache for local C++ and CUDA builds, including PR qualification
  builds and cache-miss runtime JIT compilation. Prefer `sccache`; fall back to
  `ccache`.
- Verify the selected launcher with `--version` before configuring or starting a
  cache-miss compile. If neither launcher is usable, report the missing prerequisite
  rather than silently starting an uncached full build or JIT compile. Python/JIT
  standard Python installations already declare the portable PyPI `sccache`
  wheel as a base dependency; source/development environments must still verify
  the executable rather than assuming dependency resolution succeeded.
- Keep CMake and runtime JIT selection consistent: automatic selection is
  `sccache` then `ccache`. Configure explicit CMake launchers with the selected
  executable when reproducibility requires pinning the launcher. Verify generated
  compiler commands actually invoke it; the launcher may be a normal CMake
  variable rather than a `CMakeCache.txt` entry.
- Runtime JIT artifact hits may replay without a compiler-cache executable because
  they do not compile. On an artifact cache miss, the C++/CUDA compiler adapters
  must wrap the actual compiler/link subprocess with the verified launcher. The
  launcher is execution provenance, not scientific/generated-source identity, and
  must not invalidate an otherwise identical JIT artifact key.
- Reuse the existing cache across builds. Do not clear or disable it for ordinary
  validation. Do not weaken cache correctness with sloppiness settings or override
  build/source identities to obtain hits.
- For `ccache` worktree builds, prefer matching in-tree build directory layouts
  and set `CCACHE_BASEDIR` to each checkout root, not just its build directory,
  so equivalent compiler input paths can reuse cached objects across worktrees.
  The automatic CMake ccache launcher overrides this environment value with
  `CMAKE_BINARY_DIR`; use explicit CXX and CUDA launcher settings when
  checkout-root normalization is required.
- Retain the selected launcher's `--show-stats` output before and after substantial
  compilation so cache use can be distinguished from cache hits. Compiler caching
  does not replace clean source provenance or real-device execution evidence.

## Release authority

- Repository cleanup, benchmark evidence retention, fixes, PRs and merges do
  not authorize creating/publishing/editing/deleting GitHub Releases or release
  assets, creating/pushing release tags, or dispatching release/distribution
  publishing workflows. Each such operation needs explicit user authorization
  for that operation in the current task.
- Do not substitute a fork Release or another external host to bypass this
  boundary. Use existing Git history and ignored local artifacts for historical
  recovery; a new external backup requires separate approval.
- A recovery link, checksum, CI result, prior instruction to continue work or
  permission to open a PR is not authorization to publish a Release.

## Scoped instructions

Read the closest applicable nested instructions before editing:

- `docs/AGENTS.md` for current-state documentation versus historical rationale;
- `python/generativeqc_compiler/AGENTS.md` for compiler ownership and generation rules;
- `src/integrals/AGENTS.md` for integral, derivative, precision, and scheduling
  constraints; and
- `.agents/notes/AGENTS.md` before adding or revising an Agent Note.

## Agent Notes

`docs/` describes the current system: what is true now and how to work with it.
`.agents/notes/` preserves why durable decisions were made, including discarded
alternatives, measured evidence, and conditions for revisiting them.

Add a note for a non-trivial change when it changes architecture/ownership,
scientific numerics or precision, algorithmic work/data movement, or a retained
compatibility/fallback policy. Ordinary bug fixes, tests, and local mechanical
refactors do not need notes. As a practical rule, if a diagnosis took substantial
investigation and a future agent could plausibly repeat a failed approach, record
it.
