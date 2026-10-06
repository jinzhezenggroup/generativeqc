# Decision: share one verified compiler-cache policy with runtime JIT

Status: implemented
Date: 2026-10-06

## Problem

CMake builds already preferred `sccache` and then `ccache`, but the runtime
`CppCompilerAdapter` and `CudaCompilerAdapter` launched the underlying C++/NVCC
compiler directly on JIT artifact-cache misses. Repeated generated-source work
therefore bypassed the repository's compiler-cache policy even when a cache
launcher was installed.

## Decision

Runtime compiler adapters use the same launcher order as CMake: verified
`sccache` first, then verified `ccache`. Discovery checks `--version` before
the first compile for a given `PATH`. If neither launcher is usable, a cache-miss
compile fails with an actionable prerequisite message instead of silently running
uncached.

The PyPI `sccache` wheel is a base Python dependency because bounded runtime JIT
is reachable from supported production calculations, not only an advanced-user
extension. A normal Python installation therefore supplies the preferred cache
launcher without a separate setup step. Existing system `ccache` remains a
supported fallback and continues to serve CI environments that already persist
its cache.

The compiler-cache launcher is deliberately outside scientific/generated-source
artifact identity. A valid existing JIT artifact can replay without a launcher;
only a cache miss resolves and requires one. Underlying compiler/toolchain,
source, headers, flags and environment remain the artifact identity inputs.

## Rejected alternatives

Requiring only `ccache` would diverge from the existing CMake preference and
would unnecessarily require a system package when a portable PyPI `sccache`
wheel is available. Keeping `sccache` behind an optional extra was rejected:
bounded JIT is reachable from supported production calculations, so a standard
installation should not discover a missing cache prerequisite only at the first
artifact miss. Including launcher version in the JIT artifact key would
invalidate scientifically identical compiled artifacts when only the cache
implementation changed.

## Invariants

- Cache-miss CPU and CUDA runtime compilation must never silently bypass the
  verified launcher.
- `sccache` remains preferred over `ccache`, matching CMake.
- Artifact-cache hits remain independent of compiler-cache availability.
- Changing compiler-cache implementation alone must not relabel source or
  scientific artifact identity.
- Compiler-cache correctness must not be weakened with sloppiness or forged
  source/toolchain identities.

## Evidence

Focused tests cover `sccache` preference, `ccache` fallback, fail-closed
behavior when neither launcher exists, and CPU/CUDA adapter command wrapping.
Repository CI already provisions `ccache`; standard Python installations now
declare `sccache` directly without changing that existing CI cache store.

## Consequences

JIT cache misses gain a second reuse layer beneath the existing native artifact
cache. The first cache miss now has an explicit compiler-cache prerequisite.
Link commands also pass through the launcher; launchers may choose to pass
non-cacheable link work through unchanged.

## Revisit when

Revisit if the project adopts a single package-managed compiler launcher for all
supported platforms or if compiler-cache identity becomes necessary to explain a
reproducibility failure.

## References

- `AGENTS.md` compiler caching policy.
- `docs/developer/build.md` compiler-cache setup.
- `python/generativeqc_compiler/common/compiler_cache.py`.
