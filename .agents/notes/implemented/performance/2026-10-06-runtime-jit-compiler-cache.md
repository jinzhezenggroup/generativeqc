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


## Bounded-process correction during review

A normal sccache client delegates compilation to a detached shared daemon.
Killing only the client's process group therefore cannot enforce the adapters'
finite compiler lifetime. Runtime sccache 0.16.0+ now uses a private Unix endpoint
and a foreground server whose process group is owned by the invocation. The
configured remote cache backends remain available. Local sccache disk LRU stores
require a single owner: startup recursively removes stale temporary files, so
private servers cannot share the user's disk directory or a subdirectory of it.
Instead, runtime workers nonblockingly lease the lowest free persistent JIT slot
under the exclusive GenerativeQC cache namespace. Sequential calls reuse a slot;
concurrent workers get separate stores. The lock FD is inherited by the server,
so caller death cannot immediately expose a still-live store to another writer.
Only closing FDs releases the lease; cleanup never explicitly unlocks a server's
shared open-file description. The private server also has a finite idle lifetime.
Disk policy is explicit through sccache environment options for this separate
pool; the shared config's disk section is not claimed to be preserved.
Launcher/slot state does not enter native artifact identity. Startup, client
requests and cleanup share one deadline. Cleanup kills only owned process groups,
including children that ignore SIGTERM; it never issues shared-server shutdown.
Distributed workers are rejected before compilation because local cancellation
cannot own their remote processes. Older sccache versions fall back to ccache.

Process regressions cover normal return, compiler failure, startup failure,
startup timeout, compile timeout, concurrent-call isolation and rejection of
remote workers. The existing fake-NVCC timeout test explicitly selects its raw
process-runner seam. A mandatory focused CI step installs the official pinned
sccache 0.16.0 wheel and checks real compiler/cache reuse, failure, timeout cleanup
and concurrent isolation, overlapping successful writes, warm reuse of both slots,
and preservation of a shared-cache temporary-file sentinel. Local environments without IPC must report that limit
rather than treat the real-sccache integration cases as passed.

Upstream evidence: sccache v0.16.0 `src/protocol.rs` has server shutdown but no
per-compilation cancellation request; `src/server.rs` awaits cache writes before
returning the successful compilation response. An unowned shared daemon and
server-wide shutdown were rejected because they respectively leak compiler work
or disrupt unrelated clients. A ccache-only replacement was rejected to preserve
the requested sccache feature.
