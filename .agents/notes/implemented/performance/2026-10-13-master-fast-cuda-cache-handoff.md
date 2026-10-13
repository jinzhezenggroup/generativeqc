# Decision: hand master fast CUDA cache misses to the ordered CLI producer

Status: implemented
Date: 2026-10-13

## Observed duplication

Master source `c4e092c457ababd8d0a985e12a2168ac6d1fe368` ran the fast CUDA
qualification and then the dependent native CLI producer in
[run 38349249133](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38349249133).
Both jobs restored transport snapshot
`ccache-linux-cuda-v4-12.9-sm120-78cb0449118cd87d7b3d6c5bebe876c9d26d3fed461582c4a6af0996f41591e7`.

- [CUDA job 115137902786](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38349249133/job/115137902786)
  compiled the production target from 05:08:59 to 05:28:23 UTC, about 19m24s.
  Its final counters were 513 cacheable calls, 259 hits and 254 misses.
- [CLI job 115144959333](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38349249133/job/115144959333)
  then configured/compiled the fast seed from 05:30:47 to 05:50:37, about
  19m50s, with 510 cacheable calls, 258 hits and 252 misses, before release.

The command, target and fast preset/build directory match. Both logs identify
GCC 13.3.0 and NVCC 12.9.86, with the same checkout root and
`build/cuda-release-sm120` path. CMake scopes `CCACHE_BASEDIR` to that binary
directory. However, qualification selects scalar CPU linear algebra and the
CLI runner selects OpenBLAS. `GenerativeQCCpuLinalg.cmake` adds different
preprocessor definitions/include directories to the main target, inherited by
host component objects. Do not pretend these complete identities are equal.

The logs have no per-object ccache trace, so they do not establish exactly
which misses are interchangeable or predict net time saved. Ninja timestamps
are completion times, not individual compilation start/end measurements.

## Decision

On master pushes only, the mandatory production fast build also uses ccache's
documented file-based secondary storage in a fresh runner-temporary directory.
Keep local storage enabled and disable reshare. Existing local hits are not
copied into the transport; only new results/manifests write through. There is
no custom parser, exporter or rewriting of ccache's internal entries.

After the existing required preflight checks, trim this secondary directory
to 2 GiB and upload it as a unique run/attempt artifact, retained for one day.
Pass the immutable artifact ID and producing attempt through the required
CUDA job's outputs. The ordered master CLI downloads by that exact ID using
download-artifact's current-run default; it must match the current attempt.
There is no name lookup, cross-run discovery or fallback to another artifact.

The existing CLI fast seed reads this secondary directory read-only after its
normal local snapshot restore. Ccache validates its ordinary compiler/source
keys, including preprocessing where appropriate, and promotes valid remote
hits into the CLI's local cache. Changed provider-dependent code still misses
and compiles normally. Both fast qualification and the full release build,
linkage, smoke tests, package checks, final cache publication and partial-master
failure saving remain mandatory and unchanged.

Upload/download are each non-gating and bounded to two minutes; trimming is
non-gating and bounded to one minute. No delta, failed upload, stale attempt,
failed/interrupted download or a secondary-storage miss leaves the seed with
permission to skip compilation. Failed download data is not enabled as a backend.
PRs, merge queues, manual runs and tags cannot publish or consume this handoff.
Existing workflow permissions and trusted full-snapshot discovery are unchanged.

## Verification and limits

Focused controls evaluate actual workflow conditions for event/ref/tier,
dependency artifact ID, current attempt and download outcome. Negative variants
remove each trust gate, read-only mode, no-reshare or the finite size cap and
must fail. Existing lifecycle/aggregate/stage tests cover the retained required
builds and partial-save behavior.

A tiny executable C++ cache fixture, using local ccache 4.14.1, established:
warm local hits produce no delta; a new result transfers; an unused macro change
can yield a preprocessed remote hit; meaningful compiler/options/header changes
miss; remote hits populate local storage; read-only data hashes/mtimes remain
unchanged; missing or unusable secondary storage falls back to compilation.
This is a cache protocol test, not a CUDA performance benchmark. Hosted CI
pins ccache 4.14; no compiler identity or correctness setting was weakened.

The transport consumes extra runner disk and artifact bandwidth. Its 2 GiB
cap may evict some new entries, safely reducing reuse. Warm builds normally
transport nothing. Actual CUDA reuse, transferred bytes and end-to-end savings
must be measured on normal hosted master runs before making a speedup claim.

## Rejected alternatives

- Do not upload the entire 3–5 GiB restored cache on every PR or master run.
- Do not copy CMake/build objects, skip the OpenBLAS fast seed, collapse provider
  contexts, change compiler fingerprints, add sloppiness or normalize real flags.
- Do not let PR/queue artifacts feed a trusted master cache or use artifact names
  as proof of provenance. Do not treat a missing transfer as successful qualification.

## Primary references

- [ccache 4.14 storage interaction and file backend](https://ccache.dev/manual/4.14.html#_storage_interaction)
- [ccache preprocessed input identity](https://ccache.dev/manual/4.14.html#_the_preprocessor_mode)
- [Immutable artifact outputs](https://github.com/actions/upload-artifact#using-outputs)
- [Download artifacts by ID](https://github.com/actions/download-artifact#download-artifacts-by-id)
