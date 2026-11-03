# Decision: separate routine correctness from exhaustive coverage and lowering

Status: implemented
Date: 2026-11-03

## Problem

Successful CI run `41619552277`, source
`ef03dc397f92fe49a7528032a819b8150e52190b`, recorded Python test steps of
1540 s (core-c), 1521 s (core-a), 1495 s (runtime-heavy), and 1336 s (core-b).
These are actual step spans, separate from runner queuing and setup. Its JUnit
reports retain the per-case worker costs. Some slow scientific replays were
already moved to nightly by #2953; this change does not count those savings again.

The same run's fast CUDA compile job reported 637/644 cacheable compiler calls
as hits, but configuration took 114 s and its two build steps took 94/384 s.
The resource jobs spent 61/89 s configuring a much larger target graph than
their validators consumed. A higher compiler-cache hit ratio alone cannot remove
uncached source lowering or unnecessary configure-time inventory discovery.

## Decision

- Keep routine PR/merge-group correctness tests, but reserve Python tracing,
  GCC instrumentation and Codecov uploads for scheduled/manual qualification.
  Preserve JUnit timing artifacts independently of coverage on every Python shard.
- Keep a representative native method-catalog oracle across both spins, method
  families and differing backend-admission outcomes. Retain the complete catalog
  and repeated all-class DF emission/determinism replay under `ci_nightly`.
  Existing focused class, source-equivalence and invalidation checks stay routine.
- Enable the existing bounded semantic source cache in the primary CUDA compile
  and CLI builds. Cache only source data with an exact science-input transport
  key; exclude bytecode and keep binary/compiler caches independent. The existing
  recipe, source hashes, inventory and byte-budget checks remain authoritative.
- Resource-only CMake contexts omit unconsumed test/CLI/stationary targets. They
  still generate their production grid or Direct J/K sources and run every
  existing release-resource validator. Full runtime/CLI gates are unchanged.

## Invariants and rejected alternatives

Do not delete the exhaustive tests, weaken tolerances, replace independent
oracles, disable the full CUDA runtime compilation, or treat a source-cache hit
as numerical/resource qualification. Do not restore generated binaries under a
source key, use broad transport fallbacks, relax ccache correctness, or change
release optimization flags. Missing source entries use ordinary generation;
corrupt/partial semantic entries fail closed. No GPU was executed for this change.

## Validation and revisit conditions

Regression tests exercise actual shell selection for all shards/events, coverage
producer/upload admission, source-cache wiring and retained CUDA resource gates.
Existing semantic-cache tests cover misses, cross-process hits, recipe changes,
corruption, truncation, byte bounds and concurrent publication. Source replay
and full native catalog checks validate the changed test tier independently.

Hosted end-to-end speedup must be measured on the resulting exact head; historical
step/worker costs above are diagnosis, not a new timing claim. Revisit the routine
catalog sample when new scientific families or admission contracts are added.

Three fresh local CPU processes emitted the shared s/p primitive source with no
cache, a cold semantic cache, and a warm semantic cache: 9.687 s, 9.632 s, and
0.346 s respectively. All outputs contain exactly 4,904,421 bytes with SHA-256
`0de331f8d0f637afdd763988eb6be590a99dcc40413b1781cf0d7291b96147c9`.
This measures source generation only, not NVCC or total CI latency. A local CUDA
configuration replay was unavailable because the installed 12.4 toolkit is below
the repository's required 12.9; no version check or resource gate was relaxed.

A separate ABBA diagnostic ran the unchanged
`test_df_lambda_frontier.py::test_frontier_native_slots_are_disjoint_and_scratch_is_already_bounded`
in four fresh pytest processes. Plain runs took 58.781/58.129 s; coverage runs
took 87.787/87.788 s. All four passed. These local process spans demonstrate
tracing overhead for that guard, not a prediction of the complete workflow's gain.

Validation completed with 873 CI-policy tests and 2269 subtests, 672 integration
tests and 75 subtests, 41 semantic-cache/reference tests, all 93 DF shell-unit
tests, and both native catalog tiers. The expanded routine catalog sample also
passed independently. Applicable pre-commit hooks passed after a transient HTTPS
fetch failure was retried through SSH with command-local Git configuration.
