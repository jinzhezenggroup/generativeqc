# Decision: separate indexed AO-grid integration from frozen GPU qualification

Status: integrated on current master; native/GPU requalification pending
Date: 2026-10-06 (local)

## Problem

The #1893 indexed AO-grid and stable-owner experiments were qualified against
the detached source base `59cbd967f23117f75c415e760c631cd029c1bc7b`. That checkout
also contains unrelated branch history not present on current master. Publishing
its whole ancestry would mix independent work into this PR, while transplanting
the changes changes the scientific/build provenance of the tested program.

## Decision

Preserve the original source and ignored receipts in its worktree on the local
`codex/1893-indexed-ao-grid-qualified-snapshot` branch, commit
`b1f0949d967ba289f806b6e0e7456783afd61358`. Transplant only that commit's focused
changes onto master `c5ab37ec9e7b3d38d2e06729319f9eef66510e5c` in a separate PR
worktree. Do not carry the original unrelated ancestry into the submitted branch.

Reconcile the three overlapping files without reverting master's shared
contraction work:

- `CudaGrid.metrics` retains the existing lowering identity/work fields and adds
  optional AO/grid work counters independently.
- Native grid density/orbital projections keep `PreparedBoundedContraction`
  execution, its owner lifetime and preparation accounting; the census observes
  those executions instead of restoring an older direct GEMM route.
- `CudaXcLayout` keeps the prepared symmetric-product host reservation as well
  as the indexed-map derivative capability. Numeric arena admission is unchanged.

Host publication stand-ins now include the actual `AoGridWork` declaration and
the explicit gather spin count. Their existing failure, lease, recovery and
ordered spin-panel assertions remain intact, and compilation uses the shared
ccache-backed host compiler fixture. The older-library metrics test models the
retained lowering ABI while checking that missing optional counters remain
absent, rather than inventing them or weakening the compatibility assertion.

## Qualification boundary

The frozen scientific/native identity
`9fd0a78b20ebbd85a0a2b1d296495780dd134f8f2b6e6a3e4029389d617d14b0`
and Slurm jobs 6113/6114/6115 qualify the preserved experiment, not a rebuilt
current-master integration. The integrated source fingerprint is
`fcc3124f92a39b8189a938ceb23353a3b70378f77c72435a1cb569e14a9a1c2f`.
Neither that fingerprint nor host test success supplies new native/GPU evidence.
Do not load an older snapshot's native library into this checkout and report it
as current-source qualification.

Compiler structure checks cover 468 modules with zero dependency errors.
Ruff, format checks, clang-format and the default-promotion inventory pass. The
isolated work-metrics/publication regression run passes 519 tests. The final
device-free regression suite passes 922 tests with 282 explicit skips; a separate
current-master shared-lowering suite passes 12 tests with 12 skips. The 519-test
run is a subset, not additional independent qualification. Receipts are retained
locally under `.artifacts/pr-validation/`.
Native-library-backed spatial tests could not run in the new worktree because
it has no rebuilt native library; they are not classified as passed or skipped
qualification gates. Real GPU opt-in skips likewise provide no device evidence.

Submit as a draft pending a provenance-matched native rebuild and Slurm GPU
qualification on this integrated source. Preserve the original rejection:
48-atom moved schedule 3 is 6.2679% slower with 13 versus 12 Focks. The default
schedule remains 0, but the original experiment's changed compiled stack footprint
still requires insulation/measurement; a runtime default alone does not preserve
the old compiled resource footprint. P0-C remains unfinished.

## Rejected alternatives

- Submitting the old branch's unrelated ancestry or dropping current-master
  lowering to reproduce an old package.
- Reusing old binary/test identities as qualification for the new integration.
- Calling a correctness-qualified opt-in schedule a production performance win.

## References

- [Indexed AO/grid domains](../architecture/2026-10-05-indexed-ao-grid-domains.md)
- [Native CSR binding](../architecture/2026-10-05-native-csr-indexed-grid-binding.md)
- [Stable owner experiment](../performance/2026-10-05-stable-owner-schedule-experiment.md)
- #1893, including the qualification/rejection comment `6005075853`.
