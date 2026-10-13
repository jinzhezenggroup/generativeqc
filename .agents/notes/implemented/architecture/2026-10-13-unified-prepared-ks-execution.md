# Decision: one native prepared KS scheduler for scalar and batch endpoints

Status: implemented
Date: 2026-10-13

## Problem

Python DFT singlepoint already constructed a one-item prepared batch, while the
C ABI single-calculation owner used an independent execution method. The SDK
and CLI consumed that scalar method. Sharing SCF kernels and force sources did
not prevent duplicated preparation checks, CUDA initial-guess/retry loops and
result completion from diverging.

## Decision

Make the native DFT prepared calculation a scalar view of the existing prepared
batch scheduler. Split retained per-item scientific state from public execution:
only the scheduler drives CPU solves or advances CUDA streams, runs admitted
stationary forces, applies D4 and commits warm-state metadata. Both factories
use one preparation helper. Remove the former scalar CUDA SCF loop and scalar
force/correction completion instead of retaining a second execution path.

The scalar view forwards the private final-state/derivative ports to the sole
item. It owns no second D/W, grid, integral source or force implementation.
Native repeated calls remain warm-enabled; Python singlepoint remains cold by
creating a fresh owner. Both therefore use the same policies with their existing
lifetime settings. A failed warm scalar solve now follows the common bounded
batch retry policy. Resource/driver failures remain ineligible for cold retry.

Keep terminal item exceptions through the scheduler so scalar execution can
rethrow the original type and message. A status-only conversion would lose C
ABI context detail and could misclassify untyped errors. Nonconvergence still
returns completed diagnostics; other scalar failures publish no partial result.
No C ABI structures or symbols change.

## Rejected alternatives

- Routing Python back to the old scalar owner would lose the batch-bound
  stationary/response snapshot contract and leave ragged execution separate.
- Copying fixes between scalar and batch loops would preserve the ownership gap.
- Promoting every Python force/response consumer to native capability during
  this structural change would bypass method/backend/property qualification.

## Invariants

The retained item still owns token identity, scientific controls, final D/W,
backend resources and source-complete force admission. Rejected replay, failed
execution, reset and close invalidate exported state. Geometry changes preserve
the existing batch rebuild and last-good seed behavior. Python compatibility
consumers remain only where native preparation does not qualify the property.
This change is a DFT execution slice of #2227, not completion of response/HVP,
resource authority, method resolution or all-family runtime migration.

## Evidence

- `tests/python/test_dft_cli_prepared_parity.py` compares public Python and
  Python-free CLI energy/force endpoints, including moved geometries, Direct/DF
  and restricted/unrestricted inputs. Admitted Python derivative fallbacks are
  forbidden during this test.
- `tests/native/test_dft_api.cpp` injects a real scalar SCF allocation failure
  and checks exact status, unchanged caller energy, revoked diagnostics and
  recovery, alongside the existing ragged failure tests.
- Existing prepared lifecycle, stationary integral-source and independent
  PySCF force tests retain warm/replay, D/W/token, nonconvergence and scientific
  acceptance coverage.

### Local qualification on 2026-10-13

Base: `fca37ebf7543265f62c4556e6fbf712b33b4edc1`. The modified scheduler source
SHA-256 is `b8cb8a904f0aec4e3c512dfc231fb6a4b466589e569f5707ad90e39bac6840ed`.
The CPU Release build uses GCC 11.4 and verified ccache 4.5.1; generated compiler
commands invoke ccache with checkout-root path normalization. No scientific
kernel/compiler source changes are included.

- The full CPU build and all 94 CTests pass, including scalar allocation-failure
  recovery, native DFT integral sources, final states, SDK and CLI contracts.
- Prepared dispatch/lifecycle contract tests: 24 pass, four explicit GPU cases
  skip. Public Python/CLI comparison plus lifecycle endpoints: 11 pass, five
  explicit GPU cases skip.
- Independent PySCF 2.14.0 checks: 11 Direct PBE/global-hybrid energy/force cases
  and 12 UKS Direct/DF final D/W, moving-grid force and nonconvergence cases pass.
- After installation, all seven Python/CLI comparison cases pass against the
  installed library and binary. The C++ ragged-batch/explicit-KS SDK consumer is
  compiled against installed headers/library and passes with an empty runtime
  PATH/PYTHONPATH. The CLI children likewise have no interpreter/compiler on PATH.
- Native API contracts, runtime ownership anchors, electronic-structure and
  provider boundaries, public Python docstrings, Ruff and whitespace checks pass.
- The modified C++ translation unit also passes a CUDA-enabled host syntax
  compile against CUDA 12.9.1 headers. This is not a CUDA build/device receipt.
  Slurm exposes only drained `node3` during this run; no real GPU test is run.
  CUDA runtime/numerical qualification remains required before merge or any
  CUDA parity claim. No performance improvement is inferred from this change.

The first parity test exposed different existing default grids for composed
SDK models and Python `GridSpec`. The test explicitly aligns grid and screening
inputs; this migration does not change either public default. Independent solves
may differ in DIIS stopping at roundoff, so parity checks use converged E/F and
positive reported work rather than mandating identical iteration counts.

Ignored local build/install logs, test receipts and source/binary hashes are in
`.artifacts/validation/` in the task worktree. Python/oracle/installed receipts
precede the final whitespace-only formatting of the force-rejection statement:
their scheduler source SHA-256 is
`c63bc825184ac8ff9722a7a17ec32e51fb6eb52b73ab95db09fe35b551d1f211`
and core SHA-256 is
`a122d2a58fecd07ed4beac1dde539ab2892d2fde3eff420b80d6674040014bfd`.
The final formatting passes changed-line clang-format checks; the rebuilt tree
passes all 94 CTests again. Its core SHA-256 is
`ed05d66b0a14176c8c97656be544e54d54976ccfbaab58c39ca67f3b0e4383ec`.

## Revisit when

Another method family needs a scalar view of a ragged scheduler. Extract a
method-neutral view only when its capability ports and failure semantics match;
do not turn the KS-specific adapter into a second runtime framework.

## References

- #2227: shared C++ runtime ownership.
- [Earlier prepared DFT owner decision](2026-10-12-dft-prepared-energy-derivative-owner.md).
