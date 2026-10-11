# Decision: one prepared DFT SCF owner for energy and derivative consumers

Status: implemented
Date: 2026-10-12

## Problem

Python singlepoint energy-only used the single-calculation route, while DFT
energy-plus-force used the prepared batch owner required by stationary snapshots.
That output-dependent fork made optimization qualification and ownership harder
to keep aligned. The optional Direct ERI value lease was also restricted to
RKS hybrids and destroyed at SCF completion, before legitimate post-SCF value
consumers. XC response re-exported the already verified frame and uploaded a
second copy of its density and quadrature.

## Decision

- Route every DFT singlepoint output selection through a one-item prepared batch.
  Keep the existing cold-solve, synchronous publication and temporary-owner
  lifetime; requesting forces adds a consumer, not a second SCF implementation.
- Admit the method-neutral full-range strict-FP64 value source for compatible
  Direct RKS/UKS, including Coulomb-only and nonlocal compositions. Keep the
  existing cold/size/mask/resource gates and exact fallback. Fitted, range,
  incremental and lower-precision providers retain their existing schedules.
- Charge UKS compensation as four canonical matrix planes, rather than two.
  Reuse the existing unrestricted distinct-orbit scatter and compensated fold;
  do not introduce another spin contraction algebra.
- At successful completion, invalidate captured borrowers and audit the source,
  but retain its values with the prepared owner until restart/destruction.
  Post-SCF fixed-density value profiling uses this source. Preserve the existing
  explicitly counted incumbent warm-energy handoff by bypassing, not destroying,
  the source for that action. Values do not cross solves or geometry changes.
- Split immutable source access from verified final-state export. Snapshot
  consumers can retrieve current geometry/basis/grid under the same token
  without another D/F/C/W export. ECP derivatives need geometry, not a new
  electronic frame. CUDA XC response borrows resident D and immutable quadrature;
  a producer-stream event orders its independent response arena. Its diagnostics
  count actual storage/transfers, including zero repeated frame exports.

## Invariants

The converged owner remains authoritative for D/W, occupations, geometry,
functional composition, XC state and integral-provider identity. Every borrowed
view requires the live owner/epoch/generation token; response actions reject
stale or closed owners before touching resident D. Geometry-source access alone
is not a new canonicality proof.

Ordinary ERIs are **not derivative ERIs**. Integral gradients still use the
original prepared first-derivative provider and contracted resident D/W, never
the zero-order value lease. Existing legal resident nonlocal XC features/grid
inputs remain reusable. Semilocal SCF tile scratch is not a complete retained
geometry derivative: missing XC jets, point adjoints and moving-grid response
still require their qualified derivative computations. No force capability or
precision domain is promoted by caching.

Public resource ledgers continue to refuse optional values until their complete
planner reserves them. Optional capacity/capability refusal keeps exact bounded
recomputation; numerical/driver failures are not downgraded to cache misses.

## Rejected alternatives

- Copying PBE0-specific optimizations into the force route would preserve the
  original divergence and fail to cover the other admitted DFT compositions.
- Lending ordinary ERIs as gradient values, or treating the final XC potential
  as geometry adjoints, would change the derivative equations.
- Persisting value sources across warm/seeded replays would require a separate
  policy/provenance and warm-baseline qualification; it is not needed to share
  the converged owner with its immediate derivative consumers.
- Keeping a second response density/grid would duplicate preparation, transfers
  and storage without improving freshness or independent response semantics.

## Evidence

- `tests/python/test_dft_shared_prepared_owner.py` exercises both DFT output
  dispatches and real complete PBE/PBE0 RKS/UKS endpoints above the resident threshold,
  with independent PySCF moving-grid analytic gradients and transfer/work gates.
- `tests/native/ks_phase_values_probe.cpp` tests restricted and signed-spin UKS
  J/K against independent PySCF matrices, Coulomb-only replay, screened masks,
  final-density/value-source reuse, warm handoff, restart and capacity refusal.
- `tests/python/test_response_native_cuda.py` retains independent RKS/UKS XC
  response, transpose, reconverged perturbation, stale-owner and failure gates,
  and now requires zero repeated preparation exports.
- `tests/python/test_ks_phase_source_auto_policy.py` protects the unchanged
  bounded selection. The executable native probe, rather than source-spelling
  assertions, protects the explicitly counted incumbent handoff.
- The default-promotion inventory records the broadened owner/source boundary.
  Its KS discovery audits the stable environment control, not policy-body text
  or internal call order. Compiled policy and native endpoint probes protect
  default selection, screening, ceilings, lifetime and resource refusal.

Real-device qualification must run in a finite Slurm allocation on a healthy
node, preserving its assigned CUDA visibility. Complete endpoint timings and
operator/source counts are separate from kernel timings; no speedup is inferred
merely from shared ownership or reduced transfer counters.

### Qualification on 2026-10-12

The source base is master `b7ebbfb7f`. The Release CUDA build targets SM 120 and
uses verified ccache 4.5.1 for C++/CUDA compilation. Slurm job 7629 on healthy
node1 uses `main`, `gpu:5090:1`, a finite 30-minute allocation and the scheduler's
unchanged device visibility. PySCF 2.14.0 / Libxc 7.0.0 provide the independent
oracles. The complete-owner fixture is the 12-atom WATER27 tetramer, spherical
def2-SVP, 96 AOs and 36,864 quadrature points, with energy/density tolerances
`1e-11` / `1e-9`.

The four complete-owner cases pass without a CPU derivative consumer. Maximum
independent energy error is `1.60e-12` Eh and maximum force component error is
`1.41e-10` Eh/Bohr, below the retained `2e-8` / `2e-7` gates. Force D/W upload,
one-electron H2D and repeated native integral frame-export counts are zero.
Each forced resident solve constructs 9,471,311 strict-FP64 ERI values; E and
E+F retain the same operator/precision selection and construction census.
Both actual iteration/operator counts are recorded rather than requiring
bitwise-reproducible DIIS stopping across independently launched cold solves.

Single complete-endpoint samples in seconds, including temporary prepared-owner
creation, synchronous publication and teardown, with compiled stationary assets:

| Method | E, values off | E, values on | E+F, values off | E+F, values on |
| --- | ---: | ---: | ---: | ---: |
| PBE RKS | 2.395 | 0.539 | 3.839 | 1.999 |
| PBE0 RKS | 3.162 | 0.562 | 4.512 | 2.061 |
| PBE UKS | 2.390 | 0.742 | 3.999 | 2.224 |
| PBE0 UKS | 3.754 | 0.771 | 5.393 | 2.263 |

These are qualification samples, not a throughput distribution or a general
speedup claim. The host ownership/resource/compatibility/inventory suite passes 1,499 tests
(four explicit CUDA gates skipped there). A separate CPU independent-gradient
suite passes eight LDA/PBE, r2SCAN RKS/UKS and global-hybrid cases in Slurm job
7626. Raw source-matched local receipts are under
`.artifacts/dft-shared-prepared/`; GPU fixture JSON includes actual endpoint work
counts and independent numerical errors.

Job 7629 passes all four focused native binaries, 16 complete-owner tests, six
resident-value tests and 16 native-response tests. Split-hybrid runtime JIT
initially fails because the validation PATH selects an NVCC wrapper without its
sibling `ptxas`; no numerical gate is relaxed. Using the actual CUDA toolkit
first in PATH while retaining verified ccache, job 7633 reruns the entire public
hybrid-force suite: 42 tests pass, including PBE0/B3LYP, M06-2X/MN15, aliased
compositions, RKS/UKS, FP64/qualified AUTO and both derivative reductions. Both
GPU qualifications execute the same core binary SHA-256
`beb66c1ee001913afb9b124457387c561bb254d87ae110a100b18cea03fb4056`.

## Revisit when

A complete planner admits optional persistent values/derivative scratch together,
or a generation-bound full-grid XC feature/adjoint cache is independently
qualified. Such work must preserve bounded fallback and distinguish value,
first-derivative and response capabilities.
