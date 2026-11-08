# Decision: actual nonlocal and quadrature primals own responses

Status: implemented source migration; final endpoint qualification incomplete
Date: 2026-11-08

For the subsequent ownership census and source-migration closure, see
`2026-11-08-derivative-retirement-closure.md`. Earlier qualification states
below describe their original experiments, not the final campaign.

## Problem

Removing native Jet rules did not eliminate independently supplied formulas in
compiler templates. Nonlocal scales/pairs/rows, cached GFN2 radial weights,
quadrature Jacobians and zero-aware Python Becke products still duplicated
derivatives. This supersedes the corresponding remaining-source items in
`2026-11-07-integral-ao-scalar-ad.md`, not its outstanding endpoint gates.

## Decision

- `method.nonlocal_pair_ad` reuses Graph. Positive log-factorization and
  implicit-energy residuals produce pair cotangents. Actual backend local
  primals feed scale responses; rVV10's inverse-kappa coordinate change is
  differentiated before evaluation at one. Serial pullbacks preserve qualified
  contraction order, signed zeros, admission and raw failure domains. Copying
  derivative strings into generators was rejected.
- Cached ES2/external-charge weights convert their actual TensorIR energy to
  Graph, differentiate it, cancel the Cartesian factor symbolically and bind
  the cached kernel. Runtime division by zero displacement is not introduced.
  Charge/kernel interleaving schedules AD factors rather than supplying a
  reciprocal-cube derivative.
- `integral.quadrature` owns neutral scalar primals, not ECP/DFT policy. Both
  Legendre consumers carry polynomial JVPs. DFT retains its n=1 initial state
  and fresh final slope; ECP retains its different bounds/stopping/final-weight
  convention. Rational and Treutler Jacobians use AD. The rational Jacobian
  uses an equivalent pole-coordinate primal, while radius evaluation retains
  the non-cancelling quotient.
- Python quadrature lowers a cached scalar DAG to straight-line arithmetic on
  complete NumPy radial vectors, not per-point Graph interpretation or a new
  differentiation algebra/native JIT. Bare generation still needs no NumPy.
  Generated Python accepts only validated operations/numeric payloads, without
  builtins or external source inputs.
- Zero-aware partition products differentiate an exponential times a
  degree-zero/one/two elementary symmetric zero-factor primal. Half-square AD
  supplies the diagonal moment's mixed response. Exact multiplicities cancel
  before runtime to avoid doubling large finite contractions. Reductions keep
  sufficient statistics and O(tile * atoms) storage, not a partition Hessian.
- AO envelope coefficients share Gaussian AD; their oracle is an independent
  test-only rounded recurrence plus sampled AO jets, not shared Python math.

## Evidence

Ignored receipts are in `.artifacts/classical-derivatives/`.

- Latest local-primal/pair checks passed 75 tests with two unavailable-Clang
  skips; earlier nonlocal/ECP method checks passed 272 with six skips. Existing
  exceptional, FP32/FP64, signed-zero and broad-exponent gates remain, with
  additional 90-digit primal-only differentiation.
- Quadrature passed 152 checks: native grids, independent PySCF rules/pruning,
  80-digit primal-only AD and Legendre moments through n=512. The pole oracle
  evaluates at the stored affine-map coordinate: rounding node+1 is amplified
  by the fourth-order spherical measure pole near the largest abscissa.
- Initial small-VV10 complete CPU lifecycles showed ~2% paired overhead. Do not
  discard it or offset it with large-grid wins. A randomized 40-pair single-core
  joint-primal rerun measured total ratios 0.99865 (VV10), 1.00108 (rVV10), with
  numerical/work parity. Preparation was noisier than execution. Per-pair/stage
  evidence is `nonlocal-small-controlled.json`; final-source repeats remain due.
- GPU preparation initially failed because source sync omitted tracked
  `src/api/build_identity.hpp.in`; full rsync fixes that omission. Preserve failed
  logs rather than fabricate generated identity. All GPU work uses finite Slurm
  jobs on n1 and verified sccache 0.16.0; never n3 or overridden device visibility.

## Remaining work

Final source-matched CPU/CUDA full endpoints, performance acceptance and the
remaining derivative census are required before a retirement PR. Source
migration and selected successes do not qualify every method or bounded
fallback. This task does not authorize release publication.
