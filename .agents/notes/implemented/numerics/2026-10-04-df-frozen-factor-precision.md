# Decision: preserve DF raw residuals through cancellation-sensitive MO dots

Status: implemented; independent numerical review pending
Date: 2026-10-04

## Problem and attribution

The strict Rys repair in #1781 left the frozen ethane230 original elementwise
gate failing: Bov had 44 failures (maximum `8.102559946772392e-10`), Bvv had
988025 (`5.8931808133303246e-8`), and oovv had 5915
(`2.9842685544956282e-9`). The gate is `atol=rtol=3e-10`, with reference on
the right side, not a flat absolute threshold. The original evidence and C
were hash-verified and these failures reproduced before changes.

Three separately measured effects require different repairs:

1. Independent 90-digit normalized s/s/s and Gaussian-center derivatives found
   inaccurate original libcint reference values. For raw `[3,3,252]`, the
   oracle is `5.55819383594290434659468908658126174153963196970096068774163733075157752279591172313491562`.
   The native error was `-3.5065421727996216e-15`; original reference error was
   `-8.610713520491127e-14`. The coordinate-conversion contribution at this
   entry is zero. Its factor weight is about `-138355.68655`; the original
   worst Bvv difference cannot be attributed to native raw error alone.
   The old reference transformation order adds up to `9.114792767772606e-9`
   relative to a staged MO-first reconstruction.
2. Extended accumulation of the exact captured native raw values and actual
   root differs from the old GPU Bvv by `3.09468185626105e-9`, with 3993
   original-gate failures. A FP64 CPU reconstruction agreeing with a FP64 GPU
   reconstruction did not exclude their shared cancellation error. This is
   distinct from the much larger old-reference discrepancy.
3. Accurate low-degree raw integrals rounded before MO projection still lose
   residuals amplified by the diffuse orbital frame (largest C about 61) and
   metric root (largest element about 225). Retaining raw high/low components
   until the first dot resolves this. Rounding the accurately accumulated
   first MO intermediate to FP64 adds only `9.374723219934822e-13` in the
   frozen experiment; a two-component large MO tensor is unnecessary here.

The original archive is retained unchanged; a stale extraction with a modified
diagnostic log failed full manifest verification and was replaced by a pristine
hash-verified extraction. No audit entry was exempted.

## Decision and ownership

The compiler emits two-component FP64 arithmetic, reusing the existing integral
component IR, geometry/Boys generators and generic Gaussian traversal. Exact
FMA product residuals and explicitly rounded sums survive CUDA contraction.
The low-degree DAG must have exactly representable scalar constants; generation
fails closed otherwise. Total angular degree <=2 preserves raw residuals;
higher primitive classes keep strict existing math with compensated weights
and contraction. No native extended dtype or production reference dependency
is introduced.

CC uses an explicit canonical raw-expansion API, one traversal per row and one
lane per output, independent of ordinary-tile tuning overrides. Both MO dots
are compensated; only the first consumes a low input buffer. Large outputs and
final cuBLAS whitening remain FP64. Runtime code owns storage/launch/lifetime,
while compiler code owns arithmetic and the three TensorIR contractions.
Ordinary HF values and derivative policies keep their previous lowering.

The reference recipe is intentionally independent: quad-moment libcint roots
plus analytic center derivatives in host extended precision. Root routines
still have FP64 final eigensolver/output. Their largest sampled relative moment
error is `5.225705865606745e-15` against 90-digit incomplete-gamma moments.
The separate moment gate of `1e-14` is not a relaxation of the factor gate.
All low-degree angular placements, including spherical d components, have
independent analytic tests. The frozen generation made 1,662,640 root calls.

## Invariants and work

- Preserve C bytes, basis normalization/order, geometry conventions, rank 488
  and relative cutoff `1e-10`. Actual native absolute cutoff remains
  `1.087578826627117e-7`. No new RHF or CC solve is used for qualification.
- Capture the actual root with its storage convention: its F-order mathematical
  view must be transposed for CPU row-major reconstruction. Do not silently
  symmetrize it or replace it with the reference root.
- The existing physical pair projection is applied on both published paths.
  Independently check actual unprojected GPU Bov/Bvv against unprojected
  reference; pair projection cannot manufacture acceptance.
- Keep the old failed reference arrays. The repaired native source still fails
  against them, which is expected from the independently attributed reference
  error. The final evidence reports those failures explicitly.
- Preserve admission before device allocation and transactional publication.
  Added storage is one row, `8*N*Q` bytes, or 897,920 bytes for ethane230.
  Large intermediates retain their previous single-double storage.
- Semantic work stays 25,815,200 raw values, 230 rows, 232 matrix products
  (231 compensated plus one cuBLAS), 24,472,809,600 transform summands and
  5,873,584,104 block summands. Two raw components write 413,043,200 bytes.
  Summands are not hardware FLOPs; compensation increases hardware arithmetic.
  Captured source timing is instrumented, not an endpoint benchmark.

## Rejected alternatives

- Looser factor tolerances, energy agreement, altered orbital frame/rank or
  one-sided pair projection violate the qualification contract.
- A metric-only replacement cannot explain the controlled common-root failure.
- Compensating only GEMM does not recover residuals already rounded out of raw
  integrals; improving only the old reference also leaves native cancellation.
- `CINTqrys_schmidt` fails the independent moment gate at large T (about
  `5.76e-10` relative at four roots and `T=1e6`). The retained shim uses Jacobi
  through `T=20` and Laguerre above it, with moment tests across the switch.
- A native long-double/quad type or CPU integral oracle would violate strict
  production FP64/native execution. Extended host arithmetic is reference-only.
- Wider large MO buffers have no demonstrated need in this frozen frame and
  would unnecessarily increase admitted storage.

## Evidence and limits

The [reproduction procedure](../../../../docs/maintainer/df_frozen_precision.md)
and compact [evidence](../../../../benchmarks/results/df-factor-1840/README.md)
record the final library/source identities and original-gate results. Initial
real-GPU qualification was n2 Slurm job2265 on RTX PRO 6000: all seven original
checks passed, as did both unprojected factors. Final qualification uses the
committed replay tool and separately generated reference from job2266.
Builds/tests use ccache and finite n2 Slurm allocations; no node3 jobs ran.

This is an ethane230 factor precision qualification, not a performance
promotion. Benzene264 and complete large-system forces remain unqualified.
Small response gates cannot close either. Independent review of the numerical
repair is still required before closing #1840. Revisit the retained precision
scope only with independent primitive/factor evidence for other ill-conditioned
frames, and measure complete endpoints before proposing performance defaults.
