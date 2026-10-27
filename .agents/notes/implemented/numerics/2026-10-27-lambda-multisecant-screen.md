# Decision: keep CCSD-history Lambda preconditioning as a numerical screen

Status: implemented (tooling prototype only; native integration remains proposed)
Date: 2026-10-27

## Problem

Native CUDA DF-CCSD already pays for a history of residuals and iterates.
Those secants could approximate the inverse Jacobian used by Lambda, saving
expensive exact actions without changing analytic response equations. It is
not yet established that the saved work pays for history retention, conversion
and construction in a complete force endpoint.

## Decision

Implement a bounded host-only prototype and a reproducible small-molecule
screen before adding a native resource/ownership contract. Nothing in native
CCSD, Lambda, the public API, CBS policy or default preconditioning changes.
The screen runs generated physical CCSD equations and existing GMRES, with an
optional minimal C ABI bridge to the **actual native** GMRES controller.

`SecantHistory` owns detached FP64 pairs, resets on approximate precision,
changed reference/coordinate epochs, denominators or damping, and recovers
incoming amplitudes from Anderson maps without another residual evaluation.
A thin SVD of jointly scaled S/Y columns constructs `H = H0 + C U.T`.
`H0 = -1/D`, not `1/D`: the existing physical Lambda diagonal is `-D`.
The callback applies `H.T`, while GMRES always applies the exact physical J.T.

## Investigation and rejected shortcuts

- Native `run_diis` with `diis_input_residual=true` stores `(G(t), R(t))`.
  Differences of G are not amplitude secants. Recovery uses the same fixed
  denominator/damping epoch and is subject to floating-point rounding. Do not
  reuse this identity after clipping or an unrelated state projection.
- `finish_mixed_warmup` clears the ring. FP32-evaluated residuals must never be
  relabeled physical FP64 merely because they are stored in double arrays.
- `RestrictedPairCoordinates` triangularizes `(i*v+a, j*v+b)`. Lambda walks
  dense ijab representatives. The weight and permutation are both required.
  The prototype's adapter/test uses the compiler coordinate owner.
- Six stored states provide at most five adjacent secants. The screen keeps
  the newest seven physical samples, including the final accepted point.
  A native exporter could combine the existing six-row ring with the final
  already-computed replay point; it must not add a hidden residual evaluation.
- Do not invert `Y.T @ Y`. Column scaling followed by thin SVD avoids squaring
  the condition number. Scaling does not change the full-column-rank inverse;
  after truncation the selected space is that of the normalized columns.
- Python GMRES with `true_residual_every=30` checks only at that interval or
  breakdown/restart. These cases all reached 30 steps, hiding the difference.
  Native GMRES also checks on predicted convergence. The retained screen uses
  that controller; the Python fallback uses interval one. No production solver
  change was made to improve these measurements.
- Ordered `(T)` amplitude cotangents need the existing pair-space adjoint
  projection before packing. Ordinary packing is not a valid conversion.

## Invariants

Only the preconditioner is approximate. GMRES's physical solve tolerance is
`1e-12`; the expanded-equation physical audit retains `1e-9`. Small independent
determinant-space derivatives verify generated J.T, RHS and all three Lambda
solutions. Tiny dense matrices appear only in tests, never in the fit or screen.
Rank/budget refusal retains H0. Invalid or nonfinite data raises rather than
being presented as usable history. No neural model or reference oracle enters
a production path.

## Evidence

The retained small-molecule experiment uses supplied pinned STO-3G Hamiltonians,
FP64 Anderson mixing with history six, ranks 0/4/6, two repetitions in reversed
order, and native GMRES. Both CCSD and CCSD(T) produced these **GMRES** J.T counts
(the independent audit adds one action to every row):

| Molecule | Diagonal | Rank 4 | Rank 6 |
| --- | ---: | ---: | ---: |
| H2O | 17 | 17 | 16 |
| NH3 | 18 | 18 | 17 |
| CH4 | 17 | 16 | 16 |

All independent Lambda residual norms are below `9e-13`; changes from the
diagonal solution are below `3e-13`. Raw identities, counts, phase times and
numeric storage estimates are in
[`ccsd.json`](../../../../benchmarks/results/cc-lambda-multisecant-20261027/ccsd.json)
and [`ccsdt.json`](../../../../benchmarks/results/cc-lambda-multisecant-20261027/ccsdt.json).
These CPU interpreter timings are diagnostic, not native CUDA performance
evidence. Scientific tests cover dense nonsymmetric pseudoinverse parity,
transpose correctness, weighted metrics/order conversion, rank deficiency,
epoch/lifetime/budget rejection, independent determinant oracles and compiled
native callback success/failure.
The focused suite passes 14 tests; Sphinx passes with warnings treated as
errors. Compiler/cache receipts and final record checksums are retained in
[`qualification.json`](../../../../benchmarks/results/cc-lambda-multisecant-20261027/qualification.json).
These submitted records were regenerated against base revision
`fa5a5c5613e816924bfe0b03734ecb3bad2cf5f9` after integrating the current native
GMRES controller; the action-count table was unchanged. Each screen binds its
tooling/controller source hashes and the rebuilt bridge binary hash.

## Consequences and revisit conditions

The measured benefit is modest: one saved J.T action for rank six, and no
benefit from rank four on two cases. This supports a guarded experiment, not
default activation or a C4/C6/C7 speedup claim. The screen's conventional small
Hamiltonians, all-FP64 history and CPU actions do not qualify DF graphs, native
FP32-to-FP64 transitions, GPU ownership, Q16 preservation, nuclear forces or CBS.

For C7 DZ (29 occupied, 149 virtual), one sqrt-weighted packed FP64 vector is
about 71.27 MiB. Two rank-six factors need about 0.835 GiB before the diagonal,
source history and fit workspace. Moving these to host avoids directly adding
GPU allocations, but the complete native host/device budget still needs an
explicit accounting contract; it does not automatically preserve Q16.

Native integration should export or transfer owned final-epoch history before
the CC owner releases its ring, preserve producer ordering/provenance, account
for all simultaneous owners, and refuse the optional preconditioner if it
displaces the baseline Q-batch. Qualify exact FP64 actions, construction and
complete Lambda times, physical residuals, and repeated full CBS E+F on C4/C6/C7
under finite Slurm allocations before considering promotion.

## References

- Follow-up: [native endpoint qualification](../../rejected/2026-10-27-lambda-multisecant-default-promotion.md)
  found no complete CBS E+F speedup under current Standard/auto controls and
  rejected default promotion of the measured diagnostic candidate. The original
  host-screen measurements above remain unchanged.
- `src/cc/cuda_solver.cu`: Anderson pairing, mixed epoch reset and ring lifetime.
- `python/generativeqc_compiler/cc/pair_coordinates.py`: DIIS storage ordering.
- `src/cc/lambda_response_cuda.cu`: Lambda coordinates, host/resident schedules.
- `src/cc/lambda_response.cpp`: physical `-D` diagonal.
- [Current procedure](../../../../docs/developer/rccsd_lambda.md#experimental-multisecant-screen).
