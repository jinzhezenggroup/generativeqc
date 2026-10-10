# Decision: incoming-residual Anderson mixing for CUDA DF energies

Status: implemented
Date: 2026-10-10

## Problem

The resident CCSD DIIS loop evaluated the current amplitudes, advanced a damped
Jacobi trial, evaluated the trial again, then extrapolated that trial with its
own physical residual. Carrying an unchanged trial already avoids a redundant
first evaluation, but mixing normally requires two expensive primal graphs per
observed iteration. Faster GEMMs do not remove this semantic work.

## Decision

Add internal `SolverOptions::diis_input_residual`, default false. CPU and CUDA
implement `(G(T), R(T))` history: minimize a combination of incoming physical
residuals and apply its coefficients to outgoing damped Jacobi maps. For fixed
shifted denominators D and damping d, `G(T)-T=(1-d)*D^-1*R(T)`, so this is a
diagonally weighted Anderson fixed-point update, not same-state residual DIIS.
Public production selection is restricted to CUDA DF energies with no retained
DF response. Conventional, CPU, force/Lambda and response owners remain legacy.

The incoming residual is live after CUDA `advance`, which modifies pinned
amplitudes, not equation output storage. Existing history insertion, FP64 Gram
metric, singular-history retirement, coefficient checks and packed-history
refusal remain the owners. CPU copies the incoming residual before the next
graph can overwrite its arena. There is no new resident storage or provider.

## Rejected alternatives

- Mixing only incoming amplitudes can trap the initial subspace; combine maps.
- Calling this a same-state DIIS optimization hides a changed numerical
  trajectory. Qualify it as a nonlinear policy, not a kernel-only refactor.
- Carrying incoming residuals after the update publishes stale energies/errors,
  including after the first insertion or a singular history. Always return no
  carry and observe the next actual amplitudes afresh.
- Broad CPU/conventional/force promotion is not supported by the scoped endpoint
  evidence. Retain the old path and opt in only at the qualified consumer.
- Frozen pre-master timing alone cannot qualify the integrated endpoint after
  master changes its RHF canonical reuse/values defaults. Preserve that ABBA and
  add one matched current-master pair, not a repeat of all numerical tests.

## Invariants

- Physical equations, damping/denominator validation and precision are unchanged.
- Every observed state has its own fresh equation evaluation in the new policy.
- The final budget observation is the updated state, without an unsolicited
  replay; success still requires the independent expanded physical replay.
- No production reference oracle, extra history allocation or vendor dependency.
- No-DIIS behavior remains unchanged; input-residual errors are never carried.
- Evidence distinguishes pilots, frozen ABBA, integrated timing and raw work
  counts; no RSS or universal convergence/speedup claim follows from these runs.

## Evidence

The retained bundle `benchmarks/results/df-cc-input-residual-20261010` preserves
raw inputs, outputs, samples, hashes and reproduction recipes losslessly.
Frozen source is reconstructed separately from current-master integration.
Independent determinant-space gates cover 48 CPU and 32 native CUDA arm cases,
plus 48 stronger CUDA arm cases and two one-update budget cases. They include
DF/dense, history 0/2/6, damping 0/0.2, explicit/canonical denominators, nonzero
level shift, packed/full history and matrix/scalar/unhoisted fallback paths.
Representative CUDA memcheck reports zero errors. These are algebraic tests;
the large molecular endpoint has its own independent PySCF energy/(T) gates.

Frozen ethane230 ABBA has 38 -> 19 primal evaluations and 20 -> 19 observed
iterations, with unchanged graph generators and capacities. Median complete
process wall speedup is 1.1345164x (11.86% shorter), CCSD speedup 1.6376045x
(38.94% shorter). This timing does not include unrelated force/Lambda work.
The prototype library changes only the CUDA solver; its standalone CPU
counterpart is qualified separately. The production runtime flag is covered
by focused CPU/CUDA enabled/default/no-DIIS/final-budget tests.

The matched production pair on master `7f342546d887796e6a92a005ba033029ed73ce2a`
includes #2205/#2206 RHF defaults: complete wall 68.6261 -> 52.1989 seconds
(1.31470x, 23.94% shorter), CCSD 42.5911 -> 26.0223 seconds (38.90% shorter),
38 -> 19 primal evaluations. Independent total-energy errors are at most
2.44e-12 Eh, (T) errors at most 9.3e-14 Eh and expanded physical residual maxima
at most 2.95e-12. Capacities are identical. This is one integrated pair, not
another ABBA, a universal timing claim or a process-RSS attribution.

After measurement, master advanced to `75c2492c3` (#2209, CPU DF-PBE native
analytic energy/forces). Its new prepared-property query defaults to the
existing capability bits for CC owners; the override and numerical changes are
DFT-specific. CC/RHF/integral production sources are unchanged. Review that
shared-interface change rather than rerunning the passing numerical matrix or
the endpoint pair; keep the measured base SHA explicit in the receipt.

Current-master integration initially exposed a build-provenance failure, not
a numerical candidate failure: extracting an overlay after a long baseline
build preserved older source mtimes, so Ninja reported no work and both library
hashes were identical. The runtime-flag test correctly detected legacy 6-graph /
4-observation behavior in the purported candidate. Preserve that failed receipt,
touch the changed source/header before incremental compilation, require distinct
library hashes and rebuild every internal header consumer. Continue from the
already-built baseline; do not repeat its full build or the prototype gate matrix.

## Consequences

The update removes outgoing trial evaluations but can change iteration counts
or convergence behavior on other Hamiltonians. Keep the scoped default and the
legacy policy; compare full endpoints with independent equations rather than
assuming half the graphs always means half the complete calculation time.

## Revisit when

Broader independent molecular/response gates justify another consumer, or a
qualified case shows unacceptable convergence behavior. A new preconditioner
that changes within a solve must revisit the fixed weighted-error rationale.

## References

- `src/cc/solver.hpp`, `src/cc/solver.cpp`, `src/cc/cuda_solver.cu`
- `src/methods/rccsd_method.cpp`, `src/methods/df_ccsdt_force.cu`
- `docs/developer/rccsd_gpu.md`
- `tests/python/test_df_cc_native_solver.py`
