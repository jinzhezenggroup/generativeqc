# Decision: reject unconditional two-sided DF ladder dressing

Status: rejected
Date: 2026-10-10

## Problem and alternative

PR #2191 retains the one-sided FP64 factorization of the paired DF ladder.
Writing `D=t1.T @ Bov` and `F=Bvv-D`, a plausible further transformation is
`F tau F.T - D tau D.T`. It is equal in exact arithmetic to the original
`Bvv tau Bvv.T - D tau Bvv.T - Bvv tau D.T`, but introduces degree-five
intermediates whose cancellation is absent from that original polynomial.

The existing TensorIR exact proof passes. An isolated two-sided CUDA library
builds; eleven original generated artifacts remain byte-identical, nine focused
solver tests and five action tests pass, and action memcheck finds zero errors.
Those observations are not a certificate for the newly introduced cancellation.

## Counterexample

Use NumPy's `default_rng(720)`, occupied=2, virtuals=3, and generate, in order:
`t1=normal(scale=1e4, size=(2,3))`,
`Bov=normal(scale=1e4, size=(2,3))`, and
`tau=normal(size=(2,2,3,3))`. Set Bvv to exact zero and replace tau by
`(tau+tau.transpose(1,0,3,2))/2`. Tau has exact occupied/virtual partner
symmetry, Bvv has the required symmetry, and every input is far below 2^128.
The existing integer-power range predicate admits these input magnitudes and
dimensions; this is not an overflow case or an asymmetric-tau counterexample.

Using the existing bounded TensorIR interpreter, maximum ladder magnitudes are:

- Original: exactly zero.
- One-sided: exactly zero.
- Generic two-sided: 16.0.
- Explicit occupied-lowrank quadratic correction: 24.0.

Without the tau symmetrization the corresponding last two magnitudes are 16.0
and 20.0. At scale=1 the symmetric two-sided residual is approximately
3.55e-15. The growth illustrates cancellation rather than a nonfinite fault.
Original and one-sided expressions contain Bvv in every term and remain zero;
the two introduced large degree-five terms do not cancel identically in FP64.

The occupied-lowrank alternative contracts tau/Bov/Bov/t1/t1 through an
occupied-only Gram intermediate. It reduces representative o9/v221 work from
1,054,175,083 to 1,036,005,568 summands per Q but adds a contraction and worsens
occupied-heavy shapes. It does not repair this numerical counterexample.

## Decision and evidence boundaries

Do not promote either unconditional two-sided variant, open a performance PR
for it, or run a full ABBA campaign merely because benign molecular gates pass.
Range safety is not a rounding or convergence certificate. Do not silently
clear introduced physical errors, add an arbitrary amplitude threshold, or
relabel exact algebraic equivalence as FP64 equivalence.

The two-endpoint feasibility pilot (Slurm 2816, node2/PRO6000, energy-only,
ethane230/o9/v221/Q488/Q8) passes independent energy/(T)/expanded-replay gates.
Complete wall is 156.8974 s for the one-sided baseline and 157.0474 s for the
two-sided candidate; CCSD is 61.0992 versus 60.9898 s. This is one observation
per variant, not evidence of a complete-endpoint improvement. No full campaign
is launched. Full source, build failures, range/work audits, sanitizer logs,
pilot receipts, and a runnable `reject-cancellation.py --symmetric-tau` remain
in ignored `.artifacts/df-cc-two-sided-ladder-20261010/` and its node2 counterpart.

## Revisit when

A justified conditioning/error certificate selects a genuine one-sided fallback
before physical execution, passes this and independent adversarial tests, and
improves fresh matched complete endpoints including its own admission cost.
Dimension-dependent work selection would also be required for the occupied
lowrank correction. Pure copy/packing reuse avoids this polynomial-degree
change and is a preferable next experiment.

## References

- PR #2191 and commit 0cf1d8d55e635b33790c253db185b0acb96f05c1.
- `../implemented/performance/2026-10-10-df-cc-ladder-dressing-factorization.md`.
