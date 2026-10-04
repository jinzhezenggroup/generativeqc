# Keep CPU UKS primary DIIS until the physical maximum is eligible

Status: implemented
Date: 2026-10-03

## Problem

The CPU final-maximum correction exposed a convergence mismatch in the unchanged
OH/PBE-UKS Cartesian def2-SVP regression (neutral doublet, O at the origin and H
at z=1.8 Bohr, E=1e-12, D=1e-10). Parent and candidate have exactly the same first
13 primary rows. The current state at row 13 passes the scalar RMS gate but has
maximum commutator 5.41723888e-10. Removing DIIS there makes the next four physical
projector maxima remain above 1e-10; spending more projector steps is unsupported.
The same regression blocks the cold OH item in the existing ragged OH+H test.

## Decision and invariants

The method-neutral self-consistent driver accepts an optional explicit eligibility
callable. The original energy/state/residual scalar conjunction is unchanged.
Only when that conjunction passes is eligibility called with const progress and
const evaluation, before the unchanged record/accept sequence. It may only veto.
The default callable is constexpr/noexcept and true; no state moves, iterations,
allocations or scientific gates are added to callers that do not opt in.

CPU UKS opts in using its existing CURRENT physical alpha/beta residual arrays.
It requires max(abs(residual)) <= min(1e-8, density_tolerance), independently of
the existing min(1e-9, density_tolerance) RMS gate. Non-CPU UKS remains eligible.
The original budget, CURRENT terminal retention, DIIS lifetime, occupation
stabilization, physical/eigen failure order, counters, resource sampling and
four-correction finalizer are unchanged. No ordinary unshifted Aufbau projection
is introduced for stabilized occupations. Public RMS telemetry remains RMS.

The trailing defaulted callable preserves all existing direct call expressions.
It changes the instantiated function type for address-taking. This is an internal
header, with no existing function-pointer users found; no forwarding overload or
extra state move is introduced solely to retain an unused pointer signature.

## Evidence and scope

The generic tests protect default move counts/order, scalar short circuit,
first-iteration exclusion, false-to-true veto, original-budget exhaustion and
CURRENT retention. The actual UKS predicate and real driver together show the
old RMS-only decision accepting a sparse maximum-large state at iteration two,
while eligibility continues to a passing state or exhausts the same budget.
Boundary/nextafter tests cover both spins, tight/loose tolerances and nonfinite
values. The unchanged molecular singlepoint and ragged batch regressions provide
the behavioral molecular red/green; a missing-callable compile failure is only a
structural control.

The local repaired OpenBLAS OH endpoint takes 14 primary iterations / 16 Fock
builds, warm replay 2/4, and cold restart 14/16. A separate budget-13 control fails
13/13 without a seed or snapshot lease. Independent fixed-density PySCF checks
use the exact native quadrature with rebuilds prohibited and verify unshifted
physical F/D/E, electron traces and idempotency. This is a correctness repair,
not a timing, universal-convergence, universal-export or CUDA qualification.

## References

- [Original final-maximum decision](2026-10-03-cpu-ks-final-maximum-closure.md)
- [Stabilized UKS export distinction](2026-09-22-uks-export-projector-closure.md)
- `tests/python/test_dft_scf.py::test_native_matches_independent_scf`
- `tests/python/test_dft_batch.py::test_open_shell_large_solver_ragged_replay_and_failure`
