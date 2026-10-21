# Decision: prove the final physical state before an extra conservative SCF sweep

Status: implemented
Date: 2026-10-22

## Problem

The [cold-SCF diagnosis](../../proposed/2026-10-22-cold-scf-stop-gates.md)
found ordinary Standard energy/density convergence several steps before the
strict AO-maximum residual margin. Simply changing that margin from 1e-9
to 1e-8 failed the unchanged canonical-density export check for PBE and
r2SCAN. The logged residual is a spin RMS, not the AO maximum. Matching
its numeric threshold to GPU4PySCF's different orbital-gradient norm is
not a scientifically meaningful alignment.

## Decision and invariants

Eligible CUDA RKS may replace the conservative AO-maximum margin with a
complete resident proof of the actual physical F[D] frame. Energy/density
and spin-RMS thresholds are unchanged; the AO maximum must already be
below min(1e-8, density_tolerance). Existing final-state validation remains
authoritative, including canonical density. Numerical rejection resumes
ordinary iterations. Device/eigensolver failures are errors, not permission
to publish an unchecked state.

The proof borrows tmp1/tmp2/effective/residual, while the DIIS density
proposal has separate storage. Rejection must invalidate the optional
warm orbital frame rather than copy its now-overwritten scratch. Last-good
warm seeds remain protected. Acceptance retains only the physical eigenframe;
no speculative token or force weights are published. The force consumer
still performs its own normal token-bound proof and weight staging.

Automatic admission is strict FP64, RKS, single-iteration execution with
the existing resident validation workspace and matrix library. Pure-J DF
is allowed; fitted exchange, Fast/budget, mixed, UKS, range/nonlocal,
incremental, ECP and chunks retain their ordinary policy. No persistent
numeric allocation is added. The explicit zero control restores the
incumbent conservative margin. See the current
[diagnostic contract](../../../../docs/developer/ks_diagnostics.md).

## Evidence and limits

The opt-in v1 prototype passed 48 complete E/F calls: twelve cold populations
and three frozen-warm calls each. Six ordinary DFT Direct methods at 96 atoms,
same-binary PBE/PBE0/r2SCAN controls, pure-J PBE DF and fitted-exchange/Fast
fallbacks were covered. Independent README gates remained energy 1e-5
Eh/atom, max force 1e-4 Eh/Bohr and RMS force 3e-5 Eh/Bohr, with the retained
quarter-gate reference checks. PBE/PBE0 reduced 15 to 14 iterations; r2SCAN
stayed at 14 and its observed candidate cold time was worse than its control.
These diagnostic samples do not establish universal speedup.

Eight Cartesian/spherical, restricted/open-shell PBE/r2SCAN/PBE0 water tests
covered original/displaced geometry and frozen warm replay (32 native E/F
calls), with independent tight GPU4PySCF oracles and stricter 1e-7 Eh/atom
and 1e-6 Eh/Bohr gates. The oracle never seeds native production. The first
test attempt's NumPy conversion error and unconverged OH oracle remain
retained; the corrected suite uses the nondegenerate water cation for UKS.
Seventy-one host lifecycle/final-validation tests passed. Compute Sanitizer
memcheck passed a PBE/96 cold plus warm scope with zero errors, including
rejected-proof continuation and later acceptance.

Ignored `.artifacts/early-final-v1/` in the cold-SCF worktree and matching
`n1:/data/jzzeng/scf-performance-20261021/early-final-v1/` retain all receipts,
source manifests, scripts, traces, failures and verified sccache statistics.
The immutable v1 runtime is `runtime-early-final-v1/source`; core SHA-256 is
`23105bf76ae5b6c5367f6d770d501e525759e524bb7c5d4965f0b97dc35ec43e`.
V1 is an opt-in build, not evidence for the later default binary.

The default v2 binary (same directory convention, `early-final-v2`) has
SHA-256 `ed8e9abcbace8758c6985cfc0d8a0efa3fdf7b400ba6197b568fa198e1b047dd`.
Its full source manifest and patch are retained separately from v1. Six
ordinary DFT Direct methods at 3/12/48/96 atoms, all four pure-J PBE DF
sizes, 96-atom PBE0 DF/Fast fallbacks and same-binary rollback controls
completed 34 populations / 136 E/F calls. Thirty-two populations passed
the independent README gates. The two failing populations are M06-2X/3
default and rollback: RMS force errors are about 3.302e-5 versus the 3e-5
gate. Historical native Standard and stock GPU4PySCF also fail this same
default-grid sample. The candidate/rollback maximum force difference is
7.51e-10 Eh/Bohr and energies agree to the recorded precision; do not hide
this existing grid failure or qualify either failed timing for ranking.

Clean 96-atom Direct same-binary cold observations are PBE 27.464 -> 26.130 s
(15 -> 14 iterations), PBE0 41.180 -> 38.878 s (15 -> 14), and r2SCAN
29.235 -> 29.445 s (14 -> 14). These single cold observations show the
mechanism and its limits, not statistical or universal leadership. All
three warm repeats remain visible in `qualification.json` and the receipts.
Thirty-two default/rollback geometry tests (128 native E/F calls) passed
for Cartesian/spherical PBE, r2SCAN, PBE0 and UKS water cation, including
explicit tighter E/D=1e-12/1e-10 tolerances and frozen warm replay. The
expanded host lifecycle/final-validation/inventory suite passed 201 tests.

## Revisit when

Broader admission requires independent endpoint, geometry, explicit-tolerance,
resource and lifetime qualification. Keep all failed and slower populations.
The saved 1-2 iterations do not resolve larger J/K, setup or DF costs and
must not be represented as closing the overall GPU4PySCF performance gap.
