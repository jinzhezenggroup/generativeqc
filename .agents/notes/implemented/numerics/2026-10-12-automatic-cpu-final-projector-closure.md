# Decision: certify automatic CPU final projectors inside the SCF budget

Status: implemented
Date: 2026-10-12

## Problem

The complete H4 RKS/UKS ordinary-program lifecycle uncovered failures hidden by
isolated point checks. Both LTA spin layouts returned converged cold states
whose snapshot export rejected the projector of the last physical F[D]. APBE
UKS stopped after an unsuccessful four-step final closure at iteration 11,
although the original 200-iteration budget still had 189 unused iterations.
The unchanged energy, density, residual and export gates must all hold.

Qualifying the standalone master-based phase also exposed a transport mismatch:
an explicit automatic MethodIR still borrowed its ingredient-carrier display
name when validating snapshot domain and stationary composition. That display
name describes neither the exact functional nor its work domain. All 22 native
states were rejected by the Python adapter despite valid native metadata.

## Decision

For a compiler-owned automatic point program, check the final physical Fock's
unshifted canonical density before publishing success. Reuse the existing
generalized eigensolver and density conventions; do not change point algebra,
work clipping, occupations, thresholds or precision. Sample the additional
canonical-projection scratch as live CPU capacity.

RKS reuses its existing bounded strict-stage loop. UKS clears nonlinear history
and resumes from the rejected physical density using only unused primary
iterations. Its four-step final correction remains bounded per stage. Preserve
global iteration/history numbering and count all physical Fock rebuilds.
Non-automatic consumers retain their existing one-attempt behavior.

Snapshot validation and stationary identity bind the resolved MethodIR and its
actual SCF domain, not a carrier name. This shared semantic-resolution correction
also occurs in the independent CUDA-plumbing work; it grants no backend or
product admission and does not require that sibling branch to merge first.

## Rejected alternatives

- Relaxing export tolerance hides an inconsistent retained determinant.
- A hidden solve during snapshot/force export violates the zero-Fock export
  contract and loses semantic work accounting.
- Increasing the user budget or adding a functional-name-specific restart
  obscures the premature stop and duplicates method policy.
- Filling empty CPU operator records with assumed counts falsely certifies
  work the producer has not actually recorded.

## Invariants

The original iteration budget, physical residual/energy/density gates, pure
semilocal ownership and fail-closed exhausted-budget behavior remain intact.
Finalization work is not free: total Fock builds can exceed history iterations.
Unavailable CPU precision/operator inventories remain unavailable; these
numerical endpoint checks do not establish zero unused K or force promotion.

## Evidence

Retain the original failing source and native artifacts under the ignored
`.cache/1118-runtime/ordinary-energy-qualification-20261012/`. The repaired
composition discovery is separate under `cpu-finalization-repair-20261012/`.
PR-source receipts belong under `cpu-finalization-phase-pr-20261012/`; do not
relabel discovery or CUDA energy measurements as qualification of this PR.

`tests/python/test_libxc_automatic_cpu_ks.py` exercises three LDA, five GGA and
three tau-MGGA registrations, both spins, cold/warm/changed geometry, stale
snapshots, independent pinned SCF and complete fixed-D Fock, legal PSD energy
directions, and explicit exhausted-budget rejection. No force or CUDA endpoint
admission follows. The separate empty-spin physical-birth question is unchanged.

## Revisit when

A shared CPU final-state owner replaces the bounded closure, or an actual CPU
operator/event producer can certify complete returned-attempt work.
