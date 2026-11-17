# Decision: separate HF force stationarity from energy-only stopping

Status: implemented
Date: 2026-11-17

## Problem

Issue #3092 retains an independent Cartesian water/def2-TZVP force failure at
the initial geometry with stock Direct CUDA RHF controls. Both auto and portable
AOT profiles on RTX 5090 and RTX PRO 6000 Blackwell report convergence while
exceeding the unchanged `1e-7` maximum-force gate. Those observations do not
isolate a mathematical kernel defect or an SM120-specific cause.

The default orbital-gradient policy admitted force endpoints even though
`standard_hf_stationarity_owner` already excluded them from relaxed AO gates.
The orbital-policy flag bypassed that exclusion: both the iteration predicate
and the retained-Fock force publication accepted the energy-oriented
`sqrt(energy_tolerance)` gradient threshold. A warm endpoint could then reuse
the insufficiently stationary determinant without further refinement.

A finite n1 diagnostic on the previously retained source
`038aedeccf3964b58acec499cc9684f423a7a93f` changes only the convergence policy,
not any public tolerance, basis, arithmetic or force gate. For both profiles,
the initial error changes from about `6.758e-7` (9 iterations) to `1.386e-9`
(12 iterations). Moved force errors also pass. This establishes the stopping
policy as a sufficient cause for that reproduced failure, not a general
force-error theorem for the strict residual threshold.

## Decision

The shared Direct/DF convergence resolver receives the requested observable
ownership explicitly. Force endpoints select the existing strict physical
AO-maximum and density-change conjunction. Explicit `orbital-gradient` requests
on force owners fail closed, like other unsupported orbital-policy owners.
Energy-only eligible owners keep the existing orbital-gradient policy. Public
energy, density and screening defaults remain `1e-9`, `1e-5`, and `1e-12`.
No card, SM, basis name or geometry participates in this decision.

Compact DF iteration and final-state selection must also withhold their relaxed
AO ceiling for force owners. Changing only the orbital selector would otherwise
leave DF force publication at a different physical stationarity level. Physical
reference, UHF, ECP, mixed-arithmetic and application-budget protections remain.
Prepared execution already binds `compute_forces` to its captured identity;
energy-to-force transitions must re-admit the native owner, not reuse a looser
energy proof as a force proof.

## Tradeoff and validation

Force endpoints can require more iterations; no speedup or unchanged benchmark
ranking is claimed. This narrows the historical Direct/DF promotions below;
their source-distinct measurements and former acceptance budgets stay intact.
The strict residual remains a solver criterion, not a universal guarantee of
independent force accuracy on ill-conditioned or differently screened systems.

Executable host probes cover default/strict/explicit requests and force-owner
admission. The independent device regression covers Cartesian RHF and UHF,
SVP/TZVP, cold/frozen-warm/moved/energy-to-force calls, and matching fitted RHF
operators. The oracle runs in a separate process; production imports are
forbidden. The original `1e-9` energy, `1e-7` force and `1e-7` translation
acceptance gates remain unchanged. Actual device receipts must identify the
source, binary, profile, physical device and finite Slurm allocation.

## Superseded scope

- [Direct orbital-gradient default](2026-11-10-default-hf-orbital-gradient.md)
- [Compact DF orbital-gradient default](2026-11-11-df-hf-orbital-stopping.md)

Revisit force admission only with a separately qualified derivative-stationarity
criterion and independent complete-endpoint evidence, not a matching energy
threshold, a microbenchmark, or a single GPU label.
