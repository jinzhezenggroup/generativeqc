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

## Retained device evidence

The [source-bound qualification](../../../../benchmarks/results/hf-force-stationarity-20261117/README.md)
retains successful finite Slurm jobs 17670 (RTX 5090) and 5324 (RTX PRO 6000
Blackwell), four fresh processes per card/profile, identical ELF bytes across
cards for each profile, and independent raw reference arrays. All 96 cases /
576 endpoints, including 480 E+force endpoints, pass the original gates.
Maximum energy/force/translation errors are `1.1221e-10`, `4.8488e-9` and
`2.2471e-13`, respectively. The original TZVP cold/warm case passes all 32
observations at maximum force error `1.3860e-9`; an earlier-source negative
regression still fails at `6.7581e-7`. Both cards are SM120; no cross-SM or
performance qualification follows from these results. Replay recomputes the
errors and DF finite differences rather than trusting stored pass flags.

## Matching the DF oracle

Do not compare different fitted operators under the direct-HF energy gate.
The shared PySCF basis exporter defaults to spherical auxiliary functions,
whereas a Cartesian PySCF molecule inherits Cartesian auxiliary functions.
Even after matching that representation, stock PySCF Cholesky DF is not the
native default spectral metric with relative cutoff `1e-10`. That cutoff is
gauge dependent: native Cartesian components have unit self-overlap, while
raw PySCF Cartesian components need not. The initially mismatched comparisons
are retained as failed setup observations, not production acceptance.

The DF reference therefore constructs its metric and three-center integrals
independently with PySCF/CINT, normalizes auxiliary components by independently
computed self-overlaps, and retains the published default spectral cutoff.
No native matrices, density seeds or fitted factors enter the oracle. Complete
forces come from reconverged displaced energies, including retained-projector
and auxiliary-center response. Two five-point steps (`0.01`, `0.02` Bohr) must
agree within `5e-8`; their Richardson extrapolation supplies the independent
reference. Displaced energies and unchanged metric ranks are retained. Public
production controls and the `1e-9`/`1e-7` acceptance gates do not change.

This explicitly fitted auxiliary model qualifies the shared force-stationarity
owner, not every auxiliary selection, fit error, or default DF endpoint.

## Superseded scope

- [Direct orbital-gradient default](2026-11-10-default-hf-orbital-gradient.md)
- [Compact DF orbital-gradient default](2026-11-11-df-hf-orbital-stopping.md)

Revisit force admission only with a separately qualified derivative-stationarity
criterion and independent complete-endpoint evidence, not a matching energy
threshold, a microbenchmark, or a single GPU label.
