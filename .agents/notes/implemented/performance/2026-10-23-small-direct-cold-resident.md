# Decision: restrict ordinary direct cold RHF value reuse to small domains

Status: implemented
Date: 2026-10-23

## Problem

Ordinary direct RHF regenerated canonical ERIs on each physical Fock build,
although the existing physical-reference owner could retain compensated FP64
values for a solve. Reuse can save recurrence work, but its provider cloning,
sorting and quartic value production must amortize against the already efficient
fused streamed Fock. A memory-fit test alone does not answer that question.

## Decision

Reuse `RhfResidentValues` and its graph/source lifetime for fresh single-system
ordinary restricted FP64 solves with an internal guess. Default admission needs
64 or more public AOs, at most 128 Cartesian AOs, through-d shells, positive
screening and at least eight permitted iterations. Reject other automatic
domains before provider cloning/sorting. Ordinary values have a 512-MiB default
ceiling; the unscreened physical-reference f-shell policy retains its 8-GiB
ceiling and eligibility. Existing explicit controls retain bounded overrides.

Warm/external seeds, reused buckets, public resource ledgers, unrestricted,
mixed/incremental/separate-JK and diagnostic paths retain the incumbent schedule.
Finite public ledgers do not reserve optional quartic values. Unlimited ordinary
admission still charges mandatory arena/provider/solver/binding, host staging,
publication and the initial-guess owner's declared numeric bound. Source
admission retains free-device reserve, checked capacity and exact fallback.

## Rejected alternatives

Full residency across larger domains was numerically correct after the force
handoff fix but slower: 48-atom water 7.00814 to 8.48815 seconds; 96-atom water
22.49069 to 27.58572 seconds. The latter retains about 15 GB of values. Repeated
192-public-AO water24 calibration already showed regression. Do not broaden
selection merely because optional storage fits or recurrence work is reusable.

The first prototype omitted derivative metadata production. Cold energy passed,
but force error reached roughly 15 Eh/Bohr while warm forces passed. Resident
Fock bypasses the streamed Cartesian density transform, pair bounds and force
queue preparation. References historically exported no forces, so their reuse
owner never needed this handoff. Correctness requires explicitly producing
these from the selected final density before force compaction. A frozen warm
check alone would have missed this defect.

## Invariants

- Construction and consumption use the same immutable screening value.
- Source reuse changes integral generation, not SCF tolerances, iteration
  stopping, physical-Fock counts, compensation or final-state acceptance.
- Energy, Pulay weight and derivative integrals consume the same final density.
- Optional source graphs retire before their borrowed values; ordinary bucket
  graphs remain available for warm/moved calls. Never publish a graph that
  borrows phase-local values after the owner dies.
- Audit resident source completion before publishing an ordinary result.
- Finite resource refusal and optional-capacity exhaustion preserve exact
  streamed fallback, including final forces.
- Independent reference work remains outside the production endpoint.

## Evidence

[Retained source-bound evidence](../../../../benchmarks/results/direct-cold-resident-20261023/README.md)
uses base `4a783a27e104e9392a7227b7efca79abcfba60ae` plus archived patches,
RTX 5090 / CUDA 12.9.1 / Release sm120 and verified sccache 0.16.0. Three clean
alternating fresh-process pairs compare actual unset selection with explicit 0.
Water12 cold median improves 1.75543 to 1.55980 seconds (11.1%); benzene improves
2.41090 to 2.11356 seconds (12.3%). Warm medians are essentially unchanged.
Water12 has 10 iterations/12 Focks; benzene has 9/11 in both arms. Source
diagnostics count 8,266,864/25,024,440 values generated once and 12/11 resident
actions. Independent energy/force gates pass all 104 final measured endpoints.

Spherical/cartesian cold, frozen warm, moved/replay and one-byte refusal pass
strict independent CPU E/F tests; existing UHF forces and reference exports
also pass. Memcheck reports zero errors; 175 host policy/lifetime/structure
checks pass. Initial over-tight CPU oracle attempts failed to converge before
CUDA work. Converged full-density CPU references achieve gradient norms below
3e-9; observable acceptance gates were not relaxed.

## Consequences and revisit conditions

The guard is a conservative source/work domain, not a convergence prediction.
Only two admitted static shapes supply repeated performance evidence; other
chemistries and hardware can have different crossovers. The first warm call
can pay ordinary graph recapture, with roughly 0.01 s observed overhead; pooled
warm medians remain stable. Tight-budget and batch paths retain fallback rather
than acquiring unplanned resident ownership.

Revisit with a cheaper provider setup, lower source-generation/data-movement
cost, a measured partial-bucket policy, or broader independent cold/warm/moved
endpoint qualification. Keep failed numerical and slower large-domain evidence
separate from any future accepted population. Later-master source or binary
changes require appropriately renewed qualification.
