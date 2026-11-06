# Decision: share strict cold range-source recurrences across AO components

Status: implemented
Date: 2026-11-07

## Problem

The source-bound A100 96-atom / 768-spherical-AO WB97M-V full-Direct diagnostic
(LSF 226178) took 106.447457 s cold, including 21.872050 s for immutable LR source
preparation. It published 1,873,221,716 FP64 values / 14,985,773,728 source bytes.
That intrusive diagnostic used source `85c68f6b865ccddd9d22f0cf193b0bb6ec9d26290d6902d4391302f9c97da324`
and ELF `ef29a74ec63fe73f39a64ecac527eb736caa661951deef78cde0d1cd25a9aa7f`;
its nested scopes are not additive clean endpoint timings.

The scalar producer reconstructed primitive geometry, Hermite coefficients and
radial recurrences independently for Cartesian AO components of one shell quartet.

## Decision

Reuse the compiler-owned materialized-pair recurrence for strict SR/LR source
orders three through nine. Share #3026's algebra through an explicit compile-time
`ResidentValues` sink, distinct from both the ordinary Fock sink and the retained
fixture-only component-map sink. Source kernel instantiations do not add a runtime
source branch to ordinary Fock component publication.

A source-only plan may prepare bounded shell metadata and primitive-pair geometry
after its mandatory canonical topology, without a second generated Fock/force
owner. A temporary quadratic inverse maps original canonical sorted AO-pair ranks
to existing compact source slots. Equal-angular bucket symmetry follows sorted
rank, not shell traversal order. Publish exact zeros, initialize optional shared
sources to NaN, and audit their complete finite inventory before admission.

The inverse must fit alongside values within the preparation allowance. It is
stream-fenced and freed. Optional metadata/inverse refusal preserves the scalar
producer; orders zero through two and ten through twelve remain scalar. Mixed
and full-range source construction do not opt into source-only geometry. Explicit
`GENERATIVEQC_DIRECT_RESIDENT_COMPONENT_SOURCE=1` restores component preparation
while retaining replay. No GPU model names enter admission.

The source-only metadata can also make the established exact uncached range-K
consumer reachable. Report complete endpoint effects, not an isolated producer
speedup. At a stricter consumer threshold with a source-row census, keep the AO
consumer: the shared consumer counts surviving components, whereas the immutable
source contract counts all original compact rows before tighter screening.

## Rejected alternatives

- Disabling the LR lease as a control changes replay/reuse, not only construction.
- Hardcoding A100, relaxing precision/screening, importing CPU oracle values, or
  substituting DF changes portability or the scientific production contract.
- A per-component runtime source branch unnecessarily changes ordinary Fock
  kernels; compile-time sink specialization removes that coupling.
- Weakening the stricter-threshold audit would hide the first failed independent
  gate rather than repair its newly reachable dispatch.

## Evidence

Initial source `362fb48621a638f873d605a466cde08f7e250b585e48f52df68904e76e3e3e0a`
retains complete paired cold results: A100 106.451900 -> 97.446155 s (-8.46%),
RTX5090 184.496538 -> 178.625509 s (-3.18%). Each warm change is below 0.7%.
Both arms retain 16 cold Focks, four actual SCF VV10 sweeps, one warm Fock and
zero new warm pair sweeps. Energy differences are zero and paired force errors
are below 1.1e-12 Ha/Bohr. These are not final master-integrated image numbers.

Final source `33b9650bd7a90bcc6c6b986512f75e2f39c7ea875860dbcf36f04293210870f2`
integrates master `b26a19f3c`, with ELF
`948b333ef7abe5f4f75b427e3cb8a4b702ce0ff135f9d76685dbac87ccaac170`.
Its complete A100 cold endpoint improves 106.309547 -> 97.375213 s (-8.40%),
while RTX5090 improves 184.298838 -> 178.414226 s (-3.19%). A100 warm changes
14.906582 -> 15.007078 s (+0.67%); RTX5090 is effectively unchanged
(13.620037 -> 13.619380 s). Both cold arms retain 16 Focks and four SCF VV10
sweeps. Worst paired energy/force errors are below 1e-12 Ha and 8e-13 Ha/Bohr.
The final same-image source controls pass both predeclared endpoint gates, so
shared construction remains default rather than opt-in.

The scoped 250 host tests and 137 default-inventory tests pass; compiler
and SCF ownership checks report zero errors. Independent GPU source gates cover
signed SR/LR RKS/UKS, complete/partial residency, stricter thresholds, displaced
replicated geometry and, in the final cohort, f shells. Preserve original image
identities, failed Slurm 16619, cancelled compilation (143), and the post-link
provenance-driver failure (2).

## Invariants and revisit conditions

Do not change the automatic lease envelope, FP64 contract, grids, convergence,
range/omega identity, sorted compact screening predicate, full-range provider or
analytic-force ownership. Requalify complete cold E+analytic-force+host-export
and warm nonregression on both devices when changing scheduling or these bounds.
Raw-matrix numerical acceptance and positive source execution counts are required;
a microbenchmark or memory bound alone is not a default-promotion gate.

## References

- Issue #2976; compiler fixture qualification #3026.
- [Source-bound evidence](../../../../benchmarks/results/cold-range-shared-source-20261106/README.md).
- [Current recurrence contract](../../../../docs/developer/direct_pair_recurrence.md).
- [Fixture-only source rationale](../../proposed/2026-11-06-bounded-materialized-lr-value-sink.md).
