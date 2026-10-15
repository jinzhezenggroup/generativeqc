# Rejected: isolate order six and evaluate only its degree-seven Coulomb DAG

Status: rejected; experimental code removed, no default or timing promotion
Date: 2026-10-15

## Problem

The earlier mixed-consumer experiment is rejected in
`../rejected/2026-10-15-r2scan3c-order-six-cooperative.md`. It reused the
order-seven consumer's degree-eight DAG for order six and widened the low-order
CTA. Its substantial endpoint regression is not evidence that a separately
scheduled, smaller mathematical domain is faster or slower.

The canonical r²SCAN-3c def2-mTZVPP basis has more d shells than SVP. Changing
that basis, screening, precision, grid, SCF gates or 3c corrections is not an
acceptable performance optimization. Direct and DF remain separate methods
and require their own endpoint measurements.

## Candidate

The archived `GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS=cooperative-six-degree7` captured an
explicit experimental schedule at preparation. Automatic selection remains the
qualified spatial-reuse schedule. Disabled/unknown settings, missing resident
primitive pairs, missing class topology/prefix, incompatible angular domains
and unavailable recurrence schedules retain the bounded incumbent.

The existing authoritative Coulomb IR supplies the new degree-seven schedule:

| Spatial degree | DAG nodes | Publication levels | Spatial roots |
| ---: | ---: | ---: | ---: |
| 7 | 1193 | 19 | 120 |
| 8 | 1991 | 24 | 165 |

These are mathematical/schedule counts, not measured runtime improvements.
The existing CoulombAux9/dual Hermite response representation is unchanged;
only the DAG/Boys workspace and consumed degree narrow. All 256 coefficient
lanes remain present. The degree-eight schedule stays the order-seven owner.

Orders zero through five use the existing 128-thread, low-only specialization,
with no shared recurrence added to it. A distinct class-major six-only queue
enumerates dd×pp, dd×ds and dp×dp directly from the prepared topology. It uses
the existing 32-entry triangle/rectangle page decoder, not another complete
shell-pair scan or host inventory readback. The mixed-f basis cooperative
domain still owns both six and seven and is not repartitioned by this selector.
Order seven and eight consumers are unchanged. Every reset/submission error
propagates before another owner can run.

The census labels the requested degree-seven route. Order-six shell and
recurrence counts become available only when the complete bounded experimental
domain actually executes. The component-local incumbent retains null for
unavailable shell/recurrence attribution; zero never substitutes for unknown.

## Validation and acceptance

- Independent authoritative-IR numerical/dependency checks cover both degrees.
- Compiled host tests cover exact class-page ownership, empty/multiple systems,
  selector capture, every resource guard and injected reset/submission failures.
- Slurm node5 job 1614 executes the standalone CUDA materialized-pair suite:
  degree-seven and degree-eight derivatives, retained raw-AD responses,
  independent host symmetry-orbit forces and exact primitive work all pass.
  This is an isolated numerical qualification, not an endpoint performance run.
- The production image must advertise the exact frozen source identity; the
  consumer inventory and official stationary AOT manifests must match. Do not
  repair a stale identity by editing a manifest. Reuse verified compiler caches
  and retain cache statistics.
- Before promotion, independently check complete J′/K′ against Libcint for both
  AO representations, both spins and pure J, including executed six-only work.
- Compare same-image complete cold, five warm, moved and five moved-warm
  endpoints with canonical independent references and unchanged 1e-8 Eh /
  1e-7 Eh/Bohr gates. Preserve SCF work, preparation, D4/gCP and host force export.
- Require repeatable >=5% 96-atom warm/moved-warm benefit, no >5% 48-atom or
  cold/moved regression, and small controls before any default/README refresh.
  A direct-force improvement alone cannot establish a DF improvement.

## Evidence location

The source-matched complete cohort is retained in
`benchmarks/results/r2scan3c-degree7-20261015/`. All 48 returned endpoints pass
the independent E/F gates: maximum errors are 1.524e-11 Eh and 3.519e-11 Eh/Bohr.
The advertised native identity equals the frozen 1742-input inventory,
`3035ff1931cb74c8a39909d28d47ab6815d1f20f42f250a31963e072c3fa071d`;
every consumer hash matches. Both arms use library SHA256
`018f81de976f28bd9147bdbcf5329cdb5adb524cf672688dadb74ca69da91ddc`.
This campaign does not inherit the earlier experiment's stale-image problem.

| Atoms | Arm | Cold (s) | Warm median (s) | Moved (s) | Moved-warm median (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 12 | control | 9.178 | 1.608 | 6.714 | 1.618 |
| 12 | degree seven | 9.077 | 1.665 | 6.644 | 1.662 |
| 48 | control | 84.139 | 22.496 | 62.893 | 22.522 |
| 48 | degree seven | 94.738 | 32.635 | 73.033 | 32.635 |

Slurm node5/1615 includes the six independent Libcint cases and the complete
12-atom populations; node5/1616 supplies both complete 48-atom populations.
The host suite passes 116 tests after supplying the existing Conda dependency
prefix to the incremental-CMake fixtures. Initial fixture configuration attempts
failed OpenSSL/LibArchive discovery, not native compilation or numerical gates;
the rerun logs retain the successful prerequisite configuration.

The 48-atom warm regression is about 45%, so the candidate fails before any
96-atom or default qualification. Removing unnecessary DAG nodes and isolating
the low consumer is insufficient. No hardware stall/occupancy explanation is
claimed. The implementation and its specific tests are removed and archived;
the selector is no longer available in production.

A separate intrusive DF/48 capture (node5/1617) passes its cold/first-warm
numerical gates. Its cProfile and CUDA-event intervals are diagnostic only,
not a complete timing population. Python array validation is about 0.108 s in
the warm ledger, so caching that proof is not the primary target. The native
snapshot takes about 0.627 s; further investigation follows the existing
resident final-state validation path rather than weakening publication gates.

Ignored raw build/test data and source freeze are under
`.artifacts/r2scan3c-degree7/`; remote real-device data are under
`/data/jzzeng/gqc-r2scan3c-degree7-20261015` on n5. The retained earlier rejection
is a separate frozen campaign and must not be overwritten or relabeled.

## Revisit

If this still regresses, retain the result rather than promoting an opt-in
experiment by default. Further work should reduce density-weighted contraction
work or recurrence liveness/barriers, not merely vary queue width or relax the
scientific endpoint. DF geometry/XC work needs its own attribution.
