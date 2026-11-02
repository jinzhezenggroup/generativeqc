# Proposal: defer the unused incumbent range warm-energy anchor

Status: proposed, opt-in prototype; qualification pending
Date: 2026-11-01

## Problem

An audited range-source solve publishes a physical energy for its final density,
then eagerly evaluates the incumbent radial provider to prepare a second warm
energy baseline. The retained source commonly handles all subsequent warm calls,
so that incumbent action is never consumed. Retained A100 WB97M-V seed traces
attribute approximately 12.15 seconds and 1.873 billion radial recurrences to
this extra cold action. This cost alone does not explain the whole GPU4PySCF gap.

## Prototype

`GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY=1` defers the incumbent action only for a
strict FP64 RKS fitted hybrid with an exact factor-backed warm density and an
audited radial source admitted for replay. A complete E+F request also supplies
its imminent force-allocation envelope, subtracting retained force storage. If
that envelope exceeds available device memory at finalization, retain eager
preparation: the upcoming force consumer is expected to evict the source. This
is a conservative work-placement decision, not a reservation or a new resource
admission rule. The existing eager path remains the
default and the fallback for unsupported owners. No source, precision, grid or
convergence admission is expanded.

If replay can keep its source, it retains the source-schedule energy and performs
the ordinary physical iteration. If replay loses the source, `begin()` restores
the frozen warm D and its own AO factor, recomputes the incumbent J/K and energy,
and only then starts the physical iteration. The preparation action is charged
as a strict Fock build in an explicit `preparation` work phase. It is neither a
post-SCF action nor an uncounted cache operation.

## Invariants

- Frozen warm updates may leave warm D older than the latest final generation.
  Retain its accepted XC energy and two grid-electron scalars with that D;
  never read current-generation XC while rebasing an older seed.
- Recompute the same occupied primary J/K schedule used by the ensuing physical
  step. The dense facade can introduce a different sub-ulp energy baseline.
- A disabled factor replay uses the corresponding dense primary schedule.
- Do not materialize the deferred anchor during force-memory eviction. The
  current final D/F/W/XC leases must survive that eviction untouched.
- Preserve compensated electronic energies and all original convergence gates.
- Publish no new successful final state after failure. Clear the deferred marker
  only after a successful incumbent reconstruction or a new accepted warm seed.
- Charge the retained three host scalars, copies, provider actions and strict
  work timeline. A saved SCF iteration is not inferred from a missing anchor.

## Pending acceptance

The prototype requires independent complete energy/force references, full and
partial/device/host/disk residency, policy invalidation, deterministic eviction,
warm updates on/off, factor disable, changed geometry, explicit/imported seeds,
failure/retry and final-state snapshot coverage. Compare full cold and warm
endpoints on A100 and Blackwell; test larger and constrained-memory controls.
Neither the design nor an isolated kernel timing authorizes default promotion.

## First qualification and rejected work placement

The first prototype, source identity
`8532cf5a883047f83c6e4ab1f5c3e66d416f44a32d6251aabc72372ff94a0755`,
passed 65 independent-reference CUDA tests on each of A100, RTX 5090 and PRO
6000. Four eviction/retry/snapshot cases passed memcheck with zero errors.
That correctness evidence did not establish safe performance promotion.

The source-identical off/on/on/off controls retain a separate conditioning
population, two fresh cold calls and three frozen warm calls per measured
owner. All 15 complete populations pass the unchanged independent README gates.
Changes below are candidate versus eager medians, with negative meaning faster:

| Device | Cold | First warm | Pooled warm median | Cold plus three warm calls |
| --- | ---: | ---: | ---: | ---: |
| A100 PCIe 40 GB | -8.74% | -0.19% | -0.04% | -5.84% |
| RTX 5090 | -6.32% | **+26.37%** | +0.03% | +0.14% |
| RTX PRO 6000 | -6.00% | -0.03% | -0.01% | -3.80% |

The 5090 force consumer evicts its range source after the cold solve. The saved
incumbent action then costs approximately 15.3 seconds in the first warm call:
one physical iteration, but two honestly counted Fock builds. Later warm calls
take approximately 57.85 seconds, so their pooled median hides the regression.
Reject unconditional deferral; never use that pooled statistic as a no-regression
claim. The resource-based force-envelope guard above is a subsequent repair,
requiring fresh evidence under its own source identity. No GPU-name exception
is introduced, and unexpected later eviction retains the bounded preparation
fallback without modifying a live stationary lease.

The larger 120-atom, 8 GiB A100 control exceeded its original 1200-second limit
before completing the eager cold endpoint (LSF 224154). Preserve that incomplete
population and timeout; it is neither a numerical pass nor a ranked timing.
Do not silently replace it with a larger-budget run.

## Force-headroom qualification

The guarded prototype, source identity
`0a9dda3c991cede2030fe3499bf730e51443701785527dda63199a61d671b59d`,
passes the same 65 GPU cases on each device and four memcheck controls with
zero errors. Fresh conditioning plus off/on/on/off populations independently
qualify on all three devices. A100 cold improves 8.66%, first warm 0.16% and
the complete cold/three-warm lifecycle 5.78%; PRO6000 improves 6.00%, 0.027%
and 3.80%, respectively. The 5090 retains eager work: cold varies +0.019%,
first warm -0.022%, lifecycle +0.002%, with one Fock in every warm call.

This repairs the measured work transfer without a GPU whitelist. It is not a
global promotion: larger qualification is still incomplete, and the independent
5090 120-atom / 8 GiB eager control also timed out before a completed call
(1650 s, Slurm15037). Keep both failed budgets and subsequent declared controls.
The separate 120-atom / 32 GiB 5090 eager arm (Slurm15038) completed four calls
but reached its 1950 s limit during moved geometry; the deferred arm did not
execute. Its partial calls are retained without a complete-lifecycle ranking.

Raw complete endpoint samples, paired work/errors, independent reference reuse
and executable replay are retained under
`benchmarks/results/deferred-range-anchor-20261101/`. The default stays eager.

The separately declared A100 120-atom / 32 GiB (224157) and 5090 108-atom /
32 GiB (15039) controls now complete both six-phase lifecycles, including
forced source eviction and moved coordinates. Both controls retain eager
anchors under the force envelope and one Fock in every warm call. Maximum
paired errors are below 9.10e-13 Eh / 1.78e-12 Eh/Bohr. These larger differential
controls supplement the independent small-system/README oracles; single-pair
latencies are not promoted to ranked speedups. The PRO6000 120-atom / 32 GiB
control (3838) also completes both six-phase lifecycles with eager work retained,
paired energy/force errors at most 4.55e-13 Eh / 1.35e-12 Eh/Bohr, and a single-pair
lifecycle change of +0.004%. The 5090 108-atom control does not
erase its 120-atom timeouts; default promotion remains a separate decision.

## Default admission follow-up

The default-off policy is superseded by the bounded candidate described in
[the default-anchor decision](../implemented/performance/2026-11-01-default-range-warm-anchor.md).
It retains this proposal's force-envelope guard and rejected unconditional
control. Its exact source has separate 5090/PRO6000 qualification; A100
requalification remains pending. Historical identities and conclusions above
remain unchanged.
