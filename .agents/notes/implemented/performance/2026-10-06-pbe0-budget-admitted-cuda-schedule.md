# Decision: budget-admitted PBE0 CUDA force scheduling

Status: implemented
Design date: 2026-10-06

## Problem

The ordinary public force owner fixed its tile to 256 points. A direct joint
schedule trial with 128 cooperative threads, two Becke rows, 16 MiB scratch and
fixed 1024-point tiles accidentally lost the existing fixed-256 active-AO profile:
the actual consumer became dense. A faster-looking isolated kernel or a larger
memory allowance therefore cannot qualify the endpoint.

## Decision

Keep the ordinary 512 MiB device / 256 MiB host budgets and use complete-owner
automatic admission, preferring 512 points before the existing 256-point and
smaller fallbacks. Add budget-auto counterparts of the two existing AO profiles;
preserve their continuous work crossover, producers, cutoffs, cache sizes,
derivative/spin/resource guards and occupancy fallback. Explicit fixed-256 callers
retain their existing profiles. No GPU product/SM/atom/AO/grid-size whitelist is
introduced. Existing backend scientific/resource limits remain unchanged.

Use 128 cooperative geometry threads, four Becke pair-panel rows and a 16 MiB
geometry scratch ceiling. Allocation remains bounded by complete-owner admission;
the ceiling is not an unconditional reservation. The shared planner still defaults
to a 1024-point preference for other owners, including the composite path. Ordinary
consumer coherence policy belongs in the runtime, not generic compiler code.

This does not parallelize the serial `geometry_point_setup` XC preparation or
change the ordered normalization kernel introduced by #2036. It changes the
schedule feeding the existing indexed AO consumer, not the mathematics.

## Why the tools matter

#2031 prepares exact source trials and retains source hashes and negative samples.
Its joint grid/budget axes must be followed by actual AO-policy/work observation,
not a benchmark wrapper that forces the desired producer.

#2029 accounts for registers, kernel limits, dynamic shared storage, driver
reservation and finite grids. On the measured device, the actual cooperative
kernel uses 255 registers, 16 static shared bytes and a 256-thread kernel limit.
With 32 threads and 256 point blocks, the whole-device warp occupancy bound is
3.137%, despite a 16.667% per-SM resource bound. With 128 threads and 512 blocks,
two register-limited blocks/SM give the same 16.667% per-SM bound and enough blocks
to reach that whole-device bound. Driver occupancy queries confirm eight versus
two resident blocks/SM. These are analytical bounds, not achieved occupancy.

Two versus four rows do not improve register-limited residency. Four rows retain
the existing ordered-AO gradient scratch reuse when
`(point_words + 3 * nactive) * sizeof(double) <= pair_capacity * sizeof(PointPair)`.
A 512-point tile also limits the extra AO union/projection work observed at 1024.

## Rejected or nonqualifying evidence

- Fixed 1024 / larger budgets: silently dense AO; substantial intrusive warm loss.
- Automatic sparse 1024 / two rows: no register-residency advantage; no promotion.
- Automatic sparse 1024 / four rows: diagnostic only. Its full-tree after-manifest
  failed because validation files changed during that run; it is not relabeled as
  clean qualification. Larger AO unions also weaken the 96-atom benefit.
- The thread-0 point-parallel XC and stable-owner grouping routes remain stopped.
- n4 could build but could not execute CUDA: loaded driver 580.173.02 versus
  installed userspace 580.178.04. No driver/visibility modifications were made.
- Initial n1 device matrix: 32 passes and 10 missing-AOT failures, not numerical
  failures. Building the missing LDA/UKS/R2SCAN manifests yields 42/42 passes.

Older #2032-base pilots and their failure exits remain separate from the final
#2036-base source-frozen, nonintrusive population. Their timings are diagnostic.

## Invariants

Do not change coefficients, grids, precision, cutoffs or convergence settings.
Retain full solver histories, native SCF/Fock work and dense point/pair/quartet
inventories. Do not remove the existing bounded resource and dense AO fallbacks.
Preserve explicit tile requests and keep AO policy independent of device names.
Do not substitute planner byte bounds for a measured global allocator peak.

## Evidence

The publication is `benchmarks/results/pbe0-cuda-schedule-20261006/`.
Its authenticated bundle retains exact raw streams, independent references,
actual runtime/native binary identities, source/harness before/after checks,
verified ccache receipts, launch assessments and failed pilots. The reconstruction
patch restores the exact measured candidate from
`21f6314f5a812e9e83a41b3c9921549d6e944778`, including #2029/#2031/#2036 and runtime
compiler caching; mutable `origin/master` is never substituted for this base.

An initial all-phase equal-work campaign (6291) is not used for promotion: the
unchanged control naturally varies from 27 to 25 cold iterations at 48 atoms and
12 to 17 moved iterations at 96 atoms, with energy/force differences around
1e-12. No convergence criterion or iteration cap is changed to force a match.

A fresh warm-focused protocol is declared before its new population: five
interleaved pairs per size, >2% median paired warm/moved-warm gain and at least
4/5 positive pairs. Actual warm SCF/Fock/AO work must match, and force-domain
inventories must match in every phase. Cold and moved retain preparation, complete
E+F, all actual histories and independent gates, but are diagnostic only: unequal
SCF work cannot substantiate a cold speed claim. The independent gates remain
1e-8 Eh / 1e-7 Eh/Bohr. Artifact populations were already primed on the same frozen
sources; another discarded priming run is not necessary. No samples from 6291
are cherry-picked into the new qualified population.

Global concurrent allocator peak and deployment/first-install compilation cost
are not measured. This is not a generic performance-envelope closure or a speed
promise for unmeasured devices/methods. Warm benefit is sufficient for this
resource-admitted scheduling choice; it is not presented as a deployment-cold win.

The complete five-pair #2036-base population passes, with every warm pair positive:

| atoms | phase | control median / s | candidate median / s | paired median gain |
| ---: | --- | ---: | ---: | ---: |
| 48 | warm | 8.928828 | 8.416576 | 5.609640% |
| 48 | moved-warm | 8.921205 | 8.409540 | 5.735386% |
| 96 | warm | 24.426597 | 23.201106 | 5.038910% |
| 96 | moved-warm | 24.483404 | 23.237256 | 5.070176% |

Maximum independent errors over both modes and all phases are
1.064109e-10 Eh and 3.022205e-11 Eh/Bohr. Each qualified warm endpoint performs
one SCF iteration and one Fock build. The resource-only #2031 preparation
manifests are diagnostic inputs, not the final promoted source identity; the
reconstruction patch and full before/after manifests bind that identity.

## Upstream integration boundary

The measured base above predates #2033's default Combined two-electron reduction
and #2038's shared incremental Direct-J/K policy. Integration is pinned to
`25675d88d64bfec13a8b7100eebcabef2a83d534`, not mutable master. Preserve the new
three-channel Combined contract, the conservative four-channel fallback reserve,
strict output validation and all endpoint mutation tests. The endpoint source pin
is deliberately re-audited with the added 512 preference, not simply inherited.
Keep a separately authenticated integration population and source patch. Do not
claim that the old timings/binaries were produced by the integration source.

The first integration device matrix (6293) preserves 38 passes and four
`KeyError: 'two_electron'` failures in semilocal source-component comparisons.
LDA/PBE have no exact exchange, so their new Combined output must be compared to
the independent Coulomb/J reference. Adapt only the oracle's name map; do not
force Separate mode, weaken vector gates, or alter the frozen production source.
Keep those failed receipts distinct from the fresh rerun and timing population.

The fresh integration population (6294) passes the same five-pair gates: paired
warm/moved-warm medians improve 5.698574% / 5.885798% at 48 atoms and
5.421262% / 5.283901% at 96 atoms, with 5/5 positive pairs in every cohort.
Every qualified warm phase retains one SCF iteration and Fock build. Independent
maxima are 1.045919e-10 Eh / 3.028999e-11 Eh/Bohr. The final 42-test CUDA matrix,
memcheck and racecheck all pass, with zero sanitizer errors/hazards/warnings.
Both full trees reconstruct exactly from the integration patch and pinned base;
the PR's four production files match that frozen candidate byte-for-byte.

## Revisit when

Requalify if geometry register usage, AO-union costs, the producer crossover,
cooperative scratch reuse or complete-owner storage accounting changes. Different
hardware may justify a different resource-admitted coherence preference, but not
a model-name whitelist or an unmeasured occupancy/bandwidth claim.
