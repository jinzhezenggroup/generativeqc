# Decision: use qualified Direct recurrence and density reuse by default

Status: implemented; unset-selector endpoint and counted-profile qualification complete
Date: 2026-10-17

## Problem

The generated relative-response and density-first consumers already pass
independent scientific gates and reproducibly improve complete Direct endpoints.
Leaving both behind an explicit environment setting prevents eligible users from
receiving those gains. The user explicitly requested default enablement of
performance-qualified options, not promotion of every experimental selector.

## Decision

Resolve the captured automatic force schedule to `DensityReuse` when the method
authenticates the existing eligible Direct RKS source. Do not widen the existing
method hint or introduce a functional, molecule, basis, AO-count or GPU-model
special case. An absent or empty environment selector is automatic; explicit
`relative-reuse`, `spatial-reuse`/`1`, and `0` preserve their historical meanings.
An unknown nonempty selector still fails closed to the incumbent.

This composes order-six relative-response reuse with density-first contraction
for pure-J orders six/seven. Existing guards retain component contraction for
nonzero exchange, derivative seeds, incomplete masks and unsafe reorderings,
and preserve the original bounded launch paths without coherent resident
topology or compatible derivative caches, angular domains and launch shapes.
Generic HF/RSH callers and callers without a qualified method hint keep their
previous automatic schedule. Explicit UKS qualification remains available; this
change does not infer an unrestricted default from restricted endpoint data.

The resolver changes no allocation, generated scientific kernel, threshold,
precision, force sign, orbit symmetry or SCF policy. Workspace and fallback
qualification from the retained implementation still apply; fixture resource
receipts must not be described as universal production occupancy measurements.

## Evidence and source boundary

The [earlier qualification](2026-10-17-direct-force-endpoint-qualification.md)
retains full source-matched forward/reverse populations, independent oracles,
actual SCF/Fock counts, counted and clean profiles, and sanitizer limitations.
Its approximately 29.6% 96-atom warm and moved-warm gains are measurements of
explicit selectors on its frozen source, not additional gains from this policy
change. It does not include occupied-XC #2577, guarded FP32 or the cold-planning
stack; none is silently composed or promoted here.

The new benchmark candidate defaults to `automatic` and removes an inherited
selector rather than setting an equivalent manual value. Executable host tests
cover absent/empty selection, explicit rollback, eligibility, force/value owner
separation, and preparation-time capture. Existing compiled dispatch tests retain
resource/failure injection and all spin/output/source fallback combinations.

Review follow-ups also restore the inherited all-order canonical FP32 fixture
(orders zero through twelve, including asymmetric/f-containing and sparse
exchange-only cases) without reverting the derivative work upper bounds. The
benchmark retires owned descendants even after leader exit, preserves primary
failures and receipts, and refuses reused/symlinked worker output before writes.
The historical reader retains authenticated, read-only Git recovery while
checking population-to-case protocols and available peer/profile scalar gates.
Those scalar checks still do not replay missing force arrays. These corrections
address reviews 5555852350 and 5556330619 and adapt the bounded behavioral
controls proposed in issue comment 6230099163; they add no production GPU work.

The final default-policy host cohort passes 234 tests with one explicit optional
history skip; the separate historical cohort passes all 28 tests. Documentation
checks pass 19 tests and 61 subtests. On node2, finite Slurm3394 runs the restored complete
native fixture, the explicit mixed-JK driver and density contraction, followed
by density memcheck, synccheck and racecheck with zero reported errors/hazards.
All fourteen generated fixture headers are byte-identical to the previously
qualified density source. This does not turn the older unfiltered linked
racecheck timeout into a passing result.

## Current-source unset-selector endpoint qualification

Finite Slurm node4 job 1111 uses its assigned RTX 5090 and the frozen source
`5eaa97feb83387814cabfc415e9b10e30035e62f`, reconciled with master
`4e94d6bf11c2e33266aed34485248e7fb0569f3c`. Native identity is
`ef867b3c2ec869e04665e18f671e1e4544b8eccde0bed5481a0375bc3541efa6`;
subsequent test, historical-reader and rationale changes leave every frozen
production input byte-identical. The later master change `e26c6ce46` concerns
static array-API take indices, not this force qualification; it is not silently
substituted for the measured source.

Both arms use that same authenticated native image. Baseline `spatial-reuse`
emulates the prior eligible automatic policy; HF uses `0`, its prior incumbent.
The candidate genuinely removes the selector, including inherited values. This
qualifies default promotion, not the entire compiler prerequisite against
unmodified master. The shared prerequisite also affects already-admitted
order-seven/eight consumers; the order-six label is not the full source delta.

All eight endpoint populations complete cold, five warm calls, changed geometry
and five changed-geometry warm calls within the original 1800-second population
cap. Canonical def2-mTZVPP, moving 48 x 16 x 32 Becke grid, exact Direct,
strict FP64, screening and SCF thresholds remain unchanged. Independent pinned
geometry oracles recheck all 96 energies and every force component; all values
are finite. Maximum errors are `2.4101609596982598e-11` Eh and
`6.036060431861834e-11` Eh/Bohr, below the original `1e-8`/`1e-7` gates.
No endpoint population is incomplete.

| r2SCAN-3c / 96 atoms / 1088 AO | Prior policy (s) | Unset default (s) | Change |
|---|---:|---:|---:|
| Cold | 257.376777 | 244.933585 | -4.835% |
| Warm median of five | 43.096976 | 30.395943 | -29.471% |
| Changed geometry | 157.561219 | 144.885886 | -8.045% |
| Changed-geometry warm median of five | 43.161365 | 30.378178 | -29.617% |

| Matched control | Cold (s, before -> after) | Warm median (s) | Moved (s) | Moved-warm median (s) |
|---|---:|---:|---:|---:|
| r2SCAN-3c / 48 | 63.384755 -> 60.064777 | 11.465641 -> 8.056844 | 45.117614 -> 41.667489 | 11.460035 -> 8.049277 |
| PBE0 / 48 | 23.479851 -> 23.214215 | 4.062626 -> 3.947513 | 22.086436 -> 21.937030 | 4.040344 -> 3.930062 |
| HF / 48 | 11.163714 -> 11.183215 | 0.745575 -> 0.744037 | 7.223661 -> 7.223939 | 0.745362 -> 0.746038 |

The corresponding four control changes are -5.238/-29.731/-7.647/-29.762%,
-1.131/-2.833/-0.676/-2.730%, and +0.175/-0.206/+0.004/+0.091%.
Retain the observed HF increases; this one policy-control cohort does not prove
they are noise or establish their reproducibility. HF's automatic schedule is
unchanged. No matched control exceeds the requested 5% regression goal. The
earlier PBE0 cold +3.06% result is not reproduced by this measured source.

Actual iteration/Fock-build counts match between arms, without normalization:

| Case | Cold iterations/Focks | Moved iterations/Focks | Every warm call |
|---|---:|---:|---:|
| r2SCAN-3c / 96 | 24/24 | 13/13 | 1/1 |
| r2SCAN-3c / 48 | 19/19 | 13/13 | 1/1 |
| PBE0 / 48 | 17/18 | 12/12 | 1/1 |
| HF / 48 | 21/22 | 16/17 | 1/1 |

This is one current-source forward cohort, not a newly executed reverse cohort.
The earlier explicit-selector forward/reverse observations remain separately
source-bound and are not relabelled as unset-default repeats. The occupied-XC
option remains excluded pending current-source composition and qualification;
an old isolated cold win is not sufficient evidence to enable that combination.

## Current-source counted profiles and executed work

The same finite allocation completes both counted cold/first-warm profiles.
All four returned energies and full force arrays independently pass the same
oracle and finite-value checks. There are zero unattributed kernel launches.
The counted cold trajectory is 24/24 baseline versus 25/25 candidate, with
complete diagnostic times 258.650838 versus 253.752671 seconds; neither that
extra SCF build nor instrumentation is normalized away. Both diagnostic warm
calls use 1/1 and take 43.469657 versus 30.542456 seconds. These are diagnostic
observations, not replacements for the uninstrumented endpoint medians.

Mutually exclusive complete warm wall owners, including synchronization and
publication within their owning scopes:

| Complete wall owner | Baseline (s) | Default (s) |
|---|---:|---:|
| Native SCF and publication | 8.286080 | 8.257640 |
| Stationary force and publication | 33.047845 | 20.280353 |
| Final-state snapshot | 0.659698 | 0.544770 |
| Final-state decode | 0.406208 | 0.403492 |
| Final-state validation | 0.615950 | 0.602717 |
| D4 | 0.445788 | 0.445579 |
| Native CPU gCP | 0.002146 | 0.002056 |
| Construction, public dispatch and final return/sync | 0.005943 | 0.005848 |
| Endpoint preparation | 0 | 0 |
| Total | 43.469657 | 30.542456 |

Every full-precision ledger sums to its returned complete wall time within
`1e-7` seconds. NVTX exclusive host ownership, device windows and the following
summed kernel device times are separate views; do not add them to each other or
to the complete wall ledger.

| Warm device kernels | Baseline (s) | Default (s) |
|---|---:|---:|
| SCF exact J | 4.226393 | 4.198163 |
| SCF XC/AO/grid | 3.962234 | 3.962411 |
| Generated force orders zero through five, aggregate | 4.628445 | 4.626712 |
| Order-six J prime | 13.868031 | 2.615874 |
| Order-seven J prime | 2.000622 | 0.492004 |
| Order-eight J prime | 0.454047 | 0.454063 |
| Other stationary force/grid/Becke response | 9.319930 | 9.320196 |
| One-electron pair resident metadata | 2.272884 | 2.269236 |
| Resident validation kernel | 0.017608 | 0.017610 |

The candidate's 9.320196-second other-response group contains geometry response
4.159853, Becke 2.490624, GEMM 1.933841, AO radial 0.509734 and other kernels
0.226144 seconds. It is not all Becke. The low-order aggregate is not an
isolated order-five measurement. The exclusive device window without kernel
activity is 2.170203 versus 2.037071 seconds; it is not a measured synchronization
kernel cost. SCF exact J already uses the MD-J consumers; disabling a ledger
guard to supposedly activate them is not a new optimization opportunity.

Actually entered warm work, not candidate enumeration or hardware FLOPs:

- Order six primitive-pair products fall from 2,400,504,912 to 8,666,248;
  admitted shells remain 4,101,784. Enumeration increases from 6,875,756 to
  117,562,880 and is explicitly not the claimed saving. The factorized consumer
  additionally executes 1,882,836,519 density terms, 3,153,588,800 Hermite joins,
  6,010,796 bra preparations and 8,455,560 ket/Coulomb preparations. There are
  4,099,576 attempts, 4,072,824 accepts and 26,752 mask declines. Covered
  component-fallback bra/ket/Coulomb preparations are 83,770/210,688/210,688;
  unavailable six-class coverage keeps aggregate preparation counts null.
- Order seven primitive-pair products remain 1,250,544. AO contractions change
  from 754,811,008 to 4,136,704 fallback contractions, plus 453,127,296 density
  terms and 870,721,600 Hermite joins. Density bra/ket/Coulomb preparations are
  1,023,448/1,243,888/1,243,888; fallback counts are 4,304/6,656/6,656. Thus
  Coulomb preparations total the same 1,250,544, not a fabricated reduction.
  There are 811,824 attempts, 809,776 accepts and 2,048 mask declines.
- Order eight remains 95,976 primitive-pair/Coulomb preparations. No measured
  class has a numerical density-contraction decline. Unavailable counters stay
  null, and fallback attempts/preparation are not omitted from the receipt.

All 96 endpoint and four profile calls qualify. Raw arrays, SQLite traces,
source/cache/device receipts and full-precision independent summaries remain in
ignored `.artifacts/default-density-20261017/`; lossless scalar observations are
published on the implementation PR. Scalar publication is not offline force
replay. No new results archive, budget increase, release or tag is introduced.

## Remaining GPU4PySCF gap

The retained same-day GPU4PySCF 1.8.1 reference on the identical RTX 5090 UUID
comes from finite Slurm1109, not the historical PR2538 measurement. Its complete
cold/warm/moved/moved-warm times are 142.375318/28.830030/120.502940/26.643178
seconds. The new default remains slower by 72.034/5.432/20.234/14.019%, or
102.558267/1.565913/24.382946/3.735001 seconds. This is not parity.

Reference actual iteration/J-K-build counts are 54/55 cold, 45/46 moved,
4/5,4/5,5/6,5/6,4/5 warm and 3/4,3/4,4/5,3/4,3/4 moved-warm. Native counts
remain the separately recorded 24/24,13/13 and 1/1. Do not normalize these
different solver trajectories into an isolated-kernel comparison.

The measured next targets are generic low-order density contraction and
stationary geometry/grid response, followed by one-electron metadata and final
state materialization. Cold remains dominated by actual repeated SCF J and XC
work; the occupied-XC composition needs independent current-source qualification
before any default promotion. Keep these separate from this bounded policy PR.

## Rejected alternatives and remaining work

Do not enable degree pruning, separate-cache XY CSE or borrowed-frame prefixes:
their complete-endpoint evidence did not justify retained production complexity.
The [negative experiments](../../rejected/2026-10-17-direct-density-followup-trials.md)
explain the measured limitations. Do not enable approximate precision merely to
claim parity, or normalize away different convergence trajectories.

Default promotion is not GPU4PySCF parity. Continue reducing the remaining
low-order force, stationary grid/response and cold SCF costs using complete
source-matched qualification. This decision does not authorize merging PRs.
