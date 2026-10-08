# Decision: retain default-off AO reuse and preserve force-provider admission

Status: implemented local candidate; cold-performance default promotion rejected
Date: 2026-10-08

Superseded for native AO default selection by
[user-accepted default reuse](2026-10-08-default-native-ao-reuse.md). The original
failed statistical gate and producer-control results below remain unchanged.

## Problem

Frozen master `4385f72751b829883407c01106186917c344317b` already contains
#2100's default bounded MD-J. Its energy-only qualification uses a different
grid and endpoint contract; it must not substitute for the large-grid complete
PBE0 energy/analytic-force comparison here. Six alternating clean observations
give 115.437826 s median versus 78.498101 s for the three primary GPU4PySCF
observations. The excluded native/reference pilot observations remain retained.

This endpoint has 96 atoms, 768 spherical def2-SVP AOs, 2,359,296 unpruned
grid points, PBE0, energy/density tolerances 1e-12/1e-10, native screening
1e-12 and reference direct tolerance 1e-14. The timer includes preparation and
the synchronized public energy/force return, excluding imports, context setup
and Calculator construction. Processes, owners and densities are fresh; caches
are persistent. Force work receipts identify packaged/native-build AOT, not a
runtime compiler miss. Primary native comparisons share Slurm allocation 6602;
diagnostic profiles never enter this clean population.

## Decision

- Expose native AO radial/axis reuse through CMake
  `GENERATIVEQC_CUDA_AO_RADIAL_REUSE`, **default OFF**. Four/ten-jet paths reuse
  identical axis DAG results (12 to 6 and 30 to 9 calls per Cartesian term),
  with one primitive exponential instead of four/ten. Multiplication and
  accumulation order remain unchanged. Scalar one/twenty-jet paths and default
  runtime JIT dispatch remain unchanged.
- Preserve the native derivative allowance already established by geometry
  planning when admitting optional device CSR maps. The map receives only the
  space above the dense owner plus `layout.native_geometry_reserve`. Optional
  map refusal retains the dense fallback; it cannot erase the native integral
  route's budget. This correctness fix does not change the force producer policy
  or automatically select a different tile.
- Do not promote AO reuse or a forced large-domain pre-AO CSR producer based on
  the kernel improvement, failed endpoints, or diagnostic stage savings.

## Root-cause evidence

Separate NSys/CPU diagnostic traces on the frozen libraries show:

| Master stage | Wall s | Device busy union s | Key kernel sums s |
| --- | ---: | ---: | --- |
| Prepare | 6.197 | 4.966 | AO 1.511; grid partition 1.420; J bounds 1.127 |
| SCF, 17 Focks | 85.616 | 81.930 | MD-J including residual 27.832; K 27.590; XC matrices 15.886; AO 5.281 |
| Forces | 24.038 | 16.978 | combined J'/K' 5.471; AO 4.727; geometry 2.156 |

The native MD-J residual `consume_source_tasks` kernels account for 11.501 s;
they are **not K queues**. Native K's largest streaming classes include ppps
2.354 s, ddds 2.303 s and dpps 2.268 s across this 17-Fock diagnostic. This
is an executed-time attribution, not an admitted-work census or a same-density
cross-engine speed ratio.

Reference profiling has **47 Focks**, versus clean histories of 38/38/44; its
85.477 s SCF trace is not a representative clean median. Force wall/busy union
are 8.901/6.011 s. Exact reference force AO collocation sums to 0.106706 s,
and the `rys_ejk_ip1*` derivative family to 1.608967 s. Collocation must include
both `_cart_kernel_deriv*` and `_sph_kernel_deriv*`. Uncovered device time is
not automatically pure CPU or compilation; API inclusive residence overlaps
device work. CPU profiling itself increases the force wall.

NCU 2025.2.1, Slurm 6622, matched first force-discovery AO workload
(512 points, 768 AOs, ten jets):

| Metric | Scalar | Reuse |
| --- | ---: | ---: |
| Duration, us | 910.144 | 256.736 |
| Warp instructions | 203,595,264 | 52,344,832 |
| Blocks x threads | 30,720 x 128 | 3,072 x 128 |
| Registers/thread | 50 | 74 |
| Achieved active warps, % | 68.626 | 42.628 |
| Local load/store requests | 0 / 0 | 0 / 0 |

This proves duplicate-work removal wins despite lower occupancy. It does not
prove a cold endpoint win. Candidate force AO kernel sum is 1.346721 s versus
4.727388 s on master, with the same 8,832 launches.

The full-set derivative NCU attempt timed out at its finite 20-minute limit
without a usable report. The small-metric retry, Slurm 6630, completes three
replay passes and yields a numerically accepted full E/F endpoint. Its single
combined derivative kernel has duration 5.479244 s, 757,837,633,032 warp
instructions, 16.666666% achieved active warps, 57,358,297,954 local-load
requests and 23,714,618,071 local-store requests. NSys reports 255 registers.
Requests are not bytes, and local-array traffic is not automatically all
register spilling. These observations motivate reducing private derivative
state and repeated work; they do not qualify a register-cap or cooperative
schedule change. Preserve the previous cooperative order-six negative result
and the order-seven candidate's existing qualification boundaries.

## Cold endpoint profit gate

All twelve native primary observations pass independent E/F gates. No rows
are normalized by Fock count or removed:

| Mode | Complete median s | Force median s | Fock histories |
| --- | ---: | ---: | --- |
| Master | 115.437826 | 22.799250 | 17,17,22,17,17,18 |
| Reuse | 111.128942 | 20.111963 | 19,17,17,18,19,17 |
| GPU4PySCF | 78.498101 | 8.546738 | 38,38,44 |

Reuse point estimate is 3.732645% less complete time. Independent bootstrap
of complete medians (100,000 draws, seed 2100) gives a 95% gain interval
**[-3.116631%, +13.435876%]**. It fails the positive-lower-bound gate. Do not
enable the option by default or open a cold-benefit PR on this result. Master
median prepare is 5.896884 s (5.108% of its complete median); compilation is
not an explanation for the remaining J/K execution gap.

## Pre-AO controls and the admission defect

Master force discovery evaluates 1,811,939,328 dense point-AO entries and
18,119,393,280 AO jets in 4,608 synchronous discoveries before evaluating
411,439,104 selected point-AO entries again. The sparse downstream path is
real, but it does not eliminate dense discovery.

Forced existing native CSR with both 32/64 MiB requested storage and automatic
512-point tiling fails to publish forces. The instrumented 32-MiB control in
Slurm 6629 records peak 536,870,912 B, clipped map reserve 9,993,344 B and
**zero integral allowance**: the native provider is never invoked. These
approximately six-second failed force calls are not performance samples.

The same total 512/256-MiB device/host budgets, cutoff and science with explicit
256-point tiles and 64-MiB CSR storage pass E/F in Slurm 6629. Discovery takes
0.181869 s with zero AO discovery jets, but selected point-AO work rises to
528,629,760 and AO-square work to 180,145,025,024, with 9,216 tiles. Diagnostic
force wall is 21.374129 s. Combining this producer with AO reuse, Slurm 6633,
also passes, but force wall is 21.049280 s; this does not establish an endpoint
advantage over reuse alone. More conservative maps and doubled tile orchestration
offset much of the discovery savings. Do not blindly replace sampled defaults.

The runtime admission fix, Slurm 6634, preserves 4,851,008 B for the native
provider, clips the optional map to 5,142,336 B and returns accepted E/F with
device peak 532,019,904 B. All 4,608 tiles take the explicit dense-budget
fallback because the CSR inventory does not fit. Force wall is 28.517974 s:
this is a successful **correctness recovery, not a performance improvement**.
The forced 96-atom producer is unqualified; these results do not prove that
master's current default 96-atom sampled policy is broken.

## Validation and evidence retention

- Initial AO/compiler host suites: 184 passed, 49 opt-in skips; 494 compiler
  modules with zero structure errors. Native AO gate: 72 checks, through-f,
  orders 0-3, full/sorted/empty maps and partial tiles; maximum independent
  error 1.705303e-13. Memcheck and synccheck report zero errors. Complete
  independent 6/48-atom gates and all primary 96-atom gates pass.
- Admission/ordinary-layout/resident-map/provider/merge-boundary host tests:
  163 passed, including real resource-model and exact-budget regression cases.
- Initial AO fixture failures in jobs 6600/6601 were harness contract errors
  (order-zero gradients, unsorted AO labels); originals remain retained. Job
  6602 corrects the harness without rebuilding the qualified candidate binary.
- All GPU/profiler/sanitizer work uses finite `srun`, preserving its CVD, on
  n1/node1 RTX 5090. CPU source checks/builds do not bypass the GPU scheduler.
  Reused ccache and library hashes distinguish compilation from execution.
- Sources, scripts and raw JSON/NSys/SQLite/NCU/pstats/receipts are retained at
  `/data/jzzeng/qc-pbe0-master-rootcause-20261008-4385f7275/evidence` on n1 and
  `.artifacts/rootcause/results` in the local worktree. A compact tracked
  ledger is generated from these original receipts, not reconstructed timings.
  Intrusive producer controls overlap other diagnostic allocations and are
  not clean performance populations.

## Consequences and revisit conditions

Prioritize J/K execution work, including MD-J residual screening/contraction,
and derivative-private-state amplification over more compile-time or occupancy
tuning. For force maps, remove synchronous discovery without substituting a
much denser domain or doubling tile orchestration. Any batched exact-map route
must preserve derivative-order coverage, immutable epochs and resource lifetime;
the present controls are not its implementation or qualification.

Revisit default AO/producer selection only after final-source full endpoints
clear a robust positive gain gate, with complete histories, independent E/F,
bounded fallbacks and applicable cold/warm/moved semantics. GPU4PySCF parity
has not been achieved.

## References

- #1965: complete PBE0 endpoint/work-count tracker.
- #1893: sparse AO/force-domain ownership and pre-AO producer controls.
- #1892: independent J/K plans; implementation closed, performance gaps remain.
- #2100: default MD-J; this candidate does not modify its J or K contracts.
- `2026-10-06-pre-ao-native-csr.md` and
  `2026-10-06-cooperative-weighted-shell-force.md`: retained qualification and
  rejected alternatives; no negative result is superseded by isolated counters.
