# Decision: exact resident AO bitmasks and generated Becke atom-adjoint cuts

Status: implemented, qualification-only; production defaults unchanged
Date: 2026-10-09
Related: #1893, #1894
Base: master `fb9586569769fccac67bc23411d902e08119024d`

## Problem

Sampled force discovery evaluates all through-order AO jets, writes a dense
panel, reduces it to labels and synchronizes each tile. Conservative pre-AO CSR
avoids jets but can admit more downstream work and require too much optional
storage. Neither reducing discovery time alone nor allocating a sparse map
proves a complete endpoint advantage.

The previous Becke coefficient experiment saves pair-panel words but adds
direction reads/multiplications to gather. Repeating that losing schedule, broad
zero pruning or the old pair-log move is not a new derivative primitive.

## Decision

Generate an exact sampled-jet bitmask producer from the existing collocation
arithmetic emitter. All configured jets are evaluated and checked; "exact"
means the identical strict-cutoff label predicate, not unscreened mathematics.
Do not materialize a discovery panel. Keep immutable tile bitmasks and counts,
and compact one sorted AO span on the consumer's stream. This reuses the existing
`AoGridBlockLayout`, gather/projection/scatter and derivative-capability checks;
it does not introduce a second sparse scientific representation.

`ExactAoMapResources` charges masks, both offset mirrors and one full-capacity
index span. Reserve this finite owner, not an arbitrary 16/64-MiB cache cap:
unused allowance is useful native integral-provider workspace. Insufficient
headroom reserves zero and takes the existing dense route. Keep the mandatory
provider reserve, full-capacity consumer arena and occupancy/cutoff guards.
Numeric capacities exclude driver/object overhead and are not allocator peaks.

For Becke, authenticate the canonical whole normalized-product AD composition,
then emit exp/log pullback cuts with the primal product already bound. Hoist
atom weights into normalization and reuse them at incident pairs. Keep the
ordinary four-word reverse and ordered gather: no coefficient expansion, pair
pruning, new allocation or extra launch. Field 7 is dead until gather and can
hold weights only between normalization and the completed reverse phase.
Single-zero pullbacks retain the excluded product; multiple zeros annihilate it.
Nonfinite or prematurely underflowed cuts request the original pair reverse.

The new Becke mode is `normalized-adjoints`; `off` and the older `coefficients`
control remain available. An older native artifact without the new configuration
ABI stays on its ordinary route. Never reconfigure a live topology owner.

## Invariants and rejected shortcuts

- Keep basis, geometry, grid, point order, device, tile and through-order map
  identities. An SCF order-1 map is not an order-2 force map.
- Bitmask storage is immutable, but its compacted span is ephemeral. All borrowed
  consumers finish enqueueing on the grid stream before the next compaction;
  the existing native geometry owner rejects a pending different stream.
- Do not add logical CSR offsets to the smaller rebased allocation. An empty
  indexed view needs a nonnull marker even though it never reads an AO label.
- Check raw nonfinite points explicitly. Infinite radii can underflow every
  exponential to zero and otherwise masquerade as a valid empty map.
- Do not call hoisting bitwise equivalent: multiplication association changes.
  Independent Decimal/finite-difference and complete E/F gates remain required.
- Count full discovery AO visits/jets even without panel writes; account for
  compaction, retained bytes and unchanged local AO-square work.

## Evidence and reproduction

Use `benchmarks.readme_pbe0_indexed_becke` with the unchanged README PBE0 protocol,
independent GPU4PySCF reference, full spherical def2-SVP, FP64 and unpruned moving
48x16x32 grid. The controls are `--force-producer exact-jets-native-bitmask` and
`--becke-primitive normalized-adjoints`. They preserve incumbent eligibility,
cutoff and endpoint budgets and do not register default profiles.

The local evidence root is `.artifacts/issue1893-1894`; the finite n1 Slurm
campaign is `/data/jzzeng/qc-exact-becke-20261009-fb9586569/evidence`. The initial
failed source/binary is retained under the sibling `failed-v1` directory. Its
unguarded envelope launch used null bounds on the exact path, and empty-map
pointer handling was incorrect. The failed memcheck's 44 errors are not a
qualification result. Both are fixed before the passing cohorts.

The corrected v2 cohort passes 267 real-device gates, with memcheck/racecheck
showing zero errors/hazards. Explicit nonfinite-point and exact-budget/tail gates
extend the final cohort to 275 passing device gates; final exact-map memcheck
passes 28 selected cases with zero errors. Becke host gates include independent
Decimal/finite differences, iterations 1/3/5, geometry rebinding, translation,
permutation, rounded zeros and overflow/premature-underflow cut fallback.

Complete endpoint records and the later tightened-map-admission population must
remain distinct. Cold/moved iteration histories are retained rather than
normalized. Grouped five-replay medians do not establish an interleaved paired
performance win or authorize default promotion.

### Initial complete endpoint population

Frozen master is `fb9586569769fccac67bc23411d902e08119024d`. The baseline,
overreserved exact producer and normalized-adjoint-only protocols ran in Slurm
job 6696 on n1, GPU visibility 0. Every native arm has 12 accepted complete calls
(cold, five warm, moved, five moved-warm) against fresh independent references.
No reference density was supplied. The energy/force gates are `1e-8` Eh and
`1e-7` Eh/bohr. The largest initial energy error is `8.186e-12` Eh and force error
is `3.790e-11` Eh/bohr. Values below are complete seconds, not isolated stages;
parentheses preserve the cold/moved iteration counts. All replay samples use
one iteration and one Fock build.

| Arm | Atoms | Cold (iterations) | Warm median | Moved (iterations) | Moved-warm median |
| --- | ---: | ---: | ---: | ---: | ---: |
| master | 48 | 49.096973 (19) | 6.380662 | 31.932178 (12) | 6.395726 |
| exact, overreserved | 48 | 52.753379 (21) | 6.404373 | 31.487743 (12) | 6.404498 |
| normalized only | 48 | 51.009009 (20) | 6.430273 | 31.920504 (12) | 6.417437 |
| master | 96 | 129.172958 (22) | 18.988425 | 77.759703 (12) | 18.982918 |
| exact, overreserved | 96 | 114.340497 (19) | 19.039873 | 81.356670 (13) | 18.969809 |
| normalized only | 96 | 106.078341 (17) | 19.058174 | 77.944057 (12) | 19.136131 |

Cold iteration histories differ, so lower cold wall time is not a source-speed
claim. Normalized adjoints do not win the warm endpoint controls. Retain this
losing result and the ordinary phased route.

### Mechanism accounting

These cold/moved force-domain work counts match the incumbent sampled-jet map.
The exact producer changes discovery writes/storage, not the omission threshold
or the local AO-square work.

| Atoms | Discovery jets | Selected point-AO visits | Selected AO-square work | Tiles / empty / compactions | Exact bytes | Incumbent retained bytes |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 48 | 4,529,848,320 | 172,813,312 | 29,924,424,704 | 2,304 / 208 / 2,096 | 150,544 | 2,700,208 |
| 96 | 18,119,393,280 | 411,439,104 | 94,349,066,240 | 4,608 / 384 / 4,224 | 522,256 | 6,428,736 |

Discovery times in the initial population are 2.208848 -> 0.274120 s at 48 atoms
and 6.717125 -> 0.971272 s at 96 atoms. Exact mode writes zero discovery panels,
downloads only 18,444 / 36,876 bytes of offsets/error status and uploads
18,440 / 36,872 bytes of offsets. AO-label H2D and host label lookup are zero.
Warm maps perform no discovery but still compact each nonempty indexed tile.

Normalized mode 2 preserves 1,330,642,944 / 10,758,389,760 pair-primal and reverse
visits, twice those counts for gather, and 16,128 / 32,256 phase launches at
48 / 96 atoms. Logical pair-panel traffic remains 32 bytes/primal pair,
32 bytes/reverse pair and 64 bytes/gathered pair; these are distinct logical
values, not hardware load/store transactions. Atom-adjoint hoisting does not
prune pairs or eliminate their full geometry work.

The initial exact population reserved 16,777,216 / 5,142,336 bytes even though
its inventory used only 150,544 / 522,256 bytes. `candidate-tight` is a separate
fresh source/build population for the finite admission fix, not a relabeling of
those initial endpoint timings. Source hashes and native/AOT identities are
retained in the raw records and `native-artifact-identities.sha256` ledger.

### Final-source regression and sanitizer gates

The final host cohort passes 807 tests, with 272 opt-in/device-related skips;
these are one cohort, not a sum of overlapping earlier runs. Compiler structure
checks 503 modules with zero dependency errors, CUDA ownership checks 338 files,
and Ruff/format/diff checks pass. Slurm job 6697 rebuilds `candidate-tight` with
the shared compiler cache and passes 294 tests: 275 real-device gates plus 19
device-free finite resource-plan gates. Its selected exact-map memcheck has nine
passing cases and zero errors.

Job 6698 independently tests the same frozen native/AOT population with the new
strict-cutoff harness retained in the evidence directory rather than mutating
the running source checkout. The harness derives cutoffs from the incumbent GPU
panel and checks equality and the adjacent FP64 values for orders 0/1/2/3; it
does not exclude values near the threshold or substitute a tolerant host AO
approximation. Both memcheck and racecheck pass 38 exact-map/nonfinite/boundary
cases and eight normalized-Becke subset cases, with zero errors, warnings or
hazards. The four new boundary cases are additional to the 275-device cohort,
not another 38 distinct tests to sum into it.

Native main-library SHA-256 identities are:

| Population | `libgenerativeqc.so` SHA-256 |
| --- | --- |
| master | `3bfba774e4f83bc4ba501e0718da5c3755e96bd5994277b1e32c2e6f9ce3474c` |
| initial candidate | `27d1efe9fef05231c530e01116bff082ae0ad9330c0de601777b0c4cdae3bbcf` |
| tightened admission | `585caec6c420e697f9a695e8d0612e9dbf5e10779251674f29e24ab3630a09bb` |

The candidate s/p/d stationary AOT identity is
`4a93342494dcd61b18e1db0b703094ee8d464ba2f2d929edab96447bfc5c2649`
in both source populations: the admission fix does not change that generated
Becke artifact. The master s/p/d identity is
`e7da467f45038d7026c27b8fccfb6f1b9ebffe79f6fddd2a74d905fb85e5b5d3`.
The ledger also retains the s/p AOT identities and full paths. Endpoint records
bind the additional experimental compiler/runtime file hashes, not just HEAD.

### Tightened admission and joint endpoint population

Job 6697 uses assigned GPU visibility 1; job 6698 uses visibility 0, with the
same frozen `candidate-tight` main/AOT artifacts. This distinction, concurrent
node work and grouped rather than interleaved controls preclude attributing the
wall-time differences solely to the admission fix. Both budgets reserve exactly
150,544 / 522,256 bytes for the map, and the complete additional device bounds
are 257,657,680 / 527,399,824 bytes, below the 536,870,912-byte allowance. Host
numeric bounds are 94,603,824 / 256,066,992 bytes, below 268,435,456 bytes; the
mandatory integral-provider reserve remains intact.

| Final-source arm | Atoms | Cold (iterations) | Warm median | Moved (iterations) | Moved-warm median |
| --- | ---: | ---: | ---: | ---: | ---: |
| tight exact only | 48 | 49.268940 (19) | 6.670138 | 34.083790 (13) | 6.655566 |
| tight exact + normalized | 48 | 48.501482 (19) | 6.404951 | 31.353383 (12) | 6.404847 |
| tight exact only | 96 | 106.688352 (17) | 19.387287 | 77.803861 (12) | 19.493642 |
| tight exact + normalized | 96 | 114.102569 (19) | 19.095233 | 76.932187 (12) | 19.052420 |

All ten native populations, 120 complete calls in total, converge with status 0
and pass independent E/F gates. Maximum errors are `8.186e-12` Eh and
`3.790e-11` Eh/bohr. Every native warm/moved-warm call retains one iteration and
one Fock build. The evidence verifier independently recomputes the existing
protocol's gate against the cold/moved oracle for each geometry, additionally
checks each matching reference replay, verifies map/local-work equality against
master, mode-2 selection, pair traffic/launch counts and both resource bounds.
Final experimental source hashes match the checkout. No raw result was edited
to normalize iterations or replace an earlier population.

Reference complete warm/moved-warm medians are 6.141895 / 3.949219 s at 48 atoms
and 10.129632 / 10.093868 s at 96 atoms. These are not work-normalized controls:
reference cold/moved iterations are 43/43 and 42/44, with 44/44 and 43/45 SCF JK
builds; its replays retain 1–5 iterations and 2–6 JK builds. The joint native
arms do not establish a stable full-endpoint advantage over master or close the
reference gap. Keep both experiments opt-in and retain the negative evidence.

Raw JSON/logs and `verified-summary.json` are retained in the ignored local
evidence root. The latter's SHA-256 is
`d22c171da07af06d0afbeeb04b5ba80b55964fe6b44642cc219307603e731bc3`.
The additional strict-threshold harness has SHA-256
`00395a02df5851da79352067e78dda6c5a3cd30576bba0d3d6c8df3b5230a5bd`.
This is RKS complete-endpoint qualification plus synthetic two-spin/changed-
geometry gates, not a promotion of every UKS, radii-adjusted or unsupported
derivative domain covered by the broader parent issues.

### Why replay performance does not improve

The measured discovery improvement is real but applies only when creating or
invalidating a map. Every incumbent warm replay already reports zero discoveries;
removing discovery panels cannot speed up work that the incumbent skips too.
Exact mode also introduces 2,096 / 4,224 same-stream tile compactions per force.
The gather/contraction/scatter domains retain exactly the incumbent selected
point-AO and AO-square counts. Reduced map capacity therefore does not imply
less steady-state scientific work or a measured endpoint speedup.

Normalized atom adjoints move a repeated scalar pullback product across a phase
boundary. They do not change the point/pair domain, four-word reverse/gather
representation, pair-panel traffic model or seven-launch-per-tile schedule.
The incumbent phases already reuse primal pair/product state, so this is a
constant-factor arithmetic change, not removal of the normalized-product or
geometry-response algorithm. At 96 atoms each force still produces and reverses
10,758,389,760 pairs and gathers twice that number of incident entries.

The master warm medians retain 1.938815 / 6.543644 s of integral derivatives and
1.836230 / 5.760378 s of aggregate semilocal geometry response at 48 / 96 atoms,
within complete calls of 6.380662 / 18.988425 s. The corresponding joint medians
are 1.944026 / 6.540678 s and 1.850974 / 5.841012 s, within 6.404951 / 19.095233 s.
Integral derivatives and the remaining SCF/endpoint work are not reduced by
these mechanisms. Aggregate geometry wall time includes AO/XC and other work;
it is not an isolated Becke reverse profile. Intrusive phase profiling was off,
so these records do not establish a DRAM-bandwidth bottleneck or quantify how
much compaction overhead offsets arithmetic/transfer savings.

## Revisit when

Promote only after complete, genuinely paired endpoint evidence with matching
semantic work, independent gates, capacity/compile accounting and appropriate
RKS/UKS capability coverage. Neither kernel savings nor sparse-policy selection
alone closes the broader issues.
