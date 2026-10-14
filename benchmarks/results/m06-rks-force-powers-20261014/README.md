# FP64 M06-2X RKS force integer powers

Campaign: October 14–15, 2026. Qualification source: master `8c770b34c`
(including #2486), candidate `8021e2d92`. This is an independent improvement;
it neither depends on nor re-credits #2464's equal-spin branch CSE.

The candidate lowers integer powers up to four **after independent polarized
AD**, only in the FP64 M06-2X RKS stationary force consumer. SCF point source,
UKS consumers, float scalar graphs, MN15 and r2SCAN defaults stay unchanged.
Strict FP64 force fallbacks use the qualified FP64 force graph. Generated
source/artifact identities change; mathematical expression identities do not.

## Complete endpoint timing

RTX 5090, node n5, finite Slurm jobs 1576/1578, CUDA 12.9.1, `sm_120`, Release,
FP64, spherical def2-SVP, matched moving quadrature and DF model. Ordinary
README native energy **plus host-returned analytic forces**, no foreign seed.
Each population includes cold + five fixed own-engine post-cold warm replays
+ moved + five fixed post-move warm replays. Cold includes preparation; all
entries use `complete_seconds`. Warm entries are medians of all five calls;
cold and moved below are single calls, not statistical assurances of small
sub-percent improvements. Raw samples and ranges remain in the capsule.

First matched population:

| Case | Cold, baseline → candidate (s) | Warm (s) | Moved (s) | Moved-warm (s) |
| --- | --- | --- | --- | --- |
| DF48 | 45.952664 → 45.384089 | 8.899711 → 8.077829 | 34.786684 → 34.070293 | 8.891267 → 8.076551 |
| DF96 | 152.750312 → 151.213377 | 37.364340 → 35.669518 | 112.652525 → 110.930658 | 37.354787 → 35.647791 |
| Direct96 | 163.911763 → 163.142795 | 20.089646 → 19.214328 | 113.078680 → 112.308974 | 20.064995 → 19.228165 |

Complete warm E+F reductions are **9.23%, 4.54%, 4.36%**; moved-warm reductions
are **9.16%, 4.57%, 4.17%**. The first population's cold/moved times also decrease,
but those small changes are not the headline claim. This is not a point-kernel
percentage relabeled as an endpoint improvement, and is not a new GPU4PySCF
comparison or a claim that every README gap is solved.

Observed iterations/Fock builds in **both** arms, in phase order:

| Case | Cold | Each warm | Moved | Each moved-warm |
| --- | --- | --- | --- | --- |
| DF48 | 18 | 1 | 13 | 1 |
| DF96 | 18 | 3 | 12 | 3 |
| Direct96 | 18 | 1 | 12 | 1 |

Second complete matched direct96 population:

| Phase | Baseline → candidate (s) | Baseline → candidate builds |
| --- | --- | --- |
| Cold | 171.732486 → 163.300249 | 19 → 18 |
| Warm | 20.093679 → 19.228675 | 1 → 1, all five |
| Moved | 120.919491 → 112.209232 | 13 → 12 |
| Moved-warm | 20.089118 → 19.224459 | 1 → 1, all five |

Warm/moved-warm reproduce **4.30%/4.30%** fixed-work complete E+F gains. The
unchanged baseline image itself varies from 18/12 to 19/13 cold/moved builds;
those larger second-population timing differences include different SCF work
and are **not** credited as a solver improvement from force-only arithmetic.
Qualification contains all **96 master endpoint calls**, with unchanged gates
passing throughout. Maximum observed errors are **2.0464e-11 Eh** and
**1.5725e-10 Eh/Bohr**. Available AO/grid/force work is retained per call; absent
integral execution counters are unknown, not zero. No SCF work is subtracted,
normalized or replaced.

## Scientific and operational acceptance

- Unchanged endpoint gates: **1e-8 Eh / 1e-7 Eh/Bohr**, independent references
  at both geometries. Every returned E/F vector and gate is retained.
- Host tests execute the production force emitter against independent Libxc
  Cartesian partials and displaced energies, including the tau/2 ABI and
  independent polarized derivatives before equal-spin binding. The focused
  suite has 124 passes and 43 explicitly optional skips; the final policy-only
  rerun has 17 passes. Configured pre-commit checks pass.
- Clean-source real-GPU force scalar acceptance on n2/3021 passes original
  Libxc 7.0.0, finite-difference, restricted-embedding and authenticated 113-bit
  Maple boundary gates. The validation script explicitly substitutes the
  **exact production force scalar body** in an ordinary ABI probe; production
  SCF generation is never patched. Source and body hashes bind the receipt.
- Memcheck on n2/3020 returns zero errors. Its M06 scalar probe body is
  byte-identical to the final scoped force body; its earlier source attribution
  is retained, not relabeled as a new run.
- Both arms use the same authenticated existing #2486 glibc loader/runtime
  bundle because n1-built libraries need glibc 2.38 and n5 has 2.35. This is
  test deployment, not a production symbol patch or CPU fallback.
- Retain the failed initial GPU compiler receipt: n2's default GCC lacks its
  C++ frontend; explicitly selecting installed G++ repairs only compilation.
  Two CPU preparation PATH failures are also retained. No failed invocation
  supplies a successful endpoint sample.

## Rejected alternatives are not erased

The separate historical capsule
`benchmarks/results/mgga-integer-powers-20261014/` retains 108 calls on the
earlier common CSE source, including global lowering's DF96 warm regression
(3 → 4 builds) and force-only direct96's moved regression (12 → 14 builds).
Those observations are not master timings. Moving the new arithmetic out of
SCF repairs the global proposal's direct operation-ordering feedback; it does
not relax the convergence threshold or import a different density.

Two complete same-image repetitions of each historical direct96 arm add 48
independently accepted calls, all with 12 moved builds. The baseline's first
cold Hartree component itself varies by one ulp, before a force evaluation.
That shows existing numerical variability but does not isolate all causes of
the original extra builds; the regression remains visible.

The inherited stricter diffuse-tail diagnostic still has four common M06
partial failures at 65,536 points against the quad oracle, and fourteen
candidate entries exceed an extra 5e-12 baseline-pair diagnostic. These remain
failures. The established independent scientific gates are unchanged; no
diagnostic row is deleted or tolerance loosened.

Do not retry #2486's AO compaction, dddd acquisition, smaller order-seven
pages or matrix tiles on the strength of this result. No r2SCAN default or
FP32 graph promotion is made. No occupancy/stall claim is inferred from
register counts; #2486's NCU hardware counters were unavailable.

## Retention and reproduction

`manifest.json` binds `summary.json.xz` by length and SHA256. The capsule keeps
all populations, full-precision E/F rows, independent references, phase
samples, observed work, source/consumer/native identities, cache/build/compiler
and scheduler receipts, point arrays, operational failures and reproduction
scripts. Historical negatives have their own unchanged manifest and capsule.

Reproduce at the pinned control/candidate source revisions, using the archived
`build_master.sh`, `repeat_endpoint.sh`, `python.sh` and
`validate_force_schedule.py`. Keep compiler caches, compile the actual sources,
and run every GPU command through a finite compatible Slurm allocation without
overriding its device visibility. Preserve every planned phase and all samples.
`retain_master_force.py` retains incomplete/failing populations and marks them
unqualified instead of selecting only successful calls. No master README
historical row is rewritten from a differently sourced population.

Rationale: `.agents/notes/implemented/performance/2026-10-15-m06-rks-force-powers.md`.
