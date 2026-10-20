# Decision: reuse positive Boys series in the existing low-order domain

Status: implemented; final qualification recorded below
Date: 2026-10-20

## Problem and decision

This follows [the reciprocal/high-order repair](2026-10-20-md-j-boys-reuse.md),
parent `f3b8af61de50f18d00c39f3c5e709bfbd708e00e` (#2641). A fresh complete
fixed-density J control reproduces 0.792419 s for 96-atom r2SCAN-3c and
0.250621 s for PBE0, versus pinned GPU4PySCF MD 0.627157/0.347445 s.

The compiler now reuses `range_detail::long_moments` also for orders 1–4,
strictly inside the already-series domains T<0.25/0.75/1.25/2 respectively.
Orders 5–8 retain T<6. Order zero, the erf/upward domains, FP64 storage and
accumulation, screening, tiling, resource policy and bounded failure fallback
remain unchanged. This removes the low-order alternating expansion's second
per-iteration division and shares the existing positive incomplete-gamma owner.
It introduces no second recurrence implementation or production oracle.

GPU4PySCF's positive series stops differently and switches at m/2+0.5. Copying
that threshold is a separate policy choice, not a prerequisite for reusing its
arithmetic strategy. This campaign measures both choices independently.

## Source work and domain census

Separate instrumented native and pinned-upstream builds count actual radial
sources on the identical MINAO density. Upstream counters are warp aggregated
inside `boys_fn`, distinguishing all evaluations from nonzero-prefactor ones;
no instrumented time enters the clean timing population. Both instrumented
matrices agree with their uninstrumented originals within 4.7e-14. The original
pinned library and source archive remain unchanged and hash-bound.

| Case | Native radial sources | Upstream evaluations | Upstream nonzero prefactor |
| --- | ---: | ---: | ---: |
| r2SCAN-3c | 1,645,838,484 | 986,115,008 | 985,883,204 |
| PBE0 | 656,368,438 | 379,428,544 | 379,279,304 |

Native therefore performs about 67% more r2 radial evaluations. This makes the
omission-contract difference measurable, but does not prove the entire timing
gap unavoidable or that all sources have equal cost. Angular distributions,
reciprocal directions, task grouping and per-source execution also differ.
Native retains its cumulative absolute omission certificate; upstream's
float-log cutoff is not substituted for it.

Only 14,214,020 native r2 sources (0.864%) enter any existing small-T branch;
1,593,318,616 (96.81%) have T>=6. Upstream's corresponding nonzero-prefactor
counts are 9,586,799 and 949,635,831. A small fraction can still cost noticeable
time in mixed-argument warps; this census is not a hardware stall attribution.
Nsight counters were unavailable (`ERR_NVGPUCTRPERM`); failed receipts are
retained, and causal claims rely on controlled endpoint ablations, not invented
counter data.

## Rejected alternatives

All arms use finite n4 RTX 5090 Slurm allocations, five clean warm complete J
samples, and independent matrices. Preparation is separate. Upload, evaluation,
AO mapping and host publication are timed. The pinned native MD target uses
released GPU4PySCF 1.8.1 frontend/helpers, not a fully pinned package.

| Arm | r2 J (s) | PBE0 J (s) | Disposition |
| --- | ---: | ---: | --- |
| Parent | 0.792419 | 0.250621 | control |
| Positive series, retained thresholds | 0.766311 | 0.243388 | retain |
| Positive series, upstream low-order thresholds | 0.793765 | 0.255081 | reject |
| Positive series plus wider tiles/hierarchy | 0.765703 | 0.243721 | reject added complexity |

The larger-tile experiment added width-64/128 hierarchy nodes, independently
reduced from completed width-32 nodes in one stream-ordered kernel. Actual
low-order shared-memory capacity selected wider tiles. It passed 12 compiled
contract tests and 16 threaded runtime/sanitizer host tests, including neutral
padding, 129 leaves, changed density, windows and independent omission sums.
It preserved every radial source, direction and summand. Yet r2 tasks fell
5,528,314 -> 4,295,906 without meaningful incremental J gain; PBE0 slightly
regressed relative to positive series alone. Owner bytes also grew by 334,176
and 196,800 respectively. Do not promote it from its lower task count alone.
The exact patch, results and source identities remain negative evidence.

The upstream-threshold experiment also preserved work counts and passed 1,142
host math checks, but expanded series work and lost the positive-only gain.
Do not confuse it with the still broader, previously rejected whole-owner
threshold m+1.5. These are separately measured negative results.

## Qualification before the parent merge

Measured scientific source identity:
`d1e23fad3ebced90ef37b0da596ab3105465a30e8d3f2386ee558f07cddf2bfe`.
Native library SHA-256:
`3adc416d5e00884becaff235a67817672d0b8d48b534864200513280b9f201ec`.

Final n4:1255 complete J medians are **0.766064 s / 0.243375 s**, improvements
of **3.33% / 2.89%** from the fresh controls. Paired pinned MD takes
0.627361/0.347913 s: r2 remains **22.11% slower**, while PBE0 is 30.05% faster.
All 15 native classes report zero local bytes. The low-order class gains range
from 1.8% to 6.0% in the separate r2 trace; order zero is unchanged.

Final source/direction/summand inventories remain exactly unchanged:

| Case | Radial sources | Directions | Summands | Owner bytes |
| --- | ---: | ---: | ---: | ---: |
| r2SCAN-3c | 1,645,838,484 | 3,058,022,000 | 89,200,193,168 | 693,472,836 |
| PBE0 | 656,368,438 | 1,217,100,552 | 18,077,298,744 | 347,775,940 |

Final matrix errors are <=4.27e-14 against the parent and <=1.614e-10 against
pinned MD, passing atol1e-9/rtol1e-10. Residual work is zero. r2 tile/screened
counts remain 5,528,314/2,329,750; PBE0 remains 1,798,645/818,053.

The baseline complete endpoints below are the immediately preceding #2641
final on n4:1246, verified to have the same source and binary as this campaign's
fresh fixed-J control. They are retained original measurements, not newly
measured endpoint samples. Final complete endpoints are n4:1256:

| Complete E/F endpoint | Parent (s) | Final (s) | Focks, both |
| --- | ---: | ---: | ---: |
| r2 cold, including preparation | 120.359919 | 119.733315 | 24 |
| r2 warm, median5 | 24.407304 | 24.412880 | 1 |
| r2 moved | 77.233266 | 77.076913 | 13 |
| r2 moved-warm, median5 | 24.408625 | 24.420752 | 1 |
| PBE0 cold, including preparation | 59.635608 | 59.722648 | 17 |
| PBE0 warm, median5 | 10.239918 | 10.213831 | 1 |

All 18 final records pass E<=1e-8 Eh and force<=1e-7 Eh/bohr; observed maxima
are 9.55e-12 Eh and 3.92e-11 Eh/bohr. Cold/moved are single observations;
every warm repeat and its range is retained. r2 warm endpoints are essentially
flat/slightly higher, and PBE0 cold is slightly higher. No complete-endpoint
speedup is claimed for those phases.

Actual native r2 SCF has 24 Focks in both profiles. Median J kernel sum falls
**0.802065 -> 0.775632 s (3.30%)**; sum J falls
**19.353563 -> 18.716536 s**. Median XC is 2.191520/2.191804 s; the median
Fock interval falls 3.020205 -> 2.993948 s. Fresh released GPU4PySCF 1.8.1 MD
on the same n4:1258 GPU has **60 counted cycles, 61 J calls**,
median complete synchronized device-returning J **0.719065 s** and sum
**43.415859 s**, with E/F gates passed. Its stock solver, full-density
builds and 1e-14 direct tolerance are retained. Native J is a narrower kernel
sum and the SCF trajectories differ; it still exceeds the reference complete
call, so actual per-Fock J parity is not established. Instrumented profiles
are never mixed into clean endpoint samples.

Final validation: **1,367 host checks; 120 GPU checks** on n5:1804, with one
unavailable fully pinned-package case deselected and the separate pinned-MD
comparison retained; memcheck/initcheck/racecheck/synccheck report zero
errors/hazards/warnings. The CUDA oracle checks **43,560 radial values and
7,920 Boys values**, with maximum radial absolute-term-scaled error
1.878e-15. Shared default/reciprocal seeds retain their previously recorded
1.5682e-12 maximum relative error under the unchanged 3e-12 gate. Compiler
structure, native ownership and repository checks pass. Verified ccache 4.5.1,
actual compiler-launcher commands and shared-cache before/after statistics are
retained; no exclusive cache-hit attribution is made.

The independent 80-digit Cartesian-derivative gate remains
`abs(error) <= 3e-12 * sum(abs(exact terms)) + 1e-290`. New host cases exercise
low-order branch boundaries over diffuse/tight exponents; the CUDA probe also
covers adjacent representable arguments on both sides. Neither numerical nor
endpoint gates are relaxed. The retained full auxiliary remains independently
qualified as well.

Exact source/binary identities, clean samples, per-class resources, work/domain
counts, upstream instrumentation, experiment patches, endpoint/Fock records,
cache provenance, tests and reproduction scripts are retained in
`benchmarks/results/md-j-low-order-series-20261020/qualification.json.xz`.
Local raw data are under `.artifacts/j-low-order-20261020/`.

Only whitespace outside strings was compacted in two older MINAO JSON samples
to retain new evidence under the existing 64-MiB cap. Tokens, numeric spelling,
strings, key order and parsed values are unchanged. Old/new hashes and a search
of all tracked bytes, including decompressed xz, are recorded. No measurement
was removed, and no Release or external backup was published.

## Qualification on the merged parent

During qualification #2641 merged. This change now follows master
`e5fa4d2f0cd13775327261a0b6ced1eab6006c62`. Incoming changes concern compiler layout/materialization
contracts, DF-HF response emission and their tests; the MD source owners and
emitted MD CUDA header are byte-identical to the measured candidate above.
The equality receipt is retained. Old measurements keep their original source
and binary identities rather than being relabeled as this build.

The submitted build's scientific source identity is
`2e030e045f1faaa4882f21711275b466622d502b2ee99983a85d20c8f7917ffb`; library SHA-256 is
`645a704c63e53dd0e39d0124535d8a8e115d2abbe73b69a4f74e3254054a56f6`.
Fresh n4:1259 complete fixed-J medians are **0.766916 s** (r2) and
**0.243402 s** (PBE0), versus paired GPU4 MD **0.627817/0.347710 s**.
Improvements over the original fresh controls are **3.22%/2.88%**; the r2 gap
is **22.16%**. All work/byte inventories remain unchanged; matrix gates pass,
with zero local bytes in all 15 classes.

The complete endpoint protocol, including all five warm repeats and moved
geometry, was rerun on n4:1261:

| Complete E/F endpoint | Median seconds | Fock builds |
| --- | ---: | ---: |
| r2scan3c cold | 119.883867 | 24 |
| r2scan3c warm | 24.391457 | 1 |
| r2scan3c moved | 80.235406 | 14 |
| r2scan3c moved-warm | 24.386074 | 1 |
| pbe0 cold | 59.843555 | 17 |
| pbe0 warm | 10.209123 | 1 |

The post-merge moved solve uses 14 Focks versus 13 in the earlier samples.
Its longer complete time must not be attributed to J throughput alone; the
actual count is retained. PBE0 cold also remains slightly slower than the parent.

All 18 additional records pass the original energy/force gates; maxima are
1.0004e-11 Eh and 4.3258e-11 Eh/bohr. These are separate source-bound post-merge
samples, not pooled with the earlier population. An initial attempt to request
only one warm repeat was rejected before execution because its repetition
metadata did not match the five-repeat oracle protocol. The oracle and strict
protocol check were preserved; the complete five-repeat protocol above passed.

Post-merge validation passes **1,410 host tests** (including incoming compiler
contracts), **120 real-GPU tests** on n5:1805, all four sanitizers, and the same
independent CUDA radial/seed probe. The one unavailable fully pinned-package
case remains explicitly deselected. Fresh build identity checks pass on both
nodes. The receipt retains both complete qualification phases and compiler-cache
provenance.

## Revisit

The remaining gap needs both work reduction under a complete cumulative error
certificate and measured execution improvements. A source-specific distance
bound cannot simply be added to the current tail cutoff: that cutoff already
spends the whole target allowance, so additional omissions need a joint proof
or an explicitly accounted remaining budget. Do not weaken screening or infer
per-Fock J from complete energy/force time divided by iterations. Revisit wider
tiles only with new evidence for their actual limiting cost.

The subsequent [same-shell primitive exchange decision](2026-10-20-md-j-same-shell-fold.md)
removes duplicate primitive products while retaining this radial arithmetic and
the cumulative omission certificate. It also records further screening-budget
and ordering experiments, including negative results.
