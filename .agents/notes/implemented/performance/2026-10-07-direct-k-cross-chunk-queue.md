# Decision: opt-in bounded cross-chunk exact-K queues

Status: implemented, experimental; not promoted
Date: 2026-10-07
References: #1892, #2059, #2060, #2065

The default-selection decision is superseded by
[default exact-K filling](2026-10-07-direct-k-fill-default.md). The original
opt-in decision and qualification record below are retained as historical evidence.

## Problem

#2060 moves survivors into the first lanes/subgroups of each original candidate
chunk, then executes that chunk immediately. It does not fill execution batches
across chunks. Its weak/negative complete cold-energy result cannot reject true
cross-chunk filling or establish a PBE0 cold advantage.

## Decision

The prepared generated raw-K owner freezes
`GENERATIVEQC_DIRECT_K_TASK_SCHEDULE=incumbent|fill|primitive`. The default is
`incumbent`. Unknown values fail during preparation. Execution never rereads the
environment. Other providers and combined-HF owners retain their existing path.

Compiler-owned packed and subgroup streams keep one bra resident and scan the
original Schwarz-ordered ket segment. `fill` accumulates admitted pair identities
and contribution bounds across original chunks. A 2W shared arena holds fewer
than W pending tasks plus at most W new survivors; W consumers then execute a
full batch. The final partial batch is flushed, including the case in which the
last scan leaves both a full batch and a nonempty residual batch. Queues are
reset at every bra/system boundary and do not materialize the quartet domain.

`primitive` additionally stably orders the admitted lookahead window by the
cached ket primitive-pair count. The bra primitive count and angular/component
shape are fixed in that row, so ket count orders the primitive-product work.
This is bounded local grouping, not a global sort or a promise of homogeneous
tasks. Its serial insertion-sort cost is part of the experiment.

The prepared topology appends the schedule identity. Integral equations, task
identity, canonical pair legality, screening predicates/tolerances, mixed-state
decision, raw-K scatter, and independent J ownership remain unchanged. Existing
Rys and unsupported-class fallbacks remain available. No symmetric-density
assumption is introduced: nonsymmetric and unrestricted inputs remain legal.

## Synchronization invariant

Packed ballots/fences retain all 32 physical lanes. Subgroup publication and
consumption use full-CTA barriers. Every lane snapshots the queue count before a
subgroup may append. Residual entries are read before any overlapping destination
is written. A bra claim is likewise snapshotted before another empty row may
advance the shared cursor; the concurrent host test exposed why the second
claim barrier is necessary with empty/inactive rows.

## Evidence

- The actual emitted workers execute concurrently on C++20 host lane groups for
  packed `psss`, subgroup `dppp`, and mixed subgroup `dsds`. Independent nested
  loops check exactly-once admission, precision counts and the per-bra lower
  bound `ceil(survivors / W)` in both optional modes.
- Cases include zero work, sparse/dense admission, W-1/W/W+1 boundaries,
  multiple chunks, partial coarse tails, canonical duplicates and inactive
  neighboring systems. A separate C++ test checks default/explicit/invalid
  selectors and frozen selection after environment mutation.
- The production artifact baseline intentionally changes only the four affected
  sm_120 shards; portable shards, registry, manifest and shell catalog do not
  change. Compiler ownership checks pass.
- 123 focused host/codegen/ABI/profile tests pass; compiler structure checks 486
  modules with zero dependency errors. Both optional modes pass the complete
  native CUDA Fock-provider suite, including independent values through f and
  Cartesian order-two derivative/CPU finite-difference gates.
- Restricted/unrestricted, symmetric/nonsymmetric, full/scaled/zero/restored
  public raw-K densities pass independent PySCF/Libcint gates. The largest small
  matrix error is 4.6863e-13. Fixed-density 48/96-atom tests retain the existing
  1e-8 gate; the largest error is 1.9094e-11. Fill synccheck and primitive
  memcheck/synccheck report zero errors.

### Frozen-density K wall evidence

RTX 5090, SM120, CUDA 12.9.1, Release, FP64, def2-SVP spherical, screening 1e-12.
Parent is `f05015e7c71809155b9be0aa69c3ff62d5d89e30`; the candidate was the
uncommitted queue source subsequently recorded as `7b6798407`. Timing includes
density upload, transformation, K build,
projection, export and synchronization, not owner preparation. Task counters and
device-window samples are collected separately from clean wall samples.

| Atoms | Density scale | Parent median ms | Fill reduction | Primitive reduction |
| --- | ---: | ---: | ---: | ---: |
| 48 | 1 | 1043.251 | 8.10% | 9.51% |
| 48 | 1e-3 | 785.812 | 14.76% | 15.59% |
| 48 | 1e-6 | 476.125 | 29.18% | 31.01% |
| 96 | 1 | 1938.155 | 13.65% | 14.36% |
| 96 | 1e-3 | 1366.041 | 23.29% | 25.24% |
| 96 | 1e-6 | 750.595 | 41.48% | 42.98% |

Every compared variant has identical per-class admitted tasks and retained
device bytes. Parent and fill have six clean wall observations per regime across
two blocks; primitive and candidate-incumbent controls have three. The latter
control is about 0.6–1.1% slower than the pooled parent on full-density inputs;
this small default-path difference is not qualified as either zero or a
statistically established regression. Synthetic scales do not represent actual
SCF delta-density trajectories, and these percentages are not cold endpoint wins.

### Complete cold qualification

Fresh processes/owners/densities use energy tolerance 1e-12 and density tolerance
1e-10, with unchanged physical Fock/residual/force gates. Preparation and complete
energy-plus-force execution are timed; compiler/runtime artifact caches are
reused, with no identity overrides. Native iterations/Fock builds are retained.

The primitive strategy passes all completed independent energy/force gates but
is not a uniformly qualified cold improvement. At 48 atoms its four samples use
23 builds, while parent builds are 23/23/26/23. One first parent force call is
also slower than subsequent calls, despite unchanged derivative code. At 96
atoms parent builds are 25/25 and primitive builds are 29/25: the slower candidate
trajectory outweighs its same-work K gain in the two-sample median. Do not drop
the extra-iteration sample or attribute the force-side first-call difference to
this value-only change. A separate fill-only cold comparison is necessary.

The completed fill-only ABBA comparison retains two fresh-process samples per
revision and size, after the initial force artifacts have been exercised:

| Atoms | Parent complete median s | Fill complete median s | Observed reduction | Parent / fill builds |
| --- | ---: | ---: | ---: | --- |
| 48 | 102.720080 | 100.349553 | 2.31% | 23,23 / 23,23 |
| 96 | 212.804516 | 202.348054 | 4.91% | 27,24 / 25,25 |

All independent energy/force gates pass. These are observed medians, not a
statistically established speedup: 96-atom trajectories differ, and two samples
per revision are insufficient to separate all iteration and timing variability.
SCF/publication medians are 75.689397 -> 73.787180 s at 48 atoms and
165.717989 -> 154.843804 s at 96 atoms. No derivative optimization is claimed.
The matched GPU4PySCF reference complete samples are 51.858825/87.218384 s at
48/96 atoms, so the native candidate is still slower than the reference.

Raw cold blocks are `parent4,fill4,fill5,parent5` at 48 atoms and
`parent2,fill2,fill3,parent3` at 96 atoms. Primitive results remain separate;
do not select only its favorable 96-atom 25-build sample. Keep both modes opt-in,
with fill the more conservative candidate for further cold qualification.

Full raw results, identities, frozen inputs and finite-time Slurm scripts are
retained under `.artifacts` and the matching isolated n1 qualification checkout.
No superiority over the matched GPU4PySCF complete endpoint is established.

## Tradeoffs and next decisions

The optional queue adds bounded shared state and generated control flow even
when the incumbent schedule is selected. Qualification must compare the candidate
incumbent mode with its frozen parent, not only fill against incumbent within one
binary. A resource/occupancy regression requires isolating optional kernel
variants before any default promotion.

Filling alone cannot reduce integral work or scatter traffic. Full-density cold
builds may already fill most chunks. K block contraction (#2065) and a separately
verified symmetric-density triangular writer remain independent follow-ups;
this change does not pretend to implement either. Keep #1892 open until its
full value/derivative and complete E+F acceptance gates are met. Promote only a
material reproducible complete cold gain, not a synthetic sparse-density win.

## PR integration on 2026-10-07

The PR integrates upstream `0ad23e791`, including #2065, #2066, #2069 and #2070.
Both the prepared block-lowering mask and the queue selector are retained.
Explicit Rys/block lowerings keep their own workers rather than claiming queue
composition. The legacy artifact fixture is regenerated for the combined
compiler: only four sm_120 shard hashes differ from this upstream baseline.
The portable shards, registry, manifest and scientific shell catalog retain
their upstream identities.

Integrated host/codegen/profile/ABI qualification reports 163 passed and one
skipped in 177.76 seconds. Compiler structure checks 488 modules with zero
dependency errors; the CUDA ownership inventory checks 332 files. Ruff
check/format and PR-scoped diff checks pass. No handwritten CUDA source changes
are introduced relative to the integrated upstream baseline.

The emitted queue workers, recurrence consumer and topology ABI are unchanged
from the measured queue source. Nevertheless, the complete cold measurements
above predate upstream integration, particularly the incremental-SCF changes
in #2069. They are historical evidence against the frozen parent, not a measured
speedup over current master or qualification of the integrated GPU binary.
Repeat the integrated real-device numerical and complete-endpoint gates before
promotion; qualify queue/block composition as a separate follow-up.
