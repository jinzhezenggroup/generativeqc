# Decision: preserve coarse screening and select only the qualified residual class

Status: implemented opt-in; production defaults unchanged
Date: 2026-10-19

## PR packaging

The retained opt-in implementation is packaged on PR #2600 commit
`aacbccc275274cb70e36b76a646750fe667acf3e`, not the original investigation's
older dense-E checkout. Both measured native/compiler source inventories match
this packaging exactly. The historical checkout warnings below refer to that
original investigation. A machine-readable evidence summary is retained under
`benchmarks/results/2026-10-19-md-j-compacted-residual/`. The unmeasured later
primitive-strip prototype is excluded; no new default is promoted.

## Decision and supersession

This note supersedes production class selection and coarse admission in the
[compacted-source experiment](2026-10-19-compacted-masked-hermite-sources.md).
Its numerical evidence and rejected alternatives remain relevant. Mode 2 now
selects the compacted exact-mask consumer only for `(2,1)`. With the other
experimental residual selectors disabled, `(2,2)` retains the incumbent,
rather than hiding a repeatable class regression behind
a positive combined-J result. The compiler still emits the `(2,2)` primitive
for independent triangular/diagonal numerical and queue tests.

The resident owner freezes `GENERATIVEQC_MD_J_RESIDUAL_TILES=0/1`. Coarse mode
uses the canonical native 8-by-8 bounds helper and outward-rounded global
density allowance before individual exact admission. Two warps each decode
one tile, retain its 16 pair IDs in registers, reduce its bounds with shuffles,
and ballot its two 32-candidate halves into the shared queue. The other two
warps join consumption. At most 128 tasks are appended while the queue has
fewer than 128 entries, so its fixed 256 slots remain sufficient. Untiled
mode retains flat 128-candidate scans. No molecular-size queue is introduced.

Zero-density tiles must be coarsely rejected when that resident policy is
enabled. Preserving numerical J while failing that observable work contract is
not enough: the preceding combined-class schedule passed only 113/114 normal
integration cases. Restoring coarse admission fixes this omission rather than
weakening the test. Disabled coarse screening has its own oracle/coverage gate.

Angular-specialized scalar contractions use the existing radial/recurrence
algebra without a materialized R matrix. Reverse density channels are formed
once per bra primitive in the unused portion of the combined-group workspace;
group admission proves that these channels do not alias bra potentials.
Diagonal/orbit-sensitive and complex masks retain the native bounded fallback.

The [full-density shortcut](../../rejected/2026-10-19-residual-full-density-shortcut.md)
was separately tested, found slower and removed. It is not part of this path.

## First source-matched qualification

The immutable `warp-coarse-hermite-20261019-49fc` snapshot adapts compact-E
PR #2600 commit `aacbccc275274cb70e36b76a646750fe667acf3e`. It is **not** a
full-library qualification of this checkout's older dense-E native HEAD.
The library SHA256 is
`7901c1aae7aa7c4b9cb421a8e2074e81af5bc3eed398d36fc1b9ff30335e6efa`.

Slurm 10348 passed 76 ordinary real-CUDA oracle cases, 76 memcheck cases,
76 racecheck cases and 76 synccheck cases, with zero reported errors/hazards.
The tests cover UKS/nonsymmetric density, opposite directional masks, threshold
equality, zero replay, ragged primitive counts, triangular/diagonal tasks,
queue refills/leftovers and coarse/individual search-domain coverage.
Slurm 10350 passed all 114 normal molecular cases in each of modes 0 and 2.
The 88 focused host tests also passed.

Slurm 10349 measured prepared-owner host-canonical-density to synchronized
host-canonical-J endpoints, excluding setup and a freshly computed independent
CPU libcint J. Four interleaved/reversed-order samples were collected per method
on one assigned RTX 5090. This is not an SCF or force timing.

| Method | Cutoff setting | Median J (s) | Maximum absolute error vs CPU J |
| --- | ---: | ---: | ---: |
| Native incumbent | `1e-12` | 4.034355 | `2.456e-11` |
| Native compacted `(2,1)` | `1e-12` | 3.717473 | `2.456e-11` |
| GPU4PySCF 1.8.1 | `1e-12` | 0.645334 | `2.389e-10` |
| GPU4PySCF 1.8.1 default | `1e-13` | 0.689749 | `2.753e-11` |
| GPU4PySCF 1.8.1 | `1e-14` | 0.739173 | `2.402e-12` |
| GPU4PySCF 1.8.1 | `1e-16` | 0.859579 | `1.599e-13` |

Candidate J time is 7.85% lower than the matched native incumbent, but is still
about 5.39 times upstream's default. This result does not isolate a benefit from
warp-local coarse scheduling alone: class fallback, restored coarse semantics
and scalar projection changes all contribute to the compared implementation.
Do not call it the fastest historical candidate or transfer the old combined
schedule's cold-SCF result to this source.

All use the same 96 atoms, 1088 spherical AOs, explicit canonical r2SCAN-3c basis
and frozen physical density. The normalization/phase map is applied on both
sides of the density and J. Density SHA256 is
`a0b22059e12adcd39771e5dcfaac0951afb298d8252ef6674889ad0349ed6090`;
basis SHA256 is
`fb2c5d3a5f1ec149f0bf1d15d1e76cdb8baf042aeaa8be677cf25f33b1b7882b`.
Identical threshold strings do not imply identical scientific screening domains:
native retains directional FP64 AO products and its global error allowance;
upstream uses float32 log conditions and a different density hierarchy.

## Work and source comparison

Slurm 10351 captures all three engines, with upstream using its default `1e-13`.
Summed captured J kernels take 3.99055 s native, 3.65874 s candidate and 0.67854 s
upstream. Candidate `(2,1)` residual takes 0.36687 s versus incumbent 0.69427 s;
`(2,2)` remains 0.36199/0.36152 s. These are profiler diagnostics, not a clean
timing distribution. Upstream complete `(2,1)` and `(2,2)` classes take 0.09911 s
and 0.04943 s across two and three launches, respectively. A native residual-only
time is not the same work scope as an upstream complete angular class.

The counted native replays match in all 25 angular domains for uniform radial
sources, density directions, Hermite summands, residual tasks and primitive
products. Totals are 2038308216 uniform radial sources, 4013012720 uniform
directions, 95323936432 uniform summands, 43920982 residual tasks and 242399046
residual primitive products. Individual candidate checks plus coarsely screened
candidates equal 526842030 in both modes. Coarse grouping/order changes the
split, not the raw domain: incumbent checks/screened are 496800846/30041184;
candidate checks/screened are 498357884/28484146. Thus the gain cannot be
reported as less admitted integral work. The fast exact-mask contraction handles
9081696 of the 9929600 `(2,1)` residual tasks; the rest use the bounded fallback.

Static linked resources for the uninstrumented `(2,1)` candidate are 126
registers/thread, 1840 stack bytes and 3085 shared bytes. These are not measured
spill traffic, active-lane ratios or proof of an isolated hardware bottleneck.
Earlier Nsight Compute evidence concerns a different source snapshot and must
remain labeled accordingly.

The upstream source comparison is pinned to v1.8.1, commit
`5b284c258a4260baef80e3d150b4e7a81a9dbd57`, matching the installed wheel version:

- `gpu4pyscf/scf/j_engine.py`: decontracted/sorted preparation, one density
  transform, descending pair bounds and 2/4/8/16/32 density maxima.
- `gpu4pyscf/lib/gvhf-md/unrolled_md_j.cu`, `md_j_2_1` and `md_j_2_2`:
  16-by-16 primitive-product threads, fixed large logical tiles, explicit scalar
  recurrence immediately consumed by density, and potentials retained across
  many partner batches. The generic resource estimator is not runtime tuning of
  these fixed specializations.
- Native mode 2 still gives one whole shell task to each thread, with
  task-dependent mask plans/primitive loops and task-local AO publication.
  Queue compaction fixes empty task slots but does not establish upstream-like
  homogeneous primitive-lane work or long-lived Hermite potentials.

Native uniform potential kernels alone still cost about 1.481 s, more than
upstream's entire captured J. Residual-only tuning cannot close this gap.
The next architecture experiment should lengthen primitive-tile potential and
density lifetimes, sharing the canonical scientific algebra and retaining exact
directional masks. Copying upstream's screening conditions would change the
contract, not merely its schedule.

## Receipts and promotion boundary

Remote immutable root:
`/home/jzzeng/generativeqc-qualification/warp-coarse-hermite-20261019-49fc`.
Local ignored evidence is under `build/warp-coarse-hermite/`, including
`j96-10349/result.json`, `traces-10351/`, source snapshots, compiler commands,
cache statistics and linked resources.

A subsequent formatting-only change preserves the emitted native primitive
byte-for-byte (SHA256
`855acd611060f9bfe97a3c39b5cd9e2694bd13a1bcdae5acb67c1c9c351b232e`).
This one-off preservation check is not a permanent source-fingerprint test.
Source-closure-dependent stationary AOT still needs a genuine rebuild; identical
emitted C++ does not authorize forging artifact identity. The final formatted
snapshot is qualified separately under `warp-coarse-final-20261019-49fc`.

### Final formatted source

The final library SHA256 is
`92d3e08bd9c945c49eb46d498052786b040f627d9b4931d85c52a972d3d96420`.
Slurm 10352 repeats all 76 cases ordinarily and under each of memcheck,
racecheck and synccheck, with zero errors/hazards. Slurm 10353 repeats the
114 normal molecular cases in each mode, and all 88 focused host cases pass.
Compiler structure (568 modules), Ruff and diff whitespace checks pass.

| Independent allocation | Incumbent J (s) | Candidate J (s) | Upstream default J (s) | Native time reduction |
| --- | ---: | ---: | ---: | ---: |
| Slurm 10353, visibility 4 | 4.026204 | 3.717649 | 0.690771 | 7.66% |
| Slurm 10354, visibility 3 | 3.883460 | 3.577326 | 0.670088 | 7.88% |

Each median has four interleaved/reversed-order samples and a newly computed
CPU oracle. Both allocations retain identical library/source, installed
upstream, basis and density identities. Candidate/incumbent maximum J
differences are `3.233e-13` and `2.984e-13`. The independent native oracle
errors remain about `2.456e-11`, versus upstream default `2.753e-11`.

Slurm 10357 repeats the traces and verifies all 25 scientific work domains.
Captured native/candidate/upstream kernel totals are 3.89631/3.56082/0.66253 s.
The `(2,1)` residual changes from 0.67660 to 0.35799 s; `(2,2)` remains on the
incumbent at approximately 0.352 s. Candidate uniform potential work remains
1.44077 s, so the broader dataflow gap is not resolved.

Slurm 10355 measures a **single matched cold SCF+host-force pair**, including
preparation, with no frozen initial density. Fresh stationary AOT manifests and
binaries are genuinely rebuilt for the formatted compiler closure. The same
48-by-16-by-32 unpruned grid, three Becke iterations, explicit basis,
`1e-12` energy/Schwarz tolerances and `1e-10` density tolerance are retained.

| Mode | Complete endpoint (s) | Preparation (s) | Iterations / Fock builds | Energy error (Ha) | Force maximum error (Ha/Bohr) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Incumbent | 271.237524 | 8.694544 | 24 / 24 | `2.274e-11` | `4.936e-11` |
| Candidate | 263.289601 | 8.790231 | 24 / 24 | `2.456e-11` | `5.844e-11` |

The complete endpoint is 2.93% shorter in this pair; this is not a repeated
distribution or a GPU4PySCF full-SCF comparison. Both final physical residual
RMS values are approximately `1.489e-12`, passing the independent `1e-10` gate.

### Final-source hardware counters

Authorized sudo profiling in Slurm 10356 preserves and verifies the scheduler's
device visibility after the account change. No credentials are stored in
scripts or receipts. All selected launches use 22 NCU replay passes with
unlocked clocks; these are execution diagnostics rather than clean timings.

| Domain | Active threads / warp instruction | Registers/thread | Achieved occupancy | Executed warp instructions |
| --- | ---: | ---: | ---: | ---: |
| Incumbent residual `(2,1)` | 4.39 | 104 | 33.27% | `269246339426` |
| Candidate residual `(2,1)` | 3.78 | 126 | 33.23% | `129765799208` |
| Upstream complete `(2,1)`, two launches | 27.37--27.42 | 109 | 22.18--30.01% | `10055829533` total |
| Upstream complete `(2,2)`, three launches | 28.19--28.58 | 128 | 23.91--31.48% | `5342628840` total |

The candidate reduces instructions substantially but does **not** improve the
instruction-wide active-lane ratio. Its theoretical local-memory sectors are
`25365216584`, versus zero for the incumbent `(2,1)` and the two upstream
`(2,1)` launches. This metric is not measured DRAM bytes or proof that all
traffic is register spilling: dynamically indexed private arrays can also live
in local storage. FP64 pipe active percentages are 86.55% incumbent, 82.32%
candidate, and 87.27--88.74% upstream `(2,1)`. None alone identifies the sole
bottleneck or justifies changing precision. The source comparison still points
to homogeneous primitive-lane work and longer-lived density/potential channels,
not merely increasing register limits or tile dimensions.

Final immutable root on node1:
`/home/jzzeng/generativeqc-qualification/warp-coarse-final-20261019-49fc`.
Its `summary.json` checks the two J allocations, all 25 work domains and the
separate cold endpoint; local receipts are under `build/warp-coarse-final/`.
Raw Nsight reports, report hashes, source snapshots and compiler-cache receipts
remain at the remote root. The rejected full-density shortcut is still absent.

Neither this result nor the old compaction endpoint is sufficient to promote a
new default for the current native checkout. Retain the explicit incumbent,
optional-memory cap and error gates. Require final-source endpoint evidence,
larger-size/cliff coverage and complete packaging/integration before promotion.
