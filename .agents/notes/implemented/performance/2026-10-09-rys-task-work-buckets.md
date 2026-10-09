# Decision: compose Rys-task K with warp-private primitive-work bins

Status: implemented
Date: 2026-10-09

## Problem

Combining #2133's nine-class task preference with landed #2135's Work schedule
does not automatically compose their performance mechanisms. The initial typed
union continued to use Fill inside each task warp. On frozen master
`6acb3e70e3031b4a7ef44ef77df551b354e34e59`, all 24 retained Cold observations
passed the independent E/F gates, but the 96-atom mean/median gate failed:

| Policy | Mean seconds | Median seconds |
| --- | ---: | ---: |
| Task + Fill | 102.891696 | 100.433881 |
| Incumbent + Work | 103.984877 | 101.457031 |
| Initial union | 101.207099 | 102.027941 |

The 48-atom initial union improved both measures, and fixed-density K also
improved, but neither justified presenting the failed 96-atom union as a winning
promotion. Retain every valid slower observation and its actual Fock count.

## Decision

Reuse the existing eight-bin, post-screening admission/flush algorithm in a
separately compiled task Work companion. Each of four 32-lane warps owns its
bins, selected bucket and bra cursor, with full-warp collectives and independent
retirement. Each bin holds at most two batches: scan only when all bins hold
fewer than 32 tasks, admitting at most another 32. Static shared queue storage
is 24,736 bytes per CTA; no new device allocation or global queue is introduced.

The prepared typed selector chooses the companion only for Rys-task + Work.
Fill, Primitive and Incumbent retain the original smaller-scratch task worker.
Explicit component-Rys/block, unsupported classes and J/HF retain their existing
contracts. Native declarations, per-profile registry function pointers and the
no-AOT stub expose a distinct typed launch, rather than repurposing precision
flags or encoding scheduling in counts. Capability remains twelve task classes;
automatic preference remains the same nine sm_120 classes.

## Invariants and rejected alternatives

- Quartet mathematics, precision, screening, normalization, primitive-pair reuse,
  symmetry, contraction/scatter and force ownership remain unchanged. Every
  control-flow collective is warp-local; quartet arithmetic has no collectives.
- Do not enlarge Fill scratch merely to add an optional Work schedule. Separate
  kernels preserve its rollback footprint and independently pinned bytes.
- Do not restore CTA barriers between independently retiring task warps. The
  concurrent host census includes fewer bra rows than warps, sparse/dense/empty
  tails, canonical pairs, work-bin homogeneity and exactly-once admission.
- Do not widen target/class preference or relax SCF or independent numerical
  tolerances to obtain a performance result. Primitive-sort diagnostics did not
  improve the original union and remain a retained negative.
- Lowering `incumbent` and schedule `fill` are separate rollback axes; set both
  for the policy preceding the two promotions.

## Scientific and source evidence

Frozen production source identity:
`cb788b62c65e98f375f29a3fbbf35cf58c61a0c329461d4667a51c7794d6eb7e`.
Library SHA256:
`1426e40973b32359dca90d0cfa106eb62d1052eddb07af50ce12c4b86f079376`.
CUDA 12.9.1, GCC 12, verified ccache 4.5.1; RTX 5090 on node4 through finite,
exclusive Slurm job 666. This is a host-compatible rebuild, not reuse of node1
binaries requiring a newer glibc.

- 320 public/prepared K matrices pass independent PySCF/Libcint comparison,
  including both spins, asymmetric/symmetric densities, four scales and frozen
  environment mutation; maximum matrix error is 6.072e-15.
- 864 matrices execute the actual production task bundle against Libcint.
  Standalone 96-case and integrated 16-case memcheck/racecheck/synccheck runs
  report zero errors, hazards and warnings.
- The native through-f value/derivative and Cartesian order-two CPU
  finite-difference suite passes. Its pre-existing generated-J routing census
  requires command-scoped `GENERATIVEQC_DISABLE_MD_J=1`; default MD-J serves J
  otherwise. All four initial-union policies reproduce that census failure.
  Clean performance retains the production MD-J default.
- Host suites report 635 passed/1314 resource skips and 178 focused passes
  (overlapping sets); a separate current-source CPU Libcint subset passes 31.
  Compiler, shared-SCF, electronic ownership, inventory and format guards pass.
- Independent emission retains all 21 incumbent/work streams, twelve original
  task shards with validation-only Work omission, component-Rys/block leaves,
  and all six retained scientific-source hashes. Only four sm_120 shards and
  registry source deliberately change; portable bundles and the generated
  registry header retain their hashes.

## Endpoint qualification

The predeclared five 96-atom and three 48-atom observations per policy pass both
mean and median gates in exclusive-node Slurm job 668, using the same frozen
binary as job 666. Fresh processes use strict FP64 PBE0 RKS, spherical def2-SVP,
the existing 48x16x32 grid, screening 1e-12 and E/D tolerances 1e-12/1e-10.
Time includes construction/context/preparation, SCF, analytic forces, host return
and owner teardown. Persistent compilation/artifact caches are equally reused;
this is execution-Cold, not cache-empty compilation timing.

| Atoms | Task + Fill median | Incumbent + Work median | Combined median | Reduction vs Fill / Work |
| --- | ---: | ---: | ---: | ---: |
| 96 | 101.927677 s | 110.618048 s | 97.393406 s | 4.45% / 11.96% |
| 48 | 47.464295 s | 45.116708 s | 42.179566 s | 11.13% / 6.51% |

Mean reductions against those baselines are 4.30%/7.65% at 96 atoms and
10.23%/7.73% at 48. Actual Fock counts are:

- 96: Task + Fill `[17,19,17,18,17]`, Incumbent + Work `[19,18,19,17,19]`,
  Combined `[19,18,17,17,17]`.
- 48: Task + Fill `[19,20,20]`, Incumbent + Work `[19,19,20]`, Combined `[19,19,19]`.

All 24 complete observations remain included, including slower valid trajectories.
Maximum independent GPU4PySCF errors are 9.095e-12 Ha and 3.785e-11 Ha/Bohr,
against unchanged 1e-8/1e-7 acceptance gates. Endpoint ratios include changed
SCF work and are not same-iteration kernel ratios. Five/three observations do
not establish statistical significance or universal-workload profitability.
Raw receipts, rejection evidence and reproduction are retained in
`benchmarks/results/unified-k-work-20261009/README.md`.

The clean complete prepared-K consumer includes upload, transformation, K,
projection, export and synchronization. Its six fresh-process admission arrays
agree at all four density scales; full-density admissions are 23,480,495 at
384 AOs and 81,907,624 at 768 AOs. These are generated shell-quartet admissions,
not primitive/root counts or proof of equal native-fallback work. No hardware
active-lane/barrier contribution or universal-workload profitability is claimed.
Full-density wall medians at 384/768 AOs are Combined 0.584464/1.079546 s,
Task + Fill 0.770219/1.328519 s and Incumbent + Work 0.729234/1.367599 s.
Relative to the faster standalone consumer these decrease by 19.85%/18.74%.
They are separate diagnostics and are not added to Cold endpoint time.

The old moved-warm harness supplied `coordinates=None`, restoring original
geometry while comparing to a moved oracle. Retain its failure and source;
the corrected harness resupplies moved coordinates on every moved-warm call.
The new preflight passes all four original/warm/moved/moved-warm E/F gates with
19/1/13/1 Fock builds. This is holdout validation, not a Cold sample.
The final three-policy holdout also passes all 24 E/F rows. Combined warm and
moved-warm medians are 6.107105/6.156997 s versus the faster standalone
6.253223/6.318093 s; moved-first time is 29.323675 s versus 31.269111 s,
with identical 1/1/13 Fock counts for those phases. These scoped timings do not
change the Cold population or establish equal cross-backend stopping rules.

## Revisit when

Additional targets, primitive distributions or full endpoints justify different
queue granularity or bin classifiers. Preserve finite queue capacity, independent
warp retirement, unchanged scientific leaves and explicit Fill fallback.
