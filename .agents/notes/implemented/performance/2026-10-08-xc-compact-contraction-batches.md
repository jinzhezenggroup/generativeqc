# Decision: retain active-AO domains through bounded contraction batches

Status: implemented (explicit qualification candidate; not promoted)
Date: 2026-10-08

## Problem

The #2089 point-batch default reduces point submissions but retains one density,
feature, weighted-panel and potential submission per original tile. The indexed
incumbent already keeps AO/density panels compact: expanding its maps or merging
point tiles would regress the scientific work domain rather than solve the
remaining contraction/submission problem. This change starts from master
`4385f72751b829883407c01106186917c344317b`.

## Decision

Add compiler-owned batch wrappers around the incumbent FP64 feature and
shared-memory matrix bodies. Immutable descriptors bind original point spans,
CSR labels and compact AO/work/local-potential offsets. Separate CTAs perform
independent density and Vxc contractions. Only discarded density output lanes
and non-authoritative/padded potential lanes skip arithmetic; valid reductions
keep the incumbent operation order and coefficient DAGs.

One CTA per spin scatters the small local matrices in original tile order.
Consecutive maps are compared by their physical labels, not merely dimensions
or pointer identity. Equal maps keep a running global entry in a register and
perform exactly the original ordered additions before one final write.
Different maps synchronize before touching overlapping entries. Point totals
also retain their original per-tile/channel reduction and addition order.

The shared group endpoint must be snapshotted into each lane's private state
before the final block barrier. Reading it in the loop increment after that
barrier races a faster lane beginning the next map comparison. Host block
simulation exercises multiple unequal/equal map groups to protect this invariant.

The original additional byte cap covers AO/features/totals, all retained work
panels, compact local matrices and the complete immutable descriptor inventory.
Fixed slot sizes charge the sum of separate maxima even when those maxima occur
in different batches. Descriptor staging stays alive through both the success
fence and the failure drain. No setup packing, allocation, descriptor upload or
new explicit fence occurs during evaluation/capture.

Use `GENERATIVEQC_CUDA_XC_COMPACT_BATCH=1` for qualification. A compact group
contains at least two original tiles, each nonempty tile having 32–128 active
AOs and at least 32 points; this bounds the serial scatter
and admits the existing 8/16/32 shared-memory candidates, not a measured
profitability domain. Empty tiles are preserved. Unsupported/resource-rejected
compact groups keep point-only batching without disabling independent eligible
groups. Only eligible groups reserve work/local matrices; physical submission
counts include both arms. Response, mixed arithmetic and optional
providers retain their incumbent paths. Allocation rejection retains one tile.
Public resource-ledger execution continues to reserve no optional batch arena.

## Work and evidence boundaries

- Semantic point/AO visits and density summands remain unchanged: maps, weights,
  quadrature, precision, functional and SCF convergence are not weakened.
- For matrix tile width `T` and active count `n`, the incumbent executes Vxc
  arithmetic in `b*(b+1)/2*T*T` lanes, where `b=ceil(n/T)`. The candidate executes
  it only in `n*(n+1)/2` authoritative lanes. This is a source work count, not a
  hardware timing claim; shared-memory loads and point padding remain.
- Density arithmetic also skips output row/point tails, not valid inner sums.
- Density/features each require one submission per batch; panel/local-Vxc/
  scatter each require one. AO submissions and canonical point algebra remain.
- Global matrix traffic decreases only across consecutive equal physical maps;
  unequal maps keep the original per-entry/tile writes. The compact local matrix
  write/read is additional traffic and can outweigh launch/writeback savings.
- The portable #2089 point-batch default is unchanged. No profitability, GPU
  sanitizer result or complete endpoint speedup is inferred from host tests.

Host tests execute the emitted resource planner, the real shared-memory
density/Vxc bodies, feature reductions and ordered scatter using CPU CUDA-block
barriers/shuffles. Independent scalar density, rho/gradient/tau and unfactored
AO bilinears check empty, noncontiguous, overlapping and repeated maps, both spin
channels, jet strides and partial points. The compiled-resource envelope requires
every new stage and the allocation/shape/provider fallback; missing evidence
fails closed.

Native `generativeqc_dft_cuda_tests --point-batches` adds independent CPU E/V and
same-tile equality gates for Cartesian/spherical bases, all curated families,
both spins, density replacement, canaries and captured replay. The masked CPU
reference selects the fixture's actual functional. Density-gradient export is
GGA-only; LDA uses ordinary enqueue, not a weakened production export guard.
The locally visible Slurm inventory contains only drained `node3`; allowed `n1`
has a separate remote Slurm controller. Real-device commands must run through
that controller's finite `srun`, preserving its assigned visibility. Do not
bypass visibility or schedule maintenance-node work to obtain evidence.

## Real-device evidence (2026-10-08)

Remote `n1`/`node1` Slurm job **6635**, `main`, `gpu:5090:1`, one task,
16 CPU workers, finite `00:15:00`, preserved scheduler visibility `0`:

- Release `sm_120`, CUDA 12.9.1; verified ccache 4.5.1 launchers for C++/CUDA.
  Generated compiler commands and before/after cache statistics are retained.
- Native `--point-batches` passed independent CPU E/V, bitwise same-tile
  schedules, Cartesian/spherical maps, all five families, both spins, tails,
  mixed eligible/fallback groups, density replacement, canaries and replay.
- Compute Sanitizer memcheck: zero errors and zero leaked allocations.
  Racecheck: zero hazards, errors or warnings.
- Final combined host qualification: 378 tests passed, covering the focused
  candidate and adjacent compiler, schedule, binding, publication, feature-export,
  lifetime, density-repreparation and descriptor-access contracts.

The fixed-density `--compact-batch-benchmark` compares **point batching against
point batching plus compact contractions**, not against the old one-tile path.
Both arms request 32 tiles / 32 MiB, retain 256-point maps and use the same
selected AO labels, FP64 point algebra and diagnostic density. Twelve atoms,
96 AOs, 294,912 points and 1,152 original tiles; six interleaved measurements
per arm/geometry, first sample excluded from the five-sample warm median:

| Fixed-density XC endpoint | Point-only | Compact candidate |
| --- | ---: | ---: |
| Original warm median (ms) | 117.812773 | 77.514228 |
| Moved warm median (ms) | 117.825519 | 77.528758 |
| Density submissions/evaluation | 1,009 | 265 |
| Feature submissions/evaluation | 1,152 | 408 |
| Vxc contraction submissions/evaluation | 1,009 | 265 |
| Ordered scatter submissions/evaluation | 0 | 24 |
| Point submissions/evaluation | 36 | 36 |
| Additional batch bytes | 24,739,840 | 32,959,520 |

Exactly 24 groups / 768 original tiles use compact contractions; the remaining
groups retain the incumbent. Both arms visit 18,395,136 selected point/AO pairs
and 1,388,045,824 selected point/AO-square summands on the original geometry;
the moved counts are 18,394,880 and 1,387,994,368. Every E/V comparison is
bitwise equal, not merely within the benchmark's historical absolute gate.
The timing includes density upload, evaluation and E/V publication, excluding
geometry/map discovery and optional preparation, which are reported separately.
This is **not complete SCF E+F evidence**, and does not authorize promotion.

Raw evidence is retained in ignored local
`build/xc-compact-qualification/gpu-job-6635/` and remote
`/data/jzzeng/qc-compact-xc-20261008-e162/results/job-6635/`, including individual
source/input hashes, build commands, cache statistics and sanitizer/endpoint logs.
Authentication anchors:

- Test binary SHA256: `446c0ea772d70a654a0393828119737a83430ae0a0269e2db202557544ea52f0`.
- `sources.sha256`: `a7261dd6588239a98c2eeba32937944e5ea230b6c9bb5d8b28751d86b7bc76e2`.
- `inputs.sha256`: `feb0736e726c0d9b1ca2df9bf2d394b8d422345c34746fd0acb6e400377f5894`.
- `compact-water-12.log`: `126bf56f134b708146eb37e916d3ff0499614cc99ff4b906b863580b6ceb7502`.

## Rejected alternatives

- Union maps or restore dense AO/density tensors: changes quadratic work and can
  destroy the locality being optimized.
- Concurrent atomic global scatter: changes addition order and reproducibility.
- Global dense-pair scans to serialize scatter: memory-bounded but not
  selected-work-bounded.
- Default enablement based on submission counts: extra resident work/local
  matrices and serial scatter need complete endpoint qualification.
- Replace optional provider bindings: would silently bypass qualified provider
  materialization and epilogues. Retain explicit bounded fallback instead.

## Revisit when

Extend the retained native E/V/graph and sanitizer gates to the intended workload
domain, and run interleaved complete cold/warm/moved SCF E+F endpoints through
finite Slurm allocations on allowed nodes.
Retain actual compact admission, source/binary identities, resource bytes,
semantic work counts, discarded-lane counts, map-group writeback counts and
convergence histories. Promote only a profitable qualified workload domain.
Larger active spaces need a separately bounded parallel ordered-scatter design.

## References

- The opt-in scheduling policy is superseded by
  `2026-10-08-xc-compact-batch-default.md`; the measured source/binary identities
  and fixed-density evidence in this note are not relabeled by that promotion.
- PR #2089; issues #2073 and #1598
- `.agents/notes/implemented/performance/2026-10-07-xc-point-batches.md`
- `docs/developer/xc_native_cuda.md`
- `tests/python/test_xc_compact_batches.py`
- `tests/native/dft_point_batch_cases.cuh`
- `benchmarks/pbe0_xc_tile_pairs.py --point-batch-tiles 32 --compact-xc-batches`
