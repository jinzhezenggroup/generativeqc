# Evidence: primitive-lane exact masked-Hermite residual J

Status: implemented experiment; default-off; performance promotion rejected
Date: 2026-10-19

## Problem and decision

The fresh GPU4PySCF comparison found a large difference in lane mapping and
Hermite intermediate lifetime. Test the former directly instead of claiming
that lowering atomic counts, copying tile dimensions, or using more lanes
necessarily reproduces upstream performance.

`GENERATIVEQC_MD_J_MASKED_TILES=2` selects a compiler-owned `(2,1)` J-only
primitive-lane experiment. Each of the two-by-four shell-pair tile's eight
tasks owns a half warp. Sixteen lanes evaluate its four-by-four primitive
products independently using the existing native Coulomb recurrence. Constant
Hermite indices and unrolled density contractions permit register allocation
of the small R matrix. Half-warp shuffles reduce Hermite potentials before a
separate AO back-projection. The old shared-recurrence tile remains value 1;
unset/0 retains the incumbent.

This is not the complete GPU4PySCF dataflow: masks, suffix setup and output
publication still have short tile/task lifetimes. Most importantly, a fixed
sixteen-product allocation is not a compacted primitive task stream.

No upstream source was vendored. The new schedule shares the existing native
recurrence and generated exact-mask helpers rather than introducing another
handwritten scientific algebra implementation.

## Invariants and resources

- Native shell, directed-rounding density and residual coarse admission remain
  authoritative. No threshold, global density-error allowance or Hamiltonian
  term is relaxed.
- Sorted directed AO entries retain the original FP64 bound-product predicate,
  reverse masks, equality behavior, nonsymmetric alpha/beta density and orbit
  deduplication. Unfavorable or unsupported masks use the incumbent AO fallback.
- All lanes named by a half-warp shuffle participate, even on ragged primitive
  edges; absent products provide zero inputs.
- The compiled primitive-lane kernel uses 41512 shared bytes, 140 registers per
  thread, and zero local/stack bytes. Shared storage has a 48-KiB static gate.
  There is no additional resident allocation or change to the original
  128-MiB optional MD allowance. Both native owners report 662672188 bytes.
- Frozen preparation owns the schedule; replay does not poll the environment.
  Fused exchange never uses this J-only route. Invalid policy values reject
  preparation, and CUDA errors retain the existing first-failure boundary.

## Source and independent qualification

The timed source is the adapted PR #2600 snapshot at
`aacbccc275274cb70e36b76a646750fe667acf3e`, with the retained experiments and
compact-E accessor. It is not the older dense-E worktree variant. Its library
SHA256 is
`3f5a73508bf45544f2bf8e0c5fcfb10b2248599a4e4f8d3be4663bc84d181ee5`.
`build/primitive-lane/` retains the paired benchmark, source hashes, validation
logs, source-matched work and summaries. Remote sources, binaries and large
profiler reports are under n1's
`/home/jzzeng/generativeqc-qualification/primitive-lane-20261019-49fc/`.

- Host mathematical/resource/error-boundary gates: 88 passed. The injected
  launch-error test covers policies 0, 1 and 2, including fused-K exclusion.
- Compiler ownership audit: 567 modules, zero dependency errors.
- Slurm 10306: 72 real-CUDA oracle tests passed, 72 under memcheck with zero
  errors, and 72 under racecheck with zero errors/warnings. These cover both
  directions, UKS/nonsymmetric density, complex-mask fallback, neighboring
  systems, stale-density replay, both shell-tile axes and 1/3/5-primitive
  contractions crossing the four-primitive chunk boundary. Slurm 10300 also
  passed the preceding 64-case version and both sanitizers.
- An initial qualification attempt used dense synthetic E arrays against the
  compact snapshot without its required slot map. The test fixture now uploads
  identity slots when that native interface exists. This fixes the independent
  test input, not the production accessor or fallback, and keeps the same
  dense mathematical oracle for both snapshots. That failed run is not counted
  as numerical evidence for either algorithm.

All GPU execution uses finite Slurm allocations and assigned device visibility.
The compiler launcher is verified sccache 0.16.0; before/after statistics and
actual wrapped compiler commands are retained. Qualification is prepared J,
not complete SCF convergence, cold startup, force, or stationary-AOT closure
qualification.

## Fresh 96-atom complete-J measurement

Slurm 10301 uses RTX 5090 UUID
`GPU-4c5ba19e-745a-9f56-6bf8-34d118bcc818`, driver 580.95.05, assigned device 7,
eight CPUs and a finite 20-minute limit. The same frozen 96-atom/1088-AO
r2SCAN-3c density, basis phases and normalization map as the preceding
comparison are used. A fresh independent CPU PySCF/libcint direct J at
`1e-15` is computed outside timed intervals. Every sample passes
`atol=3e-9, rtol=3e-11` against it.

Prepare and warm all owners, then interleave four samples per policy with
reversed order on alternating rounds. Primary synchronized host-density to
host-J times include transfers and full J work. Native additionally includes
test-only input readback/checking; CUDA event boundaries differ by engine and
are secondary evidence, not identical kernel-only timing scopes.

| Policy | Cutoff | Median host wall (s) | Largest error vs CPU J |
| --- | ---: | ---: | ---: |
| Native incumbent | `1e-12` | 3.94408 | `2.456e-11` |
| Shared-recurrence tile, value 1 | `1e-12` | 4.31341 | `2.456e-11` |
| Primitive-lane tile, value 2 | `1e-12` | 4.36859 | `2.456e-11` |
| GPU4PySCF 1.8.1 default | `1e-13` | 0.67596 | `2.753e-11` |
| GPU4PySCF 1.8.1 | `1e-14` | 0.72287 | `2.402e-12` |
| GPU4PySCF 1.8.1 | `1e-16` | 0.83822 | `1.847e-13` |

The new tile is about 10.8% slower than the incumbent. Its retained output
differs from the incumbent by at most `3.376e-13`, below the separate
`3e-11` incumbent gate. There is no reason to promote it or open a performance
PR. Upstream screening contracts still differ; observed single-density error
is not proof of identical admitted domains or general SCF/force equivalence.

## Work and hardware-counter evidence

Slurm 10302 separately captures warmed complete J calls with Nsight Systems.
These profiler intervals are attribution, not clean timing samples. Incumbent
residual kernels total 2.50344 s versus 2.91402 s for the new tile. Uniform
potential work remains approximately 1.46 s in both.

Fresh counted replays agree for every angular class's uniform roots,
directions/summands and admitted residual tasks/primitive products. Candidate
probe counts differ because the coarse tile geometry differs, not because
different retained integral contributions are allowed. In `(2,1)` specifically:

- Both policies admit 9929600 shell tasks and 55373760 primitive products,
  only about 5.58 products per shell task on average.
- The candidate handles 9081696 tasks with masked Hermite contraction and
  27664960 nonempty cutoff groups.
- Its strip refreshes revisit 24092815 angular-one primitives and 14557348
  angular-two primitives; suffix density/E summands total 2562515640.
- Logical AO/Hermite output terms are 2551357568 versus 5513255552 without
  chunk fusion. They are semantic terms, not instruction or FLOP counts.

Slurm 10305 profiles the two matching `(2,1)` residual domains, one selected
launch each and 22 replay passes each, on RTX 5090 UUID
`GPU-57db1e6a-5ff9-570d-67b0-6965fb3a77e6`, assigned device 2. Authorized sudo
runs inside Slurm and restores/checks the assigned visibility after PAM.
No credential is retained. Clocks are unlocked: use these counters to diagnose,
not as a clean endpoint timing distribution. Both outputs pass the CPU oracle.

| Metric | Incumbent residual `(2,1)` | Primitive-lane tile |
| --- | ---: | ---: |
| Mean active threads per warp instruction | 4.39 | 5.66 |
| Achieved occupancy | 33.25% | 16.67% |
| Registers per thread | 104 | 140 |
| Executed instructions | 269245132864 | 177616630928 |
| DFMA/DMUL/DADD opcode instances | 12218070082 | 6744779903 |
| FP64 atomic opcode instances | 79282176 | 24946624 |
| FP64 pipeline active-cycle metric | 86.50% | 28.97% |

Arithmetic opcode instances drop about 44.8%, instructions about 34.0%, and
atomic opcode instances about 68.5%, yet the complete consumer loses. These
opcode instances are not FLOPs or semantic integral counts. Lane activity is
an instruction-wide average, not isolated FP64 utilization or proof of one
unique stall root cause. It improves only modestly, nowhere near upstream's
27--29 active threads, while achieved occupancy halves. Copying a rectangular
primitive layout without compacting the ragged work is not upstream's actual
execution organization.

## Next experiment and promotion boundary

The requested GPU4PySCF-like performance is not achieved. Do not redefine the
goal as a successful numerical prototype or reduced arithmetic count.

Prioritize a compact source/primitive work stream across different shell
tasks, rather than reserving sixteen lanes for every small ragged task. Move
stable directional ordering and reusable density preparation out of the
repeated short tile lifetime. The native optional allowance has room for a
small immutable order table but not all dense Hermite suffixes, so any cache
must have explicit capacity accounting and allocation failure fallback.

Then extend Hermite potential lifetime across source partners with stable
output-bound/direction channels. Partner-local cutoff group IDs cannot simply
be added together. Generate these schedules from the existing algebra; do not
replace exact FP64 masks with upstream's different local/log-domain screening
or introduce a reference-provider production dependency. Uniform J must also
improve eventually: its remaining 1.46 s alone exceeds upstream complete J.

Current production defaults remain unchanged. Future promotion requires a
reproducible native-vs-native complete-J improvement and the original
independent SCF/convergence/force and source-identity qualification gates.

## References

- [Superseding compacted-source schedule](2026-10-19-compacted-masked-hermite-sources.md)
  replaces mode 2 while preserving this padded-lane experiment's negative evidence.
- [Fresh upstream comparison](2026-10-19-gpu4pyscf-residual-j-comparison.md)
- [Bounded shared-recurrence tile](2026-10-19-bounded-tiled-masked-hermite.md)
- [Bounded resident suffix cache](2026-10-19-bounded-residual-density-preprojection.md)
- Local `build/primitive-lane/summary.json` validates endpoint, oracle and
  equal native work; `counter-summary.json` validates both selected launches.
  `counter-reports-10305.sha256` pins the large remote hardware-counter reports.
