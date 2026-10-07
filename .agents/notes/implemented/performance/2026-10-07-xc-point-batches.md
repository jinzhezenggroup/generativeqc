# Decision: batch independent XC point domains without merging AO maps

Status: implemented (qualification-only; default remains one tile)
Date: 2026-10-07

The initial opt-in policy recorded here is superseded by
`2026-10-07-xc-point-batch-default.md`. Retain this note's original measurement
identities and evidence limitations; promotion does not relabel those binaries.

## Problem

#2073 records an eight-CTA physical-PBE point launch repeated for each of
9,216 256-point SCF tiles. CUDA Graph replay changes host submission overhead,
not the independent device work supply. Increasing individual tile size can
merge AO support and silently increase selected point-by-AO-squared work.

## Decision

Retain several independent compact AO panels, produce each tile's features
with the existing density provider, then submit their combined point domain.
The same canonical FP64 point consumer handles tile-local channel-major slots.
The tail is compact; a point maps directly to its slot without a device descriptor
or gather kernel. Reuse the original density/potential scratch because those
panels do not need to survive the point submission. Scatter Vxc and reduce totals
serially in the historical tile order.

The scientific compiler owns admission and point-domain mapping. Native code
owns optional allocation, lifetime, validated resource binding and stream
orchestration. Admission scans actual AO counts and checks products against a
finite byte cap before multiplying; it does not key on molecule/device identity.
The independent experimental controls request tiles and an additional allowance.
Actual retained bytes participate in the shared numeric resource ledger and the
ordinary KS XC resource diagnostic. An allocation failure retains the original
maps and one-tile executor, not a dense/CPU substitute.

## Rejected alternatives

- Merge selected AO maps: changes the admitted scientific work domain and can
  restore dense quadratic work merely to enlarge launches.
- Re-evaluate AO panels after batching features: conceals extra AO/jet work.
- Concurrent potential scatter with atomics: races or changes deterministic
  accumulation semantics without a separate numerical qualification.
- Retain every density-product panel: unnecessary data residency; scratch is
  dead after each feature contraction and can be reused.
- Promote on launch counts alone: complete cold/warm/moved E+F profitability
  and the independent source-specialization/composed ablations remain required.

## Invariants

- Original tiles, FP64 arithmetic, basis/grid/screening and AO maps are unchanged.
- AO, density, contractions, scalar reductions and matrix scatter retain their
  original work counts; only independent point submissions are batched.
- Responses and mixed arithmetic retain the one-tile executor. An admitted
  owner cannot subsequently discover another map or rebind mixed density.
- No allocation, host descriptor packing, point staging or additional explicit
  synchronization occurs inside evaluation/capture. Owner replacement on moved
  geometry recalculates residency; density changes do not reuse stale features.
- Optional residency is additional to the original full-capacity scratch, not
  described as replacing it. Tiny domains and insufficient resources fall back.

## Evidence

CPU generation/host-selector tests execute the emitted C++ planner and cover
ragged counts, high occupancy, empty tiles, compact tails, overflow-sized tile
requests, response/mixed/tiny rejection and exact byte bounds.
Native `--point-batches` exercises independent CPU E/V and exact same-tile
baseline equality, both spin layouts, spherical/cartesian bases, scaled PBE,
captured density changes, geometry-specific maps, resource rejection and teardown.
Complete endpoint evidence is retained separately; this note does not claim
default profitability or qualify #2072's source-specialization arm.

## Consequences and revisit conditions

More independent CTAs require AO residency. Contractions remain ordered, so this
does not address their launch count or promise potential-kernel acceleration.
Consider grouped provider contractions only with a separate ordered reduction
plan and independent complete E/V and moving-grid E/F gates. Promote batching
only after clean interleaved fresh-process/cold/warm/moved endpoint populations,
retained source/binary/generated identities, explicit work/memory counts, and
separate scheduling/source-specialization/composed comparisons establish a guard.

## Retained October 7 qualification

Frozen master: `2b0feff1d5577e58f3a44fbecbc1974be92df4e3`.
Measured implementation: `282dabad0cfaad612d93eb7789f55d3296f03d05`.
Both clean trees were rebuilt in Release for sm_120 with verified ccache
launchers. Slurm assigned an exclusive RTX 5090 on n1, eight host CPUs and
finite job times; device visibility and persistent compilation caches were
preserved. No cache correctness or source/build identity was weakened.

Native library SHA-256 identities:

- Master: `dc294cbe3cf4b739ab5cc6881b9de7d424f455293444bcb0a3976522fb580d38`.
- Candidate: `2644741071f0e15c9a2a317692a4ee13aa930bb66777645f0e21957b5cdd23d4`.
- Candidate generated grid source:
  `6ec2fddb973af34294a3e7ec361b93e2b80f99a506b365d978522ad790f8b6a2`.

The later registry-fact spelling of feature shapes produces byte-identical
native XC emission, checked before/after with SHA-256
`86556f953d9b50871434f3fbcdc7e861460240aae04fba73c86969161e2bb8ac`.
This does not relabel the measured binary as a later commit's build.

Post-measurement source-contract repairs select the exact unbatched
`evaluate_points<feature_terms,false,false>` instantiation for incumbent
compiled-resource evidence. A batched kernel cannot fill a missing incumbent
scope. The extracted host-owner probes retain their publication/provider checks
and cover batch forwarding, immutable maps/precision and admitted-byte accounting.
The expanded focused compiler/schedule/host-owner suite passes all 222 tests,
including the remaining shared schedule and region-selection resource fixtures.

### Scientific and lifecycle gates

- Compiler/schedule suite: 99 passed, two optional checks skipped; compiler
  structure, CUDA ownership, native-complexity and default-promotion audits pass.
- Native `--point-batches`, `--local-ao`, `--pbe0-local-ao` pass, including
  independent CPU bilinears and bitwise same-tile E/V equality.
- Compute Sanitizer memcheck: zero errors and zero bytes leaked.
- Every cold 12/96-atom and warm/moved 48-atom complete PBE0 E+F observation
  passes independent reference gates with unchanged FP64, spherical def2-SVP,
  unpruned 48×16×32 grid, convergence/screening and full moving-grid response.
  Cold maximum errors are below `7.83e-11 Eh` / `3.02e-11 Eh/bohr`, versus
  acceptance limits `1e-8 Eh` / `1e-7 Eh/bohr`.

### Work and memory: fixed-density 96-atom XC

A request of 32 tiles with a 32-MiB additional allowance admits **eight** tiles
on the original high-occupancy maps, not the requested count. Submissions per
traversal change **9,216 → 1,152**, with **8 → 64 CTAs** and the same 32-thread
block. Selected point×AO² work stays **75,674,112,000**, versus dense
**1,391,569,403,904**; all 768 empty tiles remain. Each traversal still evaluates
1,495,474,176 selected AO jet values, has 8,448 nonempty density/potential
contractions and 295,602,000 logical mapped matrix entries across tiles.
Scalar potential fallbacks/empty-tile reducers are separately accounted in NSys,
not hidden in those nonempty contraction counts.

Additional residency is **21,823,488 bytes**. The diagnostic fixture's original
arena is 144,837,920 bytes and candidate retained arena storage is 166,661,408
bytes. These are arena counts, not total process peaks including shared grid,
input density, graphs and libraries. Complete-endpoint diagnostics retain the
encompassing native reservations. The census also reports preparation/discovery,
host maps, transfers, logical gathers/scatters and explicit synchronizations.
No point gather, descriptor upload, extra AO/jet/contraction traversal or
concurrent scatter is introduced.

Clean fixed-density XC warm medians (five post-first-call samples) are
2.340563 → 1.522296 s originally and 2.337143 → 1.519399 s after displacement.
This endpoint includes density upload and E/V host return; geometry, discovery
and optional batch preparation are reported separately. It is not complete
SCF/force default-promotion evidence.

### Complete endpoint populations

| Scope | Samples per arm | Baseline median s | Candidate median s |
| --- | ---: | ---: | ---: |
| 12-atom fresh-process cold E+F | 3 | 91.631693 | 88.869089 |
| 96-atom fresh-process cold E+F | 2 | 217.112518 | 205.049690 |
| 48-atom frozen warm E+F | 5 | 7.966658 | 7.516457 |
| 48-atom displaced frozen warm E+F | 5 | 8.000996 | 7.540783 |

Cold includes preparation plus the first complete host-return endpoint,
excluding imports/context initialization/Calculator construction. Fresh
processes are interleaved AB/BA and reuse compiler caches. Native/reference
screening is `1e-12`/`1e-14`, not equal admitted work. Warm observations each
perform one physical iteration/Fock build without warm fallback; owners freeze
separately converged public snapshots, not identical density bytes. Setup,
independent reconvergence, priming and moved histories are retained outside
replay timing.

**Trajectory work is not identical:** 12-atom cold builds are 23 in all samples;
96-atom baseline builds are `[25, 27]` and candidate builds `[27, 28]`. Per-build
selected AO work is unchanged, but cumulative work differs. The roughly 5.6%
96-atom cold median reduction is only two pairs, not a statistically qualified
default-promotion claim. Five warm/moved pairs show roughly 5.7% complete-
endpoint reductions with the same one-build semantic work per observation.

### Separate diagnostics and outstanding evidence

NSys fixed-density runs contain 12 traversals per arm: point calls
**110,592 → 13,824**, and summed point GPU time **11.151889 → 1.416920 s**.
Diagnostic times are not substituted for clean endpoint populations. Density,
potential and jet work are not credited as reduced. No generic DRAM-bandwidth
diagnosis is inferred from the point timing.

NCU matched-kernel collection was attempted through Slurm but failed with
`ERR_NVGPUCTRPERM`; sudo also requires a password. Thus waves/SM, register/
occupancy and hardware throughput counters are **not newly qualified**. No
permission or device-visibility bypass was attempted. #2072's source-only and
composed arms are absent from the frozen master. These gaps, the small cold
population and residency tradeoff keep the route opt-in.

Raw vectors, source/binary/generated checksums, cache stats, work census,
sanitizer output, fresh-process JSONs, NSys capture/statistics and failed NCU
records remain local under `/data/jzzeng/qc-2073-indexed-xc-20261007/results/`.
Reproduction drivers are in its parent; tracked
`benchmarks/pbe0_xc_tile_pairs.py --point-batch-tiles 32` reproduces warm/moved
histories. No release/tag or external raw-artifact backup was published.

## References

- #2073; source-specialization sibling #2072; evidence owner #1965.
- `docs/developer/xc_native_cuda.md`
- `docs/maintainer/performance_engineering.md`
