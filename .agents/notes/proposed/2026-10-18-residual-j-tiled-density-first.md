# Proposal: reuse residual-J density projections across tasks

Status: proposed architecture; bounded tiled prototype implemented, default off
Date: 2026-10-18

## Problem

The exact masked-Hermite prototype is numerically qualified but slower on the
96-atom r2SCAN-3c case. Its algebraic contraction estimate omits repeated density
projection, ordering, indexing and publication work. Adding more runtime mask
admission tests does not address those reuse scopes.

The preceding decision and complete-endpoint evidence remain in
`../implemented/performance/2026-10-18-masked-hermite-residual-prototype.md`.
This note records the upstream audit and newly accessible hardware counters;
it does not promote the prototype or claim a replacement speedup.

The subsequent default-off resident preprojection experiment and its
non-reproducible 96-atom timing changes are recorded in
`../implemented/performance/2026-10-19-bounded-residual-density-preprojection.md`.
The subsequent bounded shared-memory tiled prototype, primitive-potential
fusion and their negative complete-J measurements are recorded in
`../implemented/performance/2026-10-19-bounded-tiled-masked-hermite.md`.
That prototype does not qualify the broader architecture as a production
replacement or establish a performance benefit.

## Pinned comparison

- GenerativeQC baseline: PR #2600 head
  `aacbccc275274cb70e36b76a646750fe667acf3e`, including its default tiled
  residual and angular-three reciprocal paths.
- Candidate: the source-matched isolated library described in the preceding
  note, with `GENERATIVEQC_MD_J_MASKED_RESIDUAL=0/1` selecting the two policies.
  Both measurements use the same binary, SHA256
  `bb089a0e07e95cd0b6ea65f0a56c6f08e394dcc508de1ac9531666bf5fc6cedd`.
- GPU4PySCF: tag `v1.8.1`, commit
  `5b284c258a4260baef80e3d150b4e7a81a9dbd57`. The installed wheel's
  `gpu4pyscf/scf/j_engine.py` is byte-identical to the tag, SHA256
  `e24c81017dd37e18664113c71bbab88c9cb1b8fe7a12a6cad76aeb3fe2cb21d3`.
- Official upstream source archive SHA256:
  `b0ea41feac9b05195832d467d8c05d29fcb6f419633781353e083a9d24d810bd`.
  Upstream CUDA is inspected and compiled unmodified for diagnostics, not
  copied into production scientific code.

The installed upstream binary contains sm70/sm80/sm90 cubins, not sm120.
The static comparison instead builds upstream `unrolled_md_j.cu` for sm120 with
CUDA 12.9.86, `-O3` and relocatable device code. This is not the exact binary
behind the historical GPU4PySCF timing and supplies no new upstream timing.
The rebuilt object SHA256 is
`dffffbf6ac1fed8f521d031c809b844119355803ac2784201603c7dd902125f5`.
Compilation used verified sccache; before/after statistics are retained.

## Verified dataflow differences

In upstream `gpu4pyscf/scf/j_engine.py`, `dm_xyz` is produced once per `get_j`
before angular tasks. Descending `q_cond` ordering and the 2/4/8/16/32 hierarchy
are present. Their log/FP32 decontracted-pair screening is not interchangeable
with our original directional AO FP64 product comparisons. Upstream also
decontracts its basis, so primitive and contracted task counts are not directly
comparable.

Upstream `gpu4pyscf/lib/gvhf-md/unrolled_md_j.cu` has a 256-thread `(16,16)`
block for both target classes. Its fixed `(2,1)` logical primitive rectangle is
768 by 480, using 6112 doubles (48,896 bytes) of dynamic shared memory for
nonnegative omega. The `(2,2)` rectangle is 224 by 224, using 5920 doubles
(47,360 bytes). Geometry and transformed density are cached, Hermite potentials
are accumulated/reduced across batches, and AO projection is separate.

The specialized dispatch has fixed tile constants. The shared-memory-budget
calculation in `_md_j_engine_quartets_scheme` configures the generic fallback;
it must not be described as runtime autotuning of those specialized kernels.
`md_contract_j.cu` returns through unrolled dispatch when it succeeds.

Our incumbent `contract_source_task` already computes the Coulomb recurrence
once per primitive product, not once per AO quartet. The remaining AO expansion
and orbit scatter follow that shared recurrence. The accepted masked prototype
already avoids constructing a full AO four-index block, but still:

- Rebuilds directed ordering/cutoffs inside each shell-pair task.
- Rebuilds bra suffixes per task/bra primitive and ket suffixes per primitive
  product, rather than sharing density projections across partners.
- Projects/publishes outputs at task scope instead of accumulating across a
  reusable two-dimensional strip or tile.

Thus changing representation locally is insufficient; the next experiment must
change the lifetime/reuse scope of transformed density and output accumulation.

## Static evidence is not a hardware diagnosis

Same-architecture static resource reports give these registers per thread:

| Class | Incumbent | Masked | Upstream rebuilt sm120 |
| --- | ---: | ---: | ---: |
| (2,1) | 104 | 100 | 108 |
| (2,2) | 99 | 104 | 128 |

All six selected kernels report zero stack/local bytes in the compiler resource
report. There is no compiler spill evidence explaining the regression. Static
instruction-site totals, barriers and atomic sites are retained in
`static-summary.json`; these are not executed instructions or per-integral work.
The candidate's disabled specialization matches the actual incumbent summary.

## Diagnostic timeline

Slurm 10212 used n1 / RTX 5090 for one fresh-process frozen-density complete-J
trace per policy. Nsight Systems CUDA event-completion tracing reported overhead;
the following is attribution, not a new unprofiled benchmark distribution:

| Residual class | Incumbent | Masked |
| --- | ---: | ---: |
| (2,1) | 0.681904490 s | 0.789010814 s |
| (2,2) | 0.353922793 s | 0.456777127 s |
| All residual classes | 2.467136664 s | 2.672041768 s |

Both complete J results pass the independent stored GPU4PySCF oracle, maximum
absolute error approximately `2.44484e-11`. Neither target class improves in
this diagnostic; there is no evidence for promoting only one class as-is.

## Hardware counters: access and scope

The unprivileged Nsight Compute probe in Slurm 10208 failed with
`ERR_NVGPUCTRPERM`. Following the user's authorization, Slurm **10221** ran
Nsight Compute 2025.2.1 with sudo on n1, through a finite 20-minute `srun` using
`main`, one node/task and `gpu:5090:1`. The allocation assigned visibility `0`;
the privileged process verified the same value before any GPU work. GPU UUID
was `GPU-45f177f9-33ac-0e4e-f473-2beeaf6d413d`, driver 580.95.05.

This host's sudo PAM environment clears `CUDA_VISIBLE_DEVICES`. A preflight
10216 exposed that issue and is not used as device-qualified evidence; the
fail-closed guard in 10220 rejected it before profiling. In 10221, `/usr/bin/env`
after sudo carries Slurm's exact assigned visibility, also supplied as a
receipt that the profiling script compares for equality. No device is chosen
independently of Slurm. No password appears in scripts or saved evidence.

No global driver/security settings were changed: `RmProfilingAdminOnly: 1`
remains in effect after collection. `--clock-control none` avoids changing GPU
clocks; measured SM frequency is approximately 2.89--2.90 GHz. Consequently the
profiler warns about unlocked clocks, and the four observations are diagnostic,
not a controlled timing distribution.

Each run selects one `CountWork=false` `(2,1)` or `(2,2)` residual kernel
from the same 96-atom / 1088-AO frozen-density replay. Kernel replay collects
SpeedOfLight, LaunchStats, Occupancy, ComputeWorkloadAnalysis,
MemoryWorkloadAnalysis, SchedulerStats, WarpStateStats, InstructionStats and
SourceCounters in 22 passes. All four post-profile complete-J oracle checks
pass with maximum absolute error `2.4448427294e-11`.

| Class / policy | FP64 pipe active | Memory throughput peak fraction | Achieved occupancy | Executed instructions | Mean active threads / warp instruction |
| --- | ---: | ---: | ---: | ---: | ---: |
| (2,1) incumbent | 86.57% | 5.14% | 33.29% | 269.245 billion | 4.39 |
| (2,1) masked | 86.45% | 11.79% | 29.11% | 333.154 billion | 3.01 |
| (2,2) incumbent | 86.21% | 7.68% | 33.00% | 183.099 billion | 5.89 |
| (2,2) masked | 79.79% | 13.15% | 18.68% | 227.424 billion | 4.58 |

The FP64 column is `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active`;
the memory column is
`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`, not DRAM
bandwidth alone. Reported DRAM traffic is only about 21--28 MB/s, and its peak
fraction is below 0.01%. These observations support substantial FP64 execution
pressure in these launches, not a global-memory-bandwidth saturation diagnosis
or a claim about every kernel in the full SCF endpoint.

The dynamic opcode instances provide a stronger comparison than static sites:

| Class / policy | DFMA + DMUL + DADD | Global FP64 atomic instructions |
| --- | ---: | ---: |
| (2,1) incumbent | 12,218,065,578 | 79,282,176 |
| (2,1) masked | 14,435,333,378 | 24,946,624 |
| (2,2) incumbent | 6,627,899,693 | 62,261,944 |
| (2,2) masked | 7,825,340,517 | 10,186,744 |

These are NCU opcode counts, not FLOPs, thread-useful work, or scientific task
counts. FP64 atomics are identified by the `.F64.` opcode modifier separately
from integer queue atomics. Masking reduces those atomic instructions by
68.53% / 83.64%, but increases the three arithmetic opcode counts by
18.15% / 18.07% and total executed instructions by 23.74% / 24.21%.
Reducing publication alone has not reduced overall execution work.

Short-scoreboard stalls account for approximately 64.8% / 57.6% of the baseline
reported issue intervals. Sampled stall PCs and neighboring SASS are retained;
a waiting instruction's PC is not proof of its dependency's root cause. These
stalls must not be equated with shared-memory bandwidth saturation. The masked
`(2,2)` specialization additionally changes the reported shared-memory occupancy
limit from 25 to 9 blocks/SM; its theoretical occupancy is 18.75%, close to the
measured 18.68%. This is a measured resource constraint, not proof that changing
occupancy alone recovers the regression. Automatic profiler speedup estimates
and partial-wave warnings are not accepted speedup predictions.

## Proposed implementation boundary

1. Keep geometry-lifetime directed AO ordering metadata with the prepared pair
   owner, invalidating on geometry, transform or bound changes. Do not cache an
   AO-quartet-sized matrix of cutoffs.
2. Generate density-dependent suffix projections once per reusable input strip
   or a budgeted resident projection, rather than once per quartet. Invalidate
   on each density replay and respect stream/event lifetimes. Account for every
   suffix variant, including duplicated directions, when estimating bytes.
3. Generate small-class Coulomb-times-density consumers and tiled reductions in
   the integral compiler. Native code owns storage admission and launch policy;
   it must not hand-maintain a second scientific formula from upstream CUDA.
4. Accumulate separate directed-output/group potentials or bounded AO partials
   across a tile before publication. Different exact output masks cannot simply
   share one unrestricted Hermite potential. Include this extra state in the
   shared/register budget rather than assuming upstream tile sizes will fit.
5. Retain the incumbent for unsupported/complex masks, invalid bounds, resource
   failures and unprofitable groups. Reject before publishing any contribution
   so fallback cannot repeat partially consumed work. Uniform sources remain
   on their existing density-first schedule.

All-suffix storage across upstream-sized tiles can exceed shared-memory capacity.
Prefer bounded strips or explicitly budgeted optional resident storage with
fallback, not an unbounded new default allocation. This is a proposed design,
not an implemented or performance-qualified kernel.

## Invariants and promotion gate

Preserve the original FP64 product comparison, including ties/underflow/overflow;
do not replace it with division or upstream log/FP32 screening. Preserve distinct
forward/reverse AO bounds and nonsymmetric UKS densities, canonical-orbit and
identical-pair exclusion, density admission/error budgets, and uniform/residual
ownership. A Schwarz suffix proof alone does not prove the whole orbit domain
is a suffix. Keep compact native E-layout accessors and bounded failure paths.

The existing same-density unprofiled complete J is 3.8710633545 s versus
4.0808392334 s (masked 5.42% slower, Slurm 10172). The single matched complete
cold E+forces pair is 262.87839 s versus 277.41127 s (5.53% slower, 24 versus
25 Fock builds, Slurm 10199). Those measurements, not replay wall time, govern
the existing no-promotion decision. No default, branch, commit or PR changes
follow from this audit.

A replacement must report actual projection/group counts and bytes, accepted
tasks/roots, dynamic instructions and full-J timing on the same input, then
repeat independent energy/force, residual and SCF-trajectory gates plus complete
endpoint timing. Include strip preparation and optional storage in the endpoint.

## Evidence retention

Local ignored receipts and scripts are in `build/md-j-dataflow-audit/`, including
`profile_counters.sh`, `export_counter_sources.sh`, `summarize_counters.py`,
`counter-summary.json`, `counter-inputs-10221.txt`, report checksums and per-mode
CSV/log/detail/SASS exports. The summary script checks all four selected kernel
identities and independent J gates. The NPZ-container checksum in the input
receipt is distinct from the density-array checksum in the preceding note.

Full `.ncu-rep` files, `.nsys-rep` files, source archives and static disassembly
remain on n1 under
`/home/jzzeng/generativeqc-qualification/md-j-dataflow-audit-20261018-49fc/`.
The large NCU reports remain remote; local SHA256 receipts identify them.
No release, GitHub posting or external artifact publication is involved.
