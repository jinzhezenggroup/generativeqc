# Decision: compact admitted exact masked-Hermite sources before contraction

Status: implemented opt-in; production defaults unchanged
Date: 2026-10-19

Production selection and coarse admission are superseded by the
[warp-coarse qualification](2026-10-19-warp-coarse-masked-hermite.md), which
retains `(2,2)` on the incumbent and restores the coarse-screening work contract.
The measurements here remain source-specific historical evidence.

## Problem and decision

The [fresh GPU4PySCF comparison](2026-10-19-gpu4pyscf-residual-j-comparison.md)
found very low active-lane ratios in native residual J. Simply assigning a
padded primitive rectangle to a half warp also failed: short contractions and
sparse admission left most of that rectangle unused. This note supersedes the
mode-2 schedule in the [primitive-lane experiment](2026-10-19-primitive-lane-masked-hermite.md),
not its retained negative evidence.

`GENERATIVEQC_MD_J_MASKED_TILES=2` now uses a bounded 256-entry shared queue.
The block scans multiple 128-candidate batches until at least 128 tasks are
admitted or candidates are exhausted. Warp ballots and a shared counter pack
admitted tasks. Each of 128 threads consumes one complete shell task, with its
own recurrence, exact directional mask plan and primitive contraction. Retained
leftovers survive the next refill. Appending only while the queue has fewer
than 128 entries bounds occupancy by 255; storage does not grow with molecule
size. The persistent block inventory remains capped at 4096.

Only requested Hermite density tails are constructed. Bra Hermite potentials
survive all opposing ket primitives before AO back-projection. Reverse ket
outputs still project per primitive product. The compiler reuses the native
Coulomb recurrence, E accessor, parity and exact mask planner; it does not own
a second recurrence implementation. Rejected masks use the original AO
consumer with thread-private scratch and single-lane synchronization.

The schedule supports `(2,1)` and triangular `(2,2)`. Same-pair diagonal tasks
publish only the bra consumer, preserving orbit deduplication. K and fused J/K
remain with the incumbent. Mode 1 retains the earlier bounded shared tile.
No additional resident allocation or change to the 128 MiB optional MD budget
is introduced.

## Source and input provenance

Timed native source is the adapted compact-E PR #2600 snapshot
`aacbccc275274cb70e36b76a646750fe667acf3e`, not the older dense-E checkout at
detached HEAD `801e6fad5f5951696abc1f7856527828603e2a4e`. The code-generation
modules are the current working-tree modules. Qualification scripts compare
staged source bytes with the compilation snapshot and retain all hashes.
Do not transfer these absolute timings to another source variant.
The linked native library SHA256 is
`a01d7b2530996a088cd77193f7d283ab9cc581fb160ea91af46846caf2d35a2f`.

- Remote retained root:
  `/home/jzzeng/generativeqc-qualification/packed22-hermite-20261019-49fc`.
- Local evidence: `build/packed22-hermite/`; `summary.json` verifies accuracy,
  identical binaries/source across clean allocations and all 25 native work
  domains. Scripts and large profiler reports are retained at the remote root.
- 96 atoms, 1088 spherical AOs, explicit canonical r2SCAN-3c shells/phases.
- Frozen physical-density SHA256:
  `a0b22059e12adcd39771e5dcfaac0951afb298d8252ef6674889ad0349ed6090`.
- Basis SHA256:
  `fb2c5d3a5f1ec149f0bf1d15d1e76cdb8baf042aeaa8be677cf25f33b1b7882b`.
- GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 13.6.0. Source comparison uses upstream
  v1.8.1 commit `5b284c258a4260baef80e3d150b4e7a81a9dbd57`; no latest-version
  claim is made. Installed Python/library hashes are in each result JSON.
- Every real-GPU run uses finite Slurm jobs on node1, preserving assigned
  visibility. Node3 is not used for GPU execution.

## Clean paired prepared-J comparison

Slurm 10330 and 10331 independently compute a fresh CPU PySCF/libcint J at
`1e-15`, outside timed intervals. The canonical density/J are transformed on
both AO sides with the established shell/phase map. Each job prepares three
native owners and four GPU4PySCF tolerances, warms each once, then takes four
interleaved samples per method, reversing order every other round.

Primary wall timing is prepared host canonical density to synchronized host
canonical J, including transfer, projection, screening and complete J work.
Native additionally includes the probe's test-only input-density readback and
assertion. CUDA-event intervals differ between engines and are secondary.
These are not SCF timings. Preparation is separately recorded, not presented
as a controlled cold-start comparison.

| Method | Job 10330 median (s) | Job 10331 median (s) | Maximum error vs fresh CPU J |
| --- | ---: | ---: | ---: |
| Native incumbent, `1e-12` | 3.92987 | 3.92880 | `2.456e-11` |
| Native compacted `(2,1)+(2,2)`, `1e-12` | 3.61357 | 3.57531 | `2.456e-11` |
| Native earlier shared tile, `1e-12` | 4.29920 | 4.30116 | `2.456e-11` |
| GPU4PySCF `1e-12` | 0.62850 | 0.62723 | `2.390e-10` |
| GPU4PySCF verified default `1e-13` | 0.67264 | 0.67392 | `2.753e-11` |
| GPU4PySCF `1e-14` | 0.71983 | 0.71933 | `2.402e-12` |
| GPU4PySCF `1e-16` | 0.83771 | 0.83600 | `1.741e-13` |

Complete prepared-J time falls by 8.05% and 9.00%, about 1.09--1.10x speedup.
Largest compacted/incumbent J differences are `3.411e-13` and `2.878e-13`.
Every output passes `atol=3e-9, rtol=3e-11` against its fresh CPU reference.
Both independent allocations use the same RTX 5090, visibility 0, UUID
`GPU-45f177f9-33ac-0e4e-f473-2beeaf6d413d`; GPU receipts, not ordinal alone,
remain authoritative. Both methods within each allocation use that same GPU.

Native preserves directional FP64 AO Schwarz admission and its directed-rounding
global density-error allowance. Upstream uses float32 log-domain pair bounds
and local density admission. Equal cutoff strings do not imply identical
retained integrals. The upstream `1e-13` results have similar observed error
on this density, not equivalent worst-case screening guarantees. The remaining
roughly 5.3x gap cannot be attributed solely to kernel throughput.

## Work census and class-specific diagnosis

Separate Nsight Systems Slurm 10332 captures one warmed J per native policy.
Counted replays verify identical uniform radial sources/directions/summands,
residual admitted shell tasks and residual primitive products in all 25 classes.
There are 43920982 residual tasks and 242399046 residual primitive products.
Candidate probes plus coarsely screened candidates agree by class. Compacted
classes currently scan more individual candidates because they do not use the
incumbent 8-by-8 coarse tile traversal; this is not a reduction in scientific
work hidden behind fewer counters.

| Nsight Systems kernel domain | Incumbent (s) | Compacted (s) |
| --- | ---: | ---: |
| Residual `(2,1)` | 0.68415 | 0.31402 |
| Residual `(2,2)` | 0.35518 | 0.39349 |
| All residual kernels | 2.46917 | 2.13136 |
| All uniform potential kernels | 1.46372 | 1.46275 |

`(2,1)` is about 2.18x faster, but `(2,2)` regresses about 10.8%. Do not enable
both classes by default merely because their combined J time improves. These
are separate instrumented traces, not clean endpoint timing distributions.
Uniform J alone still costs about twice upstream's complete J.

Masked admission covers 9081696 of 9929600 `(2,1)` tasks and 4525808 of
4586580 `(2,2)` tasks; other tasks retain the AO fallback. Logical output terms
fall from 5513255552 to 2780230272 for `(2,1)` and from 4115122880 to
3078692840 for `(2,2)`. The slower `(2,2)` result confirms that these algebraic
counts are not hardware instruction counts or a speedup prediction.

## Actual upstream source comparison

Read v1.8.1 sources from the local upstream Git repository, not a third-party
description or assumptions about current HEAD. The earlier comparison retains
binary architecture checks and source-replayed upstream work domains.

| Mechanism | GPU4PySCF 1.8.1 | Current compacted experiment |
| --- | --- | --- |
| Hermite density | `_dm_to_Rt` before angular work, reused across partners | Exact selected tails still prepared per shell task and primitive |
| Potential lifetime | Hermite accumulation over many source partners, then `_Rt_to_dm` | Bra potential survives one opposing primitive contraction, not other shell partners |
| Lane mapping | Independent primitive-pair products in 16-by-16 blocks | One admitted complete shell task per thread after compaction |
| Small-class algebra | Scalar R/density DAGs in `md_j_2_1` and `md_j_2_2` | Native recurrence and compact E access, dynamic exact masks/groups |
| Screening organization | Descending `q_cond`, hierarchical maxima at 2/4/8/16/32 | Native FP64 admission; compacted classes bypass incumbent coarse tiles |
| Shared storage | Fixed specialized primitive-pair rectangles | Fixed shared task queue plus private per-task algebra |

Relevant upstream anchors: `gpu4pyscf/scf/j_engine.py:140` density projection,
`:245` sorted pair cache, `:276` hierarchy, `:353` density transform and `:398`
potential projection. `gpu4pyscf/lib/gvhf-md/unrolled_md_j.cu:914` and `:1232`
own `(2,1)`/`(2,2)`; fixed dispatch is near `:4691` and `:4697`. The `(2,1)`
logical rectangle is 768-by-480 primitive pairs with 48896 shared bytes;
`(2,2)` is 224-by-224 with 47360 bytes. Do not compare those areas directly
with a shell-pair queue or mistake the generic resource estimator for runtime
autotuning of these fixed kernels.

## Fresh hardware-counter comparison

Slurm 10335 profiles both native policies and upstream in one allocation on
visibility 2, UUID `GPU-57db1e6a-5ff9-570d-67b0-6965fb3a77e6`. Authorized
sudo access obtains counters; the fail-closed receipt preserves the Slurm
device visibility through the account change. No credentials are retained in
scripts or receipts. Each selected launch uses 22 replay passes, with unlocked
clocks. These are diagnostics, not clean endpoint timing samples.

| Domain | Active threads / warp instruction | Registers/thread | Achieved occupancy |
| --- | ---: | ---: | ---: |
| Native residual `(2,1)` | 4.39 | 104 | 33.28% |
| Compacted residual `(2,1)` | 5.17 | 134 | 24.89% |
| GPU4PySCF full `(2,1)`, two launches | 27.43--27.54 | 109 | 23.86--31.58% |
| Native residual `(2,2)` | 5.88 | 99 | 33.08% |
| Compacted residual `(2,2)` | 4.14 | 242 | 16.56% |
| GPU4PySCF full `(2,2)`, three launches | 28.26--28.70 | 128 | 22.33--31.27% |

Instruction-wide active-lane ratios are not isolated FP64-lane utilization.
Compaction alone does not make branch-dependent AO plans or primitive loops
uniform. Native `(2,1)` executed warp instructions fall from 269.247 billion
to 104.545 billion; FP64 arithmetic opcode instances fall from 12.218 billion
to 6.108 billion, and FP64 atomic opcode instances from 79.282 million to
40.324 million. Native `(2,2)` instructions also fall, from 183.010 billion to
103.423 billion, but its active-lane ratio and occupancy worsen and its clean
timing regresses. These observations are consistent with a costly private
working set/divergent schedule; they do not isolate a single causal bottleneck.
Do not conclude that fewer instructions or atomic publications guarantee a win.

The upstream launches have different screening/source domains and include the
complete angular class, unlike these native residual-only launches. Do not
present their counter ratios as same-work kernel-efficiency factors.
`counter-summary.json` retains per-launch opcode and metric details;
`counter-reports-10335.sha256` pins the retained remote NCU reports.
Linked-library resource inspection records 1840 stack bytes for `(2,1)` and
4160 for `(2,2)`, with 3085 shared bytes each. `LOCAL:0` in that static report
does not mean absence of stack/local scratch. Do not substitute an unlinked
object's resource report for the linked kernel executed here.

## Complete cold 96-atom SCF and analytic-force gate

After rebuilding both stationary AOT profiles, Slurm 10336 runs one matched
cold pair on an RTX 5090, visibility 3, UUID
`GPU-d05490f6-a867-d1a0-f13f-8bf4b1116387`. Each policy uses a separate fresh
Python process with no preliminary density or frozen-density seed. The
calculator uses exact-direct FP64 spherical r2SCAN-3c, 48 radial by 16 polar
by 32 azimuth points, an unpruned grid and three Becke iterations. SCF gates
remain energy `1e-12`, density `1e-10`, Schwarz `1e-12`, at most 100 iterations.
Both return full analytic forces as host arrays with D4 and gCP corrections.

Timing includes calculator construction/preparation and the complete SCF plus
force execution, following the retained cold endpoint definition. It does not
include Python import time or prior AOT compilation. This is a native-vs-native
complete endpoint; it is not a full-SCF comparison against GPU4PySCF.

| Policy | Prepare (s) | Execute energy + forces (s) | Complete cold (s) | Iterations / Fock builds |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 8.64472 | 252.94442 | 261.58913 | 24 / 24 |
| Compacted `(2,1)+(2,2)` | 8.62526 | 244.79836 | 253.42362 | 24 / 24 |

This single matched pair has 3.12% lower complete-cold time; it is not a timing
distribution or a claim that prepared-J's 8--9% gain carries over unchanged.
Both policies report `warm_start_used=False`. Both pass the retained independent
reference's energy (`1e-8` Ha), maximum force component (`1e-7` Ha/Bohr),
convergence, host-force shape/finiteness and physical residual (`1e-10`) gates.
Maximum energy/force errors across policies are `2.4102e-11` Ha and
`6.2821e-11` Ha/Bohr. Policy-to-policy differences are `1.3642e-12` Ha in
energy and `2.9139e-11` Ha/Bohr in force components; residual RMS is about
`1.4894e-12`. Full force arrays, source/library/basis/reference hashes and
stationary manifest identities are in `cold96-10336/mode-{0,2}.json`.

## Qualification and promotion boundary

- 88 focused host tests pass; Ruff and the compiler ownership audit pass
  (568 modules, no dependency errors).
- Slurm 10329: 74 independent real-CUDA tests pass ordinarily and under each
  of memcheck, racecheck and synccheck, with zero errors/hazards. Cases include
  directed nonsymmetric bounds, UKS/nonsymmetric density, exact threshold
  equality, diagonal orbit exclusion, ragged 1/3/5 primitive contractions and
  a filtered 300-candidate stream forcing queue refill/leftover retention.
- Slurm 10333 runs the pinned source's normal molecular integration suite.
  Incumbent passes 114/114; compacted passes 113/114. The remaining failure is
  `test_residual_tiles_preserve_domain_tasks_and_replay_policy`: zero-density
  replay has no contracted roots but still scans candidates in the compacted
  classes, conflicting with coarse-tile policy's zero-candidate expectation.
  Do not remove the gate, relax the assertion or describe all integration
  tests as passing. Integrate compatible coarse admission before promotion.
- Stationary force AOT artifacts are genuinely rebuilt for the new compiler
  closure using verified sccache 0.16.0. Emitted NVCC commands use the launcher,
  sm120 and `--fmad=false`; manifest identities and library hashes are retained.
  AOT identity is not edited to reuse old force artifacts.

No production default, Git commit, GitHub PR or publishing operation is made
from these results. The cold SCF/force gate passes for this source-matched
pair, but the coarse-policy failure and `(2,2)` regression still preclude
promoting the combined schedule. Any revised promotion candidate needs its
own source-matched timing and numerical acceptance.

## Next bounded experiments

1. Keep the successful `(2,1)` schedule separate from the regressing `(2,2)`
   default-promotion decision. Preserve a selectable reference for both.
2. Restore conservative batch/coarse admission while retaining queue packing;
   verify zero-density behavior and all-class scientific work again.
3. Move stable directional ordering and reusable requested density preparation
   out of repeated task lifetimes under the original optional-memory cap.
4. Extend potential lifetime across shell partners using stable output-bound
   and direction channels, not partner-local group IDs. This is the important
   remaining difference from upstream, not simply a larger tile constant.
5. Optimize uniform potential work as well; residual improvements alone cannot
   close the observed complete-J gap. Keep exact masks and global error policy.
