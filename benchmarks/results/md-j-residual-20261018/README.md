# MD-J residual scheduling and GPU4PySCF diagnosis (2026-10-18)

This is diagnostic evidence, **not completed default-promotion qualification**.
PR #2600 proposes the tiled residual default but remains draft: only one of three
complete-endpoint pairs finishes, and that pair is slightly slower in aggregate.
Prepared-J observations must not be substituted for complete cold/warm timings.

The [follow-up contraction trials](contraction-trials.md) retain two rejected
residual rewrites and the opt-in angular-three reciprocal-reuse experiment.
Its [subsequent user-authorized default decision](../../../../.agents/notes/implemented/performance/2026-10-18-md-j-high-reciprocal-default.md)
changes the uniform angular-three selector only, not this historical residual
qualification record.
The [default-path validation](default-reuse.md) retains the distinct default
binary, new complete off/unset endpoint pair and explicit/unset work gates.

## Identity and scope

- Base: master `e4aff7810b0462204831eb5168f19b6e217ef520`.
- Measured implementation: `80f2d1830caa95f3bdccf5d0d4285ed7a6b7c4e7`.
- Final native library SHA-256:
  `5ca1225535852f611b5814d2872455b7c1bfbe8be31543b25e8be76069d8c314`.
- CUDA 12.9.1, GNU 13.3, Release, sm_120; reused verified ccache 4.5.1.
- n1/RTX 5090, finite Slurm allocations; scheduler device visibility preserved.
- Common accepted 96-atom water / 1088 spherical-AO r2SCAN-3c density SHA-256:
  `a0b22059e12adcd39771e5dcfaac0951afb298d8252ef6674889ad0349ed6090`.
- Basis file SHA-256:
  `fb2c5d3a5f1ec149f0bf1d15d1e76cdb8baf042aeaa8be677cf25f33b1b7882b`.
- GPU4PySCF distribution `gpu4pyscf-cuda12x==1.8.1`, PySCF 2.14.0, CuPy 13.6.0.
  Installed `j_engine.py` is byte-identical to tag commit
  `5b284c258a4260baef80e3d150b4e7a81a9dbd57`; `libgvhf_md.so` SHA-256 is
  `96000ade4694214b13eb9aedb7de085c843145aedb303cbbf08d69a258c8fe5f`.

## Prepared resident J, not SCF or forces

Slurm 10109 uses the native resident seam, not public compatibility Fock
evaluation. Three fresh processes share one assigned GPU. After preparation and
one warm-up, collect four unprofiled replays per engine; separate invocations
collect one Nsight Systems 2025.1.3 trace per engine. Density/AO unit conversion
uses `benchmarks._md_j_reference.reference_molecule`, including canonical shell
normalization/phases. Native screening is `1e-12`, GPU4PySCF `1e-14`; these
numeric strings do **not** identify equivalent screening/error contracts.

| Engine | Median device seconds | Maximum independent J error |
| --- | ---: | ---: |
| native, explicit legacy `=0` | 4.199605 | 2.445e-11 |
| native, unset tiled default | 4.107982 | 2.445e-11 |
| GPU4PySCF MD-J | 0.712396 | reference |

The tiled median improves 2.18% over legacy but remains 5.77 times GPU4PySCF.
Both native outputs pass `atol=3e-9, rtol=3e-11` against the independent J matrix.
No high-angular CPU fallback is used by GPU4PySCF for this s/p/d basis.

Summed CUDA kernel times from the separate traces:

| Stage | Legacy seconds | Tiled default seconds |
| --- | ---: | ---: |
| residual selection **and contraction** | 2.546919 | 2.450146 |
| uniform reciprocal | 1.017183 | 1.019684 |
| uniform higher-order oriented | 0.642221 | 0.640492 |
| density bounds / transform / projection / other | 0.000620 | 0.000623 |
| total | 4.206943 | 4.110946 |

GPU4PySCF sums to 0.710180 seconds: unrolled MD-J 0.557391, generic MD-J
0.151558, other kernels 0.001231. Native residual `(2,1)` alone takes 0.680100
seconds. The source still constructs public-AO integral components, applies AO
orientation masks and performs density-weighted atomic updates; GPU4PySCF's
density and output are Hermite representations across the integral contraction.

## Fixed-domain no-contraction controls

Slurm 10113 reuses one unchanged legacy owner for physical density, exact zero,
and physical density scaled by `1e-200`. All three check **526,842,030** residual
candidates. Separate counted replays confirm 43,920,982 admitted physical tasks
versus zero tasks/roots for both controls; both control J matrices are exactly
zero. The scaled input retains the physical input's nonzero-entry count.

| Residual kernel sum | Physical | Zero | Scaled nonzero |
| --- | ---: | ---: | ---: |
| seconds | 2.535387 | 0.009439 | 0.010073 |

These are changed-density, changed-branch controls, not an additive physical
enumeration/contraction decomposition. They do show that the candidate count
alone cannot explain the roughly 2.5-second physical residual cost. Prioritize
admitted contraction/representation and reuse before another enumeration-only
optimization; do not label an exact percentage as contraction time from this test.

## Semantic work and limits

Final-library native instrumented work (Slurm 10112):

| Counter | Count |
| --- | ---: |
| uniform radial evaluations | 2,225,962,468 |
| residual primitive products | 242,399,046 |
| residual individual checks | 498,500,700 |
| residual tile-screened candidates | 28,341,330 |
| residual admitted shell tasks | 43,920,982 |

Individual checks plus tile-screened candidates conserve the legacy 526,842,030 domain. Tile
grouping depends on geometry-compaction order, so the checked/rejected partition
can differ between fresh owners; it is not an invariant or a universal 5.88% cut.

A GPU4PySCF **source-replayed**, not instrumented, census (Slurm 10110) predicts
6,948,715 visited microtiles, 1,696,028 density-rejected microtiles and 5,252,687
executed microtiles. These contain 1,304,707,584 recurrence slots, of which
1,304,434,004 have nonzero symmetry/padding factors. All 21 actual J launch names,
grids and block shapes match the replay. Bounded vectorized replay is also
checked against scalar domain/branch/padding loops on 35 synthetic cases.

Native raw radial evaluations/products total 2,468,361,514, about 1.89 times the
replayed GPU4PySCF slots. This includes duplicated oriented higher-order sources
and unequal angular/screening mixes; it does not quantify how much time is due
to screening rather than instruction/data movement. Native divides its density
error allowance over contracted shell pairs; GPU4PySCF uses decontracted pair
bounds, maximum density and a per-interaction log cutoff. Do not weaken the
native aggregate bound or AO masks to make these work counts match.

Slurm 10111's Nsight Compute attempt is blocked by `ERR_NVGPUCTRPERM` for both
engines. No hardware-counter report exists; timeline metadata and source replay
do not establish achieved occupancy, FP64 utilization or bandwidth. Permissions,
drivers and scheduler visibility are unchanged.

## Complete endpoints and qualification

Slurm 10105 uses the same final library and assigned GPU for fresh legacy/default
processes, all J diagnostics disabled. Each completes cold, warm, moved and
moved-warm native energy/force endpoints with one warm replay per geometry.
The retained independent reference keeps all scientific settings and selected
numerical records unchanged; only repeat protocol metadata is projected from
five warm replays to one, retaining an original warm record per geometry.

| Phase | Legacy complete seconds | Default complete seconds | Legacy/default Focks |
| --- | ---: | ---: | ---: |
| cold | 268.291989 | 266.256284 | 24 / 24 |
| warm | 57.433399 | 56.486135 | 1 / 1 |
| moved | 170.116598 | 176.995186 | 13 / 14 |
| moved-warm | 57.535810 | 56.561501 | 1 / 1 |

All independent E/F gates pass, with maximum energy error 2.31922e-11 hartree,
force error 6.93632e-11 hartree/bohr and physical residual RMS 1.48954e-12.
The four-phase sum is 553.377795 seconds legacy versus 556.299106 default,
**0.53% slower**. Moved has an extra Fock, so do not call every phase a speedup.
Slurm 10104/10103 hit 1000-second first-cold timeouts, one per schedule, without
a returned endpoint. They are unresolved, unusable timing samples, not a proven
pre-existing master bug or completed performance cohorts.

Slurm 10102 passes 41 independent resident CUDA cases. Isolated sanitizer runs
10106/10108 pass unsuppressed memcheck and, in the six-run counted legacy/default
matrix, memcheck/racecheck/synccheck with zero errors (racecheck also zero hazards
and warnings). The earlier multi-owner 64-GiB sanitizer API OOM is not a passing
receipt; the cause is not proven by the changed retry scope/memory request.
The n2/Slurm 3435 attempt stops before CUDA testing due to kernel/NVML mismatch.

## Retention and reproduction

Raw reports, unchanged endpoint/failed-arm JSON, reference projection receipts,
input density, independent J, scripts and member checksums remain in the ignored
local `.artifacts/mdj-remote/` tree; original n1 outputs remain under
`/data/jzzeng/gqc-mdj-residual-20261018/`. They are not published release assets.

- Archive: `diagnostic-evidence-20261018.tar.xz`, 7,327,680 bytes, 153 members.
- SHA-256: `5cb7cbfd1e1cfc752edf6757ad7590d2cdbc17339cd6e18481e8de3910933e7d`.
- Manifest: `diagnostic-evidence-manifest.json` contains source/library identity,
  archive hash and each member hash; every restored member is verified. No
  originals are removed and no external host or Release is created.
- Restore the task scripts/input from that archive beside a source-matched build.
  Run `run-compare-j.sh`, `g4-j-census.py`, `native-j-census.py` and
  `enumeration-control.py` only inside finite Slurm allocations as in the
  retained receipts; preserve `CUDA_VISIBLE_DEVICES`. `run-default-cohort.sh`
  records the distinct complete-endpoint protocol. Diagnostic J, source replay,
  sanitizer and clean endpoint samples must stay separate.
