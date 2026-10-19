# Decision: bounded resident residual-J density preprojection stays experimental

Status: implemented (default-off prototype; not promoted)
Date: 2026-10-19

## Problem

The first exact masked-Hermite consumer rebuilt sorted directed plans and
Hermite density suffixes inside shell-pair tasks. Its 96-atom complete-J and
SCF endpoint regressions are recorded in
`2026-10-18-masked-hermite-residual-prototype.md`. The subsequent
`../../proposed/2026-10-18-residual-j-tiled-density-first.md` proposed moving that
work to a reusable scope. This experiment tests geometry-lifetime ordering and
density-replay-lifetime projections, not a full two-dimensional tiled rewrite.

## Decision and ownership

- The compiler emits compact immutable direction permutations, density refresh,
  and a cached specialization of the existing exact masked consumer.
- Native code owns admission, storage, error propagation and launch ordering.
  `GENERATIVEQC_MD_J_MASKED_PREPROJECT=1` is latched at owner creation; unset/0
  retains the incumbent and invalid active values fail preparation.
- Enable only J-only `(2,1)` cached contraction. Keep the old masked selector
  at zero when qualifying this experiment; it independently controls the older
  `(2,1)/(2,2)` prototype.
- Store permutations rather than duplicated AO bounds or density values. The
  refresh reads current alpha/beta density and overwrites suffixes on the owner
  stream before every residual replay, including zero-density replays.
- Reserve metadata and suffixes only inside the remaining original 128 MiB MD
  allowance and request budget. Alternate admission between angular-one and
  angular-two pairs in stable class-local order; skip pairs that do not fit.
  Uncached/invalid pairs and unfavorable masks retain the original AO consumer.
- Allocate the cache in a separate optional transaction after the incumbent MD
  owner. Allocation failure removes only optional cache buffers, not the
  already-prepared MD route. No AO-quartet-sized mask matrix is allocated.

The cached workspace removes task-local suffix construction, but retains the
existing grouped radial consumer, task-specific cutoff selection and bounded
AO fallback. It is not equivalent to GPU4PySCF's shared-memory tiled dataflow.

## Scientific invariants

Keep the original FP64 directional Schwarz multiplication/comparison, including
equality, rather than dividing the threshold. Preserve independent reversed AO
bounds, nonsymmetric alpha/beta density, canonical E transforms, primitive
orientations, diagonal pairs and orbit deduplication. Uniform J, K, density
admission and thresholds do not change. Invalid bounds select fallback before
any publication. Reordered sums are validated numerically, not bitwise.

## Source-matched 96-atom evidence

Measurements use n1 / RTX 5090, CUDA 12.9.1, finite Slurm allocations and the
assigned `CUDA_VISIBLE_DEVICES=0`. Both policies use one isolated library based
on PR #2600 commit `aacbccc275274cb70e36b76a646750fe667acf3e`, preserving compact
E transforms, residual tiles and angular-three reciprocal dispatch. This pins
the comparison source; it is not a statement of the PR's current merge status.

- Library SHA256:
  `89329930149b20e8dc027044736b38292c7f2b5732446b1afd0a554c1fc5201f`.
- Frozen density: 96 atoms, 1088 spherical AOs, r2SCAN-3c basis, screening
  `1e-12`, request device budget 1 GiB.
- Density-array SHA256:
  `a0b22059e12adcd39771e5dcfaac0951afb298d8252ef6674889ad0349ed6090`.
- Independent J: the retained GPU4PySCF result in
  `/data/jzzeng/gqc-mdj-residual-20261018/results/j-only-10083/gpu4pyscf.npy`.
- After warming each prepared owner, run `0,1,1,0,0,1,1,0`, four clean complete-J
  samples per policy. CUDA-event timing includes density refresh and all J
  work. Profiling/work-count replays are separate from clean timing.

| Slurm job | Incumbent J median (s) | Cached J median (s) | Cached time change |
| --- | ---: | ---: | ---: |
| 10256 | 3.9067733154 | 3.9165872803 | +0.2512% |
| 10257 | 3.9290139160 | 3.9242623291 | -0.1209% |

Host-wall medians also change sign: `3.91630 -> 3.92479` s and
`3.93881 -> 3.93252` s. This is no reproducible complete-J benefit; do not select
the favorable repeat as a speedup. These are prepared J-only timings, not SCF
energy/force endpoint timings. Since this stage fails to establish benefit,
this experiment does not proceed to default promotion or a performance PR.
No new cold SCF speed claim is made. Changing compiler source also invalidates
stationary AOT closure identity; any later force qualification must rebuild the
proper artifacts, not reuse or edit the old manifests.

The cache is demonstrably consumed, not merely allocated:

- Uncounted owner bytes: `662672188 -> 672097628` (about 9 MiB extra).
- Diagnostic suffix/metadata bytes: `5220576 + 4202496`; diagnostic counters
  themselves consume part of the same cap, so do not equate this allocation
  exactly with the uncounted owner's allocation.
- Refreshed active primitives: angular one `3365`, angular two `2748`;
  density/E summands `80760 + 361090 = 441850`.
- `(2,1)` cached masked tasks: `129052` of `9929600` residual tasks, about
  `1.30%`; grouped radial summands `112117520`.
- All angular-class uniform roots/directions/summands and admitted residual
  tasks/primitive products agree between the diagnostic policies. Tile probe
  counts can differ because compacted pair order changes tile grouping; they
  are not scientific work counts.
- Maximum independent-J error over these samples is `2.44485e-11`; maximum
  candidate/incumbent difference is below `3.34e-13`. Both pass the explicit
  `atol=3e-9, rtol=3e-11` oracle and `3e-11` incumbent gates.

The measured cache coverage limits this particular implementation. It does not
disprove density-first algebra or establish the hardware bottleneck of a fully
tiled replacement. Previous hardware-counter conclusions concern the older
consumer, not this cached specialization.

## Rejected admission attempts

Full resident suffix coverage exceeds the remaining MD allowance and correctly
falls back (job 10249). A later retry accidentally compiled that same admission
policy again (10252), and is also excluded. The qualification build script
copies staged sources into its source snapshot; updating only that snapshot is
not sufficient. Source/staging equality checks were added to the benchmark.

Angular-one-first admission allocates a cache but leaves no angular-two cache
coverage on this case (10253). Its apparent `0.41%` timing gain is not evidence
for cached contraction: `MD_J_MASKED (2,1)` tasks are zero. The benchmark rejects
it. Balanced admission and a host executable regression now verify that both
classes receive storage before either class can consume the remaining budget.
Do not increase the published MD allowance to hide this resource condition.

## Validation and retention

- Focused host gates: 84 passed, 4 public-library tests deselected; the final
  admission/failure-injection subset was repeated: 18 passed.
- Real CUDA oracle gate: 60 passed; 60 passed under memcheck with zero errors;
  60 passed under racecheck with zero hazards (Slurm 10251). Each case exercises
  six policies, including counted/uncounted cache use, missing-side cache
  fallback and original/zero/original density refresh in one owner.
- Actual pinned-PR molecular normal-J suite: 114 passed with the cache off and
  114 passed with it on (10256), including independent reference checks.
- Compiler structure: 566 modules, zero dependency errors; Ruff lint/format
  checks and `git diff --check` pass.
- Native builds use verified sccache 0.16.0 and retain before/after statistics.
  Both MD CUDA units and every translation unit embedding the modified owner
  layout are rebuilt; the original baseline build is not overwritten.

Ignored local scripts/results are in `build/preproject-hermite/`, especially
`benchmark_96.py`, `run_96.sh`, `run_normal_qualification.sh`, `96-balanced/`,
`96-repeat/`, sanitizer logs, build commands and cache statistics.
Remote results and artifacts remain under
`/home/jzzeng/generativeqc-qualification/preproject-hermite-20261019-49fc/`.
`source-matched-10256.tar.gz` preserves candidate sources, SHA256
`75024a712d3f66a7e279f55af70b86508cfd52630230575569e7292124e08a61`.
No default change, commit, PR, release or driver-configuration change is made.

## Revisit when

Generate bounded shared-memory source strips/2D tiles that cover a meaningful
fraction of `(2,1)` residual work without a molecular-size suffix cache. Share
geometry and density projections across tasks, reduce repeated AO restoration,
and measure the entire J owner including preparation/refresh. Preserve exact
directional masks and the bounded fallback. Only reproducible complete-J wins
followed by independent cold SCF energy/force qualification justify promotion.
