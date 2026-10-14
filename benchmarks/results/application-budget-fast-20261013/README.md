# Budget-selected FP32 XC: complete PBE0 endpoints

This is genuine FP32 **XC warmup**, not the exact FP64 source-reuse optimization
in PR [#2398](https://github.com/jinzhezenggroup/generativeqc/pull/2398).
The comparator already includes that optimization. J/K and all analytic forces
remain FP64. The measured gains are modest, not a 2–7x endpoint improvement.

## Accepted comparison

Slurm job **8030**, node1 RTX 5090, assigned visibility `1`, Release/sm_120,
CUDA 12.9.1, fast-compile disabled. Each mode/case/seed has one retained first
call and five alternating-order measured repetitions: **96 complete endpoints**,
including 16 first calls. Timings include fresh construction, native preparation,
SCF/refinement, all analytic forces, host export, cache release and synchronization.
Process caches are warm; no converged density is reused between timed endpoints.

PBE0/RKS, spherical def2-SVP, exact full-range J/K, no DF, moving grid `(48,16,32)`,
energy/density/screening tolerances `1e-10 / 1e-8 / 1e-12`, at most 180 iterations.
Both timed modes receive the same initial-guess policy. The automatic policy uses
the native strict MINAO seed; its preparation cost is included. Independent CPU
PySCF SCF and full moving-grid analytic forces use Hcore and never supply native
densities, integrals or gradients. Both native and oracle protocols are retained.

| Initial guess | System | Optimized FP64 (s) | Fast (s) | FP64/Fast |
| --- | --- | ---: | ---: | ---: |
| Hcore | Water, 24 AOs | 1.889413 | 1.787591 | 1.0570 |
| Hcore | Water tetramer, 96 AOs | 3.522350 | 3.457885 | 1.0186 |
| Hcore | Ammonia, 29 AOs | 2.090117 | 1.936046 | 1.0796 |
| Hcore | Water octamer S4, 192 AOs | 8.603568 | 8.307334 | 1.0357 |
| Automatic MINAO | Water | 1.807437 | 1.746586 | 1.0348 |
| Automatic MINAO | Water tetramer | 3.396368 | 3.323361 | 1.0220 |
| Automatic MINAO | Ammonia | 1.970700 | 1.877602 | 1.0496 |
| Automatic MINAO | Water octamer S4 | 7.955756 | 7.893785 | 1.0079 |

These are medians, not confidence bounds. The automatic-seed octamer is effectively
near parity; a sub-percent ratio is not a meaningful demonstrated speed benefit.
The raw comparison samples, including outliers and first calls, are in
[`summary.json.xz`](summary.json.xz). No sample is selected by iteration count.

All 96 endpoints pass independent gates of `1e-7 Eh` and `1e-6 Eh/bohr` maximum
force component. Largest errors are `3.070e-12 Eh` and `3.752e-10 Eh/bohr`.
Small final errors reflect strict final SCF/forces; they do not measure the error
of an FP32 gradient, since no such gradient runs here. Basis/grid/model error,
electronic-root/stability and trajectory behavior are outside these gates.

## Work and actual precision

Every accepted Fast sample reports one solve, one force evaluation and
`qualified_strict_final`, not an invented per-call observable audit.

| Seed | System | FP64 Focks | Fast mixed + strict Focks |
| --- | --- | ---: | ---: |
| Hcore | Water | 13 | 9 + 4 |
| Hcore | Tetramer | 14 | 10 + 6 |
| Hcore | Ammonia | 14 | 10 + 4 |
| Hcore | Octamer | 17 | 10 + 9 |
| Automatic MINAO | Water | 10 | 6 + 4 |
| Automatic MINAO | Tetramer | 12 | 6 + 7 |
| Automatic MINAO | Ammonia | 11 | 6 + 5 |
| Automatic MINAO | Octamer | 14 | 7 + 9 |

Mixed XC uses FP32 AO arithmetic, PBE correlation points, tiled matrix products
**and inner sums**, plus local point totals. Global storage/scatter, normalization,
exchange, unqualified tails/small contractions, J/K and gradients retain FP64.
The census counts composite operator submissions, not FP32 instructions or point
fallback fractions. Do not interpret the effective-bit summary as all-FP32.

The extra refinement work is a real performance cost. A separate earlier-handoff
screen (job 8028, thresholds `1e-2` and `1e-3`) improved the Hcore octamer but made
the tetramer slower than FP64 with eighteen iterations. Those variants were not
promoted. Point-only and point-plus-AO negative ablations are retained locally.

## Source and validation provenance

- Base: `da7185262dd77c67d16dd72614b96a30e315bebf` plus the changed-source
  manifest in `summary.json.xz`. Source/library identities are recorded per cohort.
- Native source identity:
  `a514e417d6ae21b91fde604baaaf2092db62f13f8bd17cb35a373bf556e858c4`.
- Library SHA-256:
  `6cf7a535f35e7579ea56d9efa34b777436a1d732587cf8eeeee08e444b9bdf60`.
- Ignored reconstructible runtime patch SHA-256:
  `8916ed73b1e0a2df63f6a90fe16c79c47d1a0931e9b253c8a044111dc064624c`.
- Public CUDA integration: **10 passed**, job 8030. Covers actual mixed admission,
  explicit/default/Hcore seeds, tight-budget/domain audits, explicit target audit,
  two-policy isolation, retained-plan replay, changed geometry, release/rebuild,
  legacy AUTO admission and malformed/truncated private descriptors.
- Earlier same-XC regression build: **244 passed**, job 8027; host/controller/
  generator/API suite: **158 passed, 72 subtests**. Compiler and shared-SCF
  dependency checks and warnings-as-errors Sphinx build pass.
- Actual CUDA point/contraction probe memcheck: **21 passed, zero errors**,
  job 8019. Independent Libxc/NumPy oracles check arithmetic and fallback domains.
- Full molecular endpoint memcheck: **10 passed, zero errors**, job **8031**, on
  the final library hash above. Exact provenance is retained in the ignored
  `source-matched-memcheck.log`; instrumented times are not speed samples.

Full diagnostics, every failed attempt, profiler/build logs, the source patch,
oracle files and negative ablations stay under
`.artifacts/application-budget-fast-20261013/` and on node1 under
`/data/jzzeng/pbe0-fast-20261013/`. The compact tracked record binds all four raw
accepted JSON files and both oracle records by SHA-256. No archive/Release is
published. Five unrelated DF provider files were initially stale in the remote
tree; they were aligned with the worktree before this accepted campaign. A
preserved-timestamp Ninja stale-object failure was then fixed by refreshing those
dependency timestamps with the existing ccache, not by changing scientific code
or disabling the cache. Failed pre-alignment/library-load runs remain separate.

## Reproduction

Use `benchmarks/pbe0_precision.py --engine reference` for water/tetramer reference
generation. Additional references use its `reference(case, grid, threads)` helper
for `ammonia-def2-svp-spherical` and `water-octamer-s4-def2-svp-spherical`; the
retained `extra_oracles.py` records that invocation. PySCF is benchmark-only.

With source `python/` and the repository on `PYTHONPATH`, and the matched Release
library selected by `GENERATIVEQC_LIBRARY`, run through a finite Slurm allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:15:00 \
  python -m benchmarks.pbe0_fast --reference REFERENCE.json \
  --output .artifacts/fast-comparison.json --repeats 5 \
  --initial-guess automatic
```

Repeat with `--initial-guess hcore`; use the extra reference file and `--cases`
for held-out molecules. Preserve assigned visibility and leave experimental
precision/residency overrides unset. Application budgets do not authorize those
overrides, looser final convergence, or omission of requested audits.

See the current [application guide](../../../docs/user/application_precision.md)
and the [decision note](../../../.agents/notes/implemented/numerics/2026-10-13-budget-selected-fp32-xc.md).
