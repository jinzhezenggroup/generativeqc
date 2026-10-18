# README quadrature benchmarks

`readme_dft` and `readme_omol25` accept `--grid RADIAL POLAR AZIMUTH`.
The shell runner also accepts a whitespace-separated `README_BENCHMARK_GRIDS`
list of comma-separated triples. Every explicit triple gets its own
`grid-RxPxA/<method>/<atoms>/{reference,native}.json`, logs and outcomes.
References and accuracy-only receipts never cross grid directories.
Leaving the variable unset retains the existing single-grid output layout.

## What changes with the grid

These are unpruned rational-Legendre radial / polar-Legendre × azimuthal
trapezoidal rules, not PySCF grid levels or Lebedev point counts. The total
number of points is `atoms * RADIAL * POLAR * AZIMUTH` before screening.
The existing `48 16 32` setting means **24,576 points per atom**, not 48 radial
points times a 32-point angular rule. For 96 atoms that is 2,359,296 points.

**Native KS currently shares the quadrature between semilocal XC and VV10.**
Changing this benchmark's triple changes both contributions, including their
analytic grid response. It does not implement independent coarse VV10 and
fine semilocal grids. Giving only GPU4PySCF a coarser `nlcgrids` would compare
different numerical models and is not a supported shortcut.

Raw journals include `grid_work`: dimensions, points per atom, total XC/VV10
points and the unscreened ordered-pair upper bound. The latter is the square
of the full VV10 point count, **not an executed-pair count or a speedup**.
Retain measured SCF/force work and full E+F times as well.

## Run a sweep

Use a source-matched Release library and the GPU4PySCF comparison environment.
Run on n1, n4 or n5 with their 5090 resource; n2 uses `gpu:pro6000:1` and is a
different hardware population. Do not use n3 or change Slurm's device visibility.

```bash
export GENERATIVEQC_LIBRARY=/path/to/source-matched/libgenerativeqc.so
export README_BENCHMARK_OUTPUT=.artifacts/readme-grids/none
export README_BENCHMARK_METHODS=wb97m-v
export README_BENCHMARK_SIZES='3 12'
export README_BENCHMARK_GRIDS='48,12,24 48,16,32 64,16,32'
export README_BENCHMARK_REPEATS=5
export README_BENCHMARK_POINT_TIMEOUT=1200
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=01:00:00 \
  bash benchmarks/run_readme_dft_benchmarks.sh both
```

For a separate DF population, set `README_BENCHMARK_DENSITY_FITTING=cuda` and
`README_BENCHMARK_OUTPUT=.artifacts/readme-grids/cuda`. Both engines use the
retained cc-pVDZ-JKFIT basis and the matched mixed-RSH exchange contract.
The grids above are an exploratory sweep, not universally qualified defaults.

## Separate two accuracy questions

1. **Same-grid engine parity:** each native E+F call must match an independent
   GPU4PySCF calculation of that exact grid (`1e-8 Eh`, `1e-7 Eh/bohr`). A
   missing oracle remains unchecked; shrinking the grid never relaxes these gates.
2. **Quadrature convergence:** compare independent reference E+F results between
   grids on both the original and displaced geometries. For FP64 publication,
   require at least `1e-8 Eh` / `1e-7 Eh/bohr` in this comparison as well; a loose
   exploratory pass cannot qualify FP64. The selected baseline is not an exact integral;
   a baseline's zero self-difference is not evidence that it is converged.

These are separate necessary checks, not a proof of total endpoint accuracy.
Converge radial and angular resolution for **both XC and NLC**, establish a
stable denser reference, and account for combined errors before publication.
The comparator reports requested and FP64 threshold gates separately, even
when explicitly given loose exploratory tolerances. Neither result automatically
qualifies Fast: Fast also needs its configured total energy, maximum-force and
RMS-force budgets, actual native arithmetic and the full benchmark cohort.

```bash
PYTHONPATH=.:python python -m tools.compare_readme_grids \
  --raw .artifacts/readme-grids/none \
  --grids 48,12,24 48,16,32 64,16,32 --baseline-grid 64,16,32 \
  --sizes 3 12 --energy-tolerance 1e-8 --force-tolerance 1e-7 \
  --output .artifacts/readme-grid-report
PYTHONPATH=.:python python -m tools.render_readme_grids \
  --raw .artifacts/readme-grids --grids 48,12,24 48,16,32 --sizes 3 12 \
  --output .artifacts/readme-grid-report
```

The renderer retains scalar samples, original/moved reference force arrays,
SCF work, scheduler metadata and raw hashes in `grid-timings.json.xz`.
Optional per-iteration trajectories and subcomponent timing profiles stay in
the hash-bound raw journals; semantic work and traffic counters are retained.
It plots only complete checked phases; failed or unchecked observations remain
visible in the accompanying table. Different grids are separate series.
The ordinary DFT renderer also accepts `--raw <sweep-root> --grid R P A`.

GPU4PySCF 1.8.1 can omit empty AO tiles in SCF, while its analytic grid response
constructs fresh integrators. The benchmark's scoped iterator adapter preserves
all offsets in both paths and represents an empty tile by exactly zero AOs.
It changes no points, weights, density threshold, XC/VV10 formula or force
mathematics, and restores the response iterator even after an exception.

## Matched publication protocol

The historical exploratory shared-grid study uses original and H(1)-z-displaced (+0.001 bohr) 3/12-atom
water clusters, spherical def2-SVP and WB97M-V/RKS. DF uses cc-pVDZ-JKFIT and the
matched full-range-DF / exact-long-range exchange contract. Both XC and VV10
share the explicit grid in the native path. Radial reductions to 32 points fail
the exploratory force gate; 40×16×32 also fails on 12 atoms. The smaller plotted
48×12×24 rule therefore keeps 48 radial points, but **neither plotted shared
grid passes the FP64 grid thresholds**. Point and pair-bound reductions
are sizing statements, not measured speedups or executed-work counts.

Inside a finite GPU allocation with a source-matched Release library selected,
retain priming separately before fresh-process publication runs:

```bash
export README_BENCHMARK_METHODS=wb97m-v README_BENCHMARK_SIZES='3 12'
export README_BENCHMARK_GRIDS='48,12,24 48,16,32'
for mode in none cuda; do
  export README_BENCHMARK_DENSITY_FITTING="$mode"
  export README_BENCHMARK_REPEATS=1
  export README_BENCHMARK_OUTPUT=".artifacts/readme-grids/priming/$mode"
  bash benchmarks/run_readme_dft_benchmarks.sh both
  export README_BENCHMARK_REPEATS=5
  export README_BENCHMARK_OUTPUT=".artifacts/readme-grids/measurements/$mode"
  bash benchmarks/run_readme_dft_benchmarks.sh both
done
```

The archived reference sweep used 64×16×32 with one replay and exploratory
`1e-5 Eh` / `1e-4 Eh/bohr` gates; retain those observations, not their use as
FP64 acceptance. New FP64 comparisons use the stricter commands above and need
further baseline convergence checks. Same-grid parity remains
`1e-8 Eh` / `1e-7 Eh/bohr`; DF parity is not separate
denser-grid convergence certification. n4's independent CPU-PySCF/GPU4PySCF
grid regression passed 12 paired/open-shell, Direct/DF, LDA/PBE/r²SCAN cases
with CUDA 12.9.1. Its earlier default-toolkit loading failure, initial empty-AO
reference failures and finite-deadline partial runs remain in raw evidence,
not accepted timing populations. These water results do not certify universal
defaults, 48/96 atoms or other molecules/bases.

## Independent XC/VV10 feasibility probe

`benchmarks.readme_split_vv10` is **reference-only**: native SCF and analytic
force owners still share one quadrature. Keep XC fixed while varying NLC;
otherwise a force error cannot be attributed to VV10-grid coarsening alone.
Both moving atomic rules are supplied independently to GPU4PySCF's SCF and
full grid response. Complete timings include preparation, SCF, J/K, both
quadratures, analytic forces, synchronization and host force export. No
reference-only improvement is a measured native speedup.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH=.:python python -m benchmarks.readme_split_vv10 \
  --atoms 3 --xc-grid 48,16,32 --nlc-grid 32,12,24 --repeats 5 \
  --finite-difference-step 0.001 --output .artifacts/split-vv10/3.json
```

Use `--density-fitting cuda` for matched cc-pVDZ-JKFIT DF. Retain priming
separately; measured populations must start new processes. Compare original
and moved geometries with the same XC grid and a denser NLC baseline, keeping
energy/force convergence gates separate from timing. The optional finite
difference checks H(1)-z at two steps; its energy-only diagnostic calls are not
timing samples. Raw records retain both grid identities, all endpoint arrays,
SCF iterations/builds, scheduler visibility and caller hashes. A baseline's
self-difference is never accepted as convergence evidence.

### Measured split-grid decision

The 2026-10-19 feasibility run used GPU4PySCF 1.8.1 / PySCF 2.14.0 on n1's
RTX 5090s: Slurm jobs 10263/10264, each limited to 30 minutes, with distinct
scheduler-assigned devices. XC stayed **48×16×32** for every population.
Direct and matched DF each tested NLC **64×16×32**, **48×16×32**,
**48×12×24**, **32×12×24** and **24×8×16**, on both 3/12-atom geometries.
All 20 populations completed: 240 full E+F calls, excluding separate priming
and finite-difference diagnostics. Each population has one cold, five warm,
one moved and five moved-warm calls; no setup time is subtracted.

**Correction: withdraw the FP64 recommendation of VV10 32×12×24 / XC 48×16×32.**
Against NLC 64×16×32 with the same XC grid, that candidate's maximum errors
over Direct/DF and original/moved 3/12-atom geometries were **6.38e-7 Eh** and
**3.23e-6 Eh/bohr**, passing only the exploratory `1e-5 Eh` / `1e-4 Eh/bohr` gates.
They exceed the FP64 energy/maximum-force thresholds by about **64× / 32×**.
These measure NLC-only differences, not convergence of the fixed XC grid.
Even NLC48×16×32 versus NLC64×16×32 has 12-atom errors up to
`7.63e-8 Eh` / `3.86e-7 Eh/bohr`, failing both FP64 thresholds. NLC64's
self-comparison cannot qualify it either. No tested grid is certified as a
complete FP64 endpoint; do not replace the README cohort with these timings.
Two-step H(1)-z finite differences at 0.001/0.0005 bohr on 3 atoms gave errors
below `1.01e-8 Eh/bohr` in Direct and DF; this is not an all-coordinate Hessian
or large-system response qualification, and does not establish grid convergence.

Complete endpoint seconds below compare **NLC 48×16×32 → 32×12×24**, with XC
unchanged. Warm entries are medians of five calls; cold/moved are single calls.

| Mode / atoms | Cold | Warm | Moved | Moved-warm |
| --- | ---: | ---: | ---: | ---: |
| Direct / 3 | 21.079 → 17.639 | 2.114 → 1.731 | 21.422 → 17.138 | 2.432 → 1.771 |
| DF / 3 | 21.608 → 16.900 | 1.768 → 1.325 | 20.625 → 16.465 | 1.741 → 1.380 |
| Direct / 12 | 102.527 → 39.562 | 8.498 → 3.787 | 92.849 → 32.823 | 8.560 → 3.030 |
| DF / 12 | 100.671 → 36.219 | 7.938 → 2.688 | 90.863 → 31.073 | 7.881 → 2.583 |

The 12-atom warm reference speedups are **2.24× Direct / 2.95× DF**. Actual
SCF work is not reduced to fabricate this comparison: cold iterations/builds
are 32/33 (3 atoms) and 45/46 (12); moved are 39/40 and 42/43. Warm calls use
1/2 except original 12-atom Direct at NLC32, which uses 2/3 versus 1/2 at
NLC48. Every population's phase work sets are retained in the archive.

All tested smaller NLC rules pass only the exploratory small-system gates; NLC24's
12-atom energy error reaches `6.64e-6 Eh`, leaving much less margin; do not
promote it from timing alone. Coarsening is not monotonically faster on
3 atoms. NLC32 is at most an **unqualified Fast candidate**; the probe has no
native split implementation, total error-budget acceptance or complete 48/96
coverage. It has 9,216 NLC points/atom versus 24,576 XC points/atom;
the 62.5% NLC-point reduction is sizing, not executed work or a native speedup.

The publication's `grid-timings.json.xz` contains a separate
`split_grid_feasibility` summary: all 20 phase medians (rounded to 5 decimals),
actual SCF work sets, both-geometry NLC errors, finite-difference diagnostics,
reference versions, caller hashes and scheduler visibility. Its column-labelled
rows bind full raw endpoint arrays/times to SHA-256 journals under
`n1:/data/jzzeng/readme-split-vv10-20261019` (also ignored locally). The dense
baseline has no self-convergence pass. These reference observations do not
change the original shared-grid DF convergence scope.

Native SCF, the C ABI, prepared plans and analytic-force owners still assume
one quadrature. A production split needs separate grid/AO ownership, model and
cache identities, resource/work accounting, final-density handoff and moving-grid
response. First converge XC and NLC against a stable denser reference under the
intended precision contract, then qualify native same-model parity, complete E+F
and 3/12/48/96 atoms. Keep all eight original HF/DFT methods, both Direct/DF and
independent GPU4PySCF in the rerun. Neither **48×12×24 shared** nor
**32×12×24 split NLC** is an accepted FP64 replacement. Defaults stay unchanged
pending evidence; retaining them is not a new convergence certification.

## Profile-aware accuracy study

Use `tools.analyze_vv10_grid_study` to compare FP64 and Fast budgets without
turning reference-only observations into native precision certification. It
requires successful process outcomes, original/moved force arrays, radial and
angular anchor checks, and replay consistency. By default 10% of each budget
is reserved for the empirical reference-refinement allowance; it is not a
rigorous bound on the infinite-grid limit. Fast checks RMS forces as well as
energy and maximum-force errors. An incomplete or unstable anchor leaves grid
admission unknown or failed even if the observed candidate difference is small.

For example, after an XC-only sweep with NLC fixed at 64×16×32:

```bash
PYTHONPATH=.:python python -m tools.analyze_vv10_grid_study \
  --raw .artifacts/vv10-grid/precision-study/xc-scan --axis xc \
  --fixed-grid 64,16,32 --anchor 144,48,96 \
  --radial-check 120,48,96 --angular-check 144,40,80 \
  --output .artifacts/vv10-grid/precision-study/xc-analysis.json
```

These dimensions are a study example, **not a recommendation**: the first
three-atom anchor fails the reserved FP64 stability allowance and requires
further refinement. An XC-only Fast pass still needs NLC convergence, combined
error qualification, actual native Fast/fallback validation and 48/96-atom
coverage. The [ongoing study rationale](../.agents/notes/proposed/2026-10-19-vv10-profile-grid-qualification.md)
records these separate requirements. Accuracy studies may use n2's PRO 6000,
but benchmark timing updates remain a separate RTX 5090 population.

For expensive dense accuracy sweeps, `benchmarks.vv10_grid_accuracy_sweep`
accepts `--grid-pairs XC_R,P,A:NLC_R,P,A ...`. It may reuse **only its own
GPU4PySCF density** between neighboring rules, reconverging every target grid
and both geometries. Its `accuracy-original` / `accuracy-moved` records are
explicitly not cold/warm timing populations. Keep this optimization out of the
fresh-process README timing runs; it does not justify subtracting setup or SCF
work from published endpoint latency.

For `readme_dft --policy fast`, the application policy selects native solver
controls from the configured budget; it is no longer pinned to the FP64
benchmark's explicit convergence ceilings. FP64 and independent-reference
controls remain unchanged. Journals record requested/initial candidate controls
and actual per-call refinement/fallback work separately. Earlier Fast journals
without this distinction are tight-control populations, not interchangeable
with default-control Fast timings.
