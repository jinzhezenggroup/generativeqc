# Cross-functional occupied-XC endpoint survey

This survey asks whether the recovered generic initializer benefits functionals
other than r2SCAN. **Yes for a qualified M06-2X population, but not universally:**
MN15 and a small M06-2X UKS population regress despite accurate final states.
No production selector, scientific tolerance, force provider or functional
capability is changed by this survey.

## Immutable provenance and controls

All native observations use the same previously qualified image:

- Scientific source: `ddadce1e29453e0f5cabc4c12948c7c798e8d66c`.
- Library SHA-256:
  `be126dbe2567668cfb7e747177d71fd811f8f1c7d832e608f7aed5f0ca9adb92`.
- Native source identity:
  `739538b11aec9053c6a8e2bd526f441f907f0d93ffc3d8824b1c4292b2583d99`.
- NVIDIA RTX 5090, Slurm `node1/main`, finite scheduler allocations on
  November 5, 2026. Each receipt retains its allocation and assigned visibility.
- Controls are **actual absent** `GENERATIVEQC_CUDA_KS_OCCUPIED_XC` versus
  exact `0`, alternating order in fresh processes. Traced diagnostics never
  contribute timing observations.

The delivered DFT owner/producer files at
`6ba9f19af11f8d7e10db28149c6530d8628b1c20` are byte-identical to the measured
source under `src/dft` and `python/generativeqc_compiler/dft`. This is not a
qualification of the complete merged native image. Later master admission,
stopping and warm-anchor changes, as well as integrated integral/force producers,
are not relabelled with these measurements.

## Complete M06-2X RKS energy plus force

The existing `benchmarks.readme_defaults` Standard/Direct protocol uses the
established water series, spherical def2-SVP, unchanged native grid and solver,
complete analytic forces and host export. Cold includes construction,
preparation and SCF; imports and CUDA context setup precede its timer. Each cold
solve has a held-state warm solve. Every cold observation takes **10 iterations /
10 Fock builds**, and every warm observation takes **1 / 1**, for both controls.

| Population | Matrix cold median | Default cold median | Less time | Cost floor |
| --- | ---: | ---: | ---: | ---: |
| 48 atoms, discovery, three pairs | 9.986051 s | 9.713744 s | **2.727%** | 2.000% |
| 48 atoms, independent confirmation, three pairs | 10.257112 s | 9.944520 s | **3.048%** | 2.130% |
| 96 atoms, three pairs | 20.694219 s | 20.425110 s | 1.300% | 2.000% |

The fixed floor is `max(2%, 2 * sum(relative MADs))`. Both independent 48-atom
cohorts clear it. The cohorts have different baseline timings and are deliberately
**not pooled** into a more attractive single percentage. There is no 96-atom or
warm speedup claim. Warm does not regress beyond the floor in either 48-atom
cohort or at 96 atoms.

All 36 complete E+F observations pass the existing independent dense-grid
GPU4PySCF references, CPU/GPU small-system spot contract, and audited reference
reuse rules in `readme-final-20261022`. Every paired vector passes unchanged
`2e-9 Eh` energy and `1e-7 Eh/Bohr` force gates. Complete vectors, actual settings,
source identity, environment and semantic work are retained losslessly.

## Other outcomes, including regressions

The new `benchmarks.occupied_xc_functionals` runner retains another **168 complete
untraced E+F observations**: M06-2X/MN15 small RKS/UKS, 12-atom RKS and 48-atom
MN15 RKS, each with cold/warm/displaced/displaced-warm phases and three
alternating pairs. These timings are compared only within their own driver and
settings; unlike the README driver, this runner includes first native CUDA
initialization in cold time.

- **MN15, 48 atoms:** cold `16.934485 -> 17.332532 s`, **2.351% slower**;
  displaced `12.780367 -> 13.408046 s`, **4.911% slower**. Both exceed the 2%
  floor. Cold Focks increase **16 -> 17**, displaced Focks **13 -> 14**.
  Warm changes remain below the floor. All independent and tight paired
  numerical gates pass; accuracy is not profitability.
- **M06-2X UKS, three atoms, tight solver:** cold is **12.055% slower**, with
  Focks **24 -> 25**; displaced is **9.976% slower**, with Focks **18 -> 20**.
  Independent and paired energy/force gates pass.
- **M06-2X RKS, 12 atoms, default solver:** independent common observable
  gates pass, but the tighter paired force gate fails at
  `1.2496081129853565e-6 Eh/Bohr`, above `1e-7`. This population is retained
  as **unqualified**, not rescued by changing SCF settings or weakening gates.
- Other small-system changes do not establish a new profitable domain. The
  capsule preserves their original timings, noise floors, work and vectors.

Small references are independently converged PySCF RKS/UKS moving-grid forces.
48-atom MN15 has fresh full-density GPU4PySCF references at grids 5 and 6, both
geometries, with the existing `1e-11 Eh` / `1e-7` orbital-gradient reference
controls. Grid stability passes half of the common observable limits; every
native vector passes both grids. Independent small CPU/GPU spot vectors also
pass the tight paired gates. Initial grid-3/incremental large-reference attempts
did not converge and supply **no acceptance data**; the retained successful
generator uses the existing full-Fock reference protocol instead.

Four separately traced RKS/UKS diagnostics retain another 16 complete E+F
vectors and prove that both functional families lend occupied frames followed
by a strict matrix tail. Production CPU derivative consumers are forbidden in
the new runner. In total the capsule retains **204 untraced complete native
E+F observations**, plus the 16 traced diagnostics. No new sanitizer or broad
Libxc/public-functional qualification is claimed.

## Reproduction

Offline replay requires NumPy for the existing dense-reference qualifier, but
does not import a native library, CUDA, PySCF or GPU4PySCF:

```bash
PYTHONPATH=.:python python benchmarks/results/occupied-xc-functionals-20261105/replay.py
PYTHONPATH=.:python python -O benchmarks/results/occupied-xc-functionals-20261105/replay.py
```

The reader rechecks images, controls, full independent/paired vectors, reference
checksums/reuse, both M06-2X replications, work, timing floors, MN15 grid stability
and retained rejections. Corrupt source, control, energy, force or missing
observations must fail even with optimized Python.

For fresh execution, build the pinned native scientific source with a verified
compiler cache, and run this change's benchmark drivers against that image.
Keep raw outputs in `.artifacts/benchmarks/`. A small CPU reference can be made
without a GPU; all native and GPU4PySCF execution must use finite `srun` jobs:

```bash
PYTHONPATH=.:python python -m benchmarks.occupied_xc_functionals oracle \
  --method mn15 --spin uks --output .artifacts/benchmarks/mn15-oracle.json.xz
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  python -m benchmarks.occupied_xc_functionals run --method mn15 --spin uks \
  --strict-solver --source-commit ddadce1e29453e0f5cabc4c12948c7c798e8d66c \
  --oracle .artifacts/benchmarks/mn15-oracle.json.xz \
  --output .artifacts/benchmarks/mn15-default.json.xz
```

Set `PYTHONPATH=.:python:tests/python`, `GENERATIVEQC_LIBRARY` and the matching
native runtime library search path before native execution. Repeat with actual
unset controls and exact `0` in alternating fresh processes at least three
times. The generic `replay` subcommand accepts those receipt paths. Use the
existing README driver for the M06-2X cold/warm populations. SCAN/RSCAN in the
new runner still require matching optional native point/force providers;
accepting a method name does not create or qualify those capabilities.
