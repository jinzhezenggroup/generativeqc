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
   grids on both the original and displaced geometries. Specify the application
   tolerances explicitly. The selected denser baseline is not an exact integral;
   a baseline's zero self-difference is not evidence that it is converged.

```bash
PYTHONPATH=.:python python -m tools.compare_readme_grids \
  --raw .artifacts/readme-grids/none \
  --grids 48,12,24 48,16,32 64,16,32 --baseline-grid 64,16,32 \
  --sizes 3 12 --energy-tolerance 1e-5 --force-tolerance 1e-4 \
  --output .artifacts/readme-grid-report
PYTHONPATH=.:python python -m tools.render_readme_grids \
  --raw .artifacts/readme-grids --grids 48,12,24 48,16,32 --sizes 3 12 \
  --output .artifacts/readme-grid-report
```

The renderer retains scalar samples, original/moved reference force arrays,
SCF work, scheduler metadata and raw hashes in `grid-timings.json.xz`.
It plots only complete checked phases; failed or unchecked observations remain
visible in the accompanying table. Different grids are separate series.
The ordinary DFT renderer also accepts `--raw <sweep-root> --grid R P A`.

GPU4PySCF 1.8.1 can omit empty AO tiles in SCF, while its analytic grid response
constructs fresh integrators. The benchmark's scoped iterator adapter preserves
all offsets in both paths and represents an empty tile by exactly zero AOs.
It changes no points, weights, density threshold, XC/VV10 formula or force
mathematics, and restores the response iterator even after an exception.
