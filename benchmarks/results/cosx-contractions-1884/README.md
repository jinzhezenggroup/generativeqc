# COSX projection/accumulation endpoint crossover

This receipt qualifies independent projection and exchange-update routes at
384/768 AOs. The four masks are generated, projection GEMM only, accumulation
GEMM only, and both GEMMs. Production selection stays generated.

Across all paired warm medians, the three library combinations change complete
endpoint time by **−0.054% to +0.108%**. This does not establish a useful
production crossover. All 192 samples pass; maximum raw/symmetric K error is
`3.33e-16`, energy error `8.88e-16`. The independent large-AO CPU subset has
maximum K/energy errors `2.78e-17 / 8.67e-19`.

| AOs | tile | geometry | generated (s) | both GEMMs (s) | time change |
| --- | --- | --- | --- | --- | --- |
| 384 | 64 | original | 1.671964 | 1.673777 | +0.108% |
| 384 | 64 | moved | 1.679716 | 1.681260 | +0.092% |
| 384 | 256 | original | 1.654142 | 1.654024 | -0.007% |
| 384 | 256 | moved | 1.653384 | 1.653567 | +0.011% |
| 768 | 64 | original | 11.542422 | 11.540475 | -0.017% |
| 768 | 64 | moved | 11.537369 | 11.534561 | -0.024% |
| 768 | 256 | original | 12.083851 | 12.077388 | -0.053% |
| 768 | 256 | moved | 12.082668 | 12.076133 | -0.054% |

The full receipt retains the independent projection-only and accumulation-only
measurements too. Each shape/tile case used its own GPU allocation; these paired
route comparisons do not establish a tile-size promotion.

Inputs are retained original/moved 48/96-atom water geometries with spherical
def2-SVP. Input blob hashes and the recovery commit are in `provenance.json`;
`prepare-inputs.py` reconstructs them without external hosting. All routes use
the identical unscreened, unfitted COSX v1 model, symmetrized exchange, RHF
spin-summed convention, strict FP64, and positive diagnostic density
`D[i,j] = 0.1*delta(i,j) + 0.001/(1+i+j)`. This is not a converged SCF density.

The explicit **3×3×6 per-atom diagnostic quadrature** uses three partition
iterations and coincident tolerance `1e-12`. It produces 2,592/5,184 points.
This coarse quadrature is deliberately identified separately from production
PBE0/COSX quadrature; neither SCF accuracy nor complete SCF/force performance
is qualified. Both 384-AO cases and the 768-AO tile-256 case exercise a partial final tile;
the 768-AO tile-64 case checks an absent tail with zero tail execution counts.

Every timed build includes density upload, all AO and ESP evaluations, weighted
ESP application, projection/accumulation, symmetrization, matrix downloads,
synchronization and host exchange-energy contraction. Six samples per route
rotate execution order. The first sample is **first replay of a prepared plan**;
the median of the other five is warm. Geometry/basis/grid setup, plan setup and
provider preparation are separate fields. CPU subset qualification precedes
measurement, so these are not process-cold CUDA/library initialization times.
Changed geometry reconstructs all owners and quadrature. There is no graph
capture, packing, matrix reuse across builds, or oracle work inside timing.

Three spread-out grid points independently check the CPU COSX oracle at the
actual AO dimension, with tile 2 exercising full and tail. Every measured
full-grid sample compares raw K, symmetric K and energy against generated
execution, under `1e-9 + 1e-11*abs(reference)` gates. The receipt verifier further
checks that every retained absolute error is below `1e-9`. The test binary also
passes `--contractions`: independent long-double matrix arithmetic, asymmetric
CPU E/K cases, finite/error, capture rejection and bounded fallback gates.
Full production lifetime/regression and sanitizer evidence is retained in #1966;
this follow-up changes only the benchmark harness.

Each route has the same admissible budget. Generated execution reserves no
provider allowance; any GEMM route shares one 96 MiB allowance, not one per site.
The four-site host reservation is 128 KiB. `sites` retain scientific/semantic/
precision and selected-candidate identities, dimensions, all offers/rejections,
actual calls and scalar summands. `summary.json` additionally derives the number
of unique ESP integrals and materialized ESP elements from the unchanged domain;
those derived counts are not profiler observations or measured memory traffic.

`verify.py` checks every retained sample, resource delta, provider and full/tail
work count and reconstructs `summary.json`. Compiler commands, ccache before/
after snapshots, source/binary/input/data hashes and Slurm identities make the
receipt reproducible. Cache statistics refer to the shared cache, not an isolated
hit rate. Raw logs and full source manifests remain in ignored
`.artifacts/1884-cosx-crossover/`. All real-device execution uses finite Slurm
`main/gpu:5090:1` allocations on n1/node1; no visibility override is used.
Different cases ran on independently allocated GPUs on the same node.

```bash
python benchmarks/results/cosx-contractions-1884/verify.py
python benchmarks/results/cosx-contractions-1884/prepare-inputs.py .artifacts/1884-cosx-crossover/inputs
# Build and qualification recipes are retained as build.sh.txt / qualify.sh.txt.
# The qualification script accepts atoms, tile, radial, polar and azimuth:
srun --partition=main --gres=gpu:5090:1 --nodelist=node1 --nodes=1 --ntasks=1 --cpus-per-task=4 --time=00:20:00 bash .artifacts/1884-cosx-crossover/qualify.sh 96 256 3 3 6
```
