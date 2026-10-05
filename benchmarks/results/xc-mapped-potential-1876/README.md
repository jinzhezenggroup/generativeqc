# Actual local-AO XC potential crossover

Generated execution wins every measured pair. At the production tile 256,
the indexed DSYR2K candidate makes the complete fixed-density XC endpoint
24.6%/21.2% slower for 384/768 AOs. Production remains generated. These are
XC component endpoints, not PBE0 SCF/force speedups.

| AOs | tile | original generated / rank-2k (s) | rank-2k time change, original / moved |
| --- | --- | --- | --- |
| 384 | 128 | 1.6476 / 1.9203 | +16.55% / +16.94% |
| 384 | 256 | 1.0809 / 1.3468 | +24.60% / +24.70% |
| 384 | 512 | 0.8000 / 1.0573 | +32.15% / +32.31% |
| 768 | 128 | 3.4225 / 3.9418 | +15.17% / +16.65% |
| 768 | 256 | 2.3539 / 2.8528 | +21.19% / +21.19% |
| 768 | 512 | 1.8155 / 2.2847 | +25.84% / +26.36% |

The four original/moved inputs come from retained commit
`b080c9bdcbf81aa952dee811bfd2dbb5a4162981`, with compressed input hashes recorded
in `provenance.json`. They contain 48/96 water atoms and spherical def2-SVP;
the benchmark uses the full v1 48×16×32 per-atom quadrature, FP64, local AO cutoff
`1e-16`, and PBE exchange/correlation scales 0.75/1.0. The positive diagnostic
density is `density(nao, 1)` from the native XC test, not a converged SCF density.

Each sample includes density H2D, AO/density/functional work, weighted panels,
potential assembly and point totals, synchronization and E/V downloads. The
candidate additionally includes compact rank-2k output and mapped scatter.
Basis/grid setup, map discovery and provider preparation are timed separately.
Both route owners are prepared before timing. Six samples per route retain the
first evaluation as cold and use the median of the remaining five as warm;
route order alternates, including after the geometry change. There is no CPU
oracle work inside the timed path and no graph capture in this benchmark.

All 144 evaluations pass paired E/V/population gates. The maximum potential
difference is `8.88e-16`; energy and population differences are zero. This
large-system comparison uses the generated route, not an independent oracle.
The indexed candidate's independent long-double/CPU, capture, resource-failure
and sanitizer qualification belongs to #1963; its native gate passes again in
this benchmark binary before timing.

Both geometries exactly reproduce the retained tile-256 AO census. The two
providers execute identical local maps and symmetric scalar summands. Original
384/768-AO tile-256 work is `25,416,412,160 / 76,047,980,544` summands per XC
evaluation. Compact triangle writes add `0.397 / 1.188` GB; scatter counts
`1.589 / 4.753` GB of FP64 accesses and `0.794 / 2.376` GB of index accesses.
These are logical access counts, not measured DRAM traffic: each triangle
entry loads compact output and the destination, writes both symmetric entries
(including the duplicate diagonal store), and loads two indices. The retained
cache is charged at global AO capacity (`1,179,648 / 4,718,592` bytes), plus the
separate 96 MiB provider allowance and 16 KiB host reservation.

The dense 768-AO crossover from #1958 therefore does not qualify this indexed
materialization. This receipt measures total endpoint cost; it does not isolate
which kernel causes the regression. Tile-size changes also alter AO domains and
need independent composed SCF/force qualification before a default change.

`verify.py` reconstructs `summary.json`, checks every paired record/sample and
work/traffic count, and matches both tile-256 geometries to the hash-bound
composed baseline. `provenance.json` records source, binary and dataset hashes,
actual ccache compiler commands and native qualification identity. The full
source manifest and raw logs remain ignored in `.artifacts/1876-mapped-potential-crossover/`.
Ccache before/after snapshots are shared-cache statistics, not an isolated hit
rate. All native execution used n1/node1 through finite Slurm `main/gpu:5090:1`.

```bash
python benchmarks/results/xc-mapped-potential-1876/verify.py
python benchmarks/results/xc-mapped-potential-1876/prepare-inputs.py .artifacts/1876-mapped-potential-crossover/inputs
# The recorded qualify.sh.txt expects those hash-verified compressed inputs.
# Run the benchmark only inside a finite GPU Slurm allocation:
gzip -dc .artifacts/1876-mapped-potential-crossover/inputs/input-48-0.txt.gz > /tmp/xc-original.txt
gzip -dc .artifacts/1876-mapped-potential-crossover/inputs/input-48-1.txt.gz > /tmp/xc-moved.txt
srun --partition=main --gres=gpu:5090:1 --nodelist=node1 --nodes=1 --ntasks=1 --time=00:20:00 build/cuda-release-sm120/generativeqc_dft_cuda_tests --indexed-potential-benchmark /tmp/xc-original.txt /tmp/xc-moved.txt
```
