# PBE0 derivative work observations

These intrusive diagnostics identify work in the exact-direct J/K derivative
owner for the retained 48/96-atom water inputs. They do **not** qualify a default,
measure complete energy-plus-force endpoints, or establish GPU4PySCF parity.
The measured source is the composed public-force-policy control, revision
`9ba032c783addfeede96890c894b7cc9447cde95`; its source and binary identities are
retained in `identities.json`. This is not an observation of shipping master.

## What was measured

Slurm 5857 solves native PBE0 at original and moved geometries, then creates a
separate derivative-capable owner and uploads the final density. It executes
the public full-range J/K derivative path and replays the same bounded kernel
with its existing optional class observer. J and K source coefficients are
`1` and `-0.25`. Both sources share the primitive recurrence, so the derivative
ledger is their union, not separate J-prime/K-prime counts. The source-major
force-versus-energy-gradient sign is checked explicitly at `1e-9`.

The observer adds 1,760 device bytes. Shell-quartet admissions and tile counts
are observed at shell admission. AO/primitive quartet counts are **upper bounds
before inner AO screening and zero-weight gates**, not executed recurrences.
The native SCF portion also retains separate actual J/K value task admissions
for every iteration. Those describe this diagnostic solve, not another run's
solver trajectory. SCF uses the existing local AO selection and strict full
density builds, with energy/density tolerances `1e-12`/`1e-10`, screening `1e-12`,
tile 256, and the frozen grid/basis inputs in `drivers/`.

Slurm 5859 additionally brackets a thirteen-pass angular replay with the CUDA
profiler API. Each pass repeats traversal and has fixed-order compilation;
register use, occupancy, launch cost, and dispatch therefore differ from the
mixed-order production kernel. These times are **neither production time
shares nor rigorous upper bounds on them**. The historical driver comment
calling the times upper bounds should be read with this correction. Orders
9–12 have no admitted classes, but still incur traversal overhead. Nothing is
subtracted from the measurements to estimate a production speedup. Nsight's
event-tracing overhead/false-dependency warnings are preserved in receipts.

## Observations

The original 96-atom census is:

| Total angular order | Shell admissions | AO quartet upper bound | Primitive quartet upper bound | Thirteen-pass kernel seconds |
| --- | ---: | ---: | ---: | ---: |
| 0 | 13,647,876 | 13,647,876 | 165,634,524 | 0.284520 |
| 1 | 27,796,096 | 83,388,288 | 744,081,936 | 0.509912 |
| 2 | 27,137,980 | 224,980,116 | 1,483,886,892 | 0.897200 |
| 3 | 15,823,664 | 345,054,564 | 1,704,263,436 | 0.826670 |
| 4 | 5,998,624 | 323,428,704 | 1,186,689,216 | 4.186318 |
| 5 | 1,536,020 | 190,606,032 | 507,813,552 | 3.484197 |
| 6 | 261,576 | 66,665,952 | 131,771,232 | 1.980489 |
| 7 | 29,328 | 14,027,904 | 19,115,136 | 1.660860 |
| 8 | 2,064 | 1,387,344 | 1,387,344 | 0.548184 |

The counts and timings above come from separate solves/jobs. Their side-by-side
display is diagnostic context, not a throughput denominator. The full 48/96,
original/moved class records and all thirteen timings remain in `summary.json`
and the raw inputs. Orders 4–6 merit investigation, even though primitive upper
bounds peak at lower orders. This does not select a production implementation.

5857 used 23/12 native builds at 48 atoms and 26/17 at 96 atoms. All derivative
observer replays agreed with production below `1e-9`; the angular replays also
passed that gate. These are same-implementation observer checks, **not an
independent scientific oracle**. Final native residuals are retained, but no
independent residual reconstruction was performed by this diagnostic.

Failed Slurm 5856 is retained too: requesting derivative order one directly on
the ordinary SCF `PreparedFockPlan` did not yield a resident value binding.
The corrected probe uses the ordinary value owner for SCF and a separate force
owner afterward. The failed original source was recovered against its recorded
SHA-256; it is not silently replaced by the successful source.

## Verification and reproduction

From the repository root:

```bash
PYTHONPATH=python:. python benchmarks/results/pbe0-derivative-work-20261005/verify.py
PYTHONPATH=python:. python -O benchmarks/results/pbe0-derivative-work-20261005/verify.py
```

The verifier authenticates retained bytes, source and driver receipts, class
inventory, actual SCF counts, native residual gates, replay errors, and derived
angular totals. Kernel names/durations are cross-checked against the event
projection exported directly from Nsight SQLite. Raw `.nsys-rep` and SQLite
files remain in ignored local storage; `profiler-retention.json` records exact
paths, sizes, and hashes. `receipts.json` retains original text logs losslessly.
No external archive or release asset is required to run this verifier.

To rebuild the historical source, start at the permanent base in
`identities.json`, decompress and apply `control-source.patch.gz`, and recompute
the normal source fingerprint. `reconstruction.json` retains the independently
verified fingerprint. Use the original compiler/build receipts with ccache
enabled. Copy the `.cpp.txt` drivers back to `.artifacts/*.cpp`, decompress the
four input files, and compile against that build's headers and library with
CUDA 12.9.1. The launcher scripts retain the exact environment and finite
per-command timeouts used by the original diagnostics. Any new real GPU run
must additionally use a finite Slurm allocation, for example:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:40:00 bash .artifacts/profile-derivative-orders-n1.sh
```

Preserve Slurm's device visibility. The source paths in historical scripts are
reproduction context and must point to the reconstructed checkout. New driver
binaries should retain their own build identity. Keep compiler caches enabled;
these diagnostics do not measure cold compilation or allocator peaks.
