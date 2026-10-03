# Independent low-order J/K source reuse

An incremental optimization on measured master
`9e9b938239d31564d6f92e9b99b922aee0dbb1cc`, not a claim that the large-system
PBE0 gap is resolved. Main README publication remains withheld.

## Matched complete endpoints

Node5 Slurm job1406 runs baseline96, candidate24, candidate96, baseline96 on
one allocated RTX 5090. Each has one cold call, two frozen one-iteration warm
calls and one separately profiled warm call. All 16 energy/force gates pass.
Only the six clean **96-atom** calls enter the matched comparison:

| Variant | Clean warm seconds |
| --- | --- |
| Baseline before | 80.164096, 80.153800 |
| Candidate | 77.709112, 77.701921 |
| Baseline after | 80.187680, 80.181607 |

Pooled baseline median **80.172851 s**, candidate **77.705516 s**:
**3.077519% less wall time, 1.031752x**. This is SCF energy plus returned analytic
forces, not an isolated kernel. Diagnostic cold excludes initial batch-owner
construction; public cold below includes it. Differing cold iterations/JIT
histories do not establish a cold speedup. Profiled calls never enter medians.

## Public protocol qualification

Node1 job5491 executes all 72 PBE0/def2-SVP spherical calls with the existing
48-by-16-by-32 unpruned grid, strict FP64, 1e-12 energy and screening tolerances,
1e-10 density tolerance, and 100 maximum SCF iterations. Each size includes
cold, five frozen warm, moved and five moved-warm complete endpoints.

| Atoms | Cold s | Warm median s | Moved s | Moved-warm median s |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 82.178565 | 0.355339 | 1.981853 | 0.356257 |
| 6 | 69.753164 | 0.763522 | 4.171776 | 0.766934 |
| 12 | 80.653458 | 1.703565 | 10.382251 | 1.700513 |
| 24 | 107.738909 | 5.140151 | 27.739105 | 5.142607 |
| 48 | 195.263519 | 18.331156 | 73.146915 | 18.363612 |
| 96 | 562.199527 | 79.803158 | 296.737738 | 79.397174 |

All full force arrays pass independent retained GPU4PySCF reference validation:
maximum errors **1.036824e-10 Eh / 3.036026e-11 Eh/Bohr**, unchanged gates
1e-8 / 1e-7. Reference data are used only for acceptance, never to seed native
density. No reference timing is remeasured or used for a new performance ratio.
Do not compare these node1 latencies against the node5 bracket.

## Work, resources and correctness

- The change shares immutable primitive geometry/Boys values and AO traversal
  across independent J/K sources in nine order-0--3 shell classes. Generated
  weighted force roots, per-channel primitive reduction order, zero-channel
  semantics and screening are unchanged. Other consumers retain their paths.
- Actual native-header host execution checks 13,026,816 components bitwise
  equal. Test-fixture geometry calls fall 1,311,456 to 1,043,280; force-root calls
  remain 1,311,456. These are **fixture**, not molecular counts or speedups.
- Molecular grid work remains 2,359,296 points, 9,216 geometry batches,
  21,516,779,520 Becke pair-state evaluations; no tile or threshold changes.
  Additional device/host bounds remain 357,022,976/197,047,712 bytes under
  unchanged 512/256 MiB allowances. Molecular post-screen primitive counts
  are not available. Both force-screened spin kernels retain 255 registers and
  90,392-byte static stack; zero-register helper records are not kernel usage.
- Node1 job5490 passes the actual public shell `[J', K']` CPU-ERI oracle for
  baseline and candidate, both spins, every coefficient mask, zero/opposing
  densities, Cartesian/spherical bases through f and moved geometry. Candidate
  memcheck/initcheck report zero errors. The host harness passes ASan/UBSan.
- Host checks pass: 91 focused cases, 147 ownership/dependency cases, 116
  benchmark/retention cases. The latter use pytest's importlib mode to avoid a
  conflicting installed `tests` package; the initial collection failure is
  retained. Formatting, compiler/SCF structure and complexity checks pass.

## Identity and retained evidence

Baseline native SHA-256:
`a80427eb5263cbcacacd36eb52f2768a4d6b114ef93961fa2dc48a6b70016cc0`.
Candidate native SHA-256:
`14d47b97f7de8e34bea8990fd9a46e8d2b27fb2a165ac053b1f2ddb0b489480b`.
Candidate compiled source identity:
`2059973ac8988a03c469597136f99ae7d779f0dd18beb2385cddb3e8aec14b75`.
The later ownership/test-only additions do not change that source identity.
Newer master `cd08953d5` is explicitly **not** the measured build.

[qualification.json.gz](qualification.json.gz) retains all scalar repeats, raw hashes,
source/build identities and the matched work/resource controls. Full force
arrays, outcomes, reference bytes, scripts, build logs, ccache receipts and
profiler data remain in ignored `.artifacts/` under
`/home/jzzeng/codes/qc-pbe0-low-order-sources-20261003`, locally and on node5.
Native libraries remain on node5; no Release, tag or new external archive is
created. The stale incremental build rejected before measurement and node4
driver-mismatch attempts remain separately labeled and excluded.

The scalar JSON is gzip-compressed without changing any decompressed bytes.
This also respects the combined checkout budget after newer master added other
evidence; it does not remove measurements or alter their acceptance. Inspect it
with `gzip -cd qualification.json.gz`.
Decompressed SHA-256:
`88dc3e898092b85be028c2aa6ab4311c3f2d785e0e07d61a2be5ba3ef0ff134c`.
Gzip SHA-256:
`23c3a9dd1b59ffe7c002ea4fe1e3ae7ab32e626ddc6e0c795f9e9a4439b87dda`.

All GPU commands use finite Slurm `main`, `--gres=gpu:5090:1`, preserving
assigned visibility. Builds use verified ccache, explicit CXX/CUDA launchers,
checkout-root `CCACHE_BASEDIR`, Release CUDA12.9.1/sm_120, no weakened caching,
and eight OpenMP/BLAS/MKL threads. Revalidate the retained data without running
GPU work:

```bash
PYTHONPATH=python:. python .artifacts/summarize_endpoints.py
python .artifacts/summarize_bracket.py
```

On the retained node5 checkout, a fresh public campaign (including new references)
can run without overwriting the historical records:

```bash
source .artifacts/gpu-environment.sh
export PBE0_BENCHMARK_PYTHON="$python"
export GENERATIVEQC_LIBRARY="$PWD/.artifacts/candidate/libgenerativeqc.so"
export PBE0_BASIS_FILE="$PWD/benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json"
export PBE0_BENCHMARK_OUTPUT="$PWD/.artifacts/reproduce-low-order-$(date +%Y%m%d-%H%M%S)"
export PBE0_POINT_TIMEOUT=3000 PBE0_ATOMS="3 6 12 24 48 96"
export PBE0_ENGINES="reference native"
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=01:20:00 bash benchmarks/run_pbe0_benchmarks.sh
```

The retained bracket script records the exact baseline/candidate sequence; copy
it and change its output paths before rerunning. Diagnostic scripts are local
historical reproduction aids, not new production APIs.
