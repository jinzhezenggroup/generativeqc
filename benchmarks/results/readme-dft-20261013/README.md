# DFT energy and analytic forces: cold and warm

Measurements are based on master `e0ad22036fe83d9bcaed277d0be9a1dbcb453e1c`
on 2026-10-13. PBE, PBE0, B3LYP and r2SCAN at **12 and 96 atoms** were remeasured
with this change's conditional force-memory handoff; other points retain the
original master measurements. Each point retains its actual native source
inventory identity and library hash in [summary.json.gz](summary.json.gz).
[manifest.json](manifest.json) identifies both snapshots and the changed files.

![Cold and warm complete DFT endpoints](dft-cold-warm.svg)

See the [complete timing table](timings.md) for all sizes and both engines.
Cold is one observed call per point; stable warm series show the median and
minimum–maximum of five repeats. Crosses/ranges retain repeats with different
SCF iteration or Fock-build counts instead of pooling them into a median.

## Measurements and acceptance

**25 of 28 native points** passed all 12 endpoint checks
(300 native observations); **26 of 28 reference points** completed.
Across accepted native observations, maximum absolute errors were **2.319e-11 Eh**
and **9.159e-11 Eh/Bohr**. All reference electronic XC evaluations
reported the CUDA Libxc backend. These are complete endpoint observations;
no iteration-normalized speedup is claimed.

| Functional | Qualified native sizes (atoms) |
| --- | --- |
| PBE | 3, 12, 48, 96 |
| PBE0 | 3, 12, 48, 96 |
| B3LYP | 3, 12, 48, 96 |
| r²SCAN | 3, 12, 48, 96 |
| r²SCAN-3c | 3, 12, 48 |
| M06-2X | 3, 12, 48, 96 |
| ωB97M-V | 3, 12 |

The 3-atom M06-2X cold call took **229.691 s**, including **226.157 s**
in the recorded force-provider lookup/construction interval. This first-call
cost remains in the cold comparison. One observed cold call with retained
caches does not establish a monotonic size trend or an empty-cache launch cost.

Unsuccessful points remain explicit; their partial timings are excluded:

| Functional | Atoms | Engine | Outcome |
| --- | ---: | --- | --- |
| r²SCAN-3c | 96 | native | Time limit; last stage `cold/0` |
| ωB97M-V | 48 | reference | Time limit; last stage `moved/0` |
| ωB97M-V | 48 | native | Skipped: complete reference unavailable |
| ωB97M-V | 96 | reference | Time limit; last stage `cold/0` |
| ωB97M-V | 96 | native | Skipped: complete reference unavailable |

A converged SCF energy with missing analytic forces cannot pass the endpoint gate.
The ωB97M-V reference has a 1800-second per-point limit and a 3600-second
outer allocation limit. Its retained outcome identifies which limit stopped
each incomplete point. No native timing is inferred from an incomplete oracle.

Host validation: 140 tests passed across force ownership/admission, ctypes
bridges, telemetry and benchmark publication. The real-device lifetime/oracle
probe and ten native CLI force/finite-difference cases also passed.

Every published native point must complete 12 independently checked endpoints:
cold, five fixed-density warm calls, changed geometry, and five moved-geometry
warm calls. All must return finite energy and host forces, converge and meet
absolute errors of **1e-8 Eh** and **1e-7 Eh/Bohr**. The comparison does not divide
wall time by iterations or assert equivalent stopping rules between engines.
The full scalar observations retain native iterations/Fock builds, reference
SCF J/K builds, residuals, errors and available resource/work diagnostics.

## Default OOM correction

The original master campaign converged SCF but ran out of memory while creating
forces at 96 atoms for PBE, PBE0, B3LYP and r2SCAN. With the optional ERI cache
still resident, the new default runs had only **234–252 MiB** free at force entry.
The handoff released **15,120,367,225 bytes (14.08 GiB)** of optional provider
storage and retained the converged state and warm seed. All four methods now
pass all 12 cold/warm/moved checks, including the formerly failing cold calls.
The 12-atom controls completed without cache eviction; the native lifetime
probe separately verifies retention of admitted ERIs when memory is sufficient.

[force-memory.json.gz](force-memory.json.gz) retains the per-call memory receipts
and additional validation. A diagnostic pressure injection left 64 MiB free:
master failed with OOM; the fixed endpoint completed all 12 checks. Its timings
are excluded from this figure. M06-2X and ωB97M-V three-atom controls also passed.
Together these checks cover **132 fixed-build endpoints**, with maximum
errors **1.046e-11 Eh** and
**9.159e-11 Eh/Bohr**. Original failed records and
an unmodified master report remain in the ignored evidence directory; they are
not presented as measurements of the fixed build. The r2SCAN-3c and ωB97M-V
reference timeouts listed above are separate limitations.

## Scientific and timing boundaries

- Nested water clusters: 3, 12, 48 and 96 atoms, batch size one, restricted FP64
  calculations. All methods use full def2-SVP spherical shells (24 AOs per water)
  except r2SCAN-3c, whose defining def2-mTZVPP basis has 34 AOs per water.
- Both engines use the same explicit 48×16×32 per-atom moving quadrature, three
  original-Becke partition iterations, no radius adjustment and analytic grid
  response. This is a matched-grid comparison, not a claim about each package's
  default grid or chemical grid-convergence accuracy.
- The public native endpoint and GPU4PySCF both include SCF energy, analytic
  forces, synchronization and force return to the host. No native calculation
  uses a reference density or reference energy/force implementation.
- Cold includes calculator/engine construction, preparation and the first
  endpoint. Process startup, provenance collection and input snapshot loading
  are excluded. Existing compiler/artifact caches are retained; these are API
  cold starts, not empty-cache compilation or process-launch timings.
- Each warm series reuses one frozen engine-local post-cold density. The moved
  geometry displaces the first hydrogen by 0.001 Bohr along z; its own post-move
  density is frozen for the five moved replays. Densities are not progressively
  improved across measured repeats.
- Native tolerances are 1e-12 Eh for energy, 1e-10 for density and 1e-12 for
  screening, with 100 SCF iterations maximum. GPU4PySCF uses 1e-12 Eh energy,
  1e-10 orbital gradient and 1e-14 direct-integral screening tolerances. Its stock
  DIIS is unchanged. Full-density potential rebuilds suppress both incremental
  inputs, as in the existing PBE0 protocol, and their work stays timed.
- r2SCAN-3c includes D4 (`s6=1`, `s8=0`, `s9=2`, `a1=0.42`, `a2=5.65`,
  `ga=2`, `gc=1`) and gCP. Its reference combines GPU4PySCF electronic calculations
  with CPU dftd4 4.3.0 and the independent CPU gCP qualification provider. Both
  corrections and their geometry-dependent setup are evaluated inside **every**
  reference endpoint. The native endpoint uses CUDA D4 and native CPU gCP; its
  corrections also remain inside every timed endpoint. The figure labels the
  reference CPU corrections.
- WB97M-V includes self-consistent VV10 and the shared molecular density-screening
  policy. No density fitting, mixed precision, experimental AO selection or
  preliminary-density override is enabled in this matrix.

## Hardware and provenance

Runs use Slurm on `n1` (`node1`), `main`, one NVIDIA GeForce RTX 5090 per
allocation, eight CPU threads and finite job/point timeouts. Independent method
groups may use separate GPUs on the same node. Slurm's device visibility is
preserved. Device state snapshots were taken **before** measurement; reduction
preserves and corrects the shared helper's legacy sampling-point label.

Runtime: Python 3.13.9, GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 13.6.0, cuTENSOR
2.2.0 and CUDA 12.9. Native compilation uses GCC 11.4, NVCC 12.9.86, the Release
sm_120 preset and verified ccache 4.5.1; generated Ninja commands invoke the
launcher. Retained cache statistics are shared-cache snapshots, not an isolated
per-build hit-rate claim. Runtime JIT uses verified sccache 0.16.0 where needed;
the execution host's default C++ compiler is GCC 13.3, as recorded in the raw
environment, distinct from the core/package build compiler. The host has two
AMD EPYC 9354 processors; each Slurm allocation requests eight CPU threads.

The remote source is a plain synchronized snapshot, so its raw Git fields are
null. The manifest records the base revision and both native source inventories and
loaded-library hashes; per-file hashes bind the actual benchmark and Python
consumers. The eight remeasured points use the fixed build; the remaining
points preserve their original master provenance.

## Reproduction

Install the normal native build dependencies plus GPU4PySCF, CuPy, PySCF, dftd4,
Matplotlib and a supported compiler cache. Use the master base plus this change
to reproduce the fixed behavior. The mixed historical matrix records each
original snapshot separately. Build with the production preset and explicit
verified launchers, retaining the shared compiler cache:

```bash
ccache --version
export CCACHE_BASEDIR="$PWD"
cmake --preset cuda-release-sm120 \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$(command -v ccache)" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$(command -v ccache)" \
  -DGENERATIVEQC_BUILD_TESTS=OFF
cmake --build --preset cuda-release-sm120 -j 6
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export README_BENCHMARK_OUTPUT=.artifacts/readme-dft
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=04:00:00 \
  bash benchmarks/run_readme_dft_benchmarks.sh both
PYTHONPATH=.:python python -m tools.render_readme_dft \
  --raw .artifacts/readme-dft --output benchmarks/results/readme-dft-20261013 \
  --source-commit "$(git rev-parse HEAD)"
```

`README_BENCHMARK_METHODS`, `README_BENCHMARK_SIZES` and
`README_BENCHMARK_POINT_TIMEOUT` select a bounded subset; defaults are the seven
documented methods, four sizes, five repeats and 1800 seconds per engine/size.
The runner also accepts `reference` and `native` separately, preserving the
independent reference JSON. Use a fresh output directory for each campaign.

Re-render the committed reduction without a GPU:

```bash
PYTHONPATH=.:python python -m tools.render_readme_dft \
  --summary benchmarks/results/readme-dft-20261013/summary.json.gz \
  --output .artifacts/readme-figure
```

The compact archive retains original-file hashes and independent cold/moved
reference vectors. Large native arrays, build products and trial logs remain in
ignored local artifact storage. Hashes establish integrity, not authentication
or a new numerical qualification of a different source revision.
