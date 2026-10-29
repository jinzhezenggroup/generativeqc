# Cross-functional stationary AO-map acceptance

Slurm job **14898**, node1/main, one RTX 5090, 30-minute limit: **14 passed**
in 936.16 seconds. The matrix retains 140 explicit stationary E+F endpoints
and 28 independent public native-force controls. LDA/PBE/PBE0/r2SCAN/WB97M-V
cover both spin domains; WB97M-V additionally covers sampled maps and separate
XC/nonlocal grids. Every explicit arm includes first call, warm, moved,
moved-warm and zero-cache fallback.

| Maximum observed error | Value | Gate |
| --- | ---: | ---: |
| Explicit energy vs GPU4PySCF (Eh) | 3.694822225952521e-12 | 1e-8 |
| Explicit force vs GPU4PySCF (Eh/Bohr) | 8.012004393265215e-10 | 3e-7 |
| Public native force vs GPU4PySCF (Eh/Bohr) | 8.016747266026414e-10 | 3e-7 |
| Selected vs dense explicit force (Eh/Bohr) | 1.9095836023552692e-13 | 1e-9 |

`acceptance.json.xz` retains full-precision reference/native/explicit outputs,
SCF work and residuals, actual force work, timing, source hashes, package
identities and scheduler provenance. Selected/native force agreement also
passes 1e-9; every physical SCF residual passes 1e-9. These small-fixture timings
are numerical acceptance observations, not a performance campaign. First-call
time includes preparation but shares the process and disk caches with other
cases. No default or crossover is promoted. The public native controller still
uses dense force AO tiles; selected maps are invoked through the explicit
stationary consumer after native SCF, without overriding native capabilities.

## Source and environment

Base: `dc526e408174ce9d22049860a6dfa8349afc7a45`. Applying `source.patch.xz`
restores the executable changes measured here; hashes in the acceptance record
were checked against the submitted source. The 20 s/p and s/p/d stationary AOT
packages were rebuilt from that source with CUDA 12.9.86, GNU 13.3.0, real
sm_120 code and verified sccache 0.16.0 CXX/CUDA launchers. Source/binary contract
checks remained enabled. Build logs and pre/post shared-cache statistics stay
in ignored task artifacts.

The core native library was reused, not rebuilt. Its SHA-256 is
`72efe46682fed58ecbb9f776a7a1dcb13977344f510afdaa5383cb368419dd69`.
Its original source identity was not independently verified; this record does
not assert a full current-source core build. The exact origin and every newly
built force-package identity are retained in the payload.

Reference environment: Python 3.13.9, NumPy 2.4.6, CuPy 13.6.0, GPU4PySCF 1.8.1,
PySCF 2.14.0 and Libxc 7.0.0. Both geometries reconverge independently with full
Fock rebuilds and matching atomic quadrature/moving-grid response. The initial
incremental LDA reference failed convergence; full-Fock reference execution
passed without changing tolerances. Initial stale/missing AOT packages were
rejected and rebuilt, rather than having their identity metadata rewritten.

## Reproduce

Provide a compatible native CUDA library, matching stationary s/p and s/p/d
packages for `lda`, `pbe`, `pbe0`, `r2scan` and `wb97mv`, both spin modes, and
the reference dependencies above. Rebuild the corresponding CMake `*_manifest`
targets using a verified compiler-cache launcher. Then run:

```bash
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --time=00:30:00 \
  env PYTHONPATH="$PWD/python:$PWD" GENERATIVEQC_DFT_CUDA_TEST=1 \
  GENERATIVEQC_LIBRARY=/absolute/path/libgenerativeqc.so \
  python -m pytest -q tests/python/test_cross_functional_force_ao_cuda.py \
  --junitxml=.artifacts/force-map-tests.xml -o junit_family=legacy
```

The JUnit properties contain the same reference, native-control and endpoint
arrays as the compact payload. Full logs/XML, build products and retry receipts
remain in ignored local artifacts. This qualification advances #1853's shared
consumer acceptance; native-controller map adoption and production performance
qualification remain open.
