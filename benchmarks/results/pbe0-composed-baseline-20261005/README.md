# Composed exact-direct PBE0 baseline

The corrected optimization baseline is master `1de139fa5` + #1847
`445fa7150` + #1830 `aae46734b` + #1833 `4c5aa3eb6`, with **explicit indexed
policy, SCF active AO, force active AO at 1e-16, and phased Becke**. The measured
clean composition is `cd121e3af089b0d59f9cffaeff381b1ac01ffb31`.
These switches are qualification choices, not production defaults.

All 144 same-geometry native/reference energy-and-force comparisons pass
absolute gates of 1e-8 Hartree and 1e-7 Hartree/Bohr. The maximum errors are
8.19e-12 / 1.80e-11 for 48 atoms and 1.05e-10 / 3.45e-11 for 96 atoms.
The 1.25x warm endpoint target in #1895 is **not met**. No default is promoted.

| Atoms | Phase | Native median (s) | GPU4PySCF median (s) | Ratio |
| --- | --- | ---: | ---: | ---: |
| 48 | cold | 131.70678 | 51.93144 | 2.536 |
| 48 | warm | 10.78222 | 6.08896 | 1.771 |
| 48 | moved | 58.39746 | 48.50742 | 1.204 |
| 48 | moved-warm | 10.77021 | 3.95170 | 2.725 |
| 96 | cold | 253.21684 | 87.68469 | 2.888 |
| 96 | warm | 33.39056 | 10.19577 | 3.275 |
| 96 | moved | 131.79104 | 86.03299 | 1.532 |
| 96 | moved-warm | 33.28310 | 10.19978 | 3.263 |

Every cold, five warm, moved and five moved-warm sample is retained without
iteration normalization or outlier removal. Native iteration trajectories are
27/1/13/1 (48 atoms) and 28/1/12/1 (96 atoms), where the warm groups are all one
iteration. Reference trajectories vary; the exact arrays are in `summary.json`.
The reference forces include analytic moving-grid response and its Fock calls
rebuild from the full current density. Neither engine uses DF or COSX.

The 96-atom warm median stationary derivative time is 14.35346 s and semilocal
geometry response is 8.80609 s. The force endpoint median is 23.48025 s, about
70% of the complete 33.39056 s endpoint. Individual component medians are not
additive timing partitions. The endpoint remainder also contains SCF and other
work. J/K optimization alone cannot remove the dominant force cost.

## Diagnostic component timing

Job 5772 uses fresh **energy-only** SCF solves, followed by five intrusive
`NativeKsSnapshot.cuda_fixed_density_profile()` CUDA-event replays per geometry.
It is a separate experiment on the same frozen scientific/source identity.
It does not measure clean E/F endpoints or actual quartet admission counts.

| Atoms | Geometry | J (s) | K (s) | AO/grid/XC (s) |
| --- | --- | ---: | ---: | ---: |
| 48 | original | 1.17920 | 1.05341 | 1.06737 |
| 48 | moved | 1.17388 | 1.05761 | 1.06854 |
| 96 | original | 2.41466 | 1.95268 | 2.34394 |
| 96 | moved | 2.39072 | 1.98814 | 2.32550 |

Its SCF trajectories are 24/13 and 25/15, respectively. They must not replace
the E/F trajectories. Native public Fock-build counts remain null; actual
indexed page counts and per-build J/K quartet admissions are unavailable.
The retained endpoint records do expose actual sparse SCF and force AO work,
native derivative routing, and full phased Becke batch coverage. Full records
retain capacity bounds and missing timer fields rather than fabricated zeros.

## Offline verification and source reconstruction

From a checkout with NumPy and the project Python package available:

```bash
PYTHONPATH=python:. python benchmarks/results/pbe0-composed-baseline-20261005/verify.py
```

The verifier authenticates the publication inventory, rechecks all 144 E/F
pairings and actual route observations, verifies all 20 diagnostic samples,
and recomputes the retained medians and trajectories. JSON gzip storage uses
the repository's shared deterministic compactor and record reader. No raw
scientific sample or negative timing is removed.

`composed-source.patch` reconstructs all library/compiler/benchmark sources and
the native AO discovery checks against the permanent master base. Unrelated
historical notes and Python regression test edits are omitted. This source was
reconstructed locally and its canonical build identity exactly matches the
measured library:
`e858a3537dd389dafbf952f9c70f5f803de37fed8b719776511c8e6b5876b6d4`.
The measured library SHA-256 is
`b9c449f51744ca1f77ba35124fc07be67ba1b72b1321b7245b540bcf9d0f94f3`.

```bash
git worktree add --detach /tmp/pbe0-composed-source 1de139fa59401802e07aa20730ec7e842195f4f4
git -C /tmp/pbe0-composed-source apply /absolute/path/to/composed-source.patch
cd /tmp/pbe0-composed-source
export CCACHE_DIR=/data/jzzeng/ccache CCACHE_BASEDIR="$PWD"
ccache --version
cmake --preset cuda-release-sm120 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build/cuda-release-sm120 --target generativeqc --parallel 6
```

Use CUDA 12.9 (nvcc 12.9.86), GCC 13.3, RTX 5090, and the captured Python
3.13.9 environment (GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 13.6.0, cuTENSOR 2.2.0,
NumPy 2.4.6). Set `GENERATIVEQC_LIBRARY` to the rebuilt library and configure
the CUDA compiler/library environment. Preserve Slurm-assigned visibility.
The exact node-specific launch scripts are retained as text for provenance;
adjust checkout/interpreter paths when rerunning elsewhere. For each size:

```bash
export PYTHONPATH="$PWD/python:$PWD"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:30:00 \
  python -m benchmarks.readme_pbe0_integrated --policy default reference --atoms 48 \
  --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json --repeats 5 --output /tmp/reference-48.json
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:30:00 \
  python -m benchmarks.readme_pbe0_integrated --policy local-indexed-phased --scf-active-ao native --atoms 48 \
  --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json --repeats 5 \
  --reference /tmp/reference-48.json --output /tmp/composed-48.json
```

Repeat with 96 atoms and corresponding output names. Job 5768 used a single
finite allocation for both engine/size pairs. No source-default timing is
claimed by these records. Build logs, profiler debris and binaries remain in
ignored local artifacts; no Release or external archive was published.

Refs #1895, #1892, #1893, #1894, #1834.
