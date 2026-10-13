# Equal-spin XC shared-branch endpoint qualification

This campaign compares master `0e6dfb453fdcd8802a9310031a9664137c4a7b47`
with `786c31a481ad6cd743c14e8b2920f3a89b5ef8c6`. The candidate shares
simultaneously demanded lazy scalar branches across equal-spin energy and
derivative outputs. General-point lowering, arithmetic expressions, FP64,
quadrature, SCF tolerances and public endpoint semantics are unchanged.

The paired campaign is separate from the October 13 README publication and
from #2450's pure-Coulomb direct derivative scheduling change. These measurements
do not attribute either of those changes to branch sharing.

## Complete 96-atom endpoints

Seconds, baseline → candidate. Warm columns are the median of all five
frozen-seed calls. Each row below passes all 12 independent E/F checks in both
arms. See [timings.md](timings.md) for every size, the full min–max ranges,
iteration/Fock-build counts and incomplete-phase status.

| Method / provider | Cold | Warm | Moved | Moved-warm |
| --- | ---: | ---: | ---: | ---: |
| M06-2X direct | 149.912 → 123.065 | 20.193 → 17.347 | 112.042 → 93.329 | 20.203 → 17.393 |
| M06-2X DF | 149.425 → 122.755 | 36.209 → 29.699 | 110.754 → 91.942 | 36.287 → 29.666 |
| r2SCAN-3c DF | 213.073 → 197.695 | 21.591 → 20.441 | 135.795 → 126.939 | 21.598 → 20.397 |

The M06-2X warm reductions are 14.1% direct and 18.0% DF. r2SCAN-3c DF warm
falls 5.3%. SCF work is identical between arms for these rows. The 48-atom
M06-2X warm reductions are 16.1% direct and 18.4% DF; r2SCAN-3c DF falls 10.2%.
PBE0 direct controls at 3/12 atoms pass all 12 calls per arm, with less than
2% change in warm and moved-warm medians.

The large **r2SCAN-3c direct moved endpoint regresses**: 296.815 → 403.783 s,
with 14 → 23 Fock builds. Cold changes from 556.277 → 503.270 s with
32 → 30 builds. At candidate moved iteration 14 the physical residual is
1.31e-13, but energy changes continue fluctuating above the unchanged 1e-12
stopping threshold until iteration 23. The full SCF histories remain retained;
these observations do not isolate a fixed-work cold/moved speedup. The direct
integral-derivative bottleneck and near-threshold SCF variability remain open.
The complete direct/96 warm series changes from 103.830 to 100.647 s, with
one Fock build per call in both arms. Both populations hit the original
1800-second limit: baseline returns 11/12 calls and candidate 10/12. Their
moved-warm series return 4/5 and 3/5 calls, so neither receives a successful
moved-warm median. This is not an all-phase direct endpoint improvement.

Across the campaign, 18 populations complete all 12 calls and all 237 returned
calls pass the independent gates. Maximum errors are 3.956e-11 Eh and
1.573e-10 Eh/Bohr. [endpoints.json.xz](endpoints.json.xz) retains every returned
scalar observation and source audit; [timings.md](timings.md) displays all
populations, including both timeouts. Paired SCF work matches in every case
except direct r2SCAN-3c/96.

## Independent qualification and diagnostic scope

The host suite passes 287 tests with PySCF/Libxc installed; one explicitly
GPU-only test is skipped. The native CUDA suite passes 14 tests: full M06-2X
DF cold/warm/moved replay, M06-2X/MN15 RKS/UKS decomposition, canonical
r2SCAN-3c RKS/UKS independent forces, reconverged directional finite
differences, and ragged geometry replay. The UKS control separately packages
`r2scan_uks` against the same native source identity.

Qualification exposed two stale fixtures, repaired in
`22aa78cd8146fffd5743fd5a51a6d7fff36d5d03`: every moved M06-2X replay must pass
the moved coordinates, and the historical r2SCAN-3c auxiliary basis must use
the existing checked schema adapter. The native implementation identity is
unchanged. Initial missing-library/profile and fixture failures remain in the
[test receipts](test-receipts.json.xz), alongside the successful reruns.

[Diagnostic records](diagnostics.json.xz) separately retain the earlier
96-atom Nsight attribution on source `9e3c5e6e6fa2cb1a832b8fbd85c100e1533f8535`.
M06-2X DF spends 22.4% of device kernel time in SCF equal-spin points and
14.8% in stationary geometry points. r2SCAN-3c direct spends most time in
integral derivatives. Isolated generated point probes give approximately
2.1× / 1.8× improvement for M06-2X / r2SCAN at 256 points, with all eight
outputs bitwise equal. The M06 probe uses 192 registers versus 255 and removes
spills. Both probes pass memcheck with zero errors at 1/127/129/257 points.
Instrumented and isolated timings are excluded from the endpoint tables.

[Build receipts](build-receipts.json.xz) retain verified cache versions,
before/after shared-cache statistics, configured compiler-command provenance,
stationary manifests and the initial portable-profile refusals. The small
[GPU qualification records](gpu-qualification.json.xz) retain independent
numerical errors. Full benchmark observations are reduced from raw arrays by
[reduce.py](reduce.py); every returned call keeps its own errors, residuals,
work counts, source/library identity, scheduler receipt and original-file hash.

## Measurement contract

Use the existing `benchmarks/run_readme_dft_benchmarks.sh native` driver with
five frozen-density warm repetitions, followed by a geometry change and five
frozen-density moved repetitions. Every measured call includes energy, analytic
forces, synchronization and host force return. API cold includes construction
and preparation; process startup and provenance collection are excluded.
Existing compiler and stationary artifact caches are retained. No reference
density or reference computation enters the native production endpoint.

Both arms use Release, CUDA 12.9, sm_120 target-specific AOT and verified
sccache 0.16.0. The stationary profiles are `r2scan_rks;m06_2x_rks;pbe0_rks`.
Every real-device command uses finite Slurm allocations on n1/node1, partition
`main`, `--gres=gpu:5090:1`, eight CPU threads, and unmodified Slurm device
visibility. Individual jobs have exclusive GPUs; independent populations can
overlap on different GPUs of that node. Per-point timeout remains 1800 seconds.

Inputs and acceptance are the README protocol: nested water clusters, spherical
def2-SVP for M06-2X/PBE0, canonical def2-mTZVPP plus D4/gCP for r2SCAN-3c,
48×16×32 moving quadrature, and the retained independent GPU4PySCF/PySCF/Libxc
references for both geometries. Direct and fitted cases use their own matching
oracles; DF includes auxiliary response. Every accepted call must pass
1e-8 Eh and 1e-7 Eh/Bohr. All observations and SCF work counts are retained;
timings are not normalized by SCF iterations.

## Reproduction and evidence

Create clean checkouts at the two revisions above. Configure each with:

```sh
sccache --version
sccache --show-stats
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGENERATIVEQC_ENABLE_CUDA=ON \
  -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF \
  -DGENERATIVEQC_CUDA_ARCHITECTURES=120 \
  -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120 \
  -DGENERATIVEQC_AOT_PROFILE=auto -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON \
  '-DGENERATIVEQC_STATIONARY_AOT_PROFILES=r2scan_rks;m06_2x_rks;pbe0_rks' \
  -DGENERATIVEQC_CUDA_COMPILE_JOBS=4 -DGENERATIVEQC_AOT_COMPILE_JOBS=2 \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$(command -v sccache)" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$(command -v sccache)" \
  -DCMAKE_INSTALL_LIBDIR=lib
cmake --build build-cuda --parallel 8
cmake --install build-cuda --prefix "$runtime_directory"
sccache --show-stats
```

Select the corresponding installed `GENERATIVEQC_LIBRARY`, CUDA library path
and Python checkout. Set `README_BENCHMARK_METHODS`,
`README_BENCHMARK_DENSITY_FITTING=none|cuda`,
`README_BENCHMARK_SIZES='48 96'`, `README_BENCHMARK_REPEATS=5` and an isolated
`README_BENCHMARK_OUTPUT`. Run the driver using:

```sh
for engine in reference native; do
  srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=8 --time=00:45:00 \
    bash benchmarks/run_readme_dft_benchmarks.sh "$engine"
done
```

The first allocation regenerates the independent references; this campaign's
native runs reuse retained references after checking complete scientific identity.
The [reducer](reduce.py) rechecks raw force arrays and source identities before
retaining scalar observations. Its input layout and invocation are documented
in its module docstring; use `--output endpoints.json.xz --table timings.md`.
The [manifest](manifest.json) binds the compact evidence and reproduction files.
Raw arrays, full compiler logs, binaries and profiler
databases stay in ignored campaign storage on n1 at
`/data/jzzeng/qc-r2scan-m062x-20261014`.
