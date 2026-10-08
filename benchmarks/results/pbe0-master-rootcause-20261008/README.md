# Current-master PBE0 cold E/F root-cause audit

Frozen master: `4385f72751b829883407c01106186917c344317b`, including #2100's
default MD-J. This bundle retains the original six-pair large-grid complete
energy/force observations, three primary reference observations, both pilots,
an MD-J-disabled control, separate diagnostic traces/counters and failed producer
controls. It does not
replace the smaller-grid energy-only #2100 qualification.

`summary.json.gz` is a deterministic, losslessly compressed JSON summary generated
from the raw receipts, with each input hashed. It contains no raw profiler arrays
or build logs. Read it with `gzip.open(path, "rt")` and `json.load`, or
`gzip -dc summary.json.gz`. Repeated protocols are shared under `protocols`; each
observation's `protocol_sha256` resolves its unchanged scientific values. The
timer includes prepare and synchronized public E/F return; imports, context
initialization and Calculator construction are excluded. Fresh processes,
owners and densities use persistent caches. The scientific protocol is PBE0,
96 atoms / 768 spherical def2-SVP AOs, 2,359,296 unpruned grid points, energy/
density tolerances 1e-12/1e-10, native screening 1e-12 and reference direct
tolerance 1e-14. Force/grid receipts use packaged/native-build AOT.

## Acceptance and interpretation

Master/candidate/reference complete medians are 115.437826 / 111.128942 /
78.498101 s. Candidate point estimate is 3.732645% less time, but the 95%
independent-bootstrap interval is [-3.116631%, +13.435876%]. All twelve native
primary independent E/F gates pass; no sample is removed or normalized by Fock
count. The original positive-lower-bound gate **fails**.

The user explicitly accepts the 3.73% point estimate and asks to adopt native
AO reuse. `GENERATIVEQC_CUDA_AO_RADIAL_REUSE` therefore defaults to `ON`, with
explicit `OFF` available. This policy decision does not make the statistical
gate pass, qualify the forced CSR producer, or establish GPU4PySCF parity.
Default-build smoke observations remain separate from the original population.

Discovery means **active-AO index construction**, not total forces. Conservative
native CSR removes full discovery jets but enlarges selected domains and, in
the 256-point control, doubles tile orchestration. Failed force endpoints are
not gains. The separately tested CSR admission fix preserves the native
derivative reserve and retains dense fallback without changing producer policy.

## Raw evidence and reproduction

Local worktree: `/data/jzzeng/qc-pbe0-master-rootcause-20261008-4385f7275`.
Raw receipts, SQLite/NSys/NCU/pstats and scripts are in `.artifacts/rootcause/`
and on n1 at the same root's `evidence/`. The original native libraries have
SHA256 `bf6465da17852183df0b340f551416a31e0b1c5df0c554e6d3ab44ab42d87f50`
(master) and `e69c70807baafdd6232c6b6666e27d605e3b9d8511bc6f7d6e2c8a2fa4e3390e`
(candidate). Retain every condition's geometry, grid and actual Fock history.

Run real GPU tests/profilers/sanitizers through finite `srun`, preserving CVD;
reuse ccache. `drivers/cold.py.txt` preserves the byte-exact endpoint driver
whose hash is recorded in the observations. `reduce.py`
computes trace busy unions and the bootstrap, and `export_compact.py` exports
this ledger without selecting fast observations. Intrusive diagnostics are
never part of the clean timing population. Reference profiling has 47 Focks,
unlike clean 38/38/44; uncovered GPU residence is not automatically compilation
or pure CPU overhead.

### Reproduce the adopted native endpoint

Use a dependency-installed GPU Python environment with GPU4PySCF/CuPy/PySCF,
working CUDA library search paths, CMake/Ninja and ccache. The following commands
run from the checkout root on the site's RTX 5090 partition. Set `GPU_PYTHON`
to that environment's Python and `CUDA_PATH` to the CUDA toolkit. Do not override
Slurm's `CUDA_VISIBLE_DEVICES`. The build uses a new directory so an older cached
`OFF` option cannot mask the adopted default; compiler caches remain intact.

```bash
export SOURCE="$PWD"
export GPU_PYTHON="$(command -v python)"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export CACHE_LAUNCHER="$(command -v ccache)"
export CCACHE_BASEDIR="$SOURCE"
export BUILD="$SOURCE/build/pbe0-ao-reuse-reproduce"
export RECEIPTS="$SOURCE/.artifacts/pbe0-ao-reuse-$(date +%Y%m%d-%H%M%S)"
export PYTHONPATH="$SOURCE/python:$SOURCE"
export GENERATIVEQC_LIBRARY="$BUILD/libgenerativeqc.so"
export GENERATIVEQC_BENCHMARK_SOURCE="$(git rev-parse HEAD)"
export GENERATIVEQC_STATIONARY_CACHE="${GENERATIVEQC_STATIONARY_CACHE:-$SOURCE/.cache/stationary-cuda}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$RECEIPTS"
test ! -e "$BUILD/CMakeCache.txt"
srun --partition=main --nodelist=n1 --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=01:00:00 bash -lc '
    set -euo pipefail
    cd "$SOURCE"
    "$CACHE_LAUNCHER" --version
    "$CACHE_LAUNCHER" --show-stats
    cmake -S . -B "$BUILD" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DGENERATIVEQC_ENABLE_CUDA=ON \
      -DGENERATIVEQC_CUDA_ARCHITECTURES=120 -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF \
      -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
      -DCMAKE_CXX_COMPILER_LAUNCHER="$CACHE_LAUNCHER" \
      -DCMAKE_CUDA_COMPILER_LAUNCHER="$CACHE_LAUNCHER" \
      -DPython_EXECUTABLE="$GPU_PYTHON" -DPython3_EXECUTABLE="$GPU_PYTHON" \
      -DGENERATIVEQC_CUDA_COMPILE_JOBS=8 -DGENERATIVEQC_AOT_COMPILE_JOBS=2 \
      -DGENERATIVEQC_AOT_SPLIT_COMPILE_THREADS=8
    cmake --build "$BUILD" --parallel 16 --target generativeqc \
      generativeqc_stationary_pbe0_rks_manifest generativeqc_stationary_pbe0_rks_spd_manifest
    grep -qx GENERATIVEQC_CUDA_AO_RADIAL_REUSE:BOOL=ON "$BUILD/CMakeCache.txt"
    sha256sum "$GENERATIVEQC_LIBRARY"
    "$CACHE_LAUNCHER" --show-stats
  '
srun --partition=main --nodelist=n1 --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:20:00 bash -lc '
    set -euo pipefail
    cd "$SOURCE"
    driver=benchmarks/results/pbe0-master-rootcause-20261008/drivers/cold.py.txt
    unset GENERATIVEQC_DISABLE_MD_J GENERATIVEQC_DIRECT_J_FOCK_LOWERING
    unset GENERATIVEQC_DIRECT_K_FOCK_LOWERING GENERATIVEQC_DIRECT_K_TASK_SCHEDULE
    unset GENERATIVEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE GENERATIVEQC_DIRECT_J_TASK_SCHEDULE
    unset GENERATIVEQC_DIRECT_J_PAIR_DENSITY GENERATIVEQC_DIRECT_J_PAIR_DENSITY_MASK
    export MODE=reference
    "$GPU_PYTHON" "$driver" reference --atoms 96 --output "$RECEIPTS/reference.json"
    export MODE=adopted-default
    "$GPU_PYTHON" "$driver" native --atoms 96 --reference "$RECEIPTS/reference.json" \
      --output "$RECEIPTS/native.json"
  '
```

This produces a new independent reference and native acceptance receipt, not a
replacement for the frozen six-pair cohort. For a new complete cold comparison,
build the frozen master in a separate checkout, use the same archived driver and
persistent cache, collect at least six alternating master/candidate pairs in
fresh processes, and retain all observations and Fock histories. Keep profiler
invocations and failed controls outside that clean population.

Rationale: [initial root-cause audit](../../../.agents/notes/implemented/performance/2026-10-08-pbe0-ao-reuse-and-csr-admission.md)
and [user-accepted default](../../../.agents/notes/implemented/performance/2026-10-08-default-native-ao-reuse.md).
Issue updates: #1965 (endpoint tracker), #1893 (AO/force-domain work).
