#!/usr/bin/env bash
# Build GenerativeQC for a USTC SCC GPU architecture on a compute node.
#
# Must run inside a GPU allocation: CMake's CUDA configuration probes the device.
#
# Environment:
#   GQC_ARCH    compute capability major*10+minor to target (default: 80 = A100;
#               use 70 for V100). Selects the cuda-release-sm<ARCH> preset.
#   JOBS        parallel build jobs (default: 32)
#   GQC_VENV    Python venv (default: <src>/.venv-scc)
set -euo pipefail

SCC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCC_DIR}/env.sh"

ARCH="${GQC_ARCH:-80}"
PRESET="cuda-release-sm${ARCH}"
PROFILE="sm_${ARCH}"

if [ ! -f "${GQC_SRC}/CMakePresets.json" ]; then
  echo "FATAL: no source tree at ${GQC_SRC}" >&2
  exit 1
fi
# The build directory must differ per architecture; the preset naming scheme
# below derives it, and env.sh defaults GQC_BUILD to the sm80 path.
export GQC_BUILD="${GQC_BUILD:-${GQC_SRC}/build/${PRESET}}"
export GENERATIVEQC_LIBRARY="${GQC_BUILD}/libgenerativeqc.so"

echo "=================================================================="
echo " target : ${PRESET} (arch ${ARCH}, profile ${PROFILE})"
echo " build  : ${GQC_BUILD}"
echo "=================================================================="

nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv || true
echo


# ---- 1. Python environment -------------------------------------------------
if [ ! -x "${GQC_VENV}/bin/python" ]; then
  echo ">> creating venv: ${GQC_VENV}"
  python3 -m venv "${GQC_VENV}"
fi
# shellcheck source=/dev/null
source "${GQC_VENV}/bin/activate"

if [ ! -f "${GQC_VENV}/.deps-ok" ]; then
  echo ">> installing dependencies (no wheel cache: tight quota)"
  pip install --no-cache-dir --upgrade pip setuptools wheel
  # sccache is a declared base dependency of GenerativeQC (see AGENTS.md); the
  # pip wheel ships the executable, so install it alongside the rest.
  pip install --no-cache-dir sccache
  # cupy must match the CUDA major version loaded above. gpu4pyscf is only used
  # by the optional comparison arm, not by GenerativeQC itself.
  #
  # threadpoolctl is load-bearing, not cosmetic: env.sh pins
  # OPENBLAS_NUM_THREADS=1, but the numpy/scipy wheels bundle libscipy_openblas,
  # which does not honour that variable unless threadpoolctl is installed.
  # Without it the GPU4PySCF arm runs with unbounded BLAS threads and loses time
  # to oversubscription, making the reference arm look slower than it is and
  # silently biasing the very comparison this run exists to produce.
  pip install --no-cache-dir "cupy-cuda12x>=13,<14" "gpu4pyscf>=1.8,<2" \
      "pyscf>=2.6" numpy scipy h5py threadpoolctl
  touch "${GQC_VENV}/.deps-ok"
fi

python3 - <<'PY'
import sys, numpy, pyscf
print(f"[deps] python {sys.version.split()[0]} numpy {numpy.__version__} pyscf {pyscf.__version__}")
try:
    import threadpoolctl
    # env.sh pins OPENBLAS_NUM_THREADS=1 for the run; assert it actually binds,
    # otherwise the GPU4PySCF arm silently loses time to BLAS oversubscription.
    pools = threadpoolctl.threadpool_info()
    print(f"[deps] threadpoolctl {threadpoolctl.__version__} pools={len(pools)}")
    wide = [p for p in pools if (p.get("num_threads") or 1) != 1]
    for p in wide:
        print(f"[deps] warning: {p.get('internal_api')} {p.get('filepath')} "
              f"num_threads={p.get('num_threads')}")
except ImportError:
    print("[deps] warning: threadpoolctl missing; BLAS thread pinning is NOT enforced")
except Exception as exc:
    print(f"[deps] warning: threadpool check failed -> {exc}")
try:
    import cupy, gpu4pyscf
    print(f"[deps] cupy {cupy.__version__} gpu4pyscf {gpu4pyscf.__version__}")
    cupy.zeros(4).sum()
    print("[deps] cupy GPU self-test passed")
except Exception as exc:
    print(f"[deps] warning: cupy/gpu4pyscf unusable -> {exc}")
    print("[deps] GenerativeQC does not require cupy; continuing")
PY

# ---- 2. configure + build --------------------------------------------------
# Compiler cache: CMake auto-selects sccache then ccache. AGENTS.md requires
# verifying the launcher rather than assuming dependency resolution installed
# one, so report what was found and show its stats around the build.
_cached_launcher="$(command -v sccache || command -v ccache || true)"
if [ -z "${_cached_launcher}" ]; then
  # AGENTS.md: report the missing prerequisite instead of silently falling back
  # to an uncached full build. AOT stable-shards makes this a ~10 minute rebuild
  # every time, so make the requirement loud rather than merely advisory.
  echo "ERROR: neither sccache nor ccache is available; refusing an uncached build." >&2
  echo "       Install one into the venv first, then re-run:" >&2
  echo "         ${GQC_VENV}/bin/pip install --no-cache-dir sccache" >&2
  exit 1
else
  echo ">> compiler cache: ${_cached_launcher} ($("${_cached_launcher}" --version 2>&1 | head -1))"
  "${_cached_launcher}" --show-stats 2>/dev/null | sed 's/^/   /' || true
fi

echo
echo ">> cmake configure (${PRESET})"
cmake --preset "${PRESET}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
    -DPython3_EXECUTABLE="${GQC_VENV}/bin/python" \
    -DCMAKE_PREFIX_PATH="${GQC_VENV}" \
    -DGENERATIVEQC_BUILD_TESTS=OFF \
    -DGENERATIVEQC_BUILD_CLI=OFF \
    -DGENERATIVEQC_COMPILER_CACHE=auto

# Confirm the cache actually reached the compiler commands instead of trusting
# the configure output: an empty launcher means every object misses.
if [ -n "${_cached_launcher}" ]; then
  _cxx_launcher="$(grep -E '^CMAKE_CXX_COMPILER_LAUNCHER:' "${GQC_BUILD}/CMakeCache.txt" 2>/dev/null | cut -d= -f2- || true)"
  _cuda_launcher="$(grep -E '^CMAKE_CUDA_COMPILER_LAUNCHER:' "${GQC_BUILD}/CMakeCache.txt" 2>/dev/null | cut -d= -f2- || true)"
  echo ">> CXX  launcher: ${_cxx_launcher:-<none>}"
  echo ">> CUDA launcher: ${_cuda_launcher:-<none>}"
  if [ -z "${_cxx_launcher}" ] && [ -z "${_cuda_launcher}" ]; then
    # AGENTS.md: report the missing prerequisite rather than silently starting
    # an uncached full build, and do not guess at a launcher CMake rejected.
    echo "ERROR: cache executable found but CMake recorded no compiler launcher." >&2
    echo "       An uncached full build is not an acceptable fallback." >&2
    echo "       Inspect ${GQC_BUILD}/CMakeCache.txt, then remove that" >&2
    echo "       build directory and re-run." >&2
    exit 1
  fi
fi

echo
echo ">> cmake build"
JOBS="${JOBS:-32}"
cmake --build --preset "${PRESET}" -j "${JOBS}"

if [ -n "${_cached_launcher}" ]; then
  echo
  echo ">> compiler cache stats after build"
  "${_cached_launcher}" --show-stats 2>/dev/null | sed 's/^/   /' || true
fi

# ---- 3. architecture self-check (the decisive step) ------------------------
# A profile mismatch does not crash: select_profile_for_device() matches the
# compute capability EXACTLY and otherwise falls back to the portable
# generic_cuda profile, silently disabling every AOT shell kernel. An sm_120
# build therefore runs on A100 and reports plausible-but-slow numbers. Verify
# the cubins rather than trusting the preset name.
echo
echo "=================================================================="
echo " architecture self-check"
echo "=================================================================="
LIB="${GQC_BUILD}/libgenerativeqc.so"
if [ ! -f "${LIB}" ]; then
  echo "FATAL: ${LIB} was not produced" >&2
  ls -la "${GQC_BUILD}" || true
  exit 1
fi

CUOBJDUMP="${CUDA_PATH:-}/bin/cuobjdump"
[ -x "${CUOBJDUMP}" ] || CUOBJDUMP="$(command -v cuobjdump || true)"
if [ -z "${CUOBJDUMP}" ] || [ ! -x "${CUOBJDUMP}" ]; then
  echo "FATAL: cuobjdump not found (looked in \${CUDA_PATH}/bin and PATH)" >&2
  exit 1
fi
echo "cuobjdump: ${CUOBJDUMP}"

# The library is a symlink; scan the real file. Use --list-elf, not -sass:
# the former lists every cubin with its .sm_XX suffix in about a second, while
# the latter disassembles a >1 GB library and does not finish in minutes.
LIB_REAL="$(readlink -f "${LIB}")"
echo "library: ${LIB_REAL} ($(( $(stat -c%s "${LIB_REAL}") / 1024 / 1024 )) MB)"
ELF_LIST="$(mktemp)"
trap 'rm -f "${ELF_LIST}"' EXIT
"${CUOBJDUMP}" --list-elf "${LIB_REAL}" > "${ELF_LIST}" 2>/dev/null || true

CUBIN_TOTAL="$(grep -c "ELF file" "${ELF_LIST}" || true)"
echo "embedded cubins: ${CUBIN_TOTAL}"
echo "--- per architecture ---"
grep -o '\.sm_[0-9]*\.' "${ELF_LIST}" | sort | uniq -c || true

MATCH_COUNT="$(grep -c "\.sm_${ARCH}\." "${ELF_LIST}" || true)"
FOREIGN="$(grep -o '\.sm_[0-9]*\.' "${ELF_LIST}" | grep -v "\.sm_${ARCH}\." | sort -u | tr '\n' ' ' || true)"
rm -f "${ELF_LIST}"

FAIL=0
[ "${MATCH_COUNT}" -gt 0 ] || { echo "FATAL: no sm_${ARCH} cubin in the library" >&2; FAIL=1; }
if [ -n "${FOREIGN}" ]; then
  echo "FATAL: library contains cubins for other architectures: ${FOREIGN}" >&2
  FAIL=1
fi
if [ "${CUBIN_TOTAL}" -eq 0 ]; then
  echo "FATAL: no cubins listed; PTX-only build or cuobjdump failed" >&2
  FAIL=1
fi

# Confirm the generated AOT registry targets this profile. Its resolve() does an
# exact compute-capability match, so a wrong name here means silent fallback.
REG="${GQC_BUILD}/generated/production_shell_kernels/generativeqc_generated_shell_registry.cu"
if [ -f "${REG}" ]; then
  echo
  echo "--- generated AOT profile identifiers ---"
  grep -o '"sm_[0-9]*"' "${REG}" | sort -u | head -5
  if ! grep -q "\"${PROFILE}\"" "${REG}"; then
    echo "FATAL: AOT registry does not contain ${PROFILE}" >&2
    FAIL=1
  fi
  echo "--- launch symbol prefixes ---"
  grep -o 'launch_sm[0-9]*_force' "${REG}" | sort -u | head -3
else
  echo "WARN: ${REG} not found" >&2
  FAIL=1
fi

if [ "${FAIL}" -ne 0 ]; then
  echo
  echo "Build artifacts failed validation; see errors above." >&2
  exit 1
fi

echo
echo "=================================================================="
echo " build complete; architecture self-check passed"
ls -la "${LIB}"
echo "=================================================================="
