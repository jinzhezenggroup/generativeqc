#!/usr/bin/env bash
# Run the water-96 benchmark cases (one arm) on a USTC SCC GPU node.
#
# Protocol matches benchmarks/readme_defaults.py: one process per case,
# --warm-repeats 3, and single-threaded BLAS/OMP so the timings are
# single-device endpoint measurements.
#
# GQC_ARM selects which implementation is measured:
#   standard    GenerativeQC's public Standard calculator (the thing under test)
#   gpu4pyscf   the GPU4PySCF reference arm (same geometry/basis/gates)
# Both arms go through the same benchmarks/readme_defaults.py protocol and emit
# the same generativeqc.readme-defaults.v2 schema, so a standard/GPU4PySCF pair
# for the same method+mode is directly comparable sample for sample. Use
# tools/scc/compare_scc.py to assemble that comparison.
#
# The frozen element-level input tables in tools/scc/inputs-w96/ are required.
# Do NOT regenerate them with `readme_defaults.py inputs`: that path hardcodes
# water-3 and does not accept --atoms, so it cannot produce water-96 tables.
#
# This is a bug hunt, not a demo. A failing case is a RESULT: it is recorded in
# failures.tsv and the sweep continues, so one bad method cannot mask the rest.
#
# Usage: run.sh [output-dir]
#   GQC_ARM       standard | gpu4pyscf (default: standard)
#   GQC_ARCH      architecture to run on (default 80 = A100; 70 for V100)
#   GQC_CASES     space-separated "method:mode" list (overrides GQC_GROUP)
#   GQC_GROUP     baseline | defects | all (default: baseline)
#   GQC_TIMEOUT   per-case seconds (default 5400 = 90 min)
#
# The two groups are separate submissions on purpose. The baseline group is the
# cheap calibration set; the defects group holds the paths that the reference
# RTX 5090 cohort showed to be slow or broken, and whose outcome on an 80 GB
# A100 is the actual measurement of interest.
#
#   baseline  hf:df hf:direct pbe:df pbe:direct
#   defects   pbe:df r2scan:df r2scan-3c:direct wb97m-v:df wb97m-v:direct r2scan-3c:df
#
# hf:* is the calibration arm: it exercises the plain Coulomb/exchange path with
# no KS diagnostic, so a timing deviation there implicates the device or the
# build rather than a method-specific code path.
#
# pbe:df appears in both groups: it is the reference point that ties the two
# submissions together.
set -euo pipefail

SCC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCC_DIR}/env.sh"

ARCH="${GQC_ARCH:-80}"
ARM="${GQC_ARM:-standard}"
case "${ARM}" in
  standard|gpu4pyscf) ;;
  *)
    echo "ERROR: unknown GQC_ARM='${ARM}' (standard|gpu4pyscf)." >&2
    echo "       The 'fast'/'gpu4pyscf-fast' arms are deliberately excluded:" >&2
    echo "       they change the measurement, not just the implementation." >&2
    exit 1
    ;;
esac

PRESET="cuda-release-sm${ARCH}"
export GQC_BUILD="${GQC_BUILD:-${GQC_SRC}/build/${PRESET}}"
export GENERATIVEQC_LIBRARY="${GQC_BUILD}/libgenerativeqc.so"

OUT="${1:-${GQC_SRC}/.artifacts/scc-a100-sm${ARCH}}"
mkdir -p "${OUT}"

# The gpu4pyscf arm does not load libgenerativeqc.so, so only require the
# library when the GenerativeQC arm is the one being measured.
if [ "${ARM}" = "standard" ] && [ ! -f "${GENERATIVEQC_LIBRARY}" ]; then
  echo "ERROR: ${GENERATIVEQC_LIBRARY} not found; run tools/scc/build.sh first." >&2
  exit 1
fi

# ---- frozen input tables ---------------------------------------------------
INPUTS="${SCC_DIR}/inputs-w96"
INPUT_COUNT="$(ls -1 "${INPUTS}" 2>/dev/null | wc -l)"
if [ "${INPUT_COUNT}" -lt 16 ]; then
  echo "ERROR: expected 16 input tables in ${INPUTS}, found ${INPUT_COUNT}." >&2
  echo "       Restore tools/scc/inputs-w96/ from the repository." >&2
  exit 1
fi
echo ">> inputs: ${INPUTS} (${INPUT_COUNT} files)"

# ---- case selection --------------------------------------------------------
GROUP="${GQC_GROUP:-baseline}"
if [ -n "${GQC_CASES:-}" ]; then
  read -r -a CASES <<< "${GQC_CASES}"
else
  case "${GROUP}" in
    baseline) read -r -a CASES <<< "hf:df hf:direct pbe:df pbe:direct" ;;
    defects)  read -r -a CASES <<< "pbe:df r2scan:df r2scan-3c:direct wb97m-v:df wb97m-v:direct r2scan-3c:df" ;;
    all)      read -r -a CASES <<< "hf:df hf:direct pbe:df pbe:direct r2scan:df r2scan-3c:direct wb97m-v:df wb97m-v:direct r2scan-3c:df" ;;
    *)
      echo "ERROR: unknown GQC_GROUP='${GROUP}' (baseline|defects|all)." >&2
      exit 1
      ;;
  esac
fi
TIMEOUT_S="${GQC_TIMEOUT:-5400}"

# Validate every requested method/mode against the frozen inputs up front, so a
# typo fails before the queue slot is consumed instead of 40 minutes in.
for case in "${CASES[@]}"; do
  method="${case%%:*}"
  mode="${case##*:}"
  if [ ! -f "${INPUTS}/${method}-reference.json" ]; then
    echo "ERROR: case '${case}' has no ${method}-reference.json in ${INPUTS}." >&2
    echo "       Available methods: $(ls -1 "${INPUTS}" | sed 's/-reference.json//;s/-auxiliary.json//' | sort -u | tr '\n' ' ')" >&2
    exit 1
  fi
  case "${mode}" in
    direct|df) ;;
    *) echo "ERROR: case '${case}' has unknown mode '${mode}' (direct|df)." >&2; exit 1 ;;
  esac
done

echo "=================================================================="
echo " output  : ${OUT}"
echo " arm     : ${ARM}"
echo " group   : ${GROUP}"
echo " cases   : ${CASES[*]}"
echo " timeout : ${TIMEOUT_S}s per case"
echo "=================================================================="

nvidia-smi --query-gpu=index,name,compute_cap,memory.total \
    --format=csv > "${OUT}/gpu-inventory.csv" || true
cat "${OUT}/gpu-inventory.csv"

# The AOT profile actually selected at runtime is a separate question from which
# cubins are embedded. build.sh proves the cubin is present; it cannot prove the
# runtime dispatch chose it. Record both so a silent generic_cuda fallback is
# detectable after the fact.
LIB_REAL="$(readlink -f "${GENERATIVEQC_LIBRARY}")"
CUOBJDUMP="${CUDA_PATH:-}/bin/cuobjdump"
[ -x "${CUOBJDUMP}" ] || CUOBJDUMP="$(command -v cuobjdump || echo '')"
{
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-all}"
  echo "generativeqc_library=${GENERATIVEQC_LIBRARY}"
  echo "generativeqc_library_real=${LIB_REAL}"
  echo "requested_arch=sm_${ARCH}"
  echo "gcc=$(gcc --version | head -1)"
  echo "nvcc=$(nvcc --version | tail -2 | head -1)"
  echo "python=$(python3 --version 2>&1)"
  echo "gqc_group=${GROUP}"
  echo "gqc_cases=${CASES[*]}"
  echo "gqc_timeout_s=${TIMEOUT_S}"
  if [ -n "${CUOBJDUMP}" ] && [ -f "${LIB_REAL}" ]; then
    echo "--- embedded cubins (cuobjdump --list-elf) ---"
    "${CUOBJDUMP}" --list-elf "${LIB_REAL}" 2>/dev/null \
      | grep -o '\.sm_[0-9]*\.' | sort | uniq -c || echo "cuobjdump failed"
  fi
} > "${OUT}/environment.txt"
cat "${OUT}/environment.txt"

# Record the exact tree revision so results stay traceable to a commit.
REV="$(git -C "${GQC_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "source_revision=${REV}"
echo "source_revision=${REV}" >> "${OUT}/environment.txt"

: > "${OUT}/failures.tsv"
printf 'case\treason\twall_s\n' >> "${OUT}/failures.tsv"

FAILED=()
for case in "${CASES[@]}"; do
  method="${case%%:*}"
  mode="${case##*:}"
  name="${ARM}-${method}-${mode}"
  echo
  echo "------------------------------------------------------------------"
  echo ">>> ${name}  ($(date -u +%H:%M:%SZ))"
  echo "------------------------------------------------------------------"

  cd "${GQC_SRC}"
  START="$(date +%s)"
  RC=0
  timeout --signal=TERM --kill-after=60 "${TIMEOUT_S}" \
    env CUDA_VISIBLE_DEVICES=0 python3 benchmarks/readme_defaults.py "${ARM}" \
        --method "${method}" \
        --mode "${mode}" \
        --atoms 96 \
        --warm-repeats 3 \
        --inputs "${INPUTS}" \
        --source-revision "${REV}" \
        --output "${OUT}/${name}.json" \
        2>&1 | tee "${OUT}/${name}.log" || RC=$?
  END="$(date +%s)"
  WALL=$((END-START))

  if [ "${RC}" -eq 0 ]; then
    echo "[ok] ${name} wall ${WALL}s"
  else
    # A non-zero exit is the interesting outcome here, not a reason to stop.
    if [ "${RC}" -eq 124 ] || [ "${RC}" -eq 137 ]; then
      REASON="timeout after ${TIMEOUT_S}s"
    else
      REASON="exit ${RC}"
    fi
    echo "[FAIL] ${name} wall ${WALL}s (${REASON})" >&2
    FAILED+=("${name}")
    printf '%s\t%s\t%s\n' "${name}" "${REASON}" "${WALL}" >> "${OUT}/failures.tsv"

    # Preserve a machine-readable record so collect.py can tell "the library
    # refused" apart from "the process died", which are different bugs.
    if [ ! -f "${OUT}/${name}.json" ]; then
      python3 - "${OUT}/${name}.json" "${method}" "${mode}" "${REASON}" "${WALL}" \
        "${REV}" "${SLURM_JOB_ID:-none}" <<'PY' || true
import json, sys
path, method, mode, reason, wall, rev, job = sys.argv[1:8]
json.dump({
    "schema": "generativeqc.readme-defaults.v2",
    "status": "failed",
    "error": f"runner: {reason}",
    "method": method,
    "mode": mode,
    "arm": "${ARM}",
    "atoms": 96,
    "source_revision": rev,
    "slurm_job_id": job,
    "runner_wall_seconds": int(wall),
    "samples": [],
}, open(path, "w"), indent=2)
PY
    fi
  fi
done

echo
echo "=================================================================="
if [ "${#FAILED[@]}" -eq 0 ]; then
  echo " all ${#CASES[@]} cases succeeded"
else
  echo " ${#FAILED[@]} of ${#CASES[@]} cases failed: ${FAILED[*]}" >&2
  echo " see ${OUT}/failures.tsv" >&2
fi
ls -la "${OUT}"/*.json 2>/dev/null || true
echo "=================================================================="

# A partial sweep still produced data worth keeping, so do not fail the job on
# case failures; report them and exit 0. Only an empty sweep is an error.
[ "${#FAILED[@]}" -lt "${#CASES[@]}" ]
