#!/usr/bin/env bash
# Run the water-96 standard-arm benchmark cases on a USTC SCC GPU node.
#
# Protocol matches benchmarks/readme_defaults.py defaults for the standard arm:
# one process per case, --warm-repeats 3, and single-threaded BLAS/OMP so the
# timings are single-device endpoint measurements.
#
# The frozen element-level input tables in tools/scc/inputs-w96/ are required.
# Do NOT regenerate them with `readme_defaults.py inputs`: that path hardcodes
# water-3 and does not accept --atoms, so it cannot produce water-96 tables.
#
# Usage: run.sh [output-dir]
#   GQC_ARCH   architecture to run on (default 80 = A100; 70 for V100)
#   GQC_CASES  space-separated "method:mode" list
#              (default: hf:direct hf:df pbe:direct pbe:df)
set -euo pipefail

SCC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCC_DIR}/env.sh"

ARCH="${GQC_ARCH:-80}"
PRESET="cuda-release-sm${ARCH}"
export GQC_BUILD="${GQC_BUILD:-${GQC_SRC}/build/${PRESET}}"
export GENERATIVEQC_LIBRARY="${GQC_BUILD}/libgenerativeqc.so"

OUT="${1:-${GQC_SRC}/.artifacts/scc-a100-sm${ARCH}}"
mkdir -p "${OUT}"

if [ ! -f "${GENERATIVEQC_LIBRARY}" ]; then
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
for method in hf pbe; do
  for kind in auxiliary reference; do
    file="${INPUTS}/${method}-${kind}.json"
    [ -f "${file}" ] && echo "   $(sha256sum "${file}" | cut -c1-16)  ${method}-${kind}.json"
  done
done

echo "=================================================================="
echo " output : ${OUT}"
echo "=================================================================="

nvidia-smi --query-gpu=index,name,compute_cap,memory.total \
    --format=csv > "${OUT}/gpu-inventory.csv" || true
cat "${OUT}/gpu-inventory.csv"
{
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-all}"
  echo "generativeqc_library=${GENERATIVEQC_LIBRARY}"
  echo "gcc=$(gcc --version | head -1)"
  echo "nvcc=$(nvcc --version | tail -2 | head -1)"
  echo "python=$(python3 --version 2>&1)"
} > "${OUT}/environment.txt"
cat "${OUT}/environment.txt"

# Record the exact tree revision so results stay traceable to a commit.
REV="$(git -C "${GQC_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)"

read -r -a CASES <<< "${GQC_CASES:-hf:direct hf:df pbe:direct pbe:df}"

FAILED=()
for case in "${CASES[@]}"; do
  method="${case%%:*}"
  mode="${case##*:}"
  name="standard-${method}-${mode}"
  echo
  echo "------------------------------------------------------------------"
  echo ">>> ${name}"
  echo "------------------------------------------------------------------"

  cd "${GQC_SRC}"
  START="$(date +%s)"
  if CUDA_VISIBLE_DEVICES=0 python3 benchmarks/readme_defaults.py standard \
        --method "${method}" \
        --mode "${mode}" \
        --atoms 96 \
        --warm-repeats 3 \
        --inputs "${INPUTS}" \
        --source-revision "${REV}" \
        --output "${OUT}/${name}.json" \
        2>&1 | tee "${OUT}/${name}.log"; then
    END="$(date +%s)"
    echo "[ok] ${name} wall $((END-START))s"
  else
    END="$(date +%s)"
    echo "[FAIL] ${name} wall $((END-START))s" >&2
    FAILED+=("${name}")
  fi
done

echo
echo "=================================================================="
if [ "${#FAILED[@]}" -eq 0 ]; then
  echo " all ${#CASES[@]} cases succeeded"
else
  echo " failed cases: ${FAILED[*]}" >&2
fi
ls -la "${OUT}"/*.json 2>/dev/null || true
echo "=================================================================="
[ "${#FAILED[@]}" -eq 0 ]
