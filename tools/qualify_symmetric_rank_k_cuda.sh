#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY SCCACHE_EXECUTABLE EXPECTED_COMMIT CUDA_TOOLKIT PYTHON_ROOT EXPECTED_HOST_COMPILER" >&2
  exit 2
fi

output_dir=$1
cache_exe=$2
expected_commit=$3
toolkit_root=$4
python_root=$5
expected_host=$6
repo_root=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$repo_root"

# These overrides can introduce files outside the fixed recipe's inventory.
for variable in NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CPATH C_INCLUDE_PATH \
                CPLUS_INCLUDE_PATH LIBRARY_PATH COMPILER_PATH GCC_EXEC_PREFIX; do
  if [[ -n "${!variable:-}" ]]; then
    echo "$variable is unsupported by the fixed rank-k qualification recipe" >&2
    exit 2
  fi
done

python_exe="$python_root/bin/python3"
export LD_LIBRARY_PATH="$python_root/lib:$toolkit_root/lib64"
if [[ ! -x "$python_exe" ]] ||
   ! "$python_exe" -I -S -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo "task-owned Python 3.10+ is required to verify the staged compiler identity" >&2
  exit 2
fi
host_exe=$(command -v "${NVCC_CCBIN:-g++}") || {
  echo "CUDA host compiler is required; select it with NVCC_CCBIN" >&2
  exit 2
}
host_exe=$(readlink -f "$host_exe")
if [[ ! -f "$expected_host" ]] || ! cmp "$host_exe" "$expected_host"; then
  echo "CUDA host compiler differs from the staged expected bytes" >&2
  exit 2
fi

snapshot_root=$(dirname "$repo_root")
if [[ ! -f "$snapshot_root/source-commit.txt" ]] ||
   [[ ! -f "$snapshot_root/source-identity.sha256" ]] ||
   [[ ! -f "$snapshot_root/generated.sha256" ]]; then
  echo "rank-k CUDA qualification requires CPU-prepared source receipts" >&2
  exit 2
fi
actual_commit=$(cat "$snapshot_root/source-commit.txt")
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "rank-k CUDA qualification source commit differs from the submitted job" >&2
  exit 2
fi
if [[ ! -x "$cache_exe" ]]; then
  echo "verified compiler cache executable is required" >&2
  exit 2
fi
cache_version=$($cache_exe --version)
echo "$cache_version"
if [[ ! "$cache_version" =~ ^sccache[[:space:]]+([0-9]+)\.([0-9]+)\.([0-9]+) ]] ||
   (( BASH_REMATCH[1] == 0 && BASH_REMATCH[2] < 16 )); then
  echo "sccache 0.16.0 or newer is required" >&2
  exit 2
fi
nvcc_exe="$toolkit_root/bin/nvcc"
if [[ ! -x "$nvcc_exe" ]]; then
  echo "explicit supported CUDA compiler is required" >&2
  exit 2
fi
nvcc_version=$($nvcc_exe --version)
if [[ "$nvcc_version" != *"V12.9.86"* ]]; then
  echo "this qualification is pinned to CUDA toolkit 12.9.86" >&2
  exit 2
fi
command -v nvidia-smi >/dev/null || { echo "NVIDIA device is required" >&2; exit 2; }

mkdir -p "$output_dir/generated" "$snapshot_root/cache"
if ! sha256sum -c "$snapshot_root/source-identity.sha256" \
     > "$output_dir/source-check.txt" 2>&1; then
  tail -20 "$output_dir/source-check.txt" >&2
  echo "rank-k CUDA source manifest mismatch" >&2
  exit 2
fi
if ! sha256sum -c "$snapshot_root/generated.sha256" \
     > "$output_dir/generated-check.txt" 2>&1; then
  cat "$output_dir/generated-check.txt" >&2
  echo "rank-k generated header mismatch" >&2
  exit 2
fi
export SCCACHE_DIR="$snapshot_root/cache"
export PATH="$toolkit_root/bin:$PATH"
echo "source_commit=$actual_commit" | tee "$output_dir/provenance.txt"
sha256sum "$snapshot_root/source-identity.sha256" >> "$output_dir/provenance.txt"
sha256sum "$snapshot_root/generated.sha256" >> "$output_dir/provenance.txt"
"$nvcc_exe" --version | tee -a "$output_dir/provenance.txt"
sha256sum "$nvcc_exe" "$toolkit_root/bin/ptxas" \
  "$toolkit_root/lib64/libcublas.so.12" \
  "$toolkit_root/lib64/libcublasLt.so.12" \
  "$toolkit_root/lib64/libcudart.so.12" >> "$output_dir/provenance.txt"
sha256sum "$host_exe" "$python_exe" >> "$output_dir/provenance.txt"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader |
  tee -a "$output_dir/provenance.txt"
"$cache_exe" --show-stats > "$output_dir/sccache-before.txt"

cp "$snapshot_root/generated/generated_symmetric_rank_k.cuh" \
  "$output_dir/generated/generated_symmetric_rank_k.cuh"
"$python_exe" -I -S tools/generate_symmetric_rank_k_cuda.py \
  --toolkit-root "$toolkit_root" --host-compiler "$host_exe" \
  --output "$output_dir/generated/verified_symmetric_rank_k.cuh"
if ! cmp "$output_dir/generated/generated_symmetric_rank_k.cuh" \
     "$output_dir/generated/verified_symmetric_rank_k.cuh"; then
  echo "rank-k staged header differs from actual source/toolchain/host/options" >&2
  exit 2
fi
set +e
"$cache_exe" "$nvcc_exe" -std=c++20 -O2 -arch=sm_90 -DGENERATIVEQC_TEST_HOOKS \
  "-ccbin=$host_exe" \
  -I "$repo_root/src" -I "$output_dir/generated" \
  -c "$repo_root/tests/native/test_symmetric_rank_k_cuda.cu" \
  -o "$output_dir/test_symmetric_rank_k_cuda.o" \
  2>&1 | tee "$output_dir/compile.log"
compile_status=${PIPESTATUS[0]}
set -e
"$cache_exe" --show-stats > "$output_dir/sccache-after.txt"
if [[ "$compile_status" -ne 0 ]]; then
  exit "$compile_status"
fi
set +e
"$cache_exe" "$nvcc_exe" --cudart shared "-ccbin=$host_exe" \
  "$output_dir/test_symmetric_rank_k_cuda.o" \
  -L "$toolkit_root/lib64" -lcublas \
  -Xlinker -rpath -Xlinker "$toolkit_root/lib64" \
  -o "$output_dir/test_symmetric_rank_k_cuda" \
  2>&1 | tee "$output_dir/link.log"
link_status=${PIPESTATUS[0]}
set -e
"$cache_exe" --show-stats > "$output_dir/sccache-after.txt"
if [[ "$link_status" -ne 0 ]]; then
  exit "$link_status"
fi
ldd "$output_dir/test_symmetric_rank_k_cuda" | \
  grep -E 'libcublas|libcudart' | tee "$output_dir/linked-cuda-libraries.txt"
for library in libcublas.so.12 libcublasLt.so.12 libcudart.so.12; do
  if ! grep -F "$library => $toolkit_root/lib64/" \
       "$output_dir/linked-cuda-libraries.txt" >/dev/null; then
    echo "$library did not resolve from the pinned toolkit" >&2
    exit 2
  fi
done
sha256sum \
  python/generativeqc_compiler/tensor/scf.py \
  python/generativeqc_compiler/tensor/weighted_gram.py \
  python/generativeqc_compiler/tensor/weighted_gram_emit.py \
  python/generativeqc_compiler/tensor/symmetric_rank_k.py \
  src/tensor/cuda_symmetric_rank_k.cuh \
  tests/native/test_symmetric_rank_k_cuda.cu \
  "$output_dir/generated/generated_symmetric_rank_k.cuh" \
  "$output_dir/test_symmetric_rank_k_cuda.o" \
  "$output_dir/test_symmetric_rank_k_cuda" \
  > "$output_dir/source-artifact.sha256"
"$output_dir/test_symmetric_rank_k_cuda" 2>&1 | tee "$output_dir/qualification.jsonl"
