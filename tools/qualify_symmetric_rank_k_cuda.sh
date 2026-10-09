#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY SCCACHE_EXECUTABLE EXPECTED_COMMIT" >&2
  exit 2
fi

output_dir=$1
cache_exe=$2
expected_commit=$3
repo_root=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$repo_root"

actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != "$expected_commit" ]] || [[ -n $(git status --porcelain) ]]; then
  echo "rank-k CUDA qualification requires the exact clean source commit" >&2
  exit 2
fi
if [[ ! -x "$cache_exe" ]]; then
  echo "verified compiler cache executable is required" >&2
  exit 2
fi
cache_version=$($cache_exe --version)
echo "$cache_version"
if [[ "$cache_version" != sccache\ 0.16.* ]] && [[ "$cache_version" != sccache\ 0.1[7-9].* ]] &&
   [[ "$cache_version" != sccache\ [1-9]* ]]; then
  echo "sccache 0.16.0 or newer is required" >&2
  exit 2
fi
command -v nvcc >/dev/null || { echo "nvcc is required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "NVIDIA device is required" >&2; exit 2; }

mkdir -p "$output_dir/generated" "$output_dir/cache"
export SCCACHE_DIR="$output_dir/cache"
echo "source_commit=$actual_commit" | tee "$output_dir/provenance.txt"
nvcc --version | tee -a "$output_dir/provenance.txt"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader |
  tee -a "$output_dir/provenance.txt"
"$cache_exe" --show-stats > "$output_dir/sccache-before.txt"

PYTHONPATH="$repo_root/python" python3 tools/generate_symmetric_rank_k_cuda.py \
  --output "$output_dir/generated/generated_symmetric_rank_k.cuh"
"$cache_exe" nvcc -std=c++20 -O2 -arch=sm_90 -DGENERATIVEQC_TEST_HOOKS \
  -I "$repo_root/src" -I "$output_dir/generated" \
  "$repo_root/tests/native/test_symmetric_rank_k_cuda.cu" \
  -o "$output_dir/test_symmetric_rank_k_cuda" -lcublas \
  2>&1 | tee "$output_dir/compile.log"
"$cache_exe" --show-stats > "$output_dir/sccache-after.txt"
sha256sum \
  python/generativeqc_compiler/tensor/scf.py \
  python/generativeqc_compiler/tensor/weighted_gram.py \
  python/generativeqc_compiler/tensor/weighted_gram_emit.py \
  python/generativeqc_compiler/tensor/symmetric_rank_k.py \
  src/tensor/cuda_symmetric_rank_k.cuh \
  tests/native/test_symmetric_rank_k_cuda.cu \
  "$output_dir/generated/generated_symmetric_rank_k.cuh" \
  "$output_dir/test_symmetric_rank_k_cuda" \
  > "$output_dir/source-artifact.sha256"
"$output_dir/test_symmetric_rank_k_cuda" 2>&1 | tee "$output_dir/qualification.jsonl"
