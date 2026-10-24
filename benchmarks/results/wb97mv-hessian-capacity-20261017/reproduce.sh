#!/usr/bin/env bash
set -euo pipefail
: "${GENERATIVEQC_REFERENCE_PYTHON:?Set the pinned GPU4PySCF/PySCF reference interpreter}"
: "${GENERATIVEQC_CLI:?Set the recorded source-618 AOT native CLI path}"
: "${GENERATIVEQC_LIBRARY:?Set the recorded source-618 native library path}"
: "${GENERATIVEQC_COMPILER_CACHE:?Set a verified sccache or ccache launcher}"
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
# Preserve a virtualenv interpreter's symlink spelling (do not resolve its target).
python=$(command -v -- "$GENERATIVEQC_REFERENCE_PYTHON")
[[ $python = /* ]] || python=$PWD/$python
cli=$(readlink -f -- "$GENERATIVEQC_CLI")
library=$(readlink -f -- "$GENERATIVEQC_LIBRARY")
[[ -x $python && -x $cli && -f $library ]] || { echo 'Missing native/reference artifact' >&2; exit 1; }
"$GENERATIVEQC_COMPILER_CACHE" --version
native_revision=618f7e3ddf616f7960e9e3fc9df6aa31fda26107
driver_revision=9abfe3b0b3560576da8c3eef23fb24772b4e76e6
# Stage on the checkout filesystem, not a login node's private /tmp.
mkdir -p "$root/.artifacts"
work=$(mktemp -d "$root/.artifacts/capacity-replay.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
mkdir "$work/native" "$work/source"
# Check the copies that will actually execute, not mutable paths checked earlier.
cp -- "$cli" "$work/native/generativeqc"
cp -- "$library" "$work/native/libgenerativeqc.so.0.1.0"
printf '%s  %s\n' \
  9b3588c62330e2277348fa1ee69b90ce88efd5ac15cc417882dc8beaabbf6bdc "$work/native/generativeqc" \
  f061b5a5829b8178c15372c58f903f517c2d2909c703e0a03f117d737a6937d2 "$work/native/libgenerativeqc.so.0.1.0" | sha256sum --check
ln -s libgenerativeqc.so.0.1.0 "$work/native/libgenerativeqc.so.0"
ln -s libgenerativeqc.so.0.1.0 "$work/native/libgenerativeqc.so"
# Source-618 lacks the shared tetramer fixture used by fresh reference workers.
# Archive the fixture-bearing driver, without relabeling the native binaries.
git -C "$root" archive "$driver_revision" | tar -xf - -C "$work/source"
export GENERATIVEQC_LIBRARY="$work/native/libgenerativeqc.so.0.1.0"
export LD_LIBRARY_PATH="$work/native:${REFERENCE_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$work/source/python:$work/source/tests/python:$work/source"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_DISABLE_PTX_JIT=1
cd "$work/source"
printf 'Native measured revision: %s; fixture driver: %s\n' "$native_revision" "$driver_revision"
printf 'Retain all stock def2-SVP comparator failures; no speed qualification is expected there.\n'
for specification in 'water-tetramer sto-3g tetramer' 'water-dimer def2-svp def2svp'; do
  read -r case_name basis label <<< "$specification"
  srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --time=00:12:00 "$python" \
    benchmarks/compare_wb97mv_hessian_gpu4pyscf.py --cases "$case_name" \
    --basis "$basis" --engines generativeqc gpu4pyscf --repeats 1 \
    --timeout-seconds 180 --gpu-memory-limit-mib 24576 \
    --cli "$work/native/generativeqc" --reference-python "$python" \
    --source-commit "$native_revision" --output "$root/.artifacts/capacity-$label.json"
done
