#!/usr/bin/env bash
# Replay on a new allocation using a verified binary from this campaign.
set -euo pipefail
root=$PWD
mkdir "$1"
cd "$1"
date -u +%FT%TZ > allocated-entrypoint.txt
trap 'printf "%s\n" "$?" > exit-code.txt; date -u +%FT%TZ > allocated-exit.txt' EXIT
environment="$root/../h100-portability/20261027-fd06"
deps="$root/../issue-3166-20261125/cli-deps"
built="$root/$2"
payload="$root/$3"
shift 3
sha256sum "$built/build/generativeqc" "$built/build/libgenerativeqc.so" > loaded-binaries.sha256
(cd "$built" && sha256sum -c build-receipts/binaries.sha256)
(cd "$payload" && sha256sum -c READY)
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader > gpu.txt
lscpu > cpu.txt
export LD_LIBRARY_PATH="$built/build:$deps/usr/lib/x86_64-linux-gnu:$environment/env/lib/python3.11/site-packages/scipy_openblas32/lib:/opt/generativeqc-reuse/cuda-12.9/lib64:/opt/generativeqc-reuse/cuda-12.9/targets/x86_64-linux/lib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:16:8
raw="$PWD/raw"
(cd "$root" && "$environment/env/bin/python" -u "$payload/run.py" "$payload" "$built/build/generativeqc" "$raw" "$@")
