#!/usr/bin/env bash
# Staged under the campaign root on the qz global-user mount.
set -euo pipefail
attempt=$1
root=$PWD
mkdir "$attempt"
cd "$attempt"
date -u +%FT%TZ > allocated-entrypoint.txt
trap 'printf "%s\n" "$?" > exit-code.txt; date -u +%FT%TZ > allocated-exit.txt' EXIT
environment="$root/../h100-portability/20261027-fd06"
deps=$(realpath "$root/../issue-3166-20261125/cli-deps")
bash "$root/build-qz.sh" "$root/source" "$environment" "$deps"
(cd "$root/payload" && sha256sum -c READY)
export LD_LIBRARY_PATH="$PWD/build:$deps/usr/lib/x86_64-linux-gnu:$environment/env/lib/python3.11/site-packages/scipy_openblas32/lib:/opt/generativeqc-reuse/cuda-12.9/lib64:/opt/generativeqc-reuse/cuda-12.9/targets/x86_64-linux/lib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:16:8
"$environment/env/bin/python" -u "$root/payload/run.py" "$root/payload" "$PWD/build/generativeqc" "$PWD/raw"
