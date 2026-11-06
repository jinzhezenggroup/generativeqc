#!/usr/bin/env bash
set -euo pipefail
: "${LSB_JOBID:?actual finite LSF GPU allocation required}"
: "${CUDA_VISIBLE_DEVICES:?scheduler visibility required}"
assigned=$CUDA_VISIBLE_DEVICES
base=/tmp/generativeqc-ecnu-a100-tzhu-e-20261029
shared=$base/issue2987-hold-fix-20261105
trial=$base/direct-gap-master-20261105
baseline=$trial/source
candidate=$trial/candidate-source
out=$trial/results-$LSB_JOBID
transfer=http://login02:18789
mkdir "$out"
exec > "$out/driver.log" 2>&1
finish() {
  status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$out/exit-code.txt"
  date -Is > "$out/completed-at.txt"
  tar -czf "$out.tar.gz" -C "$out" .
  curl --fail --max-time 90 --data-binary @"$out.tar.gz" "$transfer/results-$LSB_JOBID.tar.gz" || status=74
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -Is > "$out/started-at.txt"
hostname > "$out/hostname.txt"
nvidia-smi > "$out/nvidia-smi.txt"
cp "$0" "$out/runner.sh"
for name in master-candidate-overlay.tar.gz master-candidate-runtime.tar.gz master-candidate-source-identity.txt master-candidate-image-identity.json master-authenticate.txt; do
  test ! -e "$trial/$name"
  curl --fail --max-time 180 "$transfer/$name" -o "$trial/$name"
done
mkdir "$candidate"
tar -C "$baseline" --exclude=./build --exclude=./.artifacts -cf - . | tar -C "$candidate" -xf -
tar -xzf "$trial/master-candidate-overlay.tar.gz" -C "$candidate"
tar -xzf "$trial/master-candidate-runtime.tar.gz" -C "$candidate"
source "$base/native-env.sh"
frozen=$base/cmake-python-openblas-20261101
loader=$shared/glibc/ld-linux-x86-64.so.2
export PYTHONHOME="$base/env"
export CUDA_HOME="$base/env/targets/x86_64-linux" CUDA_PATH="$base/env/targets/x86_64-linux"
export CXX="$base/env/bin/x86_64-conda-linux-gnu-g++" NVCC_CCBIN="$CXX" CCACHE="$base/env/bin/sccache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GENERATIVEQC_PROFILE=off GENERATIVEQC_PROFILE_CACHE="$out/profile-cache"
export XDG_CACHE_HOME="$base/native-cache" CUPY_CACHE_DIR="$base/native-cupy"
unset GENERATIVEQC_DF_TRACE GENERATIVEQC_DIRECT_FORCE_CENSUS
unset GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES
unset GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_PAIR_SEPARABLE GENERATIVEQC_DIRECT_WEIGHTED_FORCE_RESPONSES
failed=0
for arm in baseline candidate; do
  source=$baseline
  identity=$trial/master-source-identity.txt
  expected=$trial/master-image-identity.json
  if [[ "$arm" = candidate ]]; then
    source=$candidate
    identity=$trial/master-candidate-source-identity.txt
    expected=$trial/master-candidate-image-identity.json
  fi
  runtime_ld="$shared/glibc:$shared/deps:$source/build:$shared/source/scipy_openblas32/lib:$base/env/lib:$base/env/targets/x86_64-linux/lib"
  cp "$base/env/bin/python" "$trial/python-runtime/python-$arm"
  "$base/env/bin/patchelf" --set-interpreter "$loader" --force-rpath --set-rpath "$runtime_ld" "$trial/python-runtime/python-$arm"
  python=("$loader" --library-path "$runtime_ld" "$trial/python-runtime/python-$arm" -u -X faulthandler)
  export PYTHONPATH="$source/python:$source:$frozen/provider:$base/results/stationary-parity/reference-packages"
  export GENERATIVEQC_LIBRARY="$source/build/libgenerativeqc.so"
  cd "$source"
  "${python[@]}" "$trial/master-authenticate.txt" "$identity" "$expected" "$out/$arm-identity.json"
  status=0
  timeout --kill-after=20s 360 "${python[@]}" "$trial/endpoint.py" \
    --source "$(cat "$identity")" --inputs "$shared/source/inputs" --warm-repeats 3 \
    --output "$out/$arm.json" > "$out/$arm.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$out/$arm.exit-code.txt"
  if [[ "$status" != 0 ]]; then failed=1; fi
done
test "$CUDA_VISIBLE_DEVICES" = "$assigned"
exit "$failed"
