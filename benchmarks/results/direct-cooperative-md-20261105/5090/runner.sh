#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?finite Slurm GPU allocation required}"
: "${CUDA_VISIBLE_DEVICES:?scheduler visibility required}"
assigned=$CUDA_VISIBLE_DEVICES
trial=/data/jzzeng/direct-gap-master-20261105
candidate=/data/jzzeng/issue2987-defaults-20261105/source
baseline=$trial/master-source
out=$trial/results-$SLURM_JOB_ID
mkdir "$out"
exec > "$out/driver.log" 2>&1
finish() {
  status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$out/exit-code.txt"
  date -Is > "$out/completed-at.txt"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cp "$0" "$out/runner.sh"
date -Is > "$out/started-at.txt"
hostname > "$out/hostname.txt"
nvidia-smi > "$out/nvidia-smi.txt"
export PATH=/group/software/cuda-12.9.1/bin:/home/jzzeng/.local/bin:/usr/bin:/bin
python=/data/jzzeng/ecnu-stationary-parity-20261030/reference-venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME=/data/jzzeng/ecnu-stationary-parity-20261030/native-cache
export GENERATIVEQC_PROFILE=off GENERATIVEQC_PROFILE_CACHE="$out/profile-cache"
unset GENERATIVEQC_DF_TRACE GENERATIVEQC_DIRECT_FORCE_CENSUS
unset GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES
unset GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_PAIR_SEPARABLE GENERATIVEQC_DIRECT_WEIGHTED_FORCE_RESPONSES
export CUDA_HOME=/group/software/cuda-12.9.1 CUDA_PATH=/group/software/cuda-12.9.1
failed=0
for arm in baseline candidate reference; do
  source=$candidate
  identity=$trial/candidate-source-identity.txt
  expected=$trial/candidate-build/identity.json
  extra=()
  if [[ "$arm" = baseline ]]; then
    source=$baseline
    identity=$trial/master-source-identity.txt
    expected=$trial/master-build/identity.json
  elif [[ "$arm" = reference ]]; then
    extra+=(--reference)
  fi
  export PYTHONPATH="$source/python:$source"
  export GENERATIVEQC_LIBRARY="$source/build/libgenerativeqc.so"
  export LD_LIBRARY_PATH="$source/build:/data/jzzeng/portable-integration-20261101/provider/scipy_openblas32/lib:/group/software/cuda-12.9.1/lib64"
  cd "$source"
  "$python" "$trial/authenticate.py" "$identity" "$expected" "$out/$arm-identity.json"
  status=0
  timeout --kill-after=20s 360 "$python" -u -X faulthandler "$trial/endpoint.py" "${extra[@]}" \
    --source "$(cat "$identity")" --inputs /data/jzzeng/ecnu-stationary-parity-20261030/inputs \
    --warm-repeats 3 --output "$out/$arm.json" > "$out/$arm.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$out/$arm.exit-code.txt"
  if [[ "$status" != 0 ]]; then failed=1; fi
done
export PYTHONPATH="$candidate/python:$candidate"
export GENERATIVEQC_LIBRARY="$candidate/build/libgenerativeqc.so"
export LD_LIBRARY_PATH="$candidate/build:/data/jzzeng/portable-integration-20261101/provider/scipy_openblas32/lib:/group/software/cuda-12.9.1/lib64"
export GENERATIVEQC_RESOURCE_CUDA_TEST=1
cd "$candidate"
status=0
timeout --kill-after=20s 480 "$python" -m pytest -q \
  tests/python/test_cuda_hybrid_snapshot.py::test_cooperative_full_range_derivatives_match_libcint \
  tests/python/test_direct_density_gradient_cuda.py \
  --basetemp "$out/independent-temp" --junitxml "$out/independent.xml" \
  > "$out/independent.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$out/independent.exit-code.txt"
if [[ "$status" != 0 ]]; then failed=1; fi
test "$CUDA_VISIBLE_DEVICES" = "$assigned"
exit "$failed"
