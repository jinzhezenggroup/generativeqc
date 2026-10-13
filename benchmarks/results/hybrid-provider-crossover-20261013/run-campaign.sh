#!/usr/bin/env bash
# Clean ABBA endpoint populations; intrusive diagnostics follow in separate processes.
set -euo pipefail
: "${SLURM_JOB_ID:?finite Slurm allocation required}"
closeout_root=/data/jzzeng/qc-closeout-20261013
closeout_python=/data/jzzeng/qc-932-validation-env-20261011/bin/python
cd "$closeout_root"
test "$(git rev-parse HEAD)" = e0ad22036fe83d9bcaed277d0be9a1dbcb453e1c
test -z "$(git status --porcelain)"
export PATH=/group/software/cuda-12.9.1/bin:/data/jzzeng/qc-provider-1886-tools/bin:$PATH
export PYTHONPATH="$closeout_root/python:$closeout_root:/data/jzzeng/dft-mixed-threshold-20261012-160bf3d23/.artifacts/pydeps"
export GENERATIVEQC_LIBRARY="$closeout_root/.artifacts/install/lib/libgenerativeqc.so"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
closeout_output="$closeout_root/.artifacts/closeout-campaign-$SLURM_JOB_ID"
mkdir "$closeout_output"
printf 'SLURM_JOB_ID=%s\nCUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "${CUDA_VISIBLE_DEVICES:-}" > "$closeout_output/allocation.txt"
git rev-parse HEAD > "$closeout_output/source.txt"
sha256sum .artifacts/install/lib/libgenerativeqc*.so* .artifacts/install/lib/generativeqc_stationary*.json > "$closeout_output/binaries.sha256"
"$closeout_python" - <<'PY' > "$closeout_output/device.json"
import json,cupy,uuid
p=cupy.cuda.runtime.getDeviceProperties(0)
print(json.dumps({'device':p['name'].decode(),'uuid':str(uuid.UUID(bytes=p['uuid'])) if 'uuid' in p else None,'visible_devices':cupy.cuda.runtime.getDeviceCount(),'runtime_version':cupy.cuda.runtime.runtimeGetVersion(),'driver_version':cupy.cuda.runtime.driverGetVersion()},indent=2))
PY
# The scheduler bounds the entire campaign. Failed receipts remain on disk.
timeout --kill-after=15s 65m "$closeout_python" tools/benchmark_hybrid_provider_crossover.py run-campaign \
  --cases water-96 water-48 formaldehyde --output "$closeout_output/native" > "$closeout_output/native.log" 2>&1
printf 'ABBA_RECORDED %s\n' "$closeout_output"
for closeout_case in water-96 water-48 formaldehyde; do
  for closeout_arm in direct df-jk-occupied; do
    closeout_profile="$closeout_output/$closeout_case-$closeout_arm-profile.json"
    set +e
    timeout --kill-after=15s 12m "$closeout_python" tools/benchmark_hybrid_provider_crossover.py run-profile \
      --case "$closeout_case" --arm "$closeout_arm" \
      --trace "$closeout_output/$closeout_case-$closeout_arm-trace.jsonl" \
      --output "$closeout_profile" > "$closeout_profile.log" 2>&1
    closeout_status=$?
    set -e
    printf '%s\n' "$closeout_status" > "$closeout_profile.exit"
  done
done
printf 'NATIVE_AND_PROFILES_RECORDED %s; independent oracle gates pending\n' "$closeout_output"
