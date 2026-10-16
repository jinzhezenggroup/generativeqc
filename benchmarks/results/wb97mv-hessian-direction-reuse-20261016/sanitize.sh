#!/usr/bin/env bash
set -euo pipefail
directory=".artifacts/sanitizers-$SLURM_JOB_ID"
mkdir -p "$directory"
printf 'source=5aa703721fdce6a220838676af615d574f0fed05 job=%s visibility=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
sha256sum libgenerativeqc.so.0.1.0 generativeqc_cuda_wb97mv_nuclear_jk_probe
for tool in memcheck initcheck racecheck synccheck; do
  /group/software/cuda-12.9.1/bin/compute-sanitizer --tool "$tool" --error-exitcode=99 \
    --log-file "$directory/$tool.txt" /usr/bin/env PATH=/nonexistent \
    PYTHONHOME=/nonexistent PYTHONPATH=/nonexistent \
    ./generativeqc_cuda_wb97mv_nuclear_jk_probe sanitizer-fixture.bin "$directory/$tool.bin" \
    > "$directory/$tool.json"
  cat "$directory/$tool.txt"
done
printf 'SANITIZERS_PASSED\n'
