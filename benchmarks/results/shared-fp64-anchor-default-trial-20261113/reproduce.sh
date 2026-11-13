#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?run through finite srun on an allowed node}"
: "${CUDA_VISIBLE_DEVICES:?preserve Slurm device visibility}"
: "${GENERATIVEQC_LIBRARY:?set the source-matched normal Release library}"
: "${EXPERIMENT_ROOT:?directory containing def2-svp.json and reference24/48.json}"
atoms=${1:?24 or 48 atoms}
output=${2:?new absolute output directory}
[[ $atoms == 24 || $atoms == 48 ]]
[[ -f $EXPERIMENT_ROOT/def2-svp.json && -f $EXPERIMENT_ROOT/reference$atoms.json ]]
[[ $output == /* && ! -e $output ]]
mkdir "$output"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GENERATIVEQC_PROFILE=off
export GENERATIVEQC_KS_RESIDENT_MAX_BYTES=2147483648
unset GENERATIVEQC_KS_FIXED_ANCHOR_DELTA
python - <<'PY'
import ctypes, os
from generativeqc.profiles import probe_device
probe = probe_device(ctypes.CDLL(os.environ['GENERATIVEQC_LIBRARY']))
assert probe['source_identity'] == '3b31877336dd65af68a072e89d993f244138aa6e53dc455b8d33b3cb954697cc'
assert probe['device']['release_build'] == 1 and probe['device']['fast_compile'] == 0
PY
for policy in fp64 fast; do
  for round in 0 1 2; do
    order='0 auto'
    if [[ $round == 1 ]]; then order='auto 0'; fi
    for selection in $order; do
      unset GENERATIVEQC_KS_FIXED_ANCHOR_DELTA
      if [[ $selection == 0 ]]; then export GENERATIVEQC_KS_FIXED_ANCHOR_DELTA=0; fi
      timeout --kill-after=30 700 python -m benchmarks.fast_anchor_default.endpoint \
        --atoms "$atoms" --policy "$policy" --delta "$selection" \
        --warm-repeats 3 --production-defaults-changed \
        --output "$output/water-$atoms-$policy-r$round-$selection.json"
    done
  done
done
