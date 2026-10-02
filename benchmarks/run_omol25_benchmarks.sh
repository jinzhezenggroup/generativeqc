#!/usr/bin/env bash
# Separate finite processes preserve reference evidence if the native point fails.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through finite srun on main with --gres=gpu:5090:1}"
: "${GENERATIVEQC_LIBRARY:?Select a Release library built from the recorded master}"
: "${OMOL25_BASIS_FILE:?Select the offline canonical def2-TZVPD H/O basis}"
omol_python="${OMOL25_BENCHMARK_PYTHON:-python}"
omol_output="${OMOL25_BENCHMARK_OUTPUT:-.artifacts/omol25-benchmarks}"
omol_timeout="${OMOL25_POINT_TIMEOUT:-900}"
omol_sizes="${OMOL25_ATOMS:-3 6 12 24 48 96}"
omol_repeats="${OMOL25_REPEATS:-5}"
omol_engines="${OMOL25_ENGINES:-reference native}"
[[ "$omol_timeout" =~ ^[1-9][0-9]*$ ]] || exit 2
export PYTHONPATH=".:python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
omol_failed=0
for atoms in $omol_sizes; do
    mkdir -p "$omol_output/$atoms"
    for engine in $omol_engines; do
        [[ "$engine" == reference || "$engine" == native ]] || exit 2
        extra=()
        if [[ "$engine" == native ]]; then
            extra=(--reference "$omol_output/$atoms/reference.json")
        fi
        exit_code=0
        timeout --kill-after=5 "$omol_timeout" "$omol_python" -m benchmarks.readme_omol25 \
            "$engine" --atoms "$atoms" --basis-file "$OMOL25_BASIS_FILE" \
            --repeats "$omol_repeats" "${extra[@]}" \
            --output "$omol_output/$atoms/$engine.json" \
            > "$omol_output/$atoms/$engine.log" 2>&1 || exit_code=$?
        printf '{"exit_code":%s,"time_limit_seconds":%s}\n' "$exit_code" "$omol_timeout" \
            > "$omol_output/$atoms/$engine.outcome"
        printf '%s atoms=%s exit=%s\n' "$engine" "$atoms" "$exit_code"
        [[ "$exit_code" == 0 ]] || omol_failed=1
    done
done
exit "$omol_failed"
