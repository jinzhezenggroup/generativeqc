#!/usr/bin/env bash
# Separate bounded processes keep reference evidence even if native fails.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through finite srun on main with --gres=gpu:5090:1}"
: "${GENERATIVEQC_LIBRARY:?Select the recorded Release native library}"
: "${PBE0_BASIS_FILE:?Select the offline spherical def2-SVP H/O snapshot}"
pbe0_python="${PBE0_BENCHMARK_PYTHON:-python}"
pbe0_output="${PBE0_BENCHMARK_OUTPUT:-.artifacts/pbe0-benchmarks}"
pbe0_timeout="${PBE0_POINT_TIMEOUT:-360}"
pbe0_sizes="${PBE0_ATOMS:-3 6 12 24 48 96}"
pbe0_engines="${PBE0_ENGINES:-reference native}"
[[ "$pbe0_timeout" =~ ^[1-9][0-9]*$ ]] || exit 2
export PYTHONPATH=".:python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
pbe0_failed=0
for atoms in $pbe0_sizes; do
    mkdir -p "$pbe0_output/$atoms"
    for engine in $pbe0_engines; do
        [[ "$engine" == reference || "$engine" == native ]] || exit 2
        extra=()
        if [[ "$engine" == native ]]; then
            if ! "$pbe0_python" -c 'import json, pathlib, sys; path = pathlib.Path(sys.argv[1]); sys.exit(0 if path.exists() and json.loads(path.read_text()).get("status") == "measured" else 1)' \
                "$pbe0_output/$atoms/reference.json"; then
                printf '{"exit_code":null,"reason":"reference_unavailable"}\n' \
                    > "$pbe0_output/$atoms/native.outcome"
                printf 'native atoms=%s skipped: no complete independent reference\n' "$atoms"
                pbe0_failed=1
                continue
            fi
            extra=(--reference "$pbe0_output/$atoms/reference.json")
        fi
        exit_code=0
        timeout --kill-after=5 "$pbe0_timeout" "$pbe0_python" -m benchmarks.readme_pbe0 \
            "$engine" --atoms "$atoms" --basis-file "$PBE0_BASIS_FILE" \
            --repeats 5 "${extra[@]}" --output "$pbe0_output/$atoms/$engine.json" \
            > "$pbe0_output/$atoms/$engine.log" 2>&1 || exit_code=$?
        printf '{"exit_code":%s,"time_limit_seconds":%s}\n' "$exit_code" "$pbe0_timeout" \
            > "$pbe0_output/$atoms/$engine.outcome"
        printf '%s atoms=%s exit=%s\n' "$engine" "$atoms" "$exit_code"
        [[ "$exit_code" == 0 ]] || pbe0_failed=1
    done
done
exit "$pbe0_failed"
