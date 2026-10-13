#!/usr/bin/env bash
# Run complete independent endpoints inside a finite, exclusive GPU allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through srun with an explicit finite --time}"
readme_engine="${1:-both}"
case "$readme_engine" in both|reference|native) ;; *) exit 2 ;; esac
if [[ "$readme_engine" != reference ]]; then
    : "${GENERATIVEQC_LIBRARY:?Select a source-matched Release native library}"
fi
readme_python="${README_BENCHMARK_PYTHON:-python}"
readme_output="${README_BENCHMARK_OUTPUT:-.artifacts/readme-dft}"
readme_timeout="${README_BENCHMARK_POINT_TIMEOUT:-1800}"
readme_repeats="${README_BENCHMARK_REPEATS:-5}"
readme_df="${README_BENCHMARK_DENSITY_FITTING:-none}"
case "$readme_df" in none|cuda) ;; *) exit 2 ;; esac
readme_aux="${README_BENCHMARK_AUXILIARY_BASIS:-benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json}"
[[ "$readme_timeout" =~ ^[1-9][0-9]*$ && "$readme_repeats" =~ ^[1-9][0-9]*$ ]] || exit 2
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=".:python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$readme_output/inputs"
"$readme_python" - "$readme_output/inputs/def2-svp.json" <<'PY'
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from generativeqc._model_resolution import snapshot_basis
basis = snapshot_basis("def2-svp", "spherical")
path = Path(sys.argv[1])
# Independent per-functional allocations may publish the same immutable input.
with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
    pending = Path(temporary) / path.name
    replace(basis, elements=tuple(basis.by_element[z] for z in (1, 8))).write(pending)
    pending.replace(path)
PY
readme_failed=0
for method in ${README_BENCHMARK_METHODS:-hf pbe pbe0 b3lyp r2scan r2scan-3c m06-2x wb97m-v}; do
    basis="$readme_output/inputs/def2-svp.json"
    [[ "$method" != r2scan-3c ]] || basis=python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json
    for atoms in ${README_BENCHMARK_SIZES:-3 12 48 96}; do
        directory="$readme_output/$method/$atoms"
        mkdir -p "$directory"
        for engine in reference native; do
            [[ "$readme_engine" == both || "$readme_engine" == "$engine" ]] || continue
            extra=(--density-fitting "$readme_df")
            [[ "$readme_df" != cuda ]] || extra+=(--auxiliary-basis-file "$readme_aux")
            if [[ "$engine" == native ]]; then
                # Reference completion must not decide whether native runs.
                # Available geometry oracles are still checked; absent ones
                # produce explicit unchecked timings, never passing accuracy.
                extra+=(--allow-missing-reference)
                extra+=(--reference "$directory/reference.json")
            fi
            result=0
            timeout --kill-after=10 "$readme_timeout" "$readme_python" -m benchmarks.readme_dft \
                --method "$method" "$engine" --atoms "$atoms" --basis-file "$basis" \
                --repeats "$readme_repeats" "${extra[@]}" --output "$directory/$engine.json" \
                > "$directory/$engine.log" 2>&1 || result=$?
            printf '{"exit_code":%s,"time_limit_seconds":%s}\n' "$result" "$readme_timeout" \
                > "$directory/$engine.outcome"
            printf '%s atoms=%s %s exit=%s\n' "$method" "$atoms" "$engine" "$result"
            [[ "$result" == 0 ]] || readme_failed=1
        done
    done
done
exit "$readme_failed"
