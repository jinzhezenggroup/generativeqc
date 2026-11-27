#!/usr/bin/env bash
# Run under a finite qz H100 Job. All software and old inputs are read-only.
set -euo pipefail
driver_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "${1:?task root}" && pwd)
historical=$(cd "${2:?frozen acquisition root}" && pwd)
cold3_inputs=$(cd "${3:?cold3 inputs}" && pwd)
action=${4:?oracle, prove, or measure required}
: "${GQC_QZ_JOB_NAME:?}"
: "${GQC_QZ_PROJECT:?}"
output="$root/$GQC_QZ_JOB_NAME"
mkdir "$output"
exec > >(tee "$output/job.log") 2>&1

projects="$(dirname "$root")"
portable="$projects/h100-portability/20261027-fd06"
site="$projects/h100-portability/20261030-h100-production-15ab-2241/reference-site"
python="$portable/env/bin/python"
cuda=/opt/generativeqc-reuse/cuda-12.9
provider="$portable/env/lib/python3.11/site-packages/scipy_openblas32/lib"
reference_libs="$site/cutensor/lib:$site/gpu4pyscf/lib:$site/gpu4pyscf/lib/deps/lib"

for name in ${!GENERATIVEQC_@}; do unset "$name"; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PATH="$portable/env/bin:$cuda/bin:/usr/bin:/bin"
export GENERATIVEQC_PROFILE=off
export GENERATIVEQC_STATIONARY_CACHE="$output/runtime-cache"
export GENERATIVEQC_JIT_SCCACHE_ROOT="$output/jit-owners"
export SCCACHE_DIR="$portable/cache/sccache"
libraries="$provider:$cuda/lib64:$cuda/targets/x86_64-linux/lib"

# Each process loads exactly one release image and the Python files that shipped with it.
select_arm() {
    local arm=$1
    export PYTHONPATH="$arm/source/python:$arm/source"
    export GENERATIVEQC_LIBRARY="$arm/build/libgenerativeqc.so"
    export LD_LIBRARY_PATH="$arm/build:$libraries"
    cd "$arm/source"
}

run() {
    local name=$1
    shift
    echo "START $name"
    local code=0
    timeout 1200 "$python" "$@" > "$output/$name.stdout" 2> "$output/$name.stderr" || code=$?
    printf '%s\t%s\n' "$name" "$code" >> "$output/commands.tsv"
    echo "END $name $code"
}

replay() {
    local arm="$root/$1" name=$2 system=$3 domain=$4
    shift 4
    select_arm "$arm"
    local trajectory="$historical/attempt-c/frozen-$system.json"
    run "$name" "$driver_dir/replay.py" --trajectory "$trajectory" \
        --expected-sha256 "${trajectory_sha[$system]}" \
        --output "$output/$name.json" \
        --build-receipt "$arm/build-receipts/build.json" \
        --inputs "$historical/attempt-c/inputs/$domain" "$@"
}

cold3() {
    local arm="$root/$1" name=$2 warms=$3 revision
    select_arm "$arm"
    revision=$("$python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["source_revision"])' \
        "$arm/build-receipts/build.json")
    run "$name" -m benchmarks.readme_defaults standard --method wb97m-v --mode df \
        --atoms 96 --warm-repeats "$warms" --source-revision "$revision" \
        --inputs "$cold3_inputs" --qz-job-name "$GQC_QZ_JOB_NAME" --output "$output/$name.json"
}

nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total --format=csv | tee "$output/device.csv"
echo "PROJECT $GQC_QZ_PROJECT JOB $GQC_QZ_JOB_NAME ACTION $action"
sha256sum "$driver_dir/replay.py" "$driver_dir/acquire_qz.sh" > "$output/drivers.sha256"

declare -A trajectory_sha=(
    [paper-water96]=2e1a4ad00ccc7cf6ff37bf89f1ef83e153d6c7344e0b608340a49a2eee895902
    [n-c32h66]=153e03fab905650a033613f946812f3ce764aa9bbd8ac83810d4fa0a549c3e20
)
systems=('paper-water96 water 8' 'n-c32h66 alkane 3')

case "$action" in
    oracle)
        export PYTHONPATH="$historical/source/python:$historical/source:$site"
        export LD_LIBRARY_PATH="$libraries:$reference_libs"
        cd "$historical/source"
        for row in "${systems[@]}"; do
            read -r system domain step <<< "$row"
            run "oracle-$system-$step" "$driver_dir/replay.py" \
                --trajectory "$historical/attempt-c/frozen-$system.json" \
                --expected-sha256 "${trajectory_sha[$system]}" \
                --mode oracle --step "$step" --output "$output/oracle-$system-$step.json"
        done
        ;;
    prove)
        for arm in master candidate; do
            for row in "${systems[@]}"; do
                read -r system domain step <<< "$row"
                for mode in cold pair moved-cold; do
                    replay "$arm" "$arm-$system-$mode" "$system" "$domain" \
                        --mode "$mode" --step "$step" --inspect-route
                done
                replay "$arm" "$arm-$system-moved-energy" "$system" "$domain" \
                    --mode pair --step "$step" --energy-only --inspect-route
            done
            GENERATIVEQC_DFT_CUDA_TEST=1 run "$arm-pytest" -m pytest -q \
                "$root/regression/tests/python/test_wb97mv_df_moved_cutoff_cuda.py"
        done
        ;;
    measure)
        for arm in master candidate; do
            cold3 "$arm" "$arm-cold3-prewarm" 1
            for row in "${systems[@]}"; do
                read -r system domain step <<< "$row"
                replay "$arm" "$arm-$system-prewarm" "$system" "$domain" --mode replay --step 0
            done
        done
        for repeat in 1 2 3; do
            arms=(master candidate)
            if [[ $repeat = 2 ]]; then arms=(candidate master); fi
            for arm in "${arms[@]}"; do
                cold3 "$arm" "$arm-cold3-r$repeat" 3
            done
        done
        for repeat in 1 2; do
            arms=(master candidate)
            if [[ $repeat = 2 ]]; then arms=(candidate master); fi
            for row in "${systems[@]}"; do
                read -r system domain step <<< "$row"
                for arm in "${arms[@]}"; do
                    replay "$arm" "$arm-$system-r$repeat" "$system" "$domain" --mode replay --step 10
                done
            done
        done
        ;;
    *) echo "Unknown acquisition action: $action" >&2; exit 2 ;;
esac
