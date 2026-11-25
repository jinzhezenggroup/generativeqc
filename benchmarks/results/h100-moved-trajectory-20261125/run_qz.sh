#!/usr/bin/env bash
# Invoke inside one allocated H100 worker. Shared dependencies are read-only.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source="$root/source"
output="$root/${1:?attempt directory required}"
: "${GQC_QZ_JOB_NAME:?qz job name required}" "${GQC_QZ_PROJECT:?qz project required}"
base="$(dirname "$root")/h100-portability/20261027-fd06"
prior="$(dirname "$root")/paper-cold3-20261120-471"
site="$(dirname "$base")/20261030-h100-production-15ab-2241/reference-site"
python="$base/env/bin/python"
cuda=/opt/generativeqc-reuse/cuda-12.9
provider="$base/env/lib/python3.11/site-packages/scipy_openblas32/lib"
reference_libs="$site/cutensor/lib:$site/gpu4pyscf/lib:$site/gpu4pyscf/lib/deps/lib"
library="$root/build/libgenerativeqc.so"
inputs="$prior/inputs"

mkdir "$output"
cd "$source"
for name in ${!GENERATIVEQC_@}; do unset "$name"; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PATH="$base/env/bin:$cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$source:$source/python"
export LD_LIBRARY_PATH="$provider:$cuda/lib64:$cuda/targets/x86_64-linux/lib"
export GENERATIVEQC_PROFILE=off
export GENERATIVEQC_STATIONARY_CACHE="$output/runtime-cache"
export GENERATIVEQC_JIT_SCCACHE_ROOT="$output/jit-owners"
export SCCACHE_DIR="$base/cache/sccache"

# The build owner records its exact source, compiler-cache stages and ELF.
"$python" -u "$root/build_qz.py"
"$python" - "$root" "$output" <<'PYRUN'
import hashlib
import json
import sys
from pathlib import Path

root, output = map(Path, sys.argv[1:])
build_receipt = root / "build-receipts/build.json"
receipt = json.loads(build_receipt.read_text())
if receipt["status"] != "complete":
    raise ValueError("no completed build receipt")
(output / "preflight.json").write_text(json.dumps({
    "source_archive_sha256": receipt["source_archive_sha256"],
    "native_source_identity": receipt["native_source_identity"],
    "library_sha256": receipt["library_sha256"],
    "build_revision": receipt["source_revision"],
    "build_receipt_sha256": hashlib.sha256(build_receipt.read_bytes()).hexdigest(),
}, indent=2) + "\n")
PYRUN

driver="$root/trajectory.py"
frozen="$output/trajectory.json"
"$python" "$driver" plan --output "$frozen"
failures=0
run_arm() {
    local arm=$1 method=$2 mode=$3
    local stem="$output/$method-$mode-$arm"
    local -a settings command
    settings=()
    if [[ "$arm" == standard ]]; then
        settings=("GENERATIVEQC_LIBRARY=$library" "LD_LIBRARY_PATH=$root/build:$LD_LIBRARY_PATH")
    else
        settings=("PYTHONPATH=$site:$PYTHONPATH" "LD_LIBRARY_PATH=$reference_libs:$LD_LIBRARY_PATH")
    fi
    command=(env "${settings[@]}" "$python" -u "$driver" "$arm" --method "$method" --mode "$mode"
        --trajectory "$frozen" --inputs "$inputs" --output "$stem.json")
    echo "START $method $mode $arm"
    local code=0
    timeout -k 20s 90m "${command[@]}" > "$stem.txt" 2>&1 || code=$?
    printf '%s %s %s %s\n' "$method" "$mode" "$arm" "$code" >> "$output/exit-codes.txt"
    echo "END $method $mode $arm $code"
    if (( code != 0 )); then
        failures=$((failures + 1))
    fi
}
for method in wb97m-v pbe r2scan pbe0; do
    run_arm standard "$method" direct
    run_arm gpu4pyscf "$method" direct
    run_arm gpu4pyscf "$method" df
    run_arm standard "$method" df
done
for method in pbe r2scan pbe0 wb97m-v; do
    run_arm oracle "$method" direct
done
printf 'failures=%s\n' "$failures"
(( failures == 0 ))
