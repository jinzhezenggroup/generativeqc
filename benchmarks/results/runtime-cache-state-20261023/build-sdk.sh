#!/usr/bin/env bash
# CPU-only Slurm preparation; never probes or allocates a CUDA device.
set -euo pipefail
root=/data/srxu/projects/gqc-runtime-cache-state-20261023
cd "$root"
: "${SLURM_JOB_ID:?requires real CPU Slurm allocation}"
test "$(id -un)" = srxu
test ! -e receipts/build-started
mkdir -p receipts compiler-cache
printf '%s\n' "$SLURM_JOB_ID" > receipts/build-started
trap 'rc=$?; printf "%s\n" "$rc" > receipts/build-exit-code' EXIT
/usr/bin/scontrol show job "$SLURM_JOB_ID" > receipts/allocation.txt
sha256sum --check source-archive.sha256
mkdir source
tar -xf source-ececb48.tar.gz -C source
/usr/bin/python3 - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path.cwd()
manifest = json.loads((root / 'source-ececb48-manifest.json').read_text())
assert manifest['revision'] == 'ececb48cd56a00a56fbb6223dc53bc61dfe6e11d'
assert len(manifest['files']) == 12353
for row in manifest['files']:
    relative = Path(row['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    assert '.env' not in relative.parts and not relative.name.startswith('.env.')
    path = root / 'source' / relative
    assert not path.is_symlink()
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
    assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == row['git_oid'], row['path']
    assert ('100755' if path.stat().st_mode & 0o111 else '100644') == row['mode'], row['path']
(root / 'receipts/source-verified.json').write_text(json.dumps({'revision': manifest['revision'], 'files': len(manifest['files']), 'status': 'PASS'}) + '\n')
PY
/usr/bin/python3 -m venv --without-pip runtime-env
# Use an existing owned pip executable to seed the new isolated Python 3.12 env.
# Explicit public index avoids recording private index/credential configuration.
/home/srxu/miniconda3/bin/python -m pip --isolated --python "$root/runtime-env/bin/python" install \
  --index-url https://pypi.org/simple pip \
  'numpy==2.4.6' 'scipy>=1.10' 'pyscf==2.14.0' 'cupy-cuda12x==13.6.0' \
  'gpu4pyscf-cuda12x==1.8.1' 'sccache==0.16.0' 'cmake>=3.28,<5' ninja \
  'array-api-compat>=1.15.0' typing-extensions 'jinja2>=3' sympy mpmath \
  'nvidia-cublas-cu12>=12.9.1.4' 'nvidia-cusolver-cu12>=11.7.5.82' \
  'nvidia-cusparse-cu12>=12.5.10.65' 'nvidia-cuda-runtime-cu12>=12.9.79' \
  'nvidia-nvjitlink-cu12>=12.9.86'
export PATH="$root/runtime-env/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin"
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64"
export PYTHONPATH="$root/source/python:$root/source"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SCCACHE_DIR="$root/compiler-cache"
export SCCACHE_CONF="$root/compiler-cache.toml"
export SCCACHE_CACHED_CONF="$root/compiler-cache-cached.toml"
export SCCACHE_SERVER_UDS="$root/compiler-cache.sock"
export CCACHE_BASEDIR="$root/source"
# The batch submission uses --export=NONE; no remote compiler-cache backend or
# experimental production flags are inherited. Never stop a shared cache daemon.
printf '' > "$SCCACHE_CONF"
sccache --version > receipts/compiler-cache-version.txt
python -c 'import re; from pathlib import Path; v=Path("receipts/compiler-cache-version.txt").read_text(); m=re.search(r"sccache (\d+)\.(\d+)\.(\d+)",v); assert m and tuple(map(int,m.groups())) >= (0,16,0)'
sccache --show-stats > receipts/compiler-cache-before.txt
cmake -S source -B source/build/cuda-release-sm120 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root/sdk" \
  -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$root/runtime-env/bin/sccache" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$root/runtime-env/bin/sccache" \
  -DPython3_EXECUTABLE="$root/runtime-env/bin/python" \
  -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF \
  -DGENERATIVEQC_ENABLE_CUDA=ON -DGENERATIVEQC_AOT_PROFILE=sm_120 \
  -DGENERATIVEQC_CUDA_ARCHITECTURES=120 -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120-real \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF \
  -DGENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=ON \
  -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=ON \
  -DGENERATIVEQC_AOT_COMPILE_JOBS=1 -DGENERATIVEQC_CUDA_COMPILE_JOBS=2
ninja -C source/build/cuda-release-sm120 -t commands > receipts/compiler-commands.txt
cmake --build source/build/cuda-release-sm120 --parallel 4
sccache --show-stats > receipts/compiler-cache-after.txt
cmake --install source/build/cuda-release-sm120
runtime-env/bin/python -m pip freeze > receipts/packages.txt
printf 'built; device qualification NOT_RUN\n' > receipts/build-complete
