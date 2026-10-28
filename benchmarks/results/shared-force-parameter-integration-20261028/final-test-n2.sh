set -euo pipefail
cd /home/jzzeng/tmp/qc-pr-unblock-2835-20261028
export PATH=/home/jzzeng/tmp/qc-1123-shared-aot-20261027/.venv/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin:/home/jzzeng/miniconda3/bin
export LD_LIBRARY_PATH=$PWD/build/integration:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=python:tests/python:.
export GENERATIVEQC_LIBRARY=$PWD/build/integration/libgenerativeqc.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GENERATIVEQC_AUTOMATIC_CUDA_FORCE_TEST=1
mkdir -p .artifacts/final-validation
python - <<'PY'
import hashlib,json,os
from pathlib import Path
p=Path('.artifacts/2835-final-inputs.json'); inv=json.loads(p.read_text()); mismatches=[]
for name,want in inv['files'].items():
 f=Path(name)
 if not f.is_file():
  mismatches.append([name,'missing']);continue
 data=f.read_bytes(); got=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 if got!=want: mismatches.append([name,got,want])
result={'source_commit':inv['source_commit'],'input_count':len(inv['files']),'input_inventory_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mismatches':mismatches,'slurm_job_id':os.environ['SLURM_JOB_ID']}
Path('.artifacts/final-validation/build-input-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
assert not mismatches
PY
set +e
python .artifacts/2835-final-force.py > .artifacts/final-validation/force.log 2>&1
force_status=$?
python .artifacts/2835-final-composite.py > .artifacts/final-validation/composite-cli.log 2>&1
cli_status=$?
ctest --test-dir build/integration --output-on-failure -R '^generativeqc_(mp2_contract|ump2_contract|scf_diagnostic|cosx_cuda)_tests$' > .artifacts/final-validation/ctest.log 2>&1
ctest_status=$?
python -m pytest -q tests/python/test_r1_native_forces.py -k cpu --junitxml=.artifacts/final-validation/r1-cpu.xml > .artifacts/final-validation/r1-cpu.log 2>&1
cpu_status=$?
python - "$force_status" "$cli_status" "$ctest_status" "$cpu_status" <<'PY'
import json,os,sys
from pathlib import Path
result=dict(zip(('force_matrix','composite_cli','ctest','r1_cpu'),map(int,sys.argv[1:]),strict=True))
Path('.artifacts/final-validation/status.json').write_text(json.dumps({'slurm_job_id':os.environ['SLURM_JOB_ID'],'exit_codes':result},indent=2)+'\n')
print(result)
sys.exit(any(result.values()))
PY
