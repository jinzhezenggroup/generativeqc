import os
from pathlib import Path
import subprocess
import time
root=Path('/data/jzzeng/readme-defaults-20261020')
for _ in range(360):
    if (root/'receipts/library.sha256').exists():
        break
    time.sleep(5)
else:
    raise RuntimeError('base build did not finish within 30 minutes')
os.environ['PATH']='/home/jzzeng/.local/bin:/data/jzzeng/qc-provider-1886-tools/bin:/home/jzzeng/codes/wb97m-20261002/gpu-env/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin'
os.environ['LD_LIBRARY_PATH']='/group/software/cuda-12.9.1/lib64'
cache='/home/jzzeng/.local/bin/sccache'
subprocess.run([cache,'--version'],check=True)
with (root/'receipts/cache-aot-before.txt').open('w') as f:
    subprocess.run([cache,'--show-stats'],stdout=f,check=True)
targets=[f'generativeqc_stationary_{method}_rks{suffix}_manifest' for method in ('pbe','pbe0','b3lyp','r2scan','m06_2x') for suffix in ('','_spd')]
subprocess.run(['cmake','--build',str(root/'source/build/cuda-release-sm120'),'--target',*targets,'-j','16'],check=True)
with (root/'receipts/cache-aot-after.txt').open('w') as f:
    subprocess.run([cache,'--show-stats'],stdout=f,check=True)
(root/'receipts/aot-complete').write_text('complete\n')
