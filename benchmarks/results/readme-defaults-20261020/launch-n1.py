"""Bounded per-case Slurm jobs, eight independent GPU workers on n1."""
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

root = Path('/data/jzzeng/readme-defaults-20261020')
methods = ('hf', 'pbe', 'pbe0', 'b3lyp', 'r2scan', 'r2scan-3c', 'm06-2x', 'wb97m-v')
arms = sys.argv[1:]
# Long populations go first; each process contains cold + three frozen-seed warm calls.
cases = [(arm, method, mode, size) for size in (96, 48, 12, 3)
         for mode in ('direct', 'df') for method in reversed(methods) for arm in arms]

def run(case):
    arm, method, mode, size = case
    name = f'{arm}-{method}-{mode}-{size}'
    directory = root / 'measurements'
    directory.mkdir(exist_ok=True)
    output = directory / f'{name}.json'
    if output.exists():
        return {'case': name, 'skipped_existing': True}
    command = ['srun', '--partition=main', '--nodelist=node1', '--gres=gpu:5090:1',
               '--nodes=1', '--ntasks=1', '--cpus-per-task=2', '--mem=48G',
               '--time=04:00:00' if size == 96 else '--time=01:30:00',
               '--job-name=readme-' + method,
               'bash', str(root / 'run.sh'), arm, method, mode, str(size), str(output)]
    with (directory / f'{name}.log').open('w') as log:
        done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    outcome = {'case': name, 'argv': command, 'exit_code': done.returncode}
    (directory / f'{name}.outcome.json').write_text(json.dumps(outcome, indent=2)+'\n')
    return outcome

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for result in pool.map(run, cases):
        print(json.dumps(result), flush=True)
