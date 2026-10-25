"""Paired full E/F endpoint receipts; run each version in its own process under srun.

Artifact cache warmups precede timing. Each cold sample includes Calculator and
prepared-context creation, the SCF, force assembly, public output and cleanup.
Warm/moved samples include complete execute() with public output and diagnostics.
No tracing is enabled during timing; unavailable counters remain null.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions

p = argparse.ArgumentParser()
p.add_argument('--samples', type=int, default=21)
p.add_argument('--warmups', type=int, default=3)
p.add_argument('--output', required=True)
p.add_argument('--case', default='all')
a = p.parse_args()
assert os.environ.get('SLURM_JOB_ID')
assert not os.environ.get('GENERATIVEQC_DF_TRACE')
water = [('O',(0.,0.,0.)),('H',(0.,-1.43233673,1.10715266)),('H',(0.,1.43233673,1.10715266))]
cases = []
for method in ('lda','pbe','pbe0','b3lyp','r2scan','wb97m-v','m06-2x','mn15'):
    for spin in (0,1):
        for fitted in (False,True):
            cases.append((f'{method}-{spin}-{int(fitted)}', method, spin, fitted, water, 'sto-3g'))
for count in (36,48):
    atoms=[(e,(x+7.0*(i%4),y+7.0*(i//4),z)) for i in range(count//3) for e,(x,y,z) in water]
    cases.append((f'pbe-water{count}-df-d', 'pbe', 0, True, atoms, 'def2-svp'))

library=Path(os.environ['GENERATIVEQC_LIBRARY']).resolve()
header={'source_revision':os.environ['R1_SOURCE_REVISION'],
        'source_manifest_sha256':hashlib.sha256(Path(os.environ['R1_SOURCE_MANIFEST']).read_bytes()).hexdigest(),
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'slurm_job_id':os.environ['SLURM_JOB_ID'], 'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'root':str(Path.cwd()), 'binary_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
        'samples':a.samples,'warmups':a.warmups,'grid':'product 12/4/8', 'precision':'fp64',
        'energy_tolerance':1e-12,'density_tolerance':1e-10,'screening_tolerance':1e-13,
        'timing':'cold includes construction, preparation, E/F, output and cleanup; warm/moved include execute output',
        'artifact_state':'warm after explicit warmups; no deletion or cache override', 'tracing':False}
f = open(a.output,'w',buffering=1)
f.write(json.dumps({'metadata':header})+'\n')

def output(result):
    assert result.converged and result.succeeded
    return dict(energy=result.energy, forces=result.forces.tolist(), iterations=result.iterations,
                fock_builds=result.fock_builds, warm_start_used=result.warm_start_used,
                executed_backend=result.executed_backend)

for name,method,spin,fitted,atoms,basis in cases:
    if a.case != 'all' and name not in a.case.split(','):
        continue
    def calculator():
        return Calculator(method=f"{method}-{'uks' if spin else 'rks'}", basis=basis,
            basis_representation='spherical',device='cuda',density_fitting='cuda' if fitted else 'none',
            ks_options=KsOptions(grid=GridSpec(12,4,8),nonlocal_grid=GridSpec(12,4,8) if method=='wb97m-v' else None),
            max_iterations=200,energy_tolerance=1e-12,density_tolerance=1e-10,screening_tolerance=1e-13)
    def prepare(calc):
        return calc.prepare_batch([atoms],charges=[spin],multiplicities=[spin+1])
    try:
        for sample in range(-a.warmups,a.samples):
            started=time.perf_counter()
            with prepare(calculator()) as batch:
                cold=batch.execute(strict=True,properties=('energy','forces')).items[0]
                native=batch._native_dft_force_eligible()
            seconds=time.perf_counter()-started
            if sample>=0:
                f.write(json.dumps(dict(case=name,phase='cold',sample=sample,seconds=seconds,native=native,**output(cold)))+'\n')
        positions=np.asarray([v for _,v in atoms]); moved=positions.copy(); moved[0,2]+=0.02
        with prepare(calculator()) as batch:
            batch.execute(strict=True,properties=('energy','forces'))
            for phase in ('warm','moved'):
                for sample in range(-a.warmups,a.samples):
                    coordinates = None if phase=='warm' else [moved if sample%2 else positions]
                    started=time.perf_counter()
                    item=batch.execute(coordinates,strict=True,properties=('energy','forces')).items[0]
                    seconds=time.perf_counter()-started
                    if sample>=0:
                        f.write(json.dumps(dict(case=name,phase=phase,sample=sample,seconds=seconds,**output(item)))+'\n')
        print('PASS',name,flush=True)
    except Exception as error:
        f.write(json.dumps(dict(case=name,error=f'{type(error).__name__}: {error}'))+'\n')
        print('ERROR',name,repr(error),flush=True)
f.close()
