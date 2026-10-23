"""Audit completed cohorts; preserve every source receipt and process observation."""
if not __debug__:
 raise SystemExit("Evidence verification requires assertions; rerun without -O/-OO or PYTHONOPTIMIZE.")

import copy, hashlib, json, lzma
from pathlib import Path
from statistics import median
from benchmarks.readme_defaults_qualification import qualify, _comparison
from benchmarks.readme_defaults_report import load_records
root=Path(__file__).resolve().parent; which='wb-pr'
def read(p):
 return json.loads(lzma.decompress(p.read_bytes()) if p.suffix=='.xz' else p.read_bytes())
def stats(b,a):
 bm,am=median(b),median(a); bd=median(abs(v-bm) for v in b); ad=median(abs(v-am) for v in a)
 floor=max(.02,2*(bd/bm+ad/am)); gain=1-am/bm
 return dict(before=b,after=a,before_median=bm,after_median=am,before_mad=bd,after_mad=ad,gain=gain,gain_floor=floor,passes_gain_gate=gain>floor)
r=root.parent; refs=r/'readme-final-20261022'
oracles=load_records([refs/'oracles.json.xz'],schema='generativeqc.readme-defaults-oracle.v1'); reuse=read(refs/'reference-reuse.json')
data=read(root/'receipts.json.xz')
revision=data['source_revision']; delta=data['source_manifest']
files=read(root/delta['base'])['files']
for name in delta['removed']: del files[name]
files.update(delta['changed'])
manifest=dict(source_revision=revision, files=dict(sorted(files.items())))
assert len(files)==delta['file_count']
assert hashlib.sha256((json.dumps(manifest,indent=2)+'\n').encode()).hexdigest()==delta['sha256']
cohorts=data['cohorts']
if which=='wb-pr':
 moved={('wb97m-v',mode):record for mode,record in data['moved_references'].items()}
else:
 moved={('pbe','df'):read(r/'df-raw-coulomb-20261023/stage2-moved-reference.json.xz')['pbe-df-96-gpu4pyscf.json'],('r2scan-3c','df'):read(r/'df-md-charge-splits-20261023/integration.json.xz')['cohorts']['stage7-pr-moved-matrix']['r2scan-3c-df-moved-gpu4pyscf.json'],('r2scan-3c','direct'):data['moved_references']['r2scan-3c/direct'],('pbe','direct'):cohorts['stage12-extra']['pbe-direct-moved-gpu4pyscf.json']}
result=dict(native_calls=0,gpu4_calls=0,records={},timing={}); records_by_cohort={}
for cohort in cohorts:
 records_by_cohort[cohort]={}
 for name,record in cohorts[cohort].items():
  path=Path(cohort)/name; records_by_cohort[cohort][name]=record
  assert record['status']=='complete',(path,record.get('error'))
  if which=='stage12' and record['arm']=='gpu4pyscf': continue
  assert record['source_revision']==revision,(path,record['source_revision'])
  view=copy.deepcopy(record); out={}
  if 'moved_geometry_bohr' in record:
   ref=moved[record['method'],record['mode']]
   assert record['moved_geometry_bohr']==ref['geometry_bohr']
   for key in ('method','mode','orbital_basis','basis_representation','charge','spin','auxiliary_input_sha256'):
    assert record.get(key)==ref.get(key),(path,key)
   out['moved']=[_comparison(s,ref['samples'][0],record['atoms']) for s in record['samples'][2:]]
   assert all(e['passed'] for e in out['moved']),(path,out['moved'])
   view['samples']=view['samples'][:2]; view['warm_repeats']=1
  q=qualify([view],oracles,reference_reuse=reuse)[0]
  assert q['status']=='qualified',(path,q)
  out['stationary']=q; result['records'][cohort+'/'+path.name]=out
  result['gpu4_calls' if record['arm']=='gpu4pyscf' else 'native_calls']+=len(record['samples'])
for mode in ('df','direct'):
 pairs=records_by_cohort[which+'-pairs-'+mode]
 for method in (('wb97m-v',) if which=='wb-pr' else ('pbe','r2scan-3c')):
  for phase in ('cold','warm'):
   if which=='wb-pr':
    names=[[f'wb-{mode}-{arm}-{i}.json' for i in range(5)] for arm in ('gpu4pyscf','standard')]
   else: names=[[f'{method}-{mode}-spatial-{variant}-{i}.json' for i in range(5)] for variant in (1,2)]
   values=[[median(s['seconds'] for s in pairs[n]['samples'] if s['phase']==phase) for n in group] for group in names]
   result['timing'][f'{method}/{mode}/{phase}']=stats(*values)
assert result['native_calls']==(56 if which=='wb-pr' else 272)
assert result['gpu4_calls']==(40 if which=='wb-pr' else 0)
print(json.dumps(result,indent=2,sort_keys=True))
