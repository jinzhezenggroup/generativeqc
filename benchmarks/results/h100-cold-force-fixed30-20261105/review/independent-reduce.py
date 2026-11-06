from pathlib import Path
import hashlib,json,lzma,math,statistics
R=Path('/Users/xshengrui/Projects/generativeqc-code/h100-cold-20261105')
O=R/'cold-force-final-audit-sol-v1'
HASH={};CHECKS=[]
def digest(p):
 p=Path(p);b=p.read_bytes();HASH[str(p)]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());return HASH[str(p)]['sha256']
def read(p):
 p=Path(p);digest(p);return json.loads(lzma.decompress(p.read_bytes()) if p.suffix=='.xz' else p.read_text())
def check(x,m):
 if not x:raise ValueError(m)
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def metric(a,b,n):
 check(len(a['forces'])==len(b['forces'])==n and all(len(z)==3 for z in a['forces']+b['forces']),'force shape')
 d=[x-y for u,v in zip(a['forces'],b['forces']) for x,y in zip(u,v)]
 out=dict(energy_abs_per_atom_hartree=abs(a['energy']-b['energy'])/n,force_max_abs_hartree_per_bohr=max(map(abs,d)),force_rms_hartree_per_bohr=math.sqrt(sum(v*v for v in d)/len(d)))
 check(all(math.isfinite(v) for v in out.values()),'nonfinite metric');return out
G={'energy_abs_per_atom_hartree':1e-5,'force_max_abs_hartree_per_bohr':1e-4,'force_rms_hartree_per_bohr':3e-5}
def passed(m,f=1):return all(m[k]<=v*f for k,v in G.items())
def close(a,b):return math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-18)
T=R/'received-fixed30-v1';P=read(T/'fixed30-protocol-v1.json');C=read(T/'fixed30-v1/campaign.json');B=read(T/'base/build-receipts-v4/build.json');SEAL=read(T/'fixed30-preparation-v1.json');INPUTS=read(T/'inputs-preparation-v1.json')
for n,h in SEAL['files'].items():check(digest(T/n)==h,'sealed file '+n)
for n,h in SEAL['base_files'].items():check(digest(T/'base'/n)==h,'sealed base '+n)
check(digest(T/'fixed30-protocol-v1.json')==C['protocol_sha256'],'campaign protocol')
check(digest(T/'fixed30-preparation-v1.json')==C['preparation_sha256'],'campaign preparation')
check(B['source']==C['source']==P['source']==read(T/'base/source-v4.json'),'source bindings')
check(C['build_artifacts']==B['artifacts'] and C['native_source_identity']==B['native_source_identity'],'build bindings')
orig=read(T/'original-fixed30-pairs-protocol.json')
for k in ['primary_physical_cases','excluded_rows','order','numerical_gates','win_rule']:check(orig[k]==P[k],'original contract '+k)
check(P['numerical_gates']==G,'original gates')
check(len(P['primary_physical_cases'])==30 and len(P['order'])==180,'original cardinality')
for group,decl in INPUTS['groups'].items():
 for n,b in decl['files'].items():check(digest(T/group/n)==b['sha256'],'input '+n)
ORACLES=read(T/'oracles.json.xz');REUSE=read(T/'reference-reuse.json')['entries'];OI={}
check(len(ORACLES)==40,'40 oracle receipts')
OM=[]
for o in ORACLES:
 key=(o['method'],o['atoms'],o['backend']);check(key not in OI,'duplicate oracle');OI[key]=o
 check(o['status']=='complete' and o['common_observable_gates']==G and o['grid_stability_gate_fraction']==.25,'oracle contract')
 for s in o['samples']:
  check(s['controls']==dict(energy=1e-11,orbital_gradient=1e-7,screening=1e-14,full_density_fock=True,force_grid_response=None if o['method']=='hf' else True),'oracle controls')
  check(s['converged'] and len(s['forces'])==o['atoms'],'oracle endpoint')
 if len(o['samples'])==2:
  m=metric(*o['samples'],o['atoms']);check(passed(m,.25),'oracle stability');OM.append(dict(kind='stability',key=key,metrics=m))
for method in sorted({o['method'] for o in ORACLES}):
 cpu=OI[method,3,'cpu']['samples'][-1];candidates=[s for s in OI[method,3,'cuda']['samples'] if s['level']==cpu['level'] and (s.get('nonlocal_grid') or {}).get('level')==(cpu.get('nonlocal_grid') or {}).get('level')]
 check(len(candidates)==1,'matching CPU quadrature');m=metric(cpu,candidates[0],3);check(passed(m,.25),'CPU/GPU spot');OM.append(dict(kind='cpu_gpu_spot',key=method,metrics=m))
HISTORY=read(R/'reference-installation-history-v1.json')['scientific_image_catalog']
INDEX={};POP=[];MAX={k:0 for k in G};ref_images=set();native_images=set()
def qualify_raw(raw,q):
 target=OI[raw['method'],raw['atoms'],'cuda'];check(raw['geometry_bohr']==target['geometry_bohr'],'exact geometry')
 for o in [target,OI[raw['method'],3,'cpu'],OI[raw['method'],3,'cuda']]:
  contract=REUSE[identity(o)];check(contract['source_revision']==o['source_revision'] and contract['driver_sha256']==o['driver_sha256'],'oracle provenance')
  check(contract['scientific_inputs']=={k:raw[k] for k in ['method','orbital_basis','basis_representation','charge','spin']},'science reuse')
  check(contract['reference_packages']==o['environment']['packages'] and raw['environment']['packages'] in contract['measurement_packages'],'exact accepted environment')
 for j,s in enumerate(raw['samples']):
  check(s['converged'] is True and s['seconds']>0 and math.isfinite(s['seconds']),'endpoint complete')
  check((s['phase'],s['repeat'])==(('cold',0) if j==0 else ('warm',j-1)),'phase/index')
  m=metric(s,target['samples'][-1],raw['atoms']);check(passed(m),'full E/F gate')
  if q:
   stored=q[s['phase']]['errors'][0 if j==0 else j-1]
   check(all(close(m[k],stored[k]) for k in G),'stored arithmetic differs')
  for k in G:MAX[k]=max(MAX[k],m[k])
 return True
for i,row in enumerate(C['runs']):
 d=P['order'][i];check(row['declaration']==d,'population order');n=row['name'];folder=T/'fixed30-v1'
 plan=read(folder/(n+'.plan.json'));raw=read(folder/(n+'.json'));q=read(folder/(n+'.qualification.json'))
 for suffix in ['.stdout','.stderr']:digest(folder/(n+suffix))
 check(digest(folder/(n+'.plan.json'))==row['plan_sha256']==raw['plan_sha256'],'plan/raw binding')
 check(digest(folder/(n+'.json'))==row['raw_sha256'],'raw binding')
 check(plan['paired_process']==d,'dispatched pair');check(plan['source_revision']==raw['source_revision']==C['source']['revision'],'run source')
 for k in ['method','atoms','mode','arm','geometry_bohr','qz_allocation']:check(raw[k]==plan[k],'run contract '+k)
 check(raw['runner_sha256']==plan['endpoint_sha256']==SEAL['files']['endpoint-fixed30-v1.py'],'producer')
 check(raw['driver_sha256']==plan['driver_sha256'],'driver')
 check(raw['status']=='complete' and len(raw['samples'])==4 and row['returncode']==0,'complete execution')
 check(raw['accelerator']['nvidia_smi']['uuid']==C['device_uuid'] and raw['accelerator']['compute_capability']==[9,0],'physical device')
 check(row['scientific_overrides']=={} and raw['occupied_xc_env'] is None,'default controls')
 qualify_raw(raw,q)
 images=raw['runtime_images_after_endpoint'];check(images,'runtime mapping')
 ni={n:b for n,b in images.items() if Path(n).name.startswith('libgenerativeqc')}
 if raw['arm']=='standard':
  core=raw['native_build']['library_path'];check(core in ni and raw['native_build']['library_sha256']==B['artifacts']['libgenerativeqc.so'],'native mapped core')
  check(raw['native_source_identity']==C['native_source_identity'],'compiled source')
  for name,b in ni.items():
   check(Path(name).parent==Path(plan['library_root']),'library location')
   expected=B['artifacts']['libgenerativeqc.so'] if name==core else B['artifacts'].get(Path(name).name)
   check(b['sha256']==expected and b['bytes']>0,'mapped native bytes');native_images.add(Path(name).name)
 else:
  check(not ni,'reference native leakage');observed={n:b for n,b in images.items() if Path(n).is_relative_to(plan['reference_site'])};check(observed,'reference mappings')
  for name,b in observed.items():check(name in HISTORY and all(b[k]==HISTORY[name][k] for k in ['bytes','sha256']),'reference mapped bytes');ref_images.add(name)
 timing={p:statistics.median([s['seconds'] for s in raw['samples'] if s['phase']==p]) for p in ['cold','warm']}
 key=(d['case_id'],d['pair_index'],d['arm']);check(key not in INDEX,'duplicate population');INDEX[key]=timing
 POP.append(dict(name=n,timing=timing,iterations=[s['iterations'] for s in raw['samples']],builds=[s.get('fock_builds',s.get('effective_potential_builds')) for s in raw['samples']],settings=[s.get('settings') for s in raw['samples']],source_revision=raw['source_revision']))
 check(all(type(s.get('fock_builds',s.get('effective_potential_builds'))) is int and s.get('fock_builds',s.get('effective_potential_builds'))>=1 for s in raw['samples']),'actual SCF counts')
 if raw['arm']=='standard' and d['case_id']=='wb97m-v-water-96-direct':
  for s in raw['samples']:check(all(s['settings'].get(k)==v for k,v in P['required_default_integration'].items() if k not in ['method','atoms','mode']),'observed automatic force/default SCF')
CASES=[]
for c in P['primary_physical_cases']:
 phase={}
 for p in ['cold','warm']:
  ratios=[INDEX[c['id'],j,'gpu4pyscf'][p]/INDEX[c['id'],j,'standard'][p] for j in range(3)];med=statistics.median(ratios)
  phase[p]=dict(ratios=ratios,median=med,win=med>1.05 and min(ratios)>1.)
 CASES.append(dict(id=c['id'],route=c['route'],phase=phase))
S={}
for p in ['cold','warm']:
 S[p]={}
 for route in ['all','direct','df']:
  cases=[c for c in CASES if route=='all' or route==c['route']]
  S[p][route]=dict(cases=len(cases),wins=sum(c['phase'][p]['win'] for c in cases),complete_set_geomean=math.exp(sum(math.log(c['phase'][p]['median']) for c in cases)/len(cases)))
analysis=read(R/'fixed30-analysis-v1.json')
for p in S:
 for route in S[p]:
  check(S[p][route]['wins']==analysis['summary'][p][route]['wins'] and close(S[p][route]['complete_set_geomean'],analysis['summary'][p][route]['complete_set_geomean']),'published reducer mismatch')
# Closed independent studies: retain the combined failure while each study is reduced separately.
STUDIES=[]
for study in ['outer','anchor','work','hf-streaming','full-cache']:
 root=R/('received-'+study+'-v1');cp=next(root.rglob('campaign.json'));campaign=read(cp);folder=cp.parent;index={}
 for row in campaign['runs']:
  n=row['name'];raw=read(folder/(n+'.json'));plan=read(folder/(n+'.plan.json'));q=read(folder/(n+'.qualification.json'))
  for suf in ['.stdout','.stderr']:digest(folder/(n+suf))
  check(digest(folder/(n+'.json'))==row['raw_sha256'] and digest(folder/(n+'.plan.json'))==row['plan_sha256']==raw['plan_sha256'],'closed-study binding')
  check(raw['status']=='complete' and row['returncode']==0 and len(raw['samples'])==4,'closed complete')
  check(raw['accelerator']['nvidia_smi']['uuid']==campaign['device_uuid'],'closed single GPU')
  qualify_raw(raw,q);index[row['study'],row['pair_index'],row['variant']]=raw
 for s in campaign['studies']:
  rows=[]
  for pair in s['pairs']:
   j=pair['pair_index'];off=index[s['id'],j,'off'];cands=[v for (sid,pidx,v) in index if sid==s['id'] and pidx==j and v not in ['off','gpu4pyscf']];check(len(cands)==1,'one candidate');candidate=index[s['id'],j,cands[0]];ref=index[s['id'],j,'gpu4pyscf']
   for a,b in zip(candidate['samples'],off['samples']):check(passed(metric(a,b,off['atoms'])),'matched closed gates')
   ratios={p:statistics.median([z['seconds'] for z in candidate['samples'] if z['phase']==p])/statistics.median([z['seconds'] for z in off['samples'] if z['phase']==p]) for p in ['cold','warm']}
   rows.append(dict(pair=j,candidate=cands[0],candidate_over_off=ratios,warm_call_ratios=[candidate['samples'][j]['seconds']/off['samples'][j]['seconds'] for j in range(1,4)],off_builds=[z.get('fock_builds') for z in off['samples']],candidate_builds=[z.get('fock_builds') for z in candidate['samples']]))
  STUDIES.append(dict(study=study,id=s['id'],pairs=rows,median_cold_reduction=1-statistics.median(p['candidate_over_off']['cold'] for p in rows),median_warm_change=statistics.median(p['candidate_over_off']['warm'] for p in rows)-1,status=s['status']))
result=dict(status='INDEPENDENT_OFFLINE_REDUCTION_PASS',method='standard-library independent full-vector arithmetic, no imports of executor analyzers or production solvers',frozen30=dict(source=C['source'],device_uuid=C['device_uuid'],populations=len(POP),returned_EF=sum(4 for _ in POP),summary=S,cases=CASES,cold_nonwins=[c for c in CASES if not c['phase']['cold']['win']],worst_endpoint_errors=MAX,oracle_checks=OM,populations_detail=POP,distinct_reference_images=len(ref_images),native_images_observed=sorted(native_images)),closed_studies=STUDIES)
(O/'independent-reduction-v1.json').write_text(json.dumps(result,indent=2)+'\n');(O/'audited-hashes-partial-v1.json').write_text(json.dumps(HASH,indent=2)+'\n')
print(json.dumps({'summary':S,'cold_nonwins':[(c['id'],c['phase']['cold']) for c in CASES if not c['phase']['cold']['win']],'closed_studies':[{k:v for k,v in s.items() if k!='pairs'} for s in STUDIES],'files_hashed':len(HASH)},indent=2))
