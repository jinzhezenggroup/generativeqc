from pathlib import Path
import json,hashlib,lzma,math,statistics
R=Path('/Users/xshengrui/Projects/generativeqc-code/h100-cold-20261105');T=R/'received-main-integration-v2';O=R/'cold-force-final-audit-sol-v1';H={}
def sha(p):
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest();H[str(p)]={'bytes':len(b),'sha256':h};return h
def read(p):
 sha(p);return json.loads(lzma.decompress(p.read_bytes()) if p.suffix=='.xz' else p.read_text())
def check(c,m):
 if not c:raise ValueError(m)
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
G={'energy_abs_per_atom_hartree':1e-5,'force_max_abs_hartree_per_bohr':1e-4,'force_rms_hartree_per_bohr':3e-5}
def errors(a,b,n):
 check(len(a['forces'])==len(b['forces'])==n and all(len(z)==3 for z in a['forces']+b['forces']),'vector shape')
 ds=[x-y for aa,bb in zip(a['forces'],b['forces']) for x,y in zip(aa,bb)]
 return {'energy_abs_per_atom_hartree':abs(a['energy']-b['energy'])/n,'force_max_abs_hartree_per_bohr':max(map(abs,ds)),'force_rms_hartree_per_bohr':math.sqrt(sum(v*v for v in ds)/len(ds))}
def close(a,b):return math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-18)
D=read(T/'main-integration-preparation-v2.json');F=T/'main-integration-v2';C=read(F/'campaign.json');B=read(T/'base/build-receipts-v6/build.json');A=read(R/'main-integration-analysis-v1.json')
check(sha(T/'main-integration-preparation-v2.json')==C['declaration_sha256'],'sealed declaration')
for n,h in D['scripts'].items():check(sha(T/n)==h,'sealed script '+n)
for n,h in D['base_files'].items():check(sha(T/'base'/n)==h,'sealed base '+n)
check(C['source']==B['source']==read(T/'base/source-v5.json') and C['build_artifacts']==B['artifacts'],'source/build binding')
check(B['native_source_identity']==C['native_source_identity']==D['native_source_identity'],'native identity')
check(sha(T/'base/build-receipts-v6/build.json')==C['build_receipt_sha256'],'build receipt binding')
OR=read(T/'base/oracles.json.xz');RE=read(T/'base/reference-reuse.json')['entries'];OI={(z['method'],z['atoms'],z['backend']):z for z in OR};check(sha(T/'base/oracles.json.xz')==sha(R/'received-fixed30-v1/oracles.json.xz'),'reuse exact independently checked original40')
CAT=read(R/'reference-installation-history-v1.json')['scientific_image_catalog'];SD={s['id']:s for s in D['studies']}
expected=[(s['id'],j,v['name']) for s in D['studies'] for j in range(3) for v in s['variants'][j%3:]+s['variants'][:j%3]]
check([(r['study'],r['pair_index'],r['variant']) for r in C['runs']]==expected,'all18 rotating process order')
I={};MAX={k:0 for k in G};native_images=set();ref_images=set()
for row in C['runs']:
 n=row['name'];p=read(F/(n+'.plan.json'));raw=read(F/(n+'.json'));q=read(F/(n+'.qualification.json'));s=SD[row['study']];v=next(v for v in s['variants'] if v['name']==row['variant'])
 for suf in ['.stdout','.stderr']:sha(F/(n+suf))
 check(sha(F/(n+'.plan.json'))==row['plan_sha256']==raw['plan_sha256'] and sha(F/(n+'.json'))==row['raw_sha256'],'plan/raw binding')
 check(p['execution_controls']==row['effective_controls']==v['controls'],'sealed variants')
 check(raw.get('observed_execution_controls')=={'GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY':v['controls'].get('GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY')},'observed0/unset')
 check(raw['status']=='complete' and len(raw['samples'])==4 and row['returncode']==0,'complete E/F')
 check(raw['source_revision']==p['source_revision']==C['source']['revision'] and raw['runner_sha256']==p['endpoint_sha256']==D['scripts']['endpoint-main-integration-v1.py'] and raw['driver_sha256']==p['driver_sha256'],'producer source')
 check(raw['accelerator']['nvidia_smi']['uuid']==C['device_uuid'] and raw['accelerator']['compute_capability']==[9,0],'physical GPU')
 target=OI[raw['method'],raw['atoms'],'cuda'];check(raw['geometry_bohr']==target['geometry_bohr']==p['geometry_bohr'],'geometry')
 for oracle in [target,OI[raw['method'],3,'cpu'],OI[raw['method'],3,'cuda']]:
  c=RE[identity(oracle)];check(c['source_revision']==oracle['source_revision'] and c['driver_sha256']==oracle['driver_sha256'],'reference receipt binding');check(c['scientific_inputs']=={k:raw[k] for k in ['method','orbital_basis','basis_representation','charge','spin']},'reference recipe');check(raw['environment']['packages'] in c['measurement_packages'] and oracle['environment']['packages']==c['reference_packages'],'reference environment')
 for j,z in enumerate(raw['samples']):
  check(z['converged'] is True and math.isfinite(z['seconds']) and z['seconds']>0 and (z['phase'],z['repeat'])==(('cold',0) if j==0 else ('warm',j-1)),'endpoint phase and convergence')
  e=errors(z,target['samples'][-1],raw['atoms']);check(all(math.isfinite(e[k]) and e[k]<=v for k,v in G.items()),'original independent full E/F gate');stored=q[z['phase']]['errors'][0 if j==0 else j-1];check(all(close(e[k],stored[k]) for k in G),'stored numerical arithmetic')
  for k in MAX:MAX[k]=max(MAX[k],e[k])
 images=raw['runtime_images_after_endpoint'];ni={n:b for n,b in images.items() if Path(n).name.startswith('libgenerativeqc')}
 if row['variant']!='gpu4pyscf':
  core=raw['native_build']['library_path'];check(core in ni and raw['native_source_identity']==C['native_source_identity'] and raw['native_build']['library_sha256']==B['artifacts']['libgenerativeqc.so'],'executed core')
  for name,b in ni.items():check(Path(name).parent==Path(p['library_root']) and b['sha256']==(B['artifacts']['libgenerativeqc.so'] if name==core else B['artifacts'].get(Path(name).name)),'mapped native bytes');native_images.add(Path(name).name)
  for z in raw['samples']:
   settings=z['settings'];check(settings['force_tile_points'] is None and settings['resolved_force_tile_points']==s['expected_force_tile'] and settings['requested_scf_tile_points'] is None and settings['scf_tile_points']==256,'automatic force256/1024 and SCF256')
 else:
  check(not ni,'reference loads no native core');observed={n:b for n,b in images.items() if Path(n).is_relative_to(p['reference_site'])};check(observed,'reference mappings')
  for name,b in observed.items():check(name in CAT and all(b[k]==CAT[name][k] for k in ['bytes','sha256']),'mapped reference bytes');ref_images.add(name)
 I[row['study'],row['pair_index'],row['variant']]=raw
RESULT=[]
for sid,s in SD.items():
 pairs=[]
 for j in range(3):
  off,cand,ref=[I[sid,j,v] for v in ['off','default','gpu4pyscf']];check(all(a['settings']==b['settings'] for a,b in zip(off['samples'],cand['samples'])),'same scientific settings')
  es=[errors(a,b,s['atoms']) for a,b in zip(cand['samples'],off['samples'])];check(all(all(e[k]<=v for k,v in s['matched_gates'].items()) for e in es),'existing3006 stronger matched E/F')
  cold=cand['samples'][0]['seconds']/off['samples'][0]['seconds'];warm=statistics.median(z['seconds'] for z in cand['samples'][1:])/statistics.median(z['seconds'] for z in off['samples'][1:]);each=[a['seconds']/b['seconds'] for a,b in zip(cand['samples'][1:],off['samples'][1:])]
  work={n:[(z['iterations'],z.get('fock_builds',z.get('effective_potential_builds'))) for z in x['samples']] for n,x in [('off',off),('default',cand),('reference',ref)]}
  pairs.append(dict(pair_index=j,cold_candidate_over_off=cold,warm_candidate_over_off=warm,warm_each_call_ratios=each,warm_each_call_guard=all(v<=1.05 for v in each),cold_reference_over_candidate=ref['samples'][0]['seconds']/cand['samples'][0]['seconds'],warm_reference_over_candidate=statistics.median(z['seconds'] for z in ref['samples'][1:])/statistics.median(z['seconds'] for z in cand['samples'][1:]),matched_errors=es,work=work,one_fewer_cold_fock_same_iterations=work['off'][0][0]==work['default'][0][0] and work['off'][0][1]==work['default'][0][1]+1))
 a=next(s for s in A['studies'] if s['id']==sid)['summary'];check(all(close(p['cold_candidate_over_off'],v) for p,v in zip(pairs,a['cold_candidate_over_off'])) and all(close(p['warm_candidate_over_off'],v) for p,v in zip(pairs,a['warm_candidate_over_off'])),'stored pair timing arithmetic')
 summary=dict(cold_ratios=[p['cold_candidate_over_off'] for p in pairs],median_cold_reduction=1-statistics.median(p['cold_candidate_over_off'] for p in pairs),warm_ratios=[p['warm_candidate_over_off'] for p in pairs],median_warm_change=statistics.median(p['warm_candidate_over_off'] for p in pairs)-1,every_warm_call_guard=all(p['warm_each_call_guard'] for p in pairs),max_warm_call_ratio=max(v for p in pairs for v in p['warm_each_call_ratios']),one_fewer_cold_fock=[p['one_fewer_cold_fock_same_iterations'] for p in pairs],median_reference_over_default_cold=statistics.median(p['cold_reference_over_candidate'] for p in pairs),median_reference_over_default_warm=statistics.median(p['warm_reference_over_candidate'] for p in pairs))
 summary['performance_2pct_target_pass']=summary['median_cold_reduction']>=.02 and all(v<1 for v in summary['cold_ratios']) and summary['every_warm_call_guard']
 RESULT.append(dict(id=sid,expected_force_tile=s['expected_force_tile'],matched_gates=s['matched_gates'],pairs=pairs,summary=summary))
check(len(I)==18 and sum(len(v['samples']) for v in I.values())==72,'complete WB2');check(A['performance_pass']==all(s['summary']['performance_2pct_target_pass'] for s in RESULT),'performance outcome')
value=dict(status='PASS_INDEPENDENT_WB2_SCIENCE_AND_RAW_REDUCTION',source_revision=C['source']['revision'],native_source_identity=C['native_source_identity'],core_sha256=B['artifacts']['libgenerativeqc.so'],device_uuid=C['device_uuid'],populations=len(I),returned_EF=72,worst_independent_errors=MAX,observed_native_images=sorted(native_images),distinct_reference_images=len(ref_images),studies=RESULT)
(O/'independent-wb2-v1.json').write_text(json.dumps(value,indent=2)+'\n');(O/'audited-hashes-wb2-v1.json').write_text(json.dumps(H,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items() if k!='studies'},indent=2));print(json.dumps([dict(id=s['id'],summary=s['summary']) for s in RESULT],indent=2))
