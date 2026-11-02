"""Recompute independent dense-reference and old-target consistency with NumPy norms."""
from pathlib import Path
import hashlib,json,lzma,math,subprocess,sys,tarfile
import numpy as np

TASK=Path(__file__).resolve().parent
ROOT=TASK/'reference-six-v5-reference-inputs/source-helper'
COHORT='reference-six-v5'
PLAN_SHA='3f6387100340e3acf044f0946611fe74daf7fb898896b2625ce19ddbaa3ffffd'
def check(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def differences(sample,target,atoms):
    a,b=(np.asarray(r['forces'],dtype=np.float64) for r in [sample,target]);check(a.shape==b.shape==(atoms,3) and np.isfinite(a).all() and np.isfinite(b).all(),'invalid complete dense force vectors')
    check(all(type(v) in [int,float] and math.isfinite(v) for v in [sample['energy'],target['energy']]),'invalid finite E/F')
    delta=a-b;return {'energy_abs_per_atom_hartree':abs(sample['energy']-target['energy'])/atoms,'force_max_abs_hartree_per_bohr':float(np.max(np.abs(delta))),'force_rms_hartree_per_bohr':float(np.sqrt(np.mean(delta*delta)))}
def controls(raw,case):
    check([s['level'] for s in raw['samples']]==case['levels'],'duplicate/missing/reordered dense-grid level')
    check(raw['native_blocked_calls']==raw['native_library_mappings_before']==raw['native_library_mappings_after']==[],'native execution/mappings contaminated the oracle')
    check(raw['native_execution_blocked_entrypoints']==['generativeqc._native.load_library','Calculator.__init__','PrecisionCalculator.__init__'],'native independence guards differ')
    for sample,level in zip(raw['samples'],case['levels'],strict=True):
        check(sample['converged'] is True and type(sample['iterations']) is int and 0<sample['iterations']<=150 and type(sample['seconds']) in [int,float] and math.isfinite(sample['seconds']) and sample['seconds']>0,'incomplete dense solver execution')
        check(sample['controls']=={'energy':1e-11,'force_grid_response':True,'full_density_fock':True,'orbital_gradient':1e-7,'screening':1e-14},'independent full-density/control contract differs')
        check(sample['xc_grid']['level']==level and sample['xc_grid']['points']>0,'actual dense XC level/points missing')
        if case['method']=='wb97m-v':check(sample['nonlocal_grid']['level']==case['nonlocal_levels'][case['levels'].index(level)] and sample['nonlocal_grid']['points']>0,'actual independent VV10 quadrature missing')
        else:check(sample['nonlocal_grid'] is None,'unexpected nonlocal functional')
def main():
    manifest_path=TASK/(COHORT+'-download-manifest.json');manifest=read(manifest_path);archive=TASK/manifest['archive'];out=TASK/COHORT;campaign=read(out/'campaign.json')
    check(manifest['cohort']==COHORT and manifest['status']=='COMPLETE_IMMUTABLE_TERMINAL_ARCHIVE' and manifest['platform_status']=='SUCCEEDED' and manifest['recorded_provenance_errors']==[],'terminal archive differs')
    check(archive.parent==TASK and archive.stat().st_size==manifest['bytes'] and sha(archive)==manifest['sha256'],'archive bytes differ')
    check(all(not Path(n).is_absolute() and '..' not in Path(n).parts and (TASK/n).is_file() and not (TASK/n).is_symlink() and (TASK/n).stat().st_size==b['bytes'] and sha(TASK/n)==b['sha256'] for n,b in manifest['files'].items()),'recovered originals differ')
    check(sha(out/'campaign.json')==manifest['campaign_sha256'] and campaign['job']=='gqc-h100-maincd2b-reference6-1102-v5' and campaign['status'] in ['COMPLETE_6_INDEPENDENT_REFERENCE_POPULATIONS_12_EF_REPRODUCED_AND_DUAL_GRID_STABLE','REFERENCE_REPLICATION_WITH_FAILURES'] and len(campaign['runs'])==6,'terminal reference six missing')
    status=[read(TASK/n) for n in manifest['files'] if n.startswith(COHORT+'-archival-admissions/') and sha(TASK/n)==manifest['platform_status_sha256']]
    check(len(status)==1 and status[0]['success'] is True and status[0]['data']['name']==campaign['job'] and status[0]['data']['status']=='SUCCEEDED','actual platform response differs')
    prereg_path=TASK/'reference-six-protocol-v1.json';prereg=read(prereg_path);check(sha(prereg_path)==campaign['preregistration_sha256']==PLAN_SHA and campaign['source_revision']==prereg['source_revision']=='000517f82dc62130459a6fd92a27518a158c49cc','frozen reference recipe differs')
    required={COHORT+'/campaign.json',COHORT+'/reference-runtime.json',COHORT+'/reference-runtime.stdout',COHORT+'/reference-runtime.stderr',COHORT+'-reference-inputs/data/oracles.json.xz'}
    for name,digest in prereg['source_files'].items():
        check(sha(ROOT/name)==digest,'archived exact scientific helper differs');required.add(COHORT+'-reference-inputs/source-helper/'+name)
    scripts={'oracle_replication6_solver_v2.py','run-reference-six-v5.py','run-reference-six-v5-job.sh','reference-six-endpoint-v1.py','reference-six-protocol-v1.json','reference-six-runtime-v5.py','cohort-gates-v4.py','cohort-gates-v3.py','cohort-gates.py','process-ownership-v1.py','run-screening-v3.py','pilot.py'}
    check(set(campaign['scripts'])==scripts,'mandatory executed producers differ')
    for name,digest in campaign['scripts'].items():
        archived=COHORT+'-executed-producers/'+name;check(sha(TASK/name)==digest==sha(TASK/archived),'executed producer differs');required.add(archived)
    for name,digest in campaign['current_native_admissions'].items():
        archived=COHORT+'-available-admissions/'+name;check(sha(TASK/name)==digest==sha(TASK/archived),'current native admission differs');required.add(archived)
    native=read(TASK/'native-qualification-v1-local-replay-v2.json');buildpath=TASK/'production-build-v1/build-receipt.json';build=read(buildpath)
    check(campaign['native_source_revision']==build['source_revision']==native['source_revision']=='cd2b4514be0a38c4920934481b1f02d157a694b4' and campaign['production_build_receipt_sha256']==sha(buildpath) and native['status']=='PASS_CURRENT_SOURCE_NATIVE_FOUR_UNFILTERED_MEMCHECK' and native['production_core_sha256']==build['artifacts']['libgenerativeqc.so'],'separate current native admission differs')
    for stage in campaign['runs']:
        required.update(COHORT+'/'+stage['case']+suffix for suffix in ['.plan.json','.json','.qualification.json','.stdout','.stderr'])
    check(required<=set(manifest['files']),'mandatory original input/log/producer missing')
    check(campaign['allocation_cgroups']=={'cpu.max':'2000000 100000','memory.max':'214748364800'} and campaign['shared_memory_bytes']==16*(1<<30),'actual allocation differs')
    runtime=read(out/'reference-runtime.json');check(sha(out/'reference-runtime.json')==campaign['reference_runtime_sha256'] and runtime['script_sha256']==campaign['scripts']['reference-six-runtime-v5.py'] and runtime['status']=='PASS_REFERENCE_RUNTIME_AND_CUTENSOR_ONLY' and campaign['device']['uuid'] in runtime['device'] and runtime['installed_files_equal_original_wheels']>0 and runtime['max_abs_error']<1e-12,'actual isolated reference runtime differs')
    oracle_path=TASK/(COHORT+'-reference-inputs/data/oracles.json.xz');check(sha(oracle_path)==prereg['old_oracle_archive_sha256']=='d61a3e8ba0a8f8a8c08ee4c4b30c8e3e397e67863b8d75e7ba974e90cf0f85a4','old immutable oracle differs');oracles=json.loads(lzma.decompress(oracle_path.read_bytes()))
    gates={'energy_abs_per_atom_hartree':1e-5,'force_max_abs_hartree_per_bohr':1e-4,'force_rms_hartree_per_bohr':3e-5};check(prereg['numerical_gates']==gates and prereg['stability_and_reproduction_gate_fraction']==0.25,'original quarter-gates changed')
    rows=[];complete_EF=0;qualified_all=True
    for case,stage in zip(prereg['cases'],campaign['runs'],strict=True):
        stem=case['id'];check(stage['case']==stem and stage['status'] in ['COMPLETE','FAIL_OR_INCOMPLETE'],'reference order/status differs');plan_path=out/(stem+'.plan.json');plan=read(plan_path)
        check(sha(plan_path)==stage['plan_sha256'] and plan['case']==case and plan['source_revision']==prereg['source_revision'] and plan['source_files']==prereg['source_files'] and plan['solver_sha256']==prereg['solver_artifact']['sha256']==sha(TASK/'oracle_replication6_solver_v2.py') and plan['endpoint_sha256']==sha(TASK/'reference-six-endpoint-v1.py'),'reference plan/source/producer differs')
        row={'case':stem,'returncode':stage['returncode'],'reproduction':[],'scope':'Accuracy only; zero majority/timing votes.'};raw_path=out/(stem+'.json')
        if not raw_path.exists():
            check(stage['status']=='FAIL_OR_INCOMPLETE' and 'receipt_sha256' not in stage,'complete reference receipt missing');row.update(status='FAIL_OR_INCOMPLETE',error='NO_RAW_RECEIPT');rows.append(row);qualified_all=False;continue
        check(sha(raw_path)==stage['receipt_sha256'],'raw reference bytes differ');raw=read(raw_path)
        check(raw['case']==case and raw['source_revision']==prereg['source_revision'] and raw['source_files']==prereg['source_files'] and raw['solver_sha256']==plan['solver_sha256'] and raw['runner_sha256']==plan['endpoint_sha256'] and raw['plan_sha256']==sha(plan_path),'raw scientific/source/producer identity differs')
        check(raw['qz_allocation']==plan['qz_allocation'] and raw['qz_allocation']['job_name']==campaign['job'] and raw['scheduler']=={'kind':'qz_job','slurm_job_id':None},'actual scheduler/allocation differs')
        check(raw['accelerator']['nvidia_smi']['uuid']==campaign['device']['uuid'] and raw['accelerator']['name']=='NVIDIA H100 80GB HBM3' and raw['accelerator']['compute_capability']==[9,0] and raw['environment']['packages']==prereg['reference_packages'],'actual reference device/environment differs')
        old=next(r for r in oracles if (r['method'],r['atoms'],r['backend'])==(case['method'],case['atoms'],'cuda'));check(old['geometry_bohr']==case['geometry_bohr'] and old['driver_sha256']==case['old_driver_sha256'],'old target scientific identity differs')
        check([s['level'] for s in raw['samples']]==case['levels'][:len(raw['samples'])] and len(raw['samples'])<=2,'duplicate/extra dense level')
        qualified=stage['returncode']==0 and raw['status']=='COMPLETE_INDEPENDENT_TWO_GRID_EF_NUMERICAL_REPLAY_PENDING' and len(raw['samples'])==2
        for sample in raw['samples']:
            target=next(s for s in old['samples'] if s['level']==sample['level']);error=differences(sample,target,case['atoms']);passed=all(v<=gates[k]*0.25 for k,v in error.items());qualified &= passed;complete_EF+=1;row['reproduction'].append({'level':sample['level'],'errors':error,'status':'PASS' if passed else 'FAIL','old_xc_grid':target['xc_grid'],'new_xc_grid':sample['xc_grid'],'old_nonlocal_grid':target['nonlocal_grid'],'new_nonlocal_grid':sample['nonlocal_grid']})
        if len(raw['samples'])==2:
            stability=differences(raw['samples'][0],raw['samples'][1],case['atoms']);qualified &= all(v<=gates[k]*0.25 for k,v in stability.items());row['new_dual_grid_errors']=stability
        if qualified:controls(raw,case)
        check(stage['qualification']==('qualified' if qualified else 'failed_or_unverified') and read(out/(stem+'.qualification.json'))['status']==stage['qualification'],'producer/independent numerical admission differs')
        row.update(status='PASS' if qualified else 'FAIL_OR_INCOMPLETE',returned_EF=len(raw['samples']),raw_status=raw['status'],error=raw.get('error'));qualified_all &= qualified;rows.append(row)
    check(not qualified_all or complete_EF==12,'missing complete dense references')
    result={'status':'PASS6_PRIMARY_KNOWN_SOURCE_REPLICATIONS12_FULL_EF_AND_DUAL_GRID_QUARTER_GATES' if qualified_all else 'DENSE_REFERENCE_REPLICATION_FAILED_OR_INCOMPLETE_ORIGINALS_RETAINED','source_revision':prereg['source_revision'],'native_source_revision':campaign['native_source_revision'],'production_build_receipt_sha256':campaign['production_build_receipt_sha256'],'preregistration_sha256':PLAN_SHA,'campaign_sha256':sha(out/'campaign.json'),'archive_sha256':sha(archive),'manifest_sha256':sha(manifest_path),'old_oracle_archive_sha256':sha(oracle_path),'returned_EF':complete_EF,'qualified_populations':sum(r['status']=='PASS' for r in rows),'rows':rows,'scope':'Independent NumPy full-vector replay of six new known-source dense references versus same-level old targets and new dual-grid stability at unchanged quarter-gates. Original14oldproducerbytes remain unrecovered; no historical source identity or raw values are altered. New results corroborate only these six primary targets, not native execution, production dispatch, timing or majority performance.'}
    path=TASK/(COHORT+'-local-replay.json');check(not path.exists(),'inspect existing reference replay');path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
