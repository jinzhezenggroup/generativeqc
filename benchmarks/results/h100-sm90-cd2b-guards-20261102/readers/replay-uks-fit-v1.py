"""Keep original default-fit failure and qualify the separate explicit refinement."""
from pathlib import Path
import hashlib,importlib.util,json,math
TASK=Path(__file__).resolve().parent
COHORT='uks-fit-v1'
def check(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    out=TASK/COHORT;m=read(TASK/(COHORT+'-download-manifest.json'));c=read(out/'campaign.json');p=read(TASK/'uks-fit-protocol-v1.json');build_path=TASK/'production-build-v1/build-receipt.json';build=read(build_path)
    check(m['platform_status']=='SUCCEEDED' and not m['recorded_provenance_errors'] and sha(out/'campaign.json')==m['campaign_sha256'],'terminal original differs')
    check(all((TASK/n).is_file() and not (TASK/n).is_symlink() and (TASK/n).stat().st_size==b['bytes'] and sha(TASK/n)==b['sha256'] for n,b in m['files'].items()),'original inventory differs')
    check(c['status']=='TERMINAL_EXPLICIT_UKS_FIT_REFINEMENT_INDEPENDENT_REPLAY_PENDING' and c['job']=='gqc-h100-maincd2b-uks-fit-1102-v1' and c['source_revision']==p['source_revision']==build['source_revision']=='cd2b4514be0a38c4920934481b1f02d157a694b4' and c['production_build_receipt_sha256']==sha(build_path),'source/production identity differs')
    check(len(c['runs'])==1 and c['protocol_sha256']==sha(TASK/'uks-fit-protocol-v1.json'),'refinement population/protocol differs')
    status_name=COHORT+'-archival-admissions/uks-fit-terminal-status-v1.json';observed=read(TASK/status_name);check(sha(TASK/status_name)==m['platform_status_sha256'] and observed['data']['name']==c['job'] and observed['data']['status']=='SUCCEEDED','actual platform response differs')
    required={COHORT+'/campaign.json',status_name};expected_scripts={'run-uks-fit-v1.py','run-uks-fit-v1-job.sh','uks-fit-protocol-v1.json','standard-guard-endpoint-v1.py','cohort-gates.py','run-screening-v3.py','pilot.py'};check(set(c['scripts'])==expected_scripts,'producer inventory differs')
    for n,h in c['scripts'].items():
        required.add(COHORT+'-executed-producers/'+n);check(sha(TASK/n)==h==sha(TASK/(COHORT+'-executed-producers')/n),'executed producer differs')
    for n,h in c['admissions'].items():
        name=COHORT+'-available-admissions/'+n;required.add(name);check(sha(TASK/name)==h==sha(TASK/n),'actual admission differs')
    refs=TASK/(COHORT+'-reference-inputs')
    for n,h in c['independent_reference_bindings'].items():
        required.add(COHORT+'-reference-inputs/'+n);check(sha(refs/n)==h,'independent reference input differs')
    cpu_path=refs/'current-cpu/uks-pbe-df-cpu-diagnostic-v1.json';cpu=read(cpu_path);cpu_producer=refs/'current-cpu/uks-pbe-df-cpu-diagnostic-v1.py'
    required.update(COHORT+'-reference-inputs/current-cpu/'+n for n in [cpu_path.name,cpu_producer.name]);check(sha(cpu_path)==p['cpu_attribution_sha256']==c['cpu_attribution_sha256'] and sha(cpu_producer)==cpu['script_sha256'],'actual CPU attribution/producer differs')
    check(cpu['status']=='COMPLETE_TWELVE_CPU_EF_ATTRIBUTION_REPLAY_PENDING' and len(cpu['samples'])==12 and cpu['environment']['pyscf']=='2.14.0','CPU reference incomplete')
    previous_path=TASK/'standard-guards-v1-local-replay.json';previous=read(previous_path);check(sha(previous_path)==p['original_guard_replay_sha256'] and previous['status']=='NATIVE_GUARDS_FAILED_OR_INCOMPLETE_WITH_ALL_ORIGINALS_RETAINED' and previous['qualified_populations']==7 and previous['returned_EF']==84,'original default-fit failure was changed')
    original_path=refs/'original-standard-campaign.json';required.add(COHORT+'-reference-inputs/original-standard-campaign.json');check(sha(original_path)==p['original_guard_campaign_sha256']==c['original_guard_campaign_sha256'],'original campaign differs')
    failed=[r for r in previous['rows'] if r['status']!='PASS'];check(len(failed)==1 and failed[0]['case']=='uks-pbe-water3-cation-df' and len(failed[0]['numerical'])==9 and all(r['errors']['energy_abs_per_atom_hartree']>1e-5 and r['errors']['force_max_abs_hartree_per_bohr']<=1e-4 and r['errors']['force_rms_hartree_per_bohr']<=3e-5 for r in failed[0]['numerical']),'original failed dense gates differ')
    spec=importlib.util.spec_from_file_location('same_source_guard_arithmetic',TASK/'replay-standard-guards-v2.py');guards=importlib.util.module_from_spec(spec);spec.loader.exec_module(guards)
    same={(s['case']['id'],s['level'],s['mode']):s for s in cpu['samples']};check(len(same)==12,'CPU duplicates/missing cases')
    basis=refs/p['new_auxiliary_prior_relative'];check(sha(basis)==p['new_auxiliary_sha256'],'refined auxiliary differs')
    row=c['runs'][0];stem=row['case'];plan_path=out/(stem+'.plan.json');raw_path=out/(stem+'.json');plan=read(plan_path);raw=read(raw_path);required.update(COHORT+'/'+stem+s for s in ['.plan.json','.json','.stdout','.stderr'])
    check(row['returncode']==0 and row['status']=='COMPLETE_REFINED_FIT_NINE_EF_PARENT_QUALIFIED_RECEIVER_PENDING' and sha(plan_path)==row['plan_sha256']==raw['plan_sha256'] and sha(raw_path)==row['receipt_sha256'],'actual refinement execution differs')
    check(plan['cases']==p['cases'] and plan['build_artifacts']==build['artifacts'] and plan['source_revision']==raw['source_revision']==c['source_revision'] and plan['endpoint_sha256']==raw['runner_sha256']==p['endpoint_sha256'],'model/source/endpoint binding differs')
    check(plan['auxiliary_basis_sha256']==p['new_auxiliary_sha256'] and plan['fit_policy']=='EXPLICIT_EXISTING_CC_PVTZ_JKFIT_UKS_GUARD_ACCURACY_REFINEMENT_NOT_DEFAULT' and raw['execution_policy']==plan['execution_policy']==c['policy'],'explicit new input/policy differs')
    check(raw['status']=='COMPLETE_NATIVE_GUARDS_INDEPENDENT_NUMERICAL_REPLAY_PENDING' and [(s['stage'],s['item'],s['oracle_case']) for s in raw['samples']]==guards.expected_samples(p['cases']),'complete ordered nine-call population differs')
    check(raw['accelerator']['nvidia_smi']['uuid']==c['device']['uuid'] and raw['accelerator']['compute_capability']==[9,0] and raw['qz_allocation']==plan['qz_allocation'] and raw['qz_allocation']['job_name']==c['job'],'actual H100/allocation differs')
    check(raw['environment']['packages']=={'cupy':'13.6.0','gpu4pyscf':'1.8.1','numpy':'2.2.6','pyscf':'2.14.0'},'actual native runtime differs')
    check(p['numerical_gates']=={'energy_abs_per_atom_hartree':1e-5,'force_max_abs_hartree_per_bohr':1e-4,'force_rms_hartree_per_bohr':3e-5},'dense numerical gates changed');results=[]
    for sample in raw['samples']:
        case=next(case for case in p['cases'] if case['id']==sample['oracle_case']);dense=read(refs/'guard-cpu-oracles-v1'/(case['id']+'-grid4.json'))
        check(sample['geometry_bohr']==case['geometry_bohr'] and sample['charge']==case['charge']==1 and sample['spin']==case['spin']==1 and sample['native_build']['library_sha256']==build['artifacts']['libgenerativeqc.so'],'actual geometry/spin/core differs')
        check(sample['native_prepared_force_before'] is sample['native_prepared_force_after'] is True and math.isfinite(sample['seconds_batch_intrusive']) and sample['seconds_batch_intrusive']>0,'native force/duration evidence differs')
        if sample['stage'] in ['base-warm','moved-warm']:check(sample['warm_start_used'] is True,'warm owner not reused')
        dense_error=guards.observable_errors(sample,dense,3);matched_error=guards.observable_errors(sample,same[case['id'],3,'existing-ccpvtz-jkfit'],3)
        check(all(v<=p['numerical_gates'][k] for k,v in dense_error.items()) and all(v<=p['cpu_same_df_gates'][k] for k,v in matched_error.items()),'refined E/F fails unchanged dense or matched-DF gates')
        results.append({'stage':sample['stage'],'item':sample['item'],'oracle_case':case['id'],'dense_errors':dense_error,'same_df_cpu_errors':matched_error})
    controls=guards.guard_controls(raw,9,c['policy']['ks_tile_points']);check(required<=set(m['files']),'mandatory raw/log/producer/reference/admission omitted')
    value={'status':'PASS_EXPLICIT_REFINED_UKS_DF_NINE_EF_ORIGINAL_DEFAULT_FAILURE_RETAINED','source_revision':c['source_revision'],'campaign_sha256':sha(out/'campaign.json'),'manifest_sha256':sha(TASK/(COHORT+'-download-manifest.json')),'protocol_sha256':sha(TASK/'uks-fit-protocol-v1.json'),'original_guard_replay_sha256':sha(previous_path),'original_default_guard_populations_qualified':7,'original_EF':84,'refined_input_EF':9,'total_retained_native_EF':93,'original_default_failure_retained':True,'cpu_attribution_sha256':sha(cpu_path),'rows':results,'controls':controls,'scope':p['scope']}
    target=TASK/(COHORT+'-local-replay.json');check(not target.exists(),'inspect previous replay');target.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items() if k!='rows'}))
if __name__=='__main__':main()
