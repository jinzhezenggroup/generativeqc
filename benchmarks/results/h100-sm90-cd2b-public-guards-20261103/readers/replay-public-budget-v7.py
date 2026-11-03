"""Independently verify supported Direct capacity executions and typed DF refusals."""
from pathlib import Path
import importlib.util
import json
import lzma

TASK=Path(__file__).resolve().parent
COHORT='public-budget-v7'
def load(name):
    spec=importlib.util.spec_from_file_location(name,TASK/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
common=load('cohort-replay-v1');capacity=load('budget-capacity-replay-v1')
check,read,sha=common.check,common.read,common.sha

def main():
    output=TASK/(COHORT+'-local-replay.json');check(not output.exists(),'inspect existing replay')
    scripts=['run-public-budget-v7.py','run-public-budget-v7-job.sh','public-budget-protocol-v6.json','public-budget-endpoint-v5.py','cohort-gates-v4.py','cohort-gates-v3.py','cohort-gates.py','process-ownership-v1.py','run-screening-v3.py','pilot.py']
    c,m,build,policy=common.authenticate(TASK,COHORT,'gqc-h100-maincd2b-budget-1102-v7',scripts,'TERMINAL_EIGHT_PUBLIC_BUDGET_ATTEMPTS_INDEPENDENT_REPLAY_PENDING')
    out=TASK/COHORT;pp=TASK/'public-budget-protocol-v6.json';p=read(pp);refs=TASK/(COHORT+'-reference-inputs')
    check(c['protocol_sha256']==sha(pp) and p['source_revision']==common.SOURCE and p['matrix_preregistration_sha256']==sha(TASK/'screening-protocol-v3.json') and p['planned_complete_EF']==12 and p['expected_negative_controls']==6 and p['expected_large_direct_host_refusals']==1 and p['expected_unsupported_df_populations']==4 and len(p['order'])==len(c['runs'])==8 and c['policy']==policy,'frozen budget protocol/policy differs')
    check(p['endpoint']=={'name':'public-budget-endpoint-v5.py','sha256':sha(TASK/'public-budget-endpoint-v5.py')} and p['budget_fractions']==[[1,1],[3,4],[1,2]],'declared endpoint/fractions differ')
    geometries=read(refs/'native-guard-geometries-v1.json');cpu_admission=read(refs/'guard-cpu-oracles-local-replay.json');cpu_campaign=read(refs/'guard-cpu-oracles-v1/campaign.json')
    check(sha(refs/'native-guard-geometries-v1.json')==p['source_geometry_sha256'] and sha(refs/'guard-cpu-oracles-local-replay.json')==p['independent_small_cpu_admission_sha256'] and cpu_admission['status']=='PASS_28_CPU_EF_CANDIDATES_DUAL_GRID_AND_8_DIRECTIONAL_GATES' and cpu_admission['campaign_sha256']==sha(refs/'guard-cpu-oracles-v1/campaign.json'),'independent CPU corpus differs')
    check(p['cases']==[case for case in geometries['cases'] if case['geometry_label'] in ['base','moved']] and p['common_observable_gates']==geometries['common_observable_gates'],'frozen scientific cases/gates differ')
    targets={}
    for case in p['cases']:
        if case['atoms']==96:continue
        path=refs/'guard-cpu-oracles-v1'/(case['id']+'-grid4.json');raw=read(path);binding=next(row for row in cpu_campaign['runs'] if row['case']==case['id'] and row['grid_level']==4)
        check(sha(path)==binding['receipt_sha256'] and raw['case']==case and raw['status']=='complete' and raw['converged'] is True,'CPU full vector differs');targets[case['id']]=raw
    oracle=refs/'data/oracles.json.xz';check(sha(oracle)==p['original_large_oracle_sha256'],'large oracle differs')
    large=next(row for row in json.loads(lzma.decompress(oracle.read_bytes())) if (row['method'],row['atoms'],row['backend'])==('pbe0',96,'cuda'))
    large_case=next(case for case in p['cases'] if case['atoms']==96);check(large_case['geometry_bohr']==large['geometry_bohr'],'large geometry differs');targets[large_case['id']]=large['samples'][-1]
    default=read(refs/'inputs-default.json')['files'];fit=read(TASK/'uks-fit-protocol-v1.json')
    check(sha(TASK/'uks-fit-protocol-v1.json')==p['explicit_uks_fit_protocol_sha256'] and sha(TASK/'uks-fit-v1-local-replay.json')==p['explicit_uks_fit_qualification_sha256'],'explicit input refinement differs')
    counts={'returned_EF':0,'verified_negative_controls':0,'typed_df_refusals':0,'typed_large_direct_host_refusals':0};rows=[]
    root=Path(build['stages'][0]['command'][build['stages'][0]['command'].index('-B')+1])
    for declared,stage in zip(p['order'],c['runs'],strict=True):
        family,mode=declared['family'],declared['mode'];stem=family+'-'+mode;cases=[case for case in p['cases'] if case['family']==family]
        check(stage['case']==stem and stage['returncode']==0,'population failed/reordered')
        planpath=out/(stem+'.plan.json');rawpath=out/(stem+'.json');plan,raw=read(planpath),read(rawpath)
        check(stage['plan_sha256']==sha(planpath)==raw['plan_sha256'] and stage['receipt_sha256']==sha(rawpath),'raw/plan bytes differ')
        check(plan['source_revision']==common.SOURCE and Path(plan['library_root'])==root and Path(plan['source_root'])==root.parent/'source' and plan['build_artifacts']==build['artifacts'] and plan['cases']==cases and plan['mode']==mode and plan['budget_fractions']==p['budget_fractions'] and plan['execution_policy']==policy and plan['endpoint_sha256']==p['endpoint']['sha256'],'actual model/source/package plan differs')
        check(raw['schema']=='generativeqc.h100-public-budget-guards.v1' and raw['endpoint_sha256']==plan['endpoint_sha256'] and (raw['family'],raw['mode'],raw['execution_policy'])==(family,mode,policy),'raw identity differs');common.native_identity(raw,plan,c)
        aux=Path(plan['auxiliary_basis']);relative=aux.relative_to(root.parent.parent/'20261030-h100-production-15ab-2241')
        expected_aux=fit['new_auxiliary_prior_relative'] if family=='uks-pbe-water3-cation' and mode=='df' else 'inputs-default/'+cases[0]['method']+'-auxiliary.json'
        expected_sha=fit['new_auxiliary_sha256'] if family=='uks-pbe-water3-cation' and mode=='df' else default[aux.name]['sha256']
        check(relative.as_posix()==expected_aux and sha(refs/relative)==plan['auxiliary_basis_sha256']==expected_sha,'actual auxiliary input differs')
        if mode=='df':
            check(stage['status']=='COMPLETE_EXPECTED_DF_PLANNING_REFUSAL' and raw['status']=='EXPECTED_PUBLIC_DFT_DF_PLANNING_REFUSAL_NO_EXECUTION' and stage['returned_EF']==0 and raw['samples']==raw['negative_controls']==raw['planning']==[],'DF refusal fabricated execution')
            check(raw['unsupported_request']=={'exception':'NotImplementedError','diagnostic':'DFT density-fitting resource plans are not qualified; use density_fitting_memory_budget_bytes for the native DF provider','planning_only':True,'execution_not_attempted_by_refusal_branch':True},'wrong typed DF refusal')
            check(not any(key in raw for key in ['unbounded_plan','selected_plan','selected_budget','detached_result_count']),'refusal claims execution/plan')
            counts['typed_df_refusals']+=1;rows.append({'case':stem,'status':'PASS_EXPECTED_UNSUPPORTED_PUBLIC_DF_PLANNING_REFUSAL','returned_EF':0});continue
        if family=='rks-pbe0-water96':
            check(stage['status']=='COMPLETE_EXPECTED_LARGE_DIRECT_HOST_PLANNING_REFUSAL' and raw['status']=='EXPECTED_PUBLIC_LARGE_DIRECT_HOST_PLANNING_REFUSAL_NO_EXECUTION' and stage['returned_EF']==0 and raw['samples']==raw['negative_controls']==[],'large host refusal fabricated execution')
            upper=capacity.replay_plan(raw['unbounded_plan'])['device'];host=64*(1<<30);check(raw['unbounded_plan']['peak_bytes']['host']>host and len(raw['planning'])==3,'large host envelope differs')
            lower_bounds=[]
            for fraction,record in zip(p['budget_fractions'],raw['planning'],strict=True):
                estimate=record['plan'];capacity.replay_plan(estimate);limit=max(1,(upper*fraction[0]+fraction[1]-1)//fraction[1])
                check(record['fraction']==fraction and estimate['budget']['host_bytes']==host and estimate['budget']['device_bytes']==limit and estimate['budget']['per_device_bytes']==[[0,limit]] and estimate['status']=='infeasible','large capacity probe differs')
                phases={a['first_phase'] for r in estimate['requests'] for c in r['candidates'] for a in c['estimates']}
                check(phases and all(r['candidates'] for r in estimate['requests']),'large host lower-bound domain missing')
                lower=max(sum(min(sum(a['bytes'] for a in candidate['estimates'] if a['space'] in ['pinned','pageable'] and a['first_phase']<=phase<=a['last_phase']) for candidate in request['candidates']) for request in estimate['requests']) for phase in phases)
                check(common.integer(lower) and lower>host,'large host infeasibility not independently established');lower_bounds.append(lower)
            refusal=raw['unsupported_request'];first=raw['planning'][0]['plan']
            check(refusal['exception']=='MemoryError' and refusal['diagnostic']==(first['diagnostic'] or first['status']) and refusal['planning_only'] is True and refusal['prepare_attempted'] is True and refusal['execution_not_attempted'] is True and refusal['host_limit_bytes']==host and refusal['declared_unbounded_host_peak_bytes']==raw['unbounded_plan']['peak_bytes']['host'],'actual large host refusal differs')
            counts['typed_large_direct_host_refusals']+=1;rows.append({'case':stem,'status':'PASS_EXPECTED_LARGE_PUBLIC_HOST_PLANNING_REFUSAL','returned_EF':0,'declared_host_lower_bounds':lower_bounds,'host_limit_bytes':host});continue
        check(stage['status']=='COMPLETE_INDEPENDENT_REPLAY_PENDING' and raw['status']=='COMPLETE_PUBLIC_REQUEST_BUDGET_GUARDS_INDEPENDENT_EF_REPLAY_PENDING','Direct endpoint failed')
        upper=capacity.replay_plan(raw['unbounded_plan'])['device'];host=max(1<<30,4*raw['unbounded_plan']['peak_bytes']['host'])
        check(0<upper<=72*(1<<30) and host<=64*(1<<30) and len(raw['planning'])==3,'finite capacity envelope differs')
        feasible=[]
        for fraction,record in zip(p['budget_fractions'],raw['planning'],strict=True):
            estimate=record['plan'];capacity.replay_plan(estimate);limit=max(1,(upper*fraction[0]+fraction[1]-1)//fraction[1])
            check(record['fraction']==fraction and estimate['budget']['host_bytes']==host and estimate['budget']['device_bytes']==limit and estimate['budget']['per_device_bytes']==[[0,limit]] and estimate['status'] in ['feasible','infeasible'],'capacity probe differs')
            if estimate['status']=='feasible':feasible.append((limit,estimate))
        check(feasible,'no feasible declared budget');limit,selected=min(feasible,key=lambda row:row[0])
        check(raw['selected_plan']==selected and raw['selected_budget']==selected['budget'] and raw['selected_device_fraction']==limit/upper and raw['selected_provider_changed']==(selected['selections']!=raw['unbounded_plan']['selections']),'lowest feasible cap selection differs')
        expected=[('public-budget-base-cold',cases[0]['id']),('public-budget-base-warm',cases[0]['id'])]+([('public-budget-moved',cases[1]['id']),('public-budget-moved-warm',cases[1]['id'])] if len(cases)==2 else [])
        check([(sample['stage'],sample['oracle_case']) for sample in raw['samples']]==expected and stage['returned_EF']==raw['detached_result_count']==len(expected),'returned/detached result sequence differs')
        controls=raw['negative_controls'];check(len(controls)==2 and controls[0]['name']=='whole-request-zero-device-budget-before-execution' and controls[0]['exception']=='MemoryError' and controls[1]['name']=='closed-public-budget-owner-execute' and controls[1]['exception']=='RuntimeError','refusal/lifetime controls differ')
        refusal=controls[0]['plan'];capacity.replay_plan(refusal);check(controls[0]['diagnostic']==(refusal['diagnostic'] or refusal['status']),'zero-budget refusal diagnostic differs');check(refusal['status']=='infeasible' and refusal['budget']['device_bytes']==0 and refusal['budget']['per_device_bytes']==[[0,0]],'zero-budget control differs')
        numerical=[]
        for sample in raw['samples']:
            case=next(case for case in cases if case['id']==sample['oracle_case']);common.loaded_core(sample,plan,build)
            check(sample['geometry_bohr']==case['geometry_bohr'] and sample['charge']==case['charge'] and sample['spin']==case['spin'] and sample['native_prepared_force'] is True and common.integer(sample['iterations']) and common.integer(sample['fock_builds']) and sample['fock_builds']>0,'returned model/work/native capability differs')
            difference=common.errors(sample,targets[case['id']],case['atoms']);check(all(difference[key]<=bound for key,bound in p['common_observable_gates'].items()),'independent dense E/F gate failed')
            if sample['stage'].endswith('warm'):check(sample['warm_start_used'] is True,'warm request lost seed')
            ledgers=[value['device_ledger'] for value in sample['resource_diagnostics'].values() if isinstance(value,dict) and value.get('device_ledger') is not None];check(ledgers,'actual ledger absent')
            for ledger in ledgers:
                a=ledger['admission'];check(all(common.integer(ledger[key]) for key in ['device','limit_bytes','live_bytes','peak_bytes','allocations']) and all(common.integer(v) for v in a.values()),'invalid capacity counters')
                check(ledger['device']==0 and 0<=ledger['live_bytes']<=ledger['peak_bytes']<=ledger['limit_bytes']<=limit and ledger['allocations']>0 and a['numeric_live_bytes']==ledger['live_bytes'] and a['numeric_peak_bytes']==ledger['peak_bytes'] and a['numeric_live_bytes']+a['provider_reserved_bytes']+a['capacity_reserved_bytes']<=a['limit_bytes']<=limit and a['combined_peak_bytes']<=a['limit_bytes'] and a['numeric_peak_bytes']<=a['combined_peak_bytes'] and a['provider_reservation_peak_bytes']<=a['combined_peak_bytes'],'owned device capacities exceed admission')
            numerical.append({'stage':sample['stage'],'oracle_case':case['id'],'errors':difference});counts['returned_EF']+=1
        counts['verified_negative_controls']+=2;rows.append({'case':stem,'status':'PASS_SUPPORTED_DIRECT_PUBLIC_BUDGET','returned_EF':len(raw['samples']),'numerical':numerical})
    check(counts=={'returned_EF':12,'verified_negative_controls':6,'typed_df_refusals':4,'typed_large_direct_host_refusals':1},'actual population differs')
    result={'status':'PASS_SMALL_DIRECT_PUBLIC_BUDGET_12_EF_6_CONTROLS_AND_FIVE_TYPED_PLANNING_REFUSALS','source_revision':common.SOURCE,'build_receipt_sha256':sha(TASK/'production-build-v1/build-receipt.json'),'core_sha256':build['artifacts']['libgenerativeqc.so'],'campaign_sha256':sha(out/'campaign.json'),'manifest_sha256':sha(TASK/(COHORT+'-download-manifest.json')),'protocol_sha256':sha(pp),**counts,'rows':rows,'scope':p['scope']}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({key:value for key,value in result.items() if key!='rows'}))

if __name__=='__main__':main()
