"""CPU replay of a separately scoped public API/fallback contract.

The original native-only Facade18 stays failed. This receiver preserves all
142 physical E/F endpoints while identifying compatibility forces, native
audits, four unsupported global-auxiliary requests and singleton fallbacks.
"""
from pathlib import Path
import importlib.util
import json
import math

TASK=Path(__file__).resolve().parent
NOTEBOOK='notebook-public-guards-v1'
SOURCE='cd2b4514be0a38c4920934481b1f02d157a694b4'
PASS='PASS_CORRECTED_PUBLIC_API_AND_EXPLICIT_SINGLETON_FALLBACK_142_EF_84_CLOSED_8_DIRECTIONS_4_REFUSALS'

def load(name):
    spec=importlib.util.spec_from_file_location(name,TASK/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

legacy=load('replay-facade-guards-v5');common=load('cohort-replay-atomic-v1');calibration=load('facade-correction-reference-replay-v1')
check,read,sha,error=legacy.check,legacy.read,legacy.sha,legacy.error

def authenticate_notebook():
    mp=TASK/(NOTEBOOK+'-download-manifest.json');manifest=read(mp)
    check(manifest['cohort']==NOTEBOOK and manifest['status']=='COMPLETE_IMMUTABLE_NOTEBOOK_PARENT_ARCHIVE' and manifest['workload']=='notebook' and manifest['platform_status']=='RUNNING' and manifest['all_parent_records_terminal'] is True,'Notebook parent/archive type differs')
    check(manifest['archive']==NOTEBOOK+'-completed.tar.xz' and sha(TASK/manifest['archive'])==manifest['sha256'],'Notebook raw archive differs')
    for name,b in manifest['files'].items():
        path=Path(name)
        check(not path.is_absolute() and '..' not in path.parts and path.name!='.env' and (TASK/path).is_file() and not (TASK/path).is_symlink(),'unsafe notebook member')
        check((TASK/path).stat().st_size==b['bytes'] and sha(TASK/path)==b['sha256'],'Notebook original bytes differ')
    index=read(TASK/NOTEBOOK/'campaign.json');check(manifest['parents']==index['parents'] and set(manifest['parents'])=={'fast-uks-capability-diagnostic-v1','fast-direct-capability-diagnostic-v2','facade-singleton-fallback-v1'},'Parent index set differs');check(sha(TASK/NOTEBOOK/'campaign.json')==manifest['campaign_sha256'] and index['workload']=='notebook','Notebook archive index differs')
    statuses=[read(TASK/n) for n in manifest['files'] if n.startswith(NOTEBOOK+'-archival-admissions/') and sha(TASK/n)==manifest['platform_status_sha256']]
    check(len(statuses)==1 and statuses[0]['success'] is True,'Exact live allocation observation missing')
    allocation=statuses[0]['data'];check(allocation['name']=='gqc-h100-current-1102' and allocation['status']=='RUNNING' and allocation['workspace']=='分布式训练空间' and allocation['project']=='原子级化学反应基座模型2.0' and allocation['image']=='generativeqc-science-py311-cuda129:20261015' and allocation['resource']['gpu_count']==1 and allocation['resource']['cpu_count']==20 and allocation['resource']['memory_gib']==200 and allocation['shared_memory_gib']==16,'Wrong notebook allocation')
    for stage_name in ['fast-uks-capability-diagnostic-preparation-v1.json','fast-direct-capability-diagnostic-preparation-v2.json','facade-singleton-fallback-stage-preparation-v1.json']:
        prefix=TASK/(NOTEBOOK+'-archival-admissions');stage=read(prefix/stage_name);receipt=read(prefix/stage['stage_receipt'])
        check(receipt['status']=='PASS_EXACT_EXTRA_PRODUCERS_STAGED_ONLY_NOT_RUN' and receipt['preparation_sha256']==sha(prefix/stage_name) and receipt['files']==stage['files'],'Exact staged-version provenance differs')
        for name,b in stage['files'].items():
            path=TASK/(NOTEBOOK+'-available-producers')/name
            check(path.stat().st_size==b['bytes'] and sha(path)==b['sha256']==sha(TASK/name),'Staged version bytes differ')
    parents={}
    for phase,b in manifest['parents'].items():
        root=TASK/NOTEBOOK/'observations'/phase;parent=read(root/b['record'])
        expected_status={'fast-uks-capability-diagnostic-v1':'TERMINAL_TWO_DIAGNOSTICS_NO_ACCEPTANCE','fast-direct-capability-diagnostic-v2':'TERMINAL_THREE_DIAGNOSTICS_NO_ACCEPTANCE','facade-singleton-fallback-v1':'TERMINAL_8_SINGLETON_FALLBACK_ATTEMPTS_INDEPENDENT_REPLAY_PENDING'}[phase]
        check(parent['status']==expected_status,'A parent is not in its actual completed-record state')
        check(sha(root/b['record'])==b['sha256'] and parent['status']==b['status'] and len(parent['runs'])==b['runs'] and parent['device']['uuid']==index['device_uuid'] and parent['source_revision']==SOURCE,'Terminal notebook parent differs')
        check(parent.get('production_build_receipt_sha256',parent.get('build_receipt_sha256'))==sha(TASK/'production-build-v1/build-receipt.json'),'Parent production artifact differs')
        for name,h in parent.get('scripts',{}).items():
            check(sha(TASK/(NOTEBOOK+'-executed-producers')/name)==h==sha(TASK/name),'Self-recorded fallback producer differs')
        for row in parent['runs']:
            check(row['returncode']==0 and sha(root/(row['case']+'.plan.json'))==row['plan_sha256'] and sha(root/(row['case']+'.json'))==row['raw_sha256'] if 'raw_sha256' in row else row['returncode']==0 and sha(root/(row['case']+'.plan.json'))==row['plan_sha256'] and sha(root/(row['case']+'.json'))==row['receipt_sha256'],'Actual notebook row failed or bytes differ')
        parents[phase]=(root,parent)
    return manifest,index,allocation,parents

def evaluate(raw,plan,population,protocol,targets,lookup,old_base,build,policy,device_uuid,notebook,allow_compat):
    expected=[(stage['stage'],i,case_id) for stage in population['stages'] for i,case_id in enumerate(stage['case_ids'])]
    check(raw['schema']=='generativeqc.h100-public-facade-guards.v1' and raw['status']=='COMPLETE_PUBLIC_FACADE_EF_AND_LIFETIME_INDEPENDENT_REPLAY_PENDING' and raw['population']==plan['population']==population and raw['source_revision']==plan['source_revision']==SOURCE and raw['protocol_sha256']==plan['protocol_sha256'] and raw['endpoint_sha256']==plan['endpoint_sha256'],'Complete raw identity differs')
    check(plan['build_artifacts']==build['artifacts'] and plan['execution_policy']==raw['execution_policy']==policy and raw['qz_allocation']==plan['qz_allocation'],'Artifact/policy/allocation changed')
    check(raw['accelerator']['nvidia_smi']['uuid']==device_uuid and raw['accelerator']['compute_capability']==[9,0] and raw['environment']['packages']==common.PACKAGES,'Actual H100/runtime differs')
    if notebook:
        check(plan['qz_allocation']['workload']=='notebook' and plan['qz_allocation']['notebook_name']=='gqc-h100-current-1102' and plan['qz_allocation']['project']=='原子级化学反应基座模型2.0' and plan['qz_allocation']['workspace']=='分布式训练空间','Notebook execution identity differs')
    else:
        check(plan['qz_allocation']['job_name']=='gqc-h100-maincd2b-facade-1102-v1' and plan['qz_allocation']['project']=='原子级化学反应基座模型2.0','Original job identity differs')
    check([(s['stage'],s['item'],s['oracle_case']) for s in raw['samples']]==expected,'Complete original physical sequence differs')
    observed={'prepared_item_requests':0,'native_force_eligible_item_requests':0,'compatibility_item_requests':0,'failed_proposal_items':0}
    for call in raw['native_calls']:
        count=call['system_count'];before=call['eligible_before'];after=call.get('eligible_after')
        check(type(count)is int and count>0 and call['requested_properties']==['energy','forces'] and len(before)==count and all(type(x)is bool for x in before),'Invalid actual prepared request')
        check(allow_compat or before==[True]*count,'Unsupported native-only path was promoted')
        check(call['status']=='COMPLETE' and after==before and len(call['transport_before'])==len(call['transport_after'])==call['returned_items']==len(call['returned_statuses'])==count and [x['index'] for x in call['returned_statuses']]==list(range(count)),'Prepared return/provenance differs')
        observed['prepared_item_requests']+=count;observed['native_force_eligible_item_requests']+=sum(before);observed['compatibility_item_requests']+=sum(not x for x in before)
        observed['failed_proposal_items']+=sum(not (x['status']==0 and x['converged'] is True) for x in call['returned_statuses'])
        check(all(type(x['warm_start_used'])is bool and type(x['warm_start_fallback'])is bool for x in call['returned_statuses']),'Prepared warm provenance missing')
    rows=[];route_rows=[]
    for sample in raw['samples']:
        case=lookup[sample['oracle_case']]
        check(sample['geometry_bohr']==case['geometry_bohr'] and sample['charge']==case['charge'] and sample['spin']==case['spin'] and sample['actual_core_sha256']==build['artifacts']['libgenerativeqc.so'],'Physical model or core changed')
        first,last=sample['native_call_range'];check(type(first)is int and type(last)is int and 0<=first<last<=len(raw['native_calls']) and all(c['phase']==sample['stage'] and c['status']=='COMPLETE' for c in raw['native_calls'][first:last]),'Returned result lacks actual request range')
        if population['arm']=='fast':
            v,b=sample['precision_validation'],sample['batch_precision_validation']
            check(v and v['profile']=='fast' and v['schema_version']==2 and v['timing_scope']=='shared_batch_endpoint' and v['endpoint_seconds'] is None and v['solve_count']==len(v['solves']) and b and b['schema_version']==1 and b['profile']=='fast' and b['candidate_calls']==1,'Fast candidate/audit costs/provenance missing')
        diff=error(sample,targets[case['id']],case['atoms']);check(all(diff[name]<=bound for name,bound in protocol['common_observable_gates'].items()),'Independent dense complete E/F gate failed')
        if case['id']=='rks-pbe0-water96-base':check(all(v<=protocol['common_observable_gates'][n] for n,v in error(sample,old_base,96).items()),'Original large dense gate failed')
        matches=[]
        for call in raw['native_calls'][first:last]:
            for index,item in enumerate(call['returned_statuses']):
                if item['status']==0 and item['converged'] is True and item['energy']==sample['energy'] and item['forces']==sample['forces']:matches.append(call['eligible_before'][index])
        check(matches,'Public value has no actual prepared-return witness')
        route_rows.append({'stage':sample['stage'],'item':sample['item'],'possible_force_routes':sorted({'native-prepared' if value else 'compatibility' for value in matches})})
        rows.append({'stage':sample['stage'],'item':sample['item'],'oracle_case':case['id'],'errors':diff})
    coverage=[]
    for stage in population['stages']:
        proof=legacy.native_stage_proof([s for s in raw['samples'] if s['stage']==stage['stage']],raw['native_calls'],population['arm'],len(stage['case_ids']))
        check(proof['status']=='PASS','Actual public candidate/audit coverage or work accounting failed')
        proof['successful_prepared_returns']=proof.pop('successful_native_returns')
        proof['witness']=[{'public_item':w['public_item'],'prepared_call':w['native_call'],'prepared_item':w['native_item']} for w in proof['witness']]
        proof['scope']='Actual PreparedBatch returns and exact candidate/audit work coverage; force route qualification is reported separately.'
        coverage.append({'stage':stage['stage'],**proof})
    controls=[{'owner':i,'name':name,'exception':'RuntimeError'} for i in range(len(population['owners'])) for name in ['closed-owner-execute','closed-owner-clear-warm-starts','closed-owner-set-warm-updates']]
    check(raw['negative_controls']==controls and raw['detached_result_count']==len(expected),'Every actual owner needs all three closed checks and stable detached outputs')
    directions=[]
    if population['kind']=='small-fast' and population['family'] in protocol['directions']:
        samples={s['stage']:s for s in raw['samples'] if s['item']==0};direction=protocol['directions'][population['family']]
        check(abs(math.fsum(v*v for row in direction for v in row)-1)<1e-14,'Direction normalization changed')
        analytic=-math.fsum(force*component for ff,dd in zip(samples['base-cold']['forces'],direction,strict=True) for force,component in zip(ff,dd,strict=True))
        for step in [0.001,0.0005]:
            central=(samples[f'direction-{step:g}-+1']['energy']-samples[f'direction-{step:g}--1']['energy'])/(2*step);delta=abs(central-analytic)
            check(math.isfinite(delta) and delta<=protocol['directional_force_gate_hartree_per_bohr'],'Directional analytic-force gate failed');directions.append({'step_bohr':step,'absolute_error':delta})
    return {'case':population['id'],'status':'PASS_SCOPED_PUBLIC_API_SCIENCE_AND_LIFETIME','public_EF':len(expected),'closed_controls':len(controls),'owners':len(population['owners']),'directions':directions,'numerical':rows,'prepared_request_coverage':coverage,'public_force_routes':route_rows,**observed}

def main():
    output=TASK/'facade-correction-v2-local-replay.json';check(not output.exists(),'Preserve previous replay')
    reader_files=[Path(__file__).name,'facade-correction-reference-replay-v1.py','replay-facade-guards-v5.py','cohort-replay-atomic-v1.py'];reader_hashes={n:sha(TASK/n) for n in reader_files}
    contract=read(TASK/'facade-correction-contract-v1.json');old_proof=read(TASK/'facade-guards-v1-local-replay.json');old_manifest=read(TASK/'facade-guards-v1-download-manifest.json');old_campaign=read(TASK/'facade-guards-v1/campaign.json')
    check(contract['original_final_protocol_sha256']==sha(TASK/'original-final-protocol-v1.json')=='2391b71fb49660f682f1b60056b2892d5516f01871d7479319f0be2cf925fdaa','Original final contract changed')
    check(contract['original_failed_proof_sha256']==sha(TASK/'facade-guards-v1-local-replay.json') and old_proof['status']=='FACADE_GUARDS_FAILED_OR_INCOMPLETE_ALL_ORIGINALS_RETAINED' and (old_proof['qualified_populations'],old_proof['public_returned_EF'],old_proof['verified_refusal_controls'],old_proof['directional_controls'])==(9,71,39,6),'Original failed cohort was relabeled')
    check(old_proof['manifest_sha256']==sha(TASK/'facade-guards-v1-download-manifest.json') and contract['original_failed_archive_sha256']==sha(TASK/'facade-guards-v1-completed.tar.xz')==old_manifest['sha256'],'Original archive changed')
    campaign,manifest,build,policy=common.authenticate(TASK,'facade-guards-v1','gqc-h100-maincd2b-facade-1102-v1',list(old_campaign['scripts']),'TERMINAL_THREE_REFERENCE_EF_AND_EIGHTEEN_FACADE_ATTEMPTS_INDEPENDENT_REPLAY_PENDING')
    protocol,refs,targets,lookup,old_base,root,fit,defaults,references=calibration.reference_targets(campaign,manifest,build,policy)
    check(contract['schema']=='generativeqc.h100-corrected-public-api-fallback-guards.v1' and contract['source_revision']==SOURCE and contract['build_receipt_sha256']==sha(TASK/'production-build-v1/build-receipt.json') and contract['core_sha256']==build['artifacts']['libgenerativeqc.so'],'Contract source/artifact identity differs')
    check(contract['original_facade_protocol_sha256']==sha(TASK/'facade-guards-protocol-v1.json') and contract['common_observable_gates']==protocol['common_observable_gates'] and contract['scope_audit_sha256']==sha(TASK/'facade-scope-audit-sol-v1/audit-report.json'),'Contract or independent review binding differs')
    check((contract['expected_physical_guard_populations'],contract['expected_actual_component_populations'],contract['expected_public_returned_EF'],contract['expected_closed_owner_controls'],contract['expected_created_owners'],contract['expected_directional_controls'],contract['expected_typed_unsupported_layout_refusals'])==(18,22,142,84,28,8,4),'Corrected counts changed')
    check((contract['original_final_cases'],contract['original_final_pairs'],contract['original_final_processes'],contract['original_final_complete_EF'])==(30,3,180,720) and contract['original_final_rules_changed'] is False and contract['numerical_gate_change'] is False,'Final scope or gates changed')
    nb_manifest,index,allocation,parents=authenticate_notebook();fallback_protocol=read(TASK/'facade-singleton-fallback-protocol-v1.json')
    check(contract['fallback_protocol_sha256']==sha(TASK/'facade-singleton-fallback-protocol-v1.json') and fallback_protocol['cases']==protocol['cases'] and fallback_protocol['common_observable_gates']==protocol['common_observable_gates'] and contract['fallback_mapping']==fallback_protocol['original_item_mapping'],'Fallback changed scientific inputs/gates')
    population_map={p['id']:p for p in protocol['populations']};reused=[r['case'] for r in old_proof['rows'] if r['status']=='PASS'];check(contract['reused_populations']==reused and len(reused)==9,'Reuse includes a failed or missing original population')
    groups=[set(reused),set(contract['fast_recovery_populations']),set(contract['unsupported_global_auxiliary_populations'])]
    check(len(groups[1])==5 and len(groups[2])==4 and set.union(*groups)==set(population_map) and sum(map(len,groups))==18,'Original physical population omitted or duplicated')
    rows=[];unsupported=[]
    def process(root_dir,stem,population,device,nb,compat,endpoint,expected_protocol):
        plan_path=root_dir/(stem+'.plan.json');raw_path=root_dir/(stem+'.json');plan=read(plan_path);raw=read(raw_path)
        check(raw['plan_sha256']==sha(plan_path) and plan['endpoint_sha256']==sha(TASK/endpoint) and plan['protocol_sha256']==sha(TASK/expected_protocol),'Endpoint/plan/protocol bytes differ')
        check(Path(plan['library_root'])==root and Path(plan['source_root'])==root.parent/'source','Another source/core selected')
        needed={case_id for owner in population['owners'] for case_id in owner['case_ids']}|{case_id for stage in population['stages'] for case_id in stage['case_ids']}
        check(plan['cases']==[case for case in protocol['cases'] if case['id'] in needed],'Frozen model/cases changed')
        aux=Path(plan['auxiliary_basis']);relative=aux.relative_to(root.parent.parent/'20261030-h100-production-15ab-2241');case0=lookup[population['owners'][0]['case_ids'][0]];refined=population['family']=='uks-pbe-water3-cation' and population['mode']=='df'
        expected_aux=fit['new_auxiliary_prior_relative'] if refined else 'inputs-default/'+case0['method']+'-auxiliary.json';expected_sha=fit['new_auxiliary_sha256'] if refined else defaults[aux.name]['sha256']
        check(relative.as_posix()==expected_aux and plan['auxiliary_basis_sha256']==expected_sha==sha(refs/relative),'Auxiliary primitive input changed')
        result=evaluate(raw,plan,population,protocol,targets,lookup,old_base,build,policy,device,nb,compat);result.update(raw_sha256=sha(raw_path),plan_sha256=sha(plan_path),dataset=root_dir.relative_to(TASK).as_posix());rows.append(result)
    for stem in reused:process(TASK/'facade-guards-v1',stem,population_map[stem],campaign['device']['uuid'],False,False,'facade-guard-endpoint-v1.py','facade-guards-protocol-v1.json')
    for phase in ['fast-uks-capability-diagnostic-v1','fast-direct-capability-diagnostic-v2']:
        directory,parent=parents[phase]
        for stage in parent['runs']:
            stem=stage['case'];check(stem in groups[1],'Unexpected recovered Fast population');process(directory,stem,population_map[stem],index['device_uuid'],True,True,'facade-native-capability-diagnostic-v1.py','facade-guards-protocol-v1.json')
    for stem in contract['unsupported_global_auxiliary_populations']:
        raw=read(TASK/'facade-guards-v1'/(stem+'.json'));stage=next(r for r in campaign['runs'] if r['case']==stem)
        check(stage['returncode']==1 and raw['status']=='FAIL_OR_INCOMPLETE' and raw['error']=='RuntimeError: '+contract['expected_original_unsupported_exception']['diagnostic'] and raw['samples']==raw['native_calls']==raw['negative_controls']==[],'Exact unsupported-layout refusal lost or fabricated execution')
        unsupported.append({'case':stem,'status':'EXPECTED_GLOBAL_MOLECULAR_AUXILIARY_LAYOUT_REFUSAL','error':raw['error'],'public_EF':0,'prepare_attempted':True,'prepared_EF_execute_observed':False})
    directory,parent=parents['facade-singleton-fallback-v1'];check(len(parent['runs'])==8,'Singleton fallback population incomplete')
    for stage,population,mapping in zip(parent['runs'],fallback_protocol['populations'],contract['fallback_mapping'],strict=True):
        check(stage['case']==population['id']==mapping['fallback_population'] and stage['original_item_mapping']==mapping,'Fallback original slot/order mapping differs')
        original=population_map[mapping['original_population']];slot=mapping['original_item']
        check(population['arm']==original['arm'] and population['mode']==original['mode'] and population['family']==original['family'] and len(population['owners'])==1 and population['owners'][0]=={**original['owners'][0],'case_ids':[original['owners'][0]['case_ids'][slot]]},'Fallback changed arm or model')
        check(population['stages']==[{**s,'case_ids':[s['case_ids'][slot]]} for s in original['stages']],'Fallback changed cold/warm/moved controls')
        process(directory,stage['case'],population,index['device_uuid'],True,False,'facade-singleton-fallback-endpoint-v1.py','facade-singleton-fallback-protocol-v1.json')
    counts={'component_populations':len(rows),'public_returned_EF':sum(r['public_EF'] for r in rows),'created_owners':sum(r['owners'] for r in rows),'closed_owner_controls':sum(r['closed_controls'] for r in rows),'directional_controls':sum(len(r['directions']) for r in rows),'unsupported_layout_refusals':len(unsupported)}
    check(counts=={'component_populations':22,'public_returned_EF':142,'created_owners':28,'closed_owner_controls':84,'directional_controls':8,'unsupported_layout_refusals':4},'Complete corrected contract not met')
    result={'status':PASS,'source_revision':SOURCE,'build_receipt_sha256':sha(TASK/'production-build-v1/build-receipt.json'),'core_sha256':build['artifacts']['libgenerativeqc.so'],'contract_sha256':sha(TASK/'facade-correction-contract-v1.json'),'original_failed_proof_sha256':sha(TASK/'facade-guards-v1-local-replay.json'),'original_failed_status_retained':old_proof['status'],'original_failed_archive_sha256':contract['original_failed_archive_sha256'],'original_final_protocol_sha256':contract['original_final_protocol_sha256'],'notebook_manifest_sha256':sha(TASK/(NOTEBOOK+'-download-manifest.json')),'notebook_archive_sha256':nb_manifest['sha256'],'physical_guard_populations':18,**counts,'observed_prepared_item_requests':sum(r['prepared_item_requests'] for r in rows),'observed_native_force_eligible_item_requests':sum(r['native_force_eligible_item_requests'] for r in rows),'observed_compatibility_item_requests':sum(r['compatibility_item_requests'] for r in rows),'observed_failed_proposal_items':sum(r['failed_proposal_items'] for r in rows),'references':references,'unsupported_layouts':unsupported,'rows':rows,'scope':contract['scope'],'native_heterogeneous_df_batch_qualified':False,'all_fast_trials_native_force_qualified':False,'clean_timing_votes':0,'scope_audit_sha256':contract['scope_audit_sha256']}
    check(all(sha(TASK/n)==h for n,h in reader_hashes.items()),'CPU receiver/helper changed during replay')
    result.update(reader=Path(__file__).name,reader_sha256=sha(Path(__file__)),reader_files=reader_hashes)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','references','unsupported_layouts']}))

if __name__=='__main__':main()
