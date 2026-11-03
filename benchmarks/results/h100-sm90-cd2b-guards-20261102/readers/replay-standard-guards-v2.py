"""Replay current Standard84 full vectors,directions and bounded owner controls."""
from pathlib import Path
import hashlib,json,lzma,math
TASK=Path(__file__).resolve().parent
COHORT='standard-guards-v1'
SOURCE='cd2b4514be0a38c4920934481b1f02d157a694b4'
def check(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dense(vector,atoms):
    check(len(vector)==atoms and all(len(row)==3 and all(type(v) in [int,float] and math.isfinite(v) for v in row) for row in vector),'invalid complete force vector')

def observable_errors(sample,target,atoms):
    dense(sample['forces'],atoms);dense(target['forces'],atoms)
    check(all(type(x) in [int,float] and math.isfinite(x) for x in [sample['energy'],target['energy']]),'invalid finite energy')
    differences=[a-b for aa,bb in zip(sample['forces'],target['forces'],strict=True) for a,b in zip(aa,bb,strict=True)]
    return {'energy_abs_per_atom_hartree':abs(sample['energy']-target['energy'])/atoms,'force_max_abs_hartree_per_bohr':max(map(abs,differences)),'force_rms_hartree_per_bohr':math.sqrt(math.fsum(v*v for v in differences)/(3*atoms))}

def expected_samples(cases):
    base=next(c for c in cases if c['geometry_label']=='base');moved=next((c for c in cases if c['geometry_label']=='moved'),base)
    order=[('base-cold',0,base['id']),('base-warm',0,base['id'])]
    if moved is not base:order += [('moved',0,moved['id']),('moved-warm',0,moved['id'])]
    order += [(c['geometry_label'],0,c['id']) for c in cases if c['geometry_label'].startswith('direction-')]
    return order+[('return-base',0,base['id']),('bounded-moved',0,moved['id']),('bounded-after-refusal',0,moved['id']),('two-neighbor-lifetime',0,base['id']),('two-neighbor-lifetime',1,moved['id'])]

def guard_controls(raw,expected,tile):
    check(raw.get('lifetime_detached_results_checked')==expected,'detached-result lifetime check missing')
    negative=raw['negative_controls'];check(len(negative)==5,'missing negative controls')
    for row,name in zip(negative[:2],['stationary-below-minimum-before-execution','stationary-below-minimum-after-success'],strict=True):
        check(row=={'name':name,'status_code':7,'expected':7},'below-minimum request did not return actual OOM7')
    check(negative[2:]==[{'name':'closed-owner-execute','exception':'RuntimeError'}]*3,'closed-owner refusal missing')
    check(len(raw['resource_controls'])==1,'missing fresh minimum control');control=raw['resource_controls'][0];values=control['capacity_values']
    check(control['name']=='fresh-stationary-minimum' and len(values)==9 and all(type(v) is int and v>=0 for v in values),'invalid actual capacity fields')
    check(0<values[4]<=values[5] and values[6]==tile and control['applied_device_allowance']==values[4] and control['cached_tier_larger']==(values[5]>values[4]),'stationary minimum/tile/allowance differs')
    return {'status':'PASS','detached_results_checked':expected,'negative_controls':5,'capacity_values':values,'scope':'Fresh stationary CUDA controller minimum only; separate SCF/DF provider arenas and full public request budgets remain outside this gate.'}

def main():
    out=TASK/COHORT;campaign=read(out/'campaign.json');manifest=read(TASK/(COHORT+'-download-manifest.json'))
    check(campaign['job']=='gqc-h100-maincd2b-standard-1102-v1' and campaign['source_revision']==SOURCE and len(campaign['runs'])==8 and campaign['status'] in ['COMPLETE_8_STANDARD_GUARD_POPULATIONS_84_EF_QUALIFIED','NATIVE_GUARDS_WITH_FAILURES'],'actual terminal8 guard populations missing')
    check(sha(out/'campaign.json')==manifest['campaign_sha256'] and manifest['platform_status']=='SUCCEEDED' and not manifest['recorded_provenance_errors'],'campaign/platform/archive differs')
    check(all((TASK/n).is_file() and not (TASK/n).is_symlink() and (TASK/n).stat().st_size==b['bytes'] and sha(TASK/n)==b['sha256'] for n,b in manifest['files'].items()),'recovered originals differ')
    status_path=TASK/(COHORT+'-archival-admissions/standard-terminal-status-v1.json');observed=read(status_path)
    check(sha(status_path)==manifest['platform_status_sha256'] and observed['success'] is True and observed['data']['name']==campaign['job'] and observed['data']['status']=='SUCCEEDED','wrong actual terminal platform response')
    names={'run-standard-guards-v1.py','run-standard-guards-v1-job.sh','standard-guard-endpoint-v1.py','cohort-gates.py','run-screening-v3.py','pilot.py'}
    check(set(campaign['scripts'])==names,'guard producer inventory differs');mandatory={COHORT+'/campaign.json'}
    for n,h in campaign['scripts'].items():
        name=COHORT+'-executed-producers/'+n;mandatory.add(name);check(sha(TASK/name)==h==sha(TASK/n),'actual executed producer differs')
    for row in campaign['runs']:
        mandatory.update(COHORT+'/'+row['case']+suffix for suffix in ['.plan.json','.stdout','.stderr'])
        if 'receipt_sha256' in row:mandatory.add(COHORT+'/'+row['case']+'.json')
    build_path=TASK/'production-build-v1/build-receipt.json';build=read(build_path)
    check(campaign['production_build_receipt_sha256']==sha(build_path) and build['source_revision']==SOURCE and len(build['artifacts'])==33,'current production package differs')
    for n,h in campaign['admissions'].items():
        archived=COHORT+'-available-admissions/'+n;mandatory.add(archived)
        check(sha(TASK/archived)==h==sha(TASK/n),'executed admission differs')
    supplemental=TASK/'native-qualification-v1-local-replay-v2.json';native=read(supplemental)
    check(native['status']=='PASS_CURRENT_SOURCE_NATIVE_FOUR_UNFILTERED_MEMCHECK' and native['source_revision']==SOURCE and native['production_core_sha256']==build['artifacts']['libgenerativeqc.so'] and sha(supplemental)==campaign['admissions']['native-qualification-v1-local-replay.json'],'strict supplemental native4 replay differs')
    prereg_path=TASK/'screening-protocol-v3.json';protocol=read(prereg_path);matrix=read(TASK/'screening-v1-local-replay.json');policy=campaign['policy']['execution_policy']
    check(matrix['source_revision']==SOURCE and matrix['status']=='PASS_RECEIVER_REPLAY_ALL66_FIXED_POLICY_ATTEMPTS_WITH_FAILURES_RETAINED' and matrix['build_receipt_sha256']==sha(build_path) and matrix['preregistration_sha256']==sha(prereg_path) and policy==protocol['fixed_policy'],'same-source66/policy gate differs')
    check(len(matrix['global_policy_summaries'])==1 and matrix['global_policy_summaries'][0]['qualified_native_reference_pairs']==matrix['global_policy_summaries'][0]['fixed_primary_inputs']==30,'all30 science missing')
    check(campaign['allocation_cgroups']=={'cpu.max':'2000000 100000','memory.max':'214748364800'},'actual allocation differs')
    refs=TASK/(COHORT+'-reference-inputs')
    for n,h in campaign['independent_reference_bindings'].items():
        name=COHORT+'-reference-inputs/'+n;mandatory.add(name);check(sha(TASK/name)==h,'independent original input differs')
    check(mandatory<=set(manifest['files']),'mandatory raw/log/producer/admission/reference inventory missing')
    geometries=read(refs/'native-guard-geometries-v1.json');admission_path=refs/'guard-cpu-oracles-local-replay.json';admission=read(admission_path);cpu_campaign=read(refs/'guard-cpu-oracles-v1/campaign.json')
    check(admission['status']=='PASS_28_CPU_EF_CANDIDATES_DUAL_GRID_AND_8_DIRECTIONAL_GATES' and admission['campaign_sha256']==sha(refs/'guard-cpu-oracles-v1/campaign.json'),'independent CPU corpus admission differs')
    cpu={}
    for case in geometries['cases']:
        if case['family']=='rks-pbe0-water96':continue
        path=refs/'guard-cpu-oracles-v1'/(case['id']+'-grid4.json');reference=read(path);binding=next(r for r in cpu_campaign['runs'] if r['case']==case['id'] and r['grid_level']==4)
        check(sha(path)==binding['receipt_sha256'] and reference['case']==case and reference['status']=='complete' and reference['converged'] is True,'CPU full-vector identity differs');cpu[case['id']]=reference
    oracle_path=refs/'data/oracles.json.xz';check(sha(oracle_path)==geometries['original_large_oracle_sha256'],'large original oracle differs')
    large=next(r for r in json.loads(lzma.decompress(oracle_path.read_bytes())) if (r['method'],r['atoms'],r['backend'])==('pbe0',96,'cuda'))
    case=next(c for c in geometries['cases'] if c['family']=='rks-pbe0-water96');check(case['geometry_bohr']==large['geometry_bohr'],'large original geometry differs');cpu[case['id']]=large['samples'][-1]
    auxiliary=read(refs/'inputs-default.json')['files'];results=[];count=0;all_qualified=True
    for declared,stage in zip(geometries['order'],campaign['runs'],strict=True):
        family,mode=declared['family'],declared['mode'];stem=family+'-'+mode;cases=[c for c in geometries['cases'] if c['family']==family];lookup={c['id']:c for c in cases};expected=expected_samples(cases)
        check(stage['case']==stem and stage['status'] in ['COMPLETE','FAIL_OR_INCOMPLETE'],'guard order/status differs')
        plan_path=out/(stem+'.plan.json');plan=read(plan_path);check(sha(plan_path)==stage['plan_sha256'],'guard plan bytes differ')
        check(plan['cases']==cases and plan['source_revision']==SOURCE and plan['arm']=='standard' and plan['mode']==mode and plan['execution_policy']==policy and plan['build_artifacts']==build['artifacts'],'plan scientific/source/policy identity differs')
        check(plan['library_root']==build['stages'][0]['command'][build['stages'][0]['command'].index('-B')+1],'molecular guard selected a fixture or different library')
        check(plan['endpoint_sha256']==sha(TASK/'standard-guard-endpoint-v1.py') and plan['driver_sha256']==protocol['native_driver_sha256'],'plan producer/driver differs')
        name=cases[0]['method']+'-auxiliary.json';check(Path(plan['auxiliary_basis']).name==name and plan['auxiliary_basis_sha256']==auxiliary[name]['sha256']==sha(refs/'inputs-default'/name),'auxiliary basis bytes differ')
        row={'case':stem,'returncode':stage['returncode'],'producer_status':stage['status'],'numerical':[],'directional':[],'scope':'Intrusive guard measurements; no majority timing vote.'};raw_path=out/(stem+'.json')
        if not raw_path.exists():
            check(stage['status']=='FAIL_OR_INCOMPLETE' and 'receipt_sha256' not in stage,'complete guard receipt missing');row.update(status='FAIL_OR_INCOMPLETE',error='NO_RAW_RECEIPT');results.append(row);all_qualified=False;continue
        check(sha(raw_path)==stage['receipt_sha256'],'raw guard bytes differ');raw=read(raw_path)
        check(raw['schema']=='generativeqc.h100-native-guards.v1' and raw['source_revision']==SOURCE and raw['driver_sha256']==plan['driver_sha256'] and raw['runner_sha256']==plan['endpoint_sha256'] and raw['plan_sha256']==sha(plan_path),'raw source/producer identity differs')
        check((raw['family'],raw['arm'],raw['mode'],raw['execution_policy'])==(family,'standard',mode,policy),'raw model/policy differs')
        check(raw['accelerator']['nvidia_smi']['uuid']==campaign['device']['uuid'] and raw['accelerator']['name']=='NVIDIA H100 80GB HBM3' and raw['accelerator']['compute_capability']==[9,0],'actual H100 identity differs')
        check(raw['qz_allocation']==plan['qz_allocation'] and raw['qz_allocation']['job_name']==campaign['job'] and raw['scheduler']['kind']=='qz_job' and raw['slurm_job_id'] is None,'allocation/scheduler differs')
        check(raw['environment']['packages']=={'cupy':'13.6.0','gpu4pyscf':'1.8.1','numpy':'2.2.6','pyscf':'2.14.0'},'native runtime package identity differs')
        observed=[(s['stage'],s['item'],s['oracle_case']) for s in raw['samples']];check(observed==expected[:len(observed)] and len(observed)<=len(expected),'guard sequence duplicated/reordered/extra')
        qualified=stage['returncode']==0 and raw['status']=='COMPLETE_NATIVE_GUARDS_INDEPENDENT_NUMERICAL_REPLAY_PENDING' and observed==expected
        for sample in raw['samples']:
            case=lookup[sample['oracle_case']];check(sample['geometry_bohr']==case['geometry_bohr'] and sample['charge']==case['charge'] and sample['spin']==case['spin'],'returned geometry/spin/charge differs')
            check(sample['native_build']['library_sha256']==build['artifacts']['libgenerativeqc.so'],'loaded native core differs')
            check(type(sample['seconds_batch_intrusive']) in [int,float] and math.isfinite(sample['seconds_batch_intrusive']) and sample['seconds_batch_intrusive']>0,'invalid diagnostic duration')
            error=observable_errors(sample,cpu[case['id']],case['atoms']);passed=all(value<=geometries['common_observable_gates'][key] for key,value in error.items()) and sample['native_prepared_force_before'] is True and sample['native_prepared_force_after'] is True
            if sample['stage'] in ['base-warm','moved-warm']:passed=passed and sample['warm_start_used'] is True
            qualified &= passed;row['numerical'].append({'stage':sample['stage'],'item':sample['item'],'oracle_case':case['id'],'status':'PASS' if passed else 'FAIL','errors':error});count+=1
        if qualified:
            row['controls']=guard_controls(raw,len(expected),policy['ks_tile_points'])
            if family in geometries['directions']:
                samples={s['stage']:s for s in raw['samples'] if s['item']==0};direction=geometries['directions'][family];analytic=-math.fsum(f*d for ff,dd in zip(samples['base-cold']['forces'],direction,strict=True) for f,d in zip(ff,dd,strict=True))
                for step in [0.001,0.0005]:
                    central=(samples[f'direction-{step:g}-+1']['energy']-samples[f'direction-{step:g}--1']['energy'])/(2*step);error=abs(central-analytic);passed=math.isfinite(error) and error<=geometries['directional_force_gate_hartree_per_bohr'];qualified &= passed
                    row['directional'].append({'step_bohr':step,'analytic_directional_gradient':analytic,'central_energy_difference':central,'absolute_error_hartree_per_bohr':error,'status':'PASS' if passed else 'FAIL'})
        row.update(status='PASS' if qualified else 'FAIL_OR_INCOMPLETE',returned_EF=len(raw['samples']),raw_status=raw['status'],error=raw.get('error'));all_qualified &= qualified;results.append(row)
    check(not all_qualified or count==84,'full guard E/F population differs')
    receipt={'status':'PASS_INDEPENDENT_8_STANDARD_POPULATIONS_84_FULL_EF_AND_BOUNDED_CONTROLS' if all_qualified else 'NATIVE_GUARDS_FAILED_OR_INCOMPLETE_WITH_ALL_ORIGINALS_RETAINED','source_revision':SOURCE,'campaign_sha256':sha(out/'campaign.json'),'manifest_sha256':sha(TASK/(COHORT+'-download-manifest.json')),'build_receipt_sha256':sha(build_path),'preregistration_sha256':sha(prereg_path),'cpu_admission_sha256':sha(admission_path),'attempted_populations':8,'returned_EF':count,'qualified_populations':sum(r['status']=='PASS' for r in results),'rows':results,'scope':'Standard small/moved/UKS/full-vector/directional/closed-owner/detached-results and fresh minimum stationary allowance, plus original large96 geometry and equal-size-neighbor batch. No Fast, larger moved, ragged-size batch, full public request budget, clean performance or final majority acceptance.'}
    path=TASK/(COHORT+'-local-replay.json');check(not path.exists(),'inspect existing local guard replay');path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}))
if __name__=='__main__':main()
