"""Replay every preregistered policy attempt, full vectors and same-GPU controls."""
from pathlib import Path
from collections import Counter
import hashlib,json,lzma,math,statistics,subprocess,sys

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[1]/'generativeqc-h100-production'

def check(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def numerical(sample,target,atoms):
    check(len(sample['forces'])==len(target['forces'])==atoms,'force atom count differs')
    differences=[a-b for x,y in zip(sample['forces'],target['forces'],strict=True) for a,b in zip(x,y,strict=True)]
    check(len(differences)==atoms*3 and all(math.isfinite(v) for v in differences),'invalid dense force vector')
    check(math.isfinite(sample['energy']) and math.isfinite(sample['seconds']) and sample['seconds']>0,'invalid complete E/F observation')
    return {'energy_abs_per_atom_hartree':abs(sample['energy']-target['energy'])/atoms,'force_max_abs_hartree_per_bohr':max(map(abs,differences)),'force_rms_hartree_per_bohr':math.sqrt(math.fsum(v*v for v in differences)/len(differences))}

def main():
    out=TASK/'screening-v1';campaign_path=out/'campaign.json';campaign=json.loads(campaign_path.read_text())
    manifest_path=TASK/'screening-v1-download-manifest.json';manifest=json.loads(manifest_path.read_text())
    archive=TASK/manifest['archive']
    check(manifest['status']=='COMPLETE_IMMUTABLE_TERMINAL_ARCHIVE' and manifest['cohort']=='screening-v1' and archive.parent==TASK and archive.stat().st_size==manifest['bytes'] and sha(archive)==manifest['sha256'],'archive identity/bytes differ')
    check(sha(TASK/'screening-v1-archival-producers/archive-terminal-v4.py')==manifest['archiver_sha256'],'archiver record differs')
    check(manifest['recorded_provenance_errors']==[] and manifest['platform_status']=='SUCCEEDED','failed/incomplete platform or missing provenance')
    check(all((TASK/n).stat().st_size==v['bytes'] and sha(TASK/n)==v['sha256'] for n,v in manifest['files'].items()),'recovered originals differ')
    check(sha(campaign_path)==manifest['campaign_sha256'],'campaign changed')
    observed_path=TASK/'screening-v1-archival-admissions/observed-platform-status.json';observed=json.loads(observed_path.read_text())
    check(sha(observed_path)==manifest['platform_status_sha256'] and observed['success'] is True and observed['data']['name']==campaign['job'] and observed['data']['status']=='SUCCEEDED','wrong terminal platform response')
    prereg_path=TASK/'screening-protocol-v3.json';prereg=json.loads(prereg_path.read_text())
    check(sha(prereg_path)==campaign['protocol_sha256'] and prereg['source_revision']==campaign['source_revision']=='7013148d8d74e714faf20ce87c533db916264f37' and prereg['source_tree']==campaign['source_tree']=='2a295aa3b05116461a0f4813e4bb9a8d720de1ce','native source/protocol changed')
    check(prereg['numerical_gates']=={'energy_abs_per_atom_hartree':1e-5,'force_max_abs_hartree_per_bohr':1e-4,'force_rms_hartree_per_bohr':3e-5},'original gates changed')
    check(campaign['status'] in ['COMPLETE_66_FIXED_POLICY_SCREENING_ALL_QUALIFIED','SCREENING_WITH_FAILURES'] and len(campaign['runs'])==len(prereg['order'])==66,'incomplete/changed66process campaign')
    mandatory={'screening-v1/campaign.json','screening-v1/reference-runtime.json','screening-v1/reference-runtime.stdout','screening-v1/reference-runtime.stderr','screening-v1-archival-admissions/build-receipt.json','screening-v1-archival-admissions/observed-platform-status.json','screening-v1-archival-producers/archive-terminal-v4.py'}|{'screening-v1-executed-producers/'+n for n in campaign['scripts']}|{'screening-v1-executed-producers/external/'+n for n in campaign['external_scripts']}
    mandatory |= {'screening-v1/retained-inputs/'+n for n in ['reference-reuse-v2.json','df-molecular-v1-local-replay.json','current64de-final-inputs-frozen-v1.json','upstream492-global-policy-selected-v1.json']}
    check(mandatory<=set(manifest['files']),'mandatory consumed raw/producer/admission missing')
    runtime=json.loads((out/'reference-runtime.json').read_text())
    check(runtime['status']=='PASS_REFERENCE_RUNTIME_AND_CUTENSOR_ONLY' and runtime['script_sha256']==campaign['scripts']['reference-runtime.py'] and runtime['installed_files_equal_original_wheels']>0 and math.isfinite(runtime['max_abs_error']) and runtime['max_abs_error']<1e-12 and campaign['device']['uuid'] in runtime['device'],'reference runtime/physical device admission differs')
    for n,h in campaign['scripts'].items():check(sha(TASK/n)==h==sha(TASK/'screening-v1-executed-producers'/n),'executed producer changed')
    check(set(campaign['scripts'])=={'run-screening-v4.py','run-screening-v4-job.sh','policy-endpoint.py','reference-runtime.py','pilot.py','process-ownership-v1.py','screening-protocol-v3.json'},'producer inventory differs')
    external=TASK/'screening-v1-executed-producers/external'
    check(set(campaign['external_scripts'])=={'endpoint.py','hf-fit-reference.py'},'external producer inventory differs')
    for n,h in campaign['external_scripts'].items():check(sha(external/n)==h==prereg['external_reference_files'][n],'external producer changed')
    build_path=TASK/'screening-v1-archival-admissions/build-receipt.json';expanded=json.loads(build_path.read_text())
    check(sha(build_path)==campaign['build_receipt_sha256'] and expanded['source_revision']==campaign['source_revision'] and expanded['source_tree']==campaign['source_tree'] and expanded['status']=='PASS_SM90_PRODUCTION33_ARTIFACT_BUILD_ONLY_NOT_SCIENTIFIC_ACCEPTANCE' and expanded['artifacts']==campaign['artifacts'] and len(expanded['artifacts'])==33,'production33artifact binding differs')
    retained=out/'retained-inputs'
    check(campaign['reference_source_revision']==prereg['reference_source_revision']=='15ab66a68a061355fda833a93e88484dbfee6360' and campaign['reference_source_tree']==prereg['reference_source_tree']=='a09359d09a300949374608d773936c8f69e30d6e','reference source differs')
    for n,h in prereg['reference_helper_files'].items():
        path=retained/'reference-source'/n
        check('screening-v1/retained-inputs/reference-source/'+n in manifest['files'] and sha(path)==h and subprocess.check_output(['git','-C',str(ROOT),'show',prereg['reference_source_revision']+':'+n])==path.read_bytes(),'frozen reference helper/source differs')
    for n,h in prereg['external_reference_files'].items():check('screening-v1/retained-inputs/'+n in manifest['files'] and sha(retained/n)==h,'reference endpoint/wheel manifest differs')
    reuse_path=retained/'reference-reuse-v2.json';check(sha(reuse_path)==prereg['reference_reuse_sha256'],'reuse contract differs');reuse=json.loads(reuse_path.read_text())
    gate_path=retained/'df-molecular-v1-local-replay.json';gate=json.loads(gate_path.read_text());check(sha(gate_path)==campaign['df_molecular_admission_sha256'] and gate['status']=='PASS_CURRENT_SOURCE44_DF_TESTS_FULL_PUBLIC_EF_AND_COMPARISON_REPLAY' and gate['source_revision']==expanded['source_revision'] and gate['build_receipt_sha256']==sha(build_path) and gate['tests']==44 and gate['public_full_ef']==299 and len(gate['per_call_projection_trace_replays'])==12,'current44DF gate differs')
    freeze_path=retained/'current64de-final-inputs-frozen-v1.json';check(sha(freeze_path)==prereg['input_freeze_sha256'],'original frozen inputs changed');freeze=json.loads(freeze_path.read_text())
    selection_path=retained/'upstream492-global-policy-selected-v1.json';selection=json.loads(selection_path.read_text());check(sha(selection_path)==prereg['prior_policy_selection_sha256'] and selection['selected_policy']==prereg['fixed_policy']=={'id':'integrated-auto-rys-public-tile1024','ks_tile_points':1024,'df_environment':{}},'previously fixed policy changed')
    inputs={kind:{n:v['sha256'] for n,v in binding['files'].items()} for kind,binding in freeze['groups'].items()}
    base={'revision':prereg['reference_source_revision'],'driver_sha256':prereg['reference_driver_sha256']}
    methods=['hf','pbe','pbe0','b3lyp','r2scan','r2scan-3c','m06-2x','wb97m-v'];primary={(m,a,mode) for m in methods for a in [48,96] for mode in ['direct','df'] if (m,mode)!=('wb97m-v','df')}
    check(len(primary)==30,'fixed30 input set differs')
    oracle_path=ROOT/'benchmarks/results/readme-repaired-20261021/oracles.json.xz';check(sha(oracle_path)==prereg['oracle_sha256']=='d61a3e8ba0a8f8a8c08ee4c4b30c8e3e397e67863b8d75e7ba974e90cf0f85a4','independent oracle differs');oracles=json.loads(lzma.decompress(oracle_path.read_bytes()))
    sys.path[:0]=[str(ROOT),str(ROOT/'python')]
    from benchmarks.readme_defaults_qualification import qualify
    # Authenticate every actually imported benchmark helper against frozen Git.
    for name,module in list(sys.modules.items()):
        if not name.startswith('benchmarks'):
            continue
        filename=getattr(module,'__file__',None)
        check(isinstance(filename,str) and bool(filename),'benchmark helper lacks a source-file binding')
        path=Path(filename)
        if path.suffix=='.py':
            relative=path.relative_to(ROOT).as_posix()
            check(subprocess.check_output(['git','-C',str(ROOT),'show',prereg['reference_source_revision']+':'+relative])==path.read_bytes(),'CPU receiver helper differs from frozen reference source')
    qualifier=ROOT/'benchmarks/readme_defaults_qualification.py'
    rows=[];indexed={};failures=Counter();all_returned=0;qualified_calls=0;actual_samples={}
    for declared,stage in zip(prereg['order'],campaign['runs'],strict=True):
        method,atoms,mode,policy_name=(declared[k] for k in ['method','atoms','mode','policy'])
        reference=policy_name=='reference';policy=None if reference else prereg['fixed_policy']
        arm='gpu4pyscf' if reference else 'standard';stem=f'{method}-{atoms}-{mode}-{policy_name}';key=(method,atoms,mode,policy_name)
        check(key not in indexed and stage['case']==stem and stage['status'] in ['COMPLETE','FAILED_OR_INCOMPLETE'],'duplicate/missing/reordered attempt')
        check({'screening-v1/'+stem+'.stdout','screening-v1/'+stem+'.stderr'}<=set(manifest['files']),'population stdout/stderr missing from inventory')
        plan_path=out/(stem+'.plan.json');check('screening-v1/'+plan_path.name in manifest['files'] and sha(plan_path)==stage['plan_sha256'],'plan bytes/inventory differ');plan=json.loads(plan_path.read_text())
        check((plan['method'],plan['atoms'],plan['mode'],plan['arm'],plan.get('execution_policy'))==(method,atoms,mode,arm,policy),'scientific case/policy differs')
        identity=base if reference else {'revision':expanded['source_revision'],'driver_sha256':prereg['native_driver_sha256']}
        check(plan['source_revision']==identity['revision'] and plan['driver_sha256']==identity['driver_sha256'],'source/driver differs')
        check(plan['primary_ratio_eligible']==declared['primary_ratio_eligible']==((method,atoms,mode) in primary) and stage['primary_ratio_eligible']==declared['primary_ratio_eligible'],'primary row admission differs')
        fitted=method=='hf' and mode=='df';kind='inputs-hf-ccpvtz' if fitted else 'inputs-default'
        check(plan['input_sha256']==inputs[kind],'scientific input/fit differs')
        if fitted:check(plan['fit_policy']=='EXPLICIT_CC_PVTZ_JKFIT_ACCURACY_CANDIDATE_NOT_FINAL','HF fitting policy differs')
        script=('hf-fit-reference.py' if fitted else 'endpoint.py') if reference else 'policy-endpoint.py'
        check(plan['endpoint_sha256']==sha((external if reference else TASK)/script),'endpoint producer differs')
        check(plan['build_artifacts']==({} if reference else expanded['artifacts']),'core/AOT plan binding differs')
        if not reference:check(Path(plan['library_root']).name=='build-sm90-production-v1','molecular plan used a fixture build')
        row={'case':stem,'method':method,'atoms':atoms,'mode':mode,'policy':policy_name,'arm':arm,'status':stage['status'],'returncode':stage['returncode'],'qualification':stage.get('qualification'),'primary_vote':False};indexed[key]=row
        raw_path=out/(stem+'.json')
        if not raw_path.exists():
            check(stage['status']=='FAILED_OR_INCOMPLETE' and 'receipt_sha256' not in stage,'missing complete receipt');row.update(failure_category='NO_RAW_RECEIPT',sample_count=0);failures[row['failure_category']]+=1;rows.append(row);continue
        check('screening-v1/'+raw_path.name in manifest['files'] and 'screening-v1/'+stem+'.qualification.json' in manifest['files'] and sha(raw_path)==stage['receipt_sha256'],'raw/qualification bytes/inventory differ');raw=json.loads(raw_path.read_text());actual_samples[key]=raw['samples']
        check((raw['method'],raw['atoms'],raw['mode'],raw['arm'])==(method,atoms,mode,arm),'raw model differs')
        check(raw['plan_sha256']==sha(plan_path) and raw['runner_sha256']==sha((external if reference else TASK)/script) and raw['source_revision']==identity['revision'] and raw.get('driver_sha256',raw.get('driver_context_sha256'))==identity['driver_sha256'],'raw source/producer binding differs')
        check(raw['qz_allocation']==plan['qz_allocation'] and raw['qz_allocation']['job_name']==campaign['job'],'actual job binding differs')
        check(raw['protocol']['common_observable_gates']==prereg['numerical_gates'],'numerical gates changed')
        check(raw['geometry_bohr']==plan['geometry_bohr'] and raw['accelerator']['nvidia_smi']['uuid']==campaign['device']['uuid'],'actual geometry/physical GPU differs')
        check(raw['accelerator']['name']=='NVIDIA H100 80GB HBM3' and raw['accelerator']['compute_capability']==[9,0] and raw['scheduler']['kind']=='qz_job' and raw['slurm_job_id'] is None,'device/scheduler differs')
        exact=qualify([raw],oracles,reference_reuse=reuse)[0]
        check(exact==json.loads((out/(stem+'.qualification.json')).read_text()) and exact['status']==stage['qualification'],'exact qualifier replay differs')
        route=True
        if not reference:
            check(raw['execution_policy']==policy,'actual requested policy differs')
            caps=raw.get('native_prepared_force_capabilities',[]);route=method=='hf' or caps==[True]*4
            check(stage['native_force_accepted']==route,'native force-route observation differs')
            if raw['status']=='complete':check(raw['resolved_execution_policy']=={'ks_tile_points':None if method=='hf' else policy['ks_tile_points'],'df_environment':policy['df_environment']},'resolved public policy differs')
            if raw['samples']:check(raw['native_build']['library_sha256']==expanded['artifacts']['libgenerativeqc.so'],'loaded core differs')
            if raw['status']=='complete':
                library_root=Path(plan['library_root']);core_path=Path(raw['native_build']['library_path'])
                check(library_root.is_absolute() and core_path.is_absolute() and core_path.parent==library_root and (core_path.name=='libgenerativeqc.so' or core_path.name.startswith('libgenerativeqc.so.')),'actual core path is outside the declared production root')
                images=raw.get('runtime_images_after_endpoint')
                check(isinstance(images,dict) and images,'missing post-owner runtime image observation')
                core_images=[name for name in images if Path(name).name=='libgenerativeqc.so' or Path(name).name.startswith('libgenerativeqc.so.')]
                check(core_images==[str(core_path)],'missing/duplicate/alternate mapped native core')
                for mapped,binding in images.items():
                    name=Path(mapped).name
                    if '.so' not in name:continue
                    expected=expanded['artifacts']['libgenerativeqc.so'] if mapped==str(core_path) else expanded['artifacts'].get(name)
                    check(expected is not None and Path(mapped).parent==library_root and binding.get('available_after_endpoint') is True and type(binding.get('bytes')) is int and binding['bytes']>0 and binding.get('sha256')==expected,'unknown/changed/unavailable mapped native package image')
                row['post_owner_mapped_native_images']=images
                row['companion_mapping_scope']='AOT companion build/admission is bound; no claim that every companion remains mapped after owner close.' 
        target=next(o for o in oracles if (o['method'],o['atoms'],o['backend'])==(method,atoms,'cuda'))
        check(target['geometry_bohr']==raw['geometry_bohr'],'independent oracle geometry differs')
        errors=[numerical(s,target['samples'][-1],atoms) for s in raw['samples']]
        passing=[all(math.isfinite(e[k]) and e[k]<=limit for k,limit in raw['protocol']['common_observable_gates'].items()) for e in errors]
        all_returned+=len(errors);row.update(sample_count=len(errors),native_route_accepted=route,raw_status=raw['status'],error=raw.get('error'),independent_errors=errors,returned_sample_numeric_pass=passing)
        if exact['status']=='qualified':
            check(stage['status']=='COMPLETE' and stage['returncode']==0 and raw['status']=='complete' and len(errors)==4 and all(passing),'unqualified complete E/F reported qualified')
            check([s['phase'] for s in raw['samples']]==['cold','warm','warm','warm'],'complete endpoint phases differ')
            row.update(warm_seconds=[s['seconds'] for s in raw['samples'][1:]],warm_median_seconds=statistics.median(s['seconds'] for s in raw['samples'][1:]),iterations=[s['iterations'] for s in raw['samples']],fock_builds=[s.get('fock_builds') for s in raw['samples']],reference_effective_potential_builds=[s.get('effective_potential_builds') for s in raw['samples']]);qualified_calls+=4
        else:
            row['failure_category']=raw.get('error') or ('INCOMPLETE_POPULATION' if len(errors)!=4 else exact['status']);failures[row['failure_category']]+=1
        if not route:failures['NATIVE_ROUTE_UNVERIFIED']+=1
        rows.append(row)
    comparisons=[];cross_comparison_count=0
    for row in rows:
        if row['arm']!='standard':continue
        reference=indexed[(row['method'],row['atoms'],row['mode'],'reference')]
        eligible=(row['method'],row['atoms'],row['mode']) in primary
        accepted=bool(row['status']=='COMPLETE' and row['qualification']=='qualified' and row.get('native_route_accepted') and reference['status']=='COMPLETE' and reference['qualification']=='qualified')
        cross=[]
        if accepted:
            native_samples=actual_samples[(row['method'],row['atoms'],row['mode'],row['policy'])]
            reference_samples=actual_samples[(row['method'],row['atoms'],row['mode'],'reference')]
            check(len(native_samples)==len(reference_samples)==4,'cross population differs')
            cross=[numerical(a,b,row['atoms']) for a in native_samples for b in reference_samples]
            check(all(all(math.isfinite(value[gate]) and value[gate]<=limit for gate,limit in prereg['numerical_gates'].items()) for value in cross),'native/reference complete4by4 cross gate failed')
            cross_comparison_count+=len(cross)
        comparisons.append({'method':row['method'],'atoms':row['atoms'],'mode':row['mode'],'policy':row['policy'],'native_warm_seconds':row.get('warm_median_seconds'),'reference_warm_seconds':reference.get('warm_median_seconds'),'exploratory_reference_over_native':reference['warm_median_seconds']/row['warm_median_seconds'] if accepted else None,'primary_input_eligible':eligible,'all_four_native_reference_EF_qualified':accepted,'native_reference_cross_errors':cross,'iterations':row.get('iterations'),'fock_builds':row.get('fock_builds'),'final_majority_vote':False})
    policy_summaries=[]
    for policy in ['tile1024']:
        selected=[row for row in comparisons if row['policy']==policy and row['primary_input_eligible']]
        check(len(selected)==30,'missing fixed primary input')
        valid=[row['exploratory_reference_over_native'] for row in selected if row['all_four_native_reference_EF_qualified']]
        policy_summaries.append({'policy':policy,'fixed_primary_inputs':30,'qualified_native_reference_pairs':len(valid),'single_process_ratios_above_1_05':sum(value>1.05 for value in valid),'complete_primary_geomean':math.exp(math.fsum(math.log(value) for value in valid)/30) if len(valid)==30 else None,'missing_or_unqualified_inputs':[{'method':row['method'],'atoms':row['atoms'],'mode':row['mode']} for row in selected if not row['all_four_native_reference_EF_qualified']],'scope':'One complete global policy on all fixed30inputs. One process per arm. Does not satisfy three independent paired-process majority, changed-geometry/lifetime/resource or publication gates. HF duplicate tile controls remain one physical case per global policy.'})
    result={'status':'PASS_RECEIVER_REPLAY_ALL66_FIXED_POLICY_ATTEMPTS_WITH_FAILURES_RETAINED','source_revision':expanded['source_revision'],'campaign_sha256':sha(campaign_path),'preregistration_sha256':sha(prereg_path),'build_receipt_sha256':sha(build_path),'attempted_populations':len(rows),'all_returned_EF':all_returned,'qualified_EF':qualified_calls,'failure_categories':dict(failures),'native_reference_cross_comparisons':cross_comparison_count,'rows':rows,'exploratory_comparisons':comparisons,'global_policy_summaries':policy_summaries,'scope':'One fixed tile1024 policy on original30 primary inputs; one native/reference process each. Full-vector/independent dense oracle/4x4 native-reference cross checks. No final3freshpair votes, default promotion, current-main universal capability or resource/lifetime acceptance.'}
    destination=TASK/'screening-v1-local-replay-v4.json';check(not destination.exists(),'inspect prior replay');destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','exploratory_comparisons']},indent=2))
if __name__=='__main__':main()
