"""Replay four affected intrusive E/F outputs and one actual SQLite trace."""
from pathlib import Path
import hashlib,json,lzma,subprocess,sys,tempfile
TASK=Path(__file__).resolve().parent
REPO=TASK.parents[1]/'generativeqc-h100-measurement-db36'
REFERENCE=TASK.parents[1]/'generativeqc-h100-production'
sys.path[:0]=[str(REFERENCE),str(REFERENCE/'python')]
from benchmarks.readme_defaults_qualification import qualify

def check(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def main():
    out=TASK/'nsys-v1';c=read(out/'campaign.json');conversion=read(out/'conversion.json');m=read(TASK/'nsys-v1-analysis-download.json');protocol=read(TASK/'measurement-protocol-v1.json');build=read(TASK/'production-build-v1/build-receipt.json')
    archive=TASK/m['archive']['name'];check(archive.stat().st_size==m['archive']['bytes'] and sha(archive)==m['archive']['sha256'],'analysis archive differs')
    check(c['job']=='gqc-h100-finaldb36-nsys-direct-1103-v1' and c['status']=='ONE_AFFECTED_INTRUSIVE_CAPTURE_ATTEMPT_RECEIVER_ANALYSIS_PENDING' and c['source_revision']==protocol['source_revision']==build['source_revision']=='db3657f08d983791c45b7d949cb9028fe0055e3f','capture source/parent differs')
    platform_path=TASK/'nsys-v1-archival-admissions/observed-platform-status.json';platform=read(platform_path);check(sha(platform_path)==m['platform_status_sha256'] and m['platform_status']=='SUCCEEDED' and m['recorded_provenance_errors']==[] and platform['success'] is True and platform['data']['name']==c['job'] and platform['data']['status']=='SUCCEEDED','actual profile job did not succeed')
    check(sha(out/'campaign.json')==m['campaign_sha256']==conversion['campaign_sha256'] and sha(out/'conversion.json')==m['conversion_sha256'] and conversion['status']=='PASS_CPU_CONVERSION_AND_ONE_ACTUAL_TIMELINE_ORIGINALS_PRESERVED','conversion/capture identity differs')
    check(all((TASK/n).is_file() and not (TASK/n).is_symlink() and (TASK/n).stat().st_size==b['bytes'] and sha(TASK/n)==b['sha256'] for n,b in m['files'].items()),'metadata/SQLite inventory differs')
    required={'nsys-v1/campaign.json','nsys-v1/conversion.json','nsys-v1-archival-admissions/observed-platform-status.json','nsys-v1-archival-admissions/nsys-v1-download-manifest-v4.json'}
    check(sha(TASK/'nsys-v1-archival-admissions/nsys-v1-download-manifest-v4.json')==m['metadata_manifest_sha256'],'original terminal metadata manifest differs')
    check(set(c['scripts'])==set(['run-nsys-affected-v3.py', 'run-nsys-affected-v3-job.sh', 'nsys-endpoint-affected-v1.py', 'run-screening-v4.py', 'measurement-protocol-v1.json', 'pilot.py', 'process-ownership-v1.py', 'nsys-tool-identity-v1.json', 'nsys-cpu-tool-observation-v1.json']) and set(conversion['scripts'])==set(['nsys-offline-affected-v2.py', 'nsys-timeline-affected-v1.py', 'process-ownership-v1.py', 'nsys-cpu-tool-observation-v1.json']),'required capture/conversion producer omitted or extra')
    for receipt in [c,conversion]:
        for n,h in receipt['scripts'].items():
            required.add('nsys-v1-executed-producers/'+n)
            check(Path(n).name==n and sha(TASK/n)==h==sha(TASK/'nsys-v1-executed-producers'/n),'executed producer differs')
    tool=read(TASK/'nsys-tool-identity-v1.json');canonical=Path(tool['path']);invoked=canonical.parents[1]/'nsys-2025.1.1/nsys'
    check(c['nsys_path']==str(invoked) and c['nsys_sha256']==tool['sha256'] and c['nsys_version']==tool['version'],'captured actual profiler identity differs')
    observation_path=TASK/'nsys-cpu-tool-observation-v1.json';observation=read(observation_path)
    check(observation['status']=='ACTUAL_POSTHOC_CPU_TOOL_OBSERVATION' and observation['observer_sha256']==sha(TASK/'observe-nsys-cpu-tools-v1.py'),'separate posthoc CPU tool observation differs')
    for name,prefix in [('nsys','nsys'),('importer','importer')]:
        observed=observation['tools'][name]
        check(conversion[prefix+'_path']==observed['path'] and conversion[prefix+'_sha256']==observed['sha256'] and conversion[prefix+'_version']==observed['version'] and '2025.1.1' in observed['version'],'actual conversion tool differs from separate posthoc observation')
    check(conversion['nsys_sha256']==tool['sha256'] and conversion['nsys_version']==tool['version'],'capture and conversion Nsight binary identities differ')
    check(c['build_receipt_sha256']==sha(TASK/'production-build-v1/build-receipt.json') and c['baseline7013_replay_sha256']==sha(TASK/'source7013-screening-v1-local-replay-v4.json') and c['artifacts']==build['artifacts'],'actual diagnostic prerequisite differs')
    matrix=read(TASK/'source7013-screening-v1-local-replay-v4.json');summary=matrix['global_policy_summaries'];check(len(summary)==1 and summary[0]['qualified_native_reference_pairs']==summary[0]['fixed_primary_inputs']==30,'all30 screening science admission missing')
    applicability=read(TASK/'source-applicability-admission-v1.json');check(matrix['source_revision']=='7013148d8d74e714faf20ce87c533db916264f37' and applicability['status']=='PASS_SOURCE_APPLICABILITY_ONLY_NOT_NEW_DEVICE_ACCEPTANCE' and applicability['source_revision']==c['source_revision'] and applicability['baseline7013_replay_sha256']==sha(TASK/'source7013-screening-v1-local-replay-v4.json'),'prior baseline relabelled or applicability missing')
    expected_admissions={'production-build-v1-local-replay.json': 'PASS_RECEIVER_SM90_PRODUCTION33_BUILD_METADATA_ONLY', 'native-qualification-v1-local-replay-v3.json': 'PASS_CURRENT_SOURCE_NATIVE_FOUR_UNFILTERED_MEMCHECK', 'native-rsh-stencils-v1-local-replay.json': 'PASS_CURRENT_SOURCE_INDEPENDENT_RSH_STENCIL_REPLAY', 'public-direct-v1-local-replay-v3.json': 'PASS_CURRENT_SOURCE_AFFECTED_PUBLIC_DIRECT_EF_AND_FAULT_TESTS', 'spatial-composite-v1-local-replay.json': 'PASS_CURRENT_SOURCE_GROUPED_DIRECT_SIX_ALL_MODES_AND_POSITIVE_WORK', 'source-applicability-admission-v1.json': 'PASS_SOURCE_APPLICABILITY_ONLY_NOT_NEW_DEVICE_ACCEPTANCE'}
    check(set(c['current_admissions'])==set(expected_admissions),'required six current build/native/public/spatial/applicability gates missing or extra')
    for name,digest in c['current_admissions'].items():
        check(sha(TASK/name)==digest==sha(out/'retained-inputs'/name),'current native/public admission differs')
        proof=read(TASK/name);check(proof['status']==expected_admissions[name] and proof['source_revision']==c['source_revision'],'failed/historical gate relabelled current')
        if name=='production-build-v1-local-replay.json':check(proof['build_receipt_sha256']==c['build_receipt_sha256'] and proof['artifacts']==build['artifacts'],'build admission/package differs')
    metadata=read(TASK/'nsys-v1-archival-admissions/nsys-v1-download-manifest-v4.json')
    check(metadata['platform_status']=='SUCCEEDED' and metadata['platform_status_sha256']==sha(platform_path) and metadata['campaign_sha256']==sha(out/'campaign.json') and metadata['recorded_provenance_errors']==[],'original terminal metadata contents differ')
    check(c['source_tree']==build['source_tree']==subprocess.check_output(['git','-C',str(REPO),'rev-parse',c['source_revision']+'^{tree}'],text=True).strip(),'source tree differs')
    for n,h in protocol['reference_helper_files'].items():check(subprocess.check_output(['git','-C',str(REPO),'show',protocol['reference_source_revision']+':'+n])==(REFERENCE/n).read_bytes() and sha(REFERENCE/n)==h,'frozen independent qualifier source differs')
    oracle_path=REFERENCE/'benchmarks/results/readme-repaired-20261021/oracles.json.xz';check(sha(oracle_path)==protocol['oracle_sha256'],'independent oracle differs');oracles=json.loads(lzma.decompress(oracle_path.read_bytes()))
    reuse_path=out/'retained-inputs/reference-reuse-v2.json';check(sha(reuse_path)==protocol['reference_reuse_sha256'],'reuse contract differs');reuse=read(reuse_path)
    check(len(c['runs'])==len(conversion['stages'])==1,'one affected capture population differs');results=[]
    for row,converted,(method,mode) in zip(c['runs'],conversion['stages'],[('pbe0','direct')],strict=True):
        stem=f'{method}-96-{mode}';check(row['case']==converted['case']==stem and row['returncode']==0 and row['status']=='NUMERICALLY_QUALIFIED_CAPTURE_TIMELINE_REPLAY_PENDING' and converted['status']=='PASS_CPU_CONVERSION_AND_ACTUAL_TIMELINE_REDUCTION','capture/conversion order or completion differs')
        plan_path=out/(stem+'.plan.json');raw_path=out/(stem+'.json');plan=read(plan_path);raw=read(raw_path)
        remote_task=Path(plan['source_root']).parent;remote_out=remote_task/'nsys-v1';remote_converted=remote_out/'converted'
        expected_capture=[str(invoked),'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(remote_out/(stem+'.capture')),str(remote_task.parent/'20261027-fd06/env/bin/python'),str(remote_task/'nsys-endpoint-affected-v1.py'),'--plan',str(remote_out/(stem+'.plan.json')),'--output',str(remote_out/(stem+'.json'))]
        check(row['command']==expected_capture,'actual profiler/endpoint capture argv differs')
        expected_conversion=[
            [conversion['importer_path'],'--input-file',str(remote_converted/(stem+'.import-work.qdstrm')),'--output-file',str(remote_converted/(stem+'.work.nsys-rep'))],
            [conversion['nsys_path'],'export','--type=sqlite','--lazy=false','--output='+str(remote_converted/(stem+'.sqlite')),str(remote_converted/(stem+'.work.nsys-rep'))],
            [conversion['python_executable'],str(remote_task/'nsys-timeline-affected-v1.py'),'--database',str(remote_converted/(stem+'.sqlite')),'--endpoint',str(remote_out/(stem+'.json')),'--plan',str(remote_out/(stem+'.plan.json')),'--source-root',str(remote_task/'source'),'--output',str(remote_converted/(stem+'.timeline.json'))]]
        captured_reports=[n for n in row['raw_capture_files'] if n.endswith('.nsys-rep')]
        if captured_reports:
            check(len(captured_reports)==1 and converted['working_raw_after'] is None and converted['outputs'][stem+'.work.nsys-rep']==row['raw_capture_files'][captured_reports[0]],'existing report copy differs from captured original')
            expected_conversion=expected_conversion[1:]
            expected_steps=['export','reduce']
        else:
            check(len([n for n in row['raw_capture_files'] if n.endswith('.qdstrm')])==1,'missing/ambiguous original import stream')
            expected_steps=['import','export','reduce']
        check([step['command'] for step in converted['commands']]==expected_conversion,'actual importer/export/reducer argv differs')
        required.update('nsys-v1/'+stem+s for s in ['.plan.json','.json','.qualification.json','.stdout','.stderr'])
        check(sha(plan_path)==row['plan_sha256']==raw['plan_sha256'] and sha(raw_path)==row['receipt_sha256'] and raw['runner_sha256']==plan['endpoint_sha256']==c['scripts']['nsys-endpoint-affected-v1.py'],'raw/plan/producer binding differs')
        check(raw['source_revision']==plan['source_revision']==c['source_revision'] and raw['driver_sha256']==plan['driver_sha256']==protocol['native_driver_sha256'] and plan['build_artifacts']==build['artifacts'],'source driver/package differs')
        check((raw['method'],raw['atoms'],raw['mode'],raw['arm'])==(method,96,mode,'standard') and raw['execution_policy']==plan['execution_policy']==protocol['fixed_policy'],'scientific identity/policy differs')
        check(raw['native_build']['library_sha256']==build['artifacts']['libgenerativeqc.so'] and Path(raw['native_build']['library_path']).parent==Path(plan['library_root']),'actual core loader differs')
        images=raw['runtime_images_after_endpoint'];check(images and all(v['sha256'] in build['artifacts'].values() and Path(n).parent==Path(plan['library_root']) for n,v in images.items()),'actual mapped native image differs')
        check(raw['accelerator']['nvidia_smi']['uuid']==c['device']['uuid'] and raw['accelerator']['compute_capability']==[9,0] and raw['qz_allocation']==plan['qz_allocation'] and raw['qz_allocation']['job_name']==c['job'],'actual device/allocation differs')
        check(raw['environment']['packages']=={'cupy':'13.6.0','gpu4pyscf':'1.8.1','numpy':'2.2.6','pyscf':'2.14.0'} and len(raw['samples'])==4 and [(s['phase'],s['repeat']) for s in raw['samples']]==[('cold',0),('warm',0),('warm',1),('warm',2)],'runtime or complete four-call order differs')
        check(raw['nsys_capture_sample']=={'phase':'warm','repeat':0} and all(s['warm_start_used'] is True for s in raw['samples'][1:]) and (method=='hf' or raw['native_prepared_force_capabilities']==[True]*4),'actual warm/native capture scope differs')
        q=qualify([raw],oracles,reference_reuse=reuse)[0];check(q==read(out/(stem+'.qualification.json')) and q['status']==row['qualification']=='qualified','all full E/F values failed independent qualifier replay')
        check(converted['input_files']==row['raw_capture_files'],'conversion input differs from actual capture')
        for n,b in row['raw_capture_files'].items():check(m['large_remote_originals']['nsys-v1/'+n]==b,'original remote raw binding differs')
        check([s['name'] for s in converted['commands']]==expected_steps and all(s['status']=='PASS' and s['returncode']==0 for s in converted['commands']),'actual conversion stages missing/failed')
        for n,b in converted['outputs'].items():
            if n.endswith('.nsys-rep'):check(m['large_remote_originals']['nsys-v1/converted/'+n]==b,'remote converted report differs')
            else:
                path=out/'converted'/n;required.add('nsys-v1/converted/'+n);check(path.stat().st_size==b['bytes'] and sha(path)==b['sha256'],'converted SQLite/timeline differs')
        database=out/'converted'/(stem+'.sqlite');timeline_path=out/'converted'/(stem+'.timeline.json');timeline=read(timeline_path)
        check(timeline['database_sha256']==sha(database) and timeline['endpoint_sha256']==sha(raw_path) and timeline['plan_sha256']==sha(plan_path) and timeline['reducer_sha256']==sha(TASK/'nsys-timeline-affected-v1.py'),'actual timeline inputs/reducer differ')
        with tempfile.TemporaryDirectory(prefix='nsys-cd2b-replay-') as tmp:
            output=Path(tmp)/'timeline.json';subprocess.run([sys.executable,str(TASK/'nsys-timeline-affected-v1.py'),'--database',str(database),'--endpoint',str(raw_path),'--plan',str(plan_path),'--source-root',str(REPO),'--output',str(output)],check=True,capture_output=True,timeout=60);check(read(output)==timeline,'actual full SQLite reduction differs locally')
        results.append({'case':stem,'qualification':q['status'],'samples':4,'actual_sqlite_replayed_exactly_locally':True,'coverage':timeline['groups'][0]['coverage'],'clean_timing_vote':False})
    check(required<=set(m['files']),'required original/log/producer/SQLite omitted')
    value={'status':'PASS_CURRENT_SOURCE_ONE_AFFECTED_DIRECT_CAPTURE4_EF_ACTUAL_SQLITE_REPLAY','source_revision':c['source_revision'],'campaign_sha256':sha(out/'campaign.json'),'conversion_sha256':sha(out/'conversion.json'),'analysis_manifest_sha256':sha(TASK/'nsys-v1-analysis-download.json'),'rows':results,'cpu_tool_observation_sha256':sha(observation_path),'cpu_tool_observation_scope':'Separate authenticated posthoc observation crosschecks converter tools,not an independently pinned before-runtime importer witness. Logical profiler launcher/canonical observed target share recorded hash/version; no local remote-path resolution claim.','scope':'Actual4 affected Direct full E/F qualify independently and the one actual SQLite reduction matches locally. HistoricalDFtraces are not rerun or relabelled. Large raw captures/report hashes are remotely recorded bindings,not locally inspected bytes. No broad guard admission,clean timing,occupancy/spill or final-majority claim.'}
    target=TASK/'nsys-v1-local-replay-affected-v1.json';check(not target.exists(),'inspect previous replay');target.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
