"""Replay all twelve intrusive E/F outputs and all three actual SQLite traces."""
from pathlib import Path
import hashlib,json,lzma,subprocess,sys,tempfile
TASK=Path(__file__).resolve().parent
REPO=TASK.parents[1]/'generativeqc-h100-measurement-cd2b'
REFERENCE=TASK.parents[1]/'generativeqc-h100-production'
sys.path[:0]=[str(REFERENCE),str(REFERENCE/'python')]
from benchmarks.readme_defaults_qualification import qualify

def check(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def main():
    out=TASK/'nsys-v1';c=read(out/'campaign.json');conversion=read(out/'conversion.json');m=read(TASK/'nsys-v1-analysis-download.json');protocol=read(TASK/'screening-protocol-v3.json');build=read(TASK/'production-build-v1/build-receipt.json')
    archive=TASK/m['archive']['name'];check(archive.stat().st_size==m['archive']['bytes'] and sha(archive)==m['archive']['sha256'],'analysis archive differs')
    check(c['job']=='gqc-h100-maincd2b-nsys-1102-v1' and c['status']=='THREE_INTRUSIVE_CAPTURE_ATTEMPTS_RECEIVER_ANALYSIS_PENDING' and c['source_revision']==protocol['source_revision']==build['source_revision']=='cd2b4514be0a38c4920934481b1f02d157a694b4','capture source/parent differs')
    platform=read(TASK/'nsys-terminal-status-v1.json');check(platform['success'] is True and platform['data']['name']==c['job'] and platform['data']['status']=='SUCCEEDED','actual profile job did not succeed')
    check(sha(out/'campaign.json')==m['campaign_sha256']==conversion['campaign_sha256'] and sha(out/'conversion.json')==m['conversion_sha256'] and conversion['status']=='PASS_CPU_CONVERSION_AND_3_ACTUAL_TIMELINES_ORIGINALS_PRESERVED','conversion/capture identity differs')
    check(all((TASK/n).is_file() and not (TASK/n).is_symlink() and (TASK/n).stat().st_size==b['bytes'] and sha(TASK/n)==b['sha256'] for n,b in m['files'].items()),'metadata/SQLite inventory differs')
    required={'nsys-v1/campaign.json','nsys-v1/conversion.json'}
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
    check(c['build_receipt_sha256']==sha(TASK/'production-build-v1/build-receipt.json') and c['matrix_replay_sha256']==sha(TASK/'screening-v1-local-replay.json') and c['artifacts']==build['artifacts'],'actual diagnostic prerequisite differs')
    matrix=read(TASK/'screening-v1-local-replay.json');summary=matrix['global_policy_summaries'];check(len(summary)==1 and summary[0]['qualified_native_reference_pairs']==summary[0]['fixed_primary_inputs']==30,'all30 screening science admission missing')
    for n,h in protocol['reference_helper_files'].items():check(subprocess.check_output(['git','-C',str(REPO),'show',protocol['reference_source_revision']+':'+n])==(REFERENCE/n).read_bytes() and sha(REFERENCE/n)==h,'frozen independent qualifier source differs')
    oracle_path=REFERENCE/'benchmarks/results/readme-repaired-20261021/oracles.json.xz';check(sha(oracle_path)==protocol['oracle_sha256'],'independent oracle differs');oracles=json.loads(lzma.decompress(oracle_path.read_bytes()))
    reuse_path=TASK/'screening-v1/retained-inputs/reference-reuse-v2.json';check(sha(reuse_path)==protocol['reference_reuse_sha256'],'reuse contract differs');reuse=read(reuse_path)
    check(len(c['runs'])==len(conversion['stages'])==3,'three capture population differs');results=[]
    for row,converted,(method,mode) in zip(c['runs'],conversion['stages'],[('hf','df'),('pbe0','df'),('pbe0','direct')],strict=True):
        stem=f'{method}-96-{mode}';check(row['case']==converted['case']==stem and row['returncode']==0 and row['status']=='NUMERICALLY_QUALIFIED_CAPTURE_TIMELINE_REPLAY_PENDING' and converted['status']=='PASS_CPU_CONVERSION_AND_ACTUAL_TIMELINE_REDUCTION','capture/conversion order or completion differs')
        plan_path=out/(stem+'.plan.json');raw_path=out/(stem+'.json');plan=read(plan_path);raw=read(raw_path)
        remote_task=Path(plan['source_root']).parent;remote_out=remote_task/'nsys-v1';remote_converted=remote_out/'converted'
        expected_capture=[str(invoked),'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(remote_out/(stem+'.capture')),str(remote_task.parent/'20261027-fd06/env/bin/python'),str(remote_task/'nsys-endpoint.py'),'--plan',str(remote_out/(stem+'.plan.json')),'--output',str(remote_out/(stem+'.json'))]
        check(row['command']==expected_capture,'actual profiler/endpoint capture argv differs')
        expected_conversion=[
            [conversion['importer_path'],'--input-file',str(remote_converted/(stem+'.import-work.qdstrm')),'--output-file',str(remote_converted/(stem+'.work.nsys-rep'))],
            [conversion['nsys_path'],'export','--type=sqlite','--lazy=false','--output='+str(remote_converted/(stem+'.sqlite')),str(remote_converted/(stem+'.work.nsys-rep'))],
            [conversion['python_executable'],str(remote_task/'nsys-timeline-v1.py'),'--database',str(remote_converted/(stem+'.sqlite')),'--endpoint',str(remote_out/(stem+'.json')),'--plan',str(remote_out/(stem+'.plan.json')),'--source-root',str(remote_task/'source'),'--output',str(remote_converted/(stem+'.timeline.json'))]]
        check([step['command'] for step in converted['commands']]==expected_conversion,'actual importer/export/reducer argv differs')
        required.update('nsys-v1/'+stem+s for s in ['.plan.json','.json','.qualification.json','.stdout','.stderr'])
        check(sha(plan_path)==row['plan_sha256']==raw['plan_sha256'] and sha(raw_path)==row['receipt_sha256'] and raw['runner_sha256']==plan['endpoint_sha256']==c['scripts']['nsys-endpoint.py'],'raw/plan/producer binding differs')
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
        check([s['name'] for s in converted['commands']]==['import','export','reduce'] and all(s['status']=='PASS' and s['returncode']==0 for s in converted['commands']),'actual conversion stages missing/failed')
        for n,b in converted['outputs'].items():
            if n.endswith('.nsys-rep'):check(m['large_remote_originals']['nsys-v1/converted/'+n]==b,'remote converted report differs')
            else:
                path=out/'converted'/n;required.add('nsys-v1/converted/'+n);check(path.stat().st_size==b['bytes'] and sha(path)==b['sha256'],'converted SQLite/timeline differs')
        database=out/'converted'/(stem+'.sqlite');timeline_path=out/'converted'/(stem+'.timeline.json');timeline=read(timeline_path)
        check(timeline['database_sha256']==sha(database) and timeline['endpoint_sha256']==sha(raw_path) and timeline['plan_sha256']==sha(plan_path) and timeline['reducer_sha256']==sha(TASK/'nsys-timeline-v1.py'),'actual timeline inputs/reducer differ')
        with tempfile.TemporaryDirectory(prefix='nsys-cd2b-replay-') as tmp:
            output=Path(tmp)/'timeline.json';subprocess.run([sys.executable,str(TASK/'nsys-timeline-v1.py'),'--database',str(database),'--endpoint',str(raw_path),'--plan',str(plan_path),'--source-root',str(REPO),'--output',str(output)],check=True,capture_output=True,timeout=60);check(read(output)==timeline,'actual full SQLite reduction differs locally')
        results.append({'case':stem,'qualification':q['status'],'samples':4,'actual_sqlite_replayed_exactly_locally':True,'coverage':timeline['groups'][0]['coverage'],'clean_timing_vote':False})
    check(required<=set(m['files']),'required original/log/producer/SQLite omitted')
    value={'status':'PASS_CURRENT_SOURCE_3_CAPTURES_12_FULL_EF_AND_ACTUAL_SQLITE_REPLAY','source_revision':c['source_revision'],'campaign_sha256':sha(out/'campaign.json'),'conversion_sha256':sha(out/'conversion.json'),'analysis_manifest_sha256':sha(TASK/'nsys-v1-analysis-download.json'),'rows':results,'cpu_tool_observation_sha256':sha(observation_path),'cpu_tool_observation_scope':'Separate authenticated posthoc observation crosschecks converter tools,not an independently pinned before-runtime importer witness. Logical profiler launcher/canonical observed target share recorded hash/version; no local remote-path resolution claim.','scope':'Actual12 full E/F qualify independently and all3 SQLite reductions match locally. Large raw captures/report hashes are remotely recorded bindings,not locally inspected bytes. No broad guard admission,clean timing,occupancy/spill or final-majority claim.'}
    target=TASK/'nsys-v1-local-replay-v2.json';check(not target.exists(),'inspect previous replay');target.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
