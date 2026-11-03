"""Qualify complete normal/memcheck components; retain whole-attempt failure."""
from pathlib import Path
import importlib.util
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[1]/'generativeqc-h100-measurement-db36'

def check(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def expected_command(remote_task, cohort, mode, tool):
    folder=remote_task/cohort/mode;source=remote_task/'source'
    command=[str(remote_task.parent/'20261027-fd06/env/bin/python'),'-m','pytest','-p','spatial-qz-plugin-v2','--assert=plain','--import-mode=prepend','--noconftest','-c','/dev/null','--rootdir',str(source),'--confcutdir',str(remote_task),'--basetemp',str(folder/'pytest'),'--junitxml',str(folder/'results.xml'),'-ra',str(source/'tests/python/test_direct_force_grouped_sources_cuda.py')]
    if mode!='normal':
        prefix=[str(remote_task.parent/'20261030-h100-production-15ab-2241'/tool['tool_relative_to_task']),'--tool',mode,'--target-processes','all','--error-exitcode','86','--log-file',str(folder/'sanitizer-%p.log')]
        if mode=='racecheck':prefix+=['--racecheck-report','all','--racecheck-detect-level','info']
        command=prefix+command
    return command

def main():
    spec=importlib.util.spec_from_file_location('original_spatial_reader',TASK/'replay-spatial-v2.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    m=read(TASK/'spatial-v1-download-manifest.json');out=TASK/'spatial-v1';c=read(out/'campaign.json');p=read(TASK/'spatial-protocol-v1.json');build=read(TASK/'production-build-v1/build-receipt.json')
    check(m['platform_status']=='FAILED' and m['recorded_provenance_errors']==[] and c['status']=='FAILED_OR_INCOMPLETE' and c['error']=="ValueError('Affected spatial test/sanitizer failed; preserve all logs')",'Original failure differs')
    archive=TASK/m['archive'];check(archive.stat().st_size==m['bytes'] and sha(archive)==m['sha256'] and sha(out/'campaign.json')==m['campaign_sha256'],'Original archive/campaign differs')
    check(all((TASK/n).is_file() and not (TASK/n).is_symlink() and (TASK/n).stat().st_size==v['bytes'] and sha(TASK/n)==v['sha256'] for n,v in m['files'].items()),'Original recovered bytes differ')
    check([v['mode'] for v in c['runs']]==['normal','memcheck','racecheck'] and [v['returncode'] for v in c['runs']]==[0,0,124] and c['runs'][2]['timeout_seconds']==3000,'Unexpected phase failure')
    check(c['source_revision']==p['source_revision']==build['source_revision']=='db3657f08d983791c45b7d949cb9028fe0055e3f' and c['source_tree']==p['source_tree']==build['source_tree'] and c['artifacts']==build['artifacts'] and c['build_receipt_sha256']==sha(TASK/'production-build-v1/build-receipt.json') and c['protocol_sha256']==sha(TASK/'spatial-protocol-v1.json'),'Source/package/protocol differs')
    for n,h in c['scripts'].items():check(sha(TASK/n)==h==sha(TASK/'spatial-v1-executed-producers'/n),'Original producer differs')
    for n,h in c['admissions'].items():check(sha(TASK/n)==h==sha(TASK/'spatial-v1-available-admissions'/n),'Original consumed admission differs')
    for n,h in p['source_files'].items():
        f=out/'executed-source'/n;check(sha(f)==h and f.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show',c['source_revision']+':'+n]),'Executed scientific source differs')
    check(c['allocation_cgroups']=={'cpu.max':'2000000 100000','memory.max':'214748364800'} and 'H100' in c['device']['snapshot'],'Allocation differs')
    toolpath=TASK/'native-qualification-v1-available-admissions/external/sanitizer129-staging-v1.json';tool=read(toolpath)
    check(c['sanitizer_staging_sha256']==sha(toolpath) and c['sanitizer_sha256']==tool['tool_sha256'] and c['sanitizer_version']==tool['version'],'Sanitizer identity differs')
    remote_task=Path(build['stages'][0]['command'][build['stages'][0]['command'].index('-B')+1]).parent
    for run in c['runs']:check(run['command']==expected_command(remote_task,'spatial-v1',run['mode'],tool),'Original complete argv differs or is filtered')
    summaries={}
    for mode in ['normal','memcheck']:
        folder=out/mode;raw=read(folder/'observations.json')
        check(raw['device']['uuid']==raw['device_after']['uuid']==c['device']['uuid'] and raw['plugin_sha256']==c['scripts']['spatial-qz-plugin-v2.py'],'Completed component device/plugin differs')
        summaries[mode]=r.validate_observation(raw,p,folder,build['artifacts']['libgenerativeqc.so'])
        xml=ET.parse(folder/'results.xml');check(len(xml.findall('.//testcase'))==6 and not any(xml.findall('.//'+tag) for tag in ['failure','error','skipped']),'Completed component JUnit differs')
        logs=list(folder.glob('sanitizer-*.log'))
        if mode=='normal':check(not logs,'Normal unexpectedly instrumented')
        else:
            check(logs,'Memcheck log missing');text='\n'.join(f.read_text() for f in logs);errors=[int(v) for v in re.findall(r'ERROR SUMMARY:\s+(\d+) errors?',text)];check(errors and all(v==0 for v in errors),'Unfiltered memcheck did not pass')
            command=c['runs'][1]['command'];check(command[command.index('--target-processes')+1]=='all' and not any(v in command for v in ['--kernel-name','--kernel-name-exclude','--launch-skip']),'Memcheck filtered')
    result={'status':'PASS_CURRENT_SOURCE_NORMAL6_AND_UNFILTERED_MEMCHECK6_COMPONENTS_ONLY','source_revision':c['source_revision'],'source_tree':c['source_tree'],'core_sha256':build['artifacts']['libgenerativeqc.so'],'original_campaign_sha256':sha(out/'campaign.json'),'original_manifest_sha256':sha(TASK/'spatial-v1-download-manifest.json'),'original_attempt_status_retained':c['status'],'original_platform_status_retained':'FAILED','summaries':summaries,'energy_only_scf':12,'source_exports':36,'full_EF':0,'clean_timing_votes':0,'racecheck_accepted':False,'scope':'Only complete normal and unfiltered memcheck components. Original unfiltered racecheck timed out without final sanitizer summary, so none of its five completed numeric tests is counted as accepted race coverage. Whole failed attempt and partial data remain unchanged. All six unfiltered racecheck cases still required.'}
    with (TASK/'spatial-v1-components-local-replay-v1.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='summaries'}))
if __name__=='__main__':main()
