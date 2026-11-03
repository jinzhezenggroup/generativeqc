"""Replay the four actual native executions, compiler-policy and loader records."""
from pathlib import Path
import hashlib, importlib.util, json, re
TASK=Path(__file__).resolve().parent
COHORT='native-qualification-v1'
def check(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def main():
    out=TASK/COHORT;c=read(out/'campaign.json');m=read(TASK/(COHORT+'-download-manifest.json'));build=read(TASK/'production-build-v1/build-receipt.json')
    check(c['job']=='gqc-h100-maincd2b-native-1102-v2' and c['status']=='PASS_CURRENT_SOURCE_SPLIT_NATIVE_FOUR_AND_UNFILTERED_MEMCHECK_RECEIVER_PENDING','actual native parent incomplete')
    check(m['platform_status']=='SUCCEEDED' and sha(out/'campaign.json')==m['campaign_sha256'] and not m['recorded_provenance_errors'],'native archive/job differs')
    check(all((TASK/n).is_file() and not (TASK/n).is_symlink() and (TASK/n).stat().st_size==v['bytes'] and sha(TASK/n)==v['sha256'] for n,v in m['files'].items()),'raw inventory differs')
    check(c['source_revision']==build['source_revision']=='cd2b4514be0a38c4920934481b1f02d157a694b4' and c['production_build_receipt_sha256']==sha(TASK/'production-build-v1/build-receipt.json') and c['production_artifacts']==build['artifacts'],'production source/package differs')
    check(c['source_tree']==build['source_tree']=='c76f5bd3cfa62416255c5d6914cf701d34e10a8d','actual native source tree differs')
    check(c['production_admission_sha256']==sha(TASK/'production-build-v1-local-replay.json'),'build admission differs')
    for n,h in c['scripts'].items():check(sha(TASK/(COHORT+'-executed-producers')/n)==h==sha(TASK/n),'executed producer differs')
    check(c['allocation_cgroups']=={'cpu.max':'2000000 100000','memory.max':'214748364800'} and 'H100' in c['device']['snapshot'],'allocation differs')
    fixture=c['fixture_artifacts'];core=build['artifacts']['libgenerativeqc.so']
    check(len(fixture)==33 and fixture['libgenerativeqc.so']!=core and all(fixture[n]==h for n,h in build['artifacts'].items() if n!='libgenerativeqc.so'),'fixture/production distinction differs')
    check(c['compiler_policy']=={'production_actions':617,'fixture_planned_actions':619,'changed_hook_actions':5,'additional_test_units':2,'production_commands_sha256':build['compiler_commands_sha256'],'fixture_commands_sha256':sha(out/'fixture-compiler-commands.txt')},'compiler policy differs')
    spec=importlib.util.spec_from_file_location('native_policy_reader',TASK/'run-native-qualification-v2.py');producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
    command=build['stages'][0]['command'];producer.SOURCE=Path(command[command.index('-S')+1]);production=Path(command[command.index('-B')+1]);fixture_root=production.parent/'build-sm90-fixtures-v1'
    baseline=TASK/'production-build-v1/compiler-commands.txt';check(sha(baseline)==build['compiler_commands_sha256'],'raw production compiler-log hash differs')
    a=producer.policy(baseline.read_text().splitlines(),production);b=producer.policy((out/'fixture-compiler-commands.txt').read_text().splitlines(),fixture_root)
    changed={k for k in a if k in b and a[k]!=b[k]};extra=set(b)-set(a);hook='-DGENERATIVEQC_TEST_HOOKS=1'
    check(len(a)==617 and len(b)==619 and not set(a)-set(b) and len(extra)==2 and {k[0] for k in extra}=={'<source>/tests/native/test_cuda_fock_composition.cpp','<source>/tests/native/test_native_stationary_cuda_binding.cpp'} and {k[0] for k in changed}==producer.HOOKS and len(changed)==5,'independent compiler action comparison differs')
    check(all(b[k].count(hook)==1 and [v for v in b[k] if v!=hook]==a[k] for k in changed) and all(hook not in v for v in a.values()) and all(hook not in b[k] for k in extra),'fixture changes more than five instrumentation sites')
    tool_name=COHORT+'-available-admissions/external/sanitizer129-staging-v1.json'
    check(tool_name in m['files'],'mandatory original sanitizer staging admission missing')
    tool_path=TASK/tool_name;tool=read(tool_path)
    check(sha(tool_path)==c['sanitizer_staging_sha256'] and tool['component_version']=='12.9.79' and c['sanitizer_sha256']==tool['tool_sha256'] and c['sanitizer_version']==tool['version'],'actual sanitizer package/hash/version differs')
    prior=production.parent.parent/'20261030-h100-production-15ab-2241';sanitizer=prior/tool['tool_relative_to_task']
    tests=producer.TESTS;expected=['configure-fixture','build-fixture']+[mode+'-'+name for mode in ['normal','memcheck'] for name in tests]
    check([s['stage'] for s in c['stages']]==expected and all(s['returncode']==0 and s['status']=='COMPLETE' for s in c['stages']),'six stages missing/reordered/failed')
    required={COHORT+'/campaign.json',COHORT+'/fixture-compiler-commands.txt',COHORT+'/sccache-before.txt',COHORT+'/sccache-after-fixture.txt'}
    for stage in c['stages']:required.update(COHORT+'/'+stage['stage']+s for s in ['.stdout','.stderr'])
    for stage in c['stages'][2:]:
        stem=stage['stage'];index=tests.index(next(n for n in tests if stem.endswith(n)));root=production if index==0 else fixture_root;digest=core if index==0 else fixture['libgenerativeqc.so'];name=tests[index]
        check(stage['core_sha256']==digest and stage['executable_sha256']==c['test_executables'][name],'stage executable/core differs')
        required.add(COHORT+'/'+stage['ldd_file']);check(sha(out/stage['ldd_file'])==stage['ldd_sha256'],'ldd bytes differ')
        linked=re.findall(r'^\s*(libgenerativeqc\.so(?:\.[0-9]+)*)\s+=>\s+(\S+)\s+\(', (out/stage['ldd_file']).read_text(),re.M)
        check(len(linked)==1 and Path(linked[0][1]).parent==root and Path(linked[0][1]).name==linked[0][0] and Path(c['loader_bindings'][name]['core_path']).parent==root and c['loader_bindings'][name]['core_sha256']==digest,'actual linked core path differs')
        check('calling init: '+linked[0][1] in (out/(stem+'.stderr')).read_text(),'execution loader initialization missing')
        executable=str(fixture_root/name)
        expected_command=[executable] if stem.startswith('normal-') else [str(sanitizer),'--tool','memcheck','--target-processes','all','--error-exitcode','86','--log-file',str(production.parent/COHORT/(stem+'.memcheck.log')),executable]
        check(stage['command']==expected_command,'actual executable or sanitizer argv differs')
        stdout=(out/(stem+'.stdout')).read_text()
        if index==0:check('CUDA common Fock composition:' in stdout and 'SCF/replay/geometry PASS' in stdout,'composition test did not finish')
        else:
            for label in ['native source inventory/lifetime gates PASS','controller numeric/provider admission/lifetime PASS']:check(sorted(re.findall(r'^([a-z0-9_]+) '+re.escape(label)+r'$',stdout,re.M))==sorted(build['profiles']),'binding profile output missing')
        if stem.startswith('memcheck-'):
            required.add(COHORT+'/'+stem+'.memcheck.log');summaries=[int(v) for v in re.findall(r'ERROR SUMMARY:\s+(\d+) errors?',(out/(stem+'.memcheck.log')).read_text())]
            check(summaries and all(v==0 for v in summaries) and summaries==stage['memcheck_summaries'],'unfiltered memory gate fails')
            check(stage['command'][1:7]==['--tool','memcheck','--target-processes','all','--error-exitcode','86'] and len(stage['command'])==10,'memcheck filters/command differ')
    required.update(COHORT+'-executed-producers/'+n for n in c['scripts']);check(required<=set(m['files']),'mandatory raw/log/producer inventory missing')
    result={'status':'PASS_CURRENT_SOURCE_NATIVE_FOUR_UNFILTERED_MEMCHECK','source_revision':c['source_revision'],'production_core_sha256':core,'fixture_core_sha256':fixture['libgenerativeqc.so'],'campaign_sha256':sha(out/'campaign.json'),'manifest_sha256':sha(TASK/(COHORT+'-download-manifest.json')),'executions':4,'source_stencil_transcript_replay':'NOT_AVAILABLE_IN_UNMODIFIED_CURRENT_NATIVE_STDOUT','scope':'Actual unmodified source-owned native tests and unfiltered all-process memcheck. Internal CPU RSH finite-difference assertions are source-owned; full displaced-energy/source-vector transcript replay is separate and not claimed here. Binary hashes are remote execution records, not local binary inspection.'}
    path=TASK/(COHORT+'-local-replay-v2.json');check(not path.exists(),'inspect previous replay');path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
