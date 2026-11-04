"""Strict CPU replay of original Libcint source-channel operands and real work."""
from pathlib import Path
import hashlib
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
import numpy as np

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[1]/'generativeqc-h100-current-evidence'
COHORT='spatial-v1'
REVISION='db3657f08d983791c45b7d949cb9028fe0055e3f'

def check(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def finite(value,shape):
    a=np.asarray(value,dtype=float)
    check(a.shape==tuple(shape) and np.isfinite(a).all(),'Nonfinite/truncated source array')
    return a

def validate_observation(raw,protocol,modeout,core):
    nodes=protocol['expected_nodeids']['production']
    check(raw['status']=='PYTEST_TERMINAL_REPLAY_PENDING' and raw['exitstatus']==0 and raw['collection_only'] is False and raw['collected_nodeids']==nodes,'Actual collection/execution differs')
    check(raw['pytest_isolation']=={'noconftest':True,'autoload_disabled':'1','PYTEST_PLUGINS':None,'PYTEST_ADDOPTS':None,'loaded_conftest_files':[]},'Pytest isolation differs')
    check(raw['actual_core']['sha256']==core and len(raw['actual_core']['paths'])==1 and raw['gpu4pyscf_imported'] is False,'Actual core/reference differs')
    check(raw['oracle_runtime']['numpy']['version']=='2.2.6' and raw['oracle_runtime']['pyscf']['version'].startswith('2.14.'),'Independent runtime differs')
    transforms={row['module']:row for row in raw['transforms']}
    check(len(transforms)==len(raw['transforms'])==2 and set(transforms)=={'test_direct_force_grouped_sources_cuda','test_cuda_hybrid_snapshot'},'Adapted module population differs')
    for name,row in transforms.items():
        check(row['all_other_ast_nodes_equal'] is True and row['source_sha256']==protocol['source_files']['tests/python/'+name+'.py'] and row['scheduler_assertion_lines']==([] if name=='test_direct_force_grouped_sources_cuda' else [20,90]),'Scientific AST changed')
    reports={(r['nodeid'],r['when']):r for r in raw['reports']}
    check(len(reports)==len(raw['reports'])==18 and all(reports[(node,phase)]['outcome']=='passed' for node in nodes for phase in ['setup','call','teardown']),'Missing/failed/skipped test phase')
    for field in ['executions','snapshots','exports','comparisons','array_equal_comparisons']:
        check(all(row['nodeid'] in nodes for row in raw[field]),'Unknown cross-node record')
    check(raw['scalar_comparisons']==[],'Unexpected scalar comparison population')
    summaries=[]
    for index,node in enumerate(nodes):
        rows={key:[r for r in raw[key] if r['nodeid']==node] for key in ['executions','snapshots','exports','comparisons','array_equal_comparisons']}
        check(len(rows['executions'])==len(rows['snapshots'])==1 and len(rows['exports'])==3 and len(rows['comparisons'])==5,'Missing/extra per-node scientific work')
        execution=rows['executions'][0]
        check(execution['properties']==['energy'] and len(execution['items'])==1,'Energy-only SCF population differs')
        item=execution['items'][0]
        check(item['index']==0 and item['status']==0 and item['converged'] is True and item['succeeded'] is True and item['executed_backend']=='cuda' and item['forces'] is None and math.isfinite(item['energy']),'Invalid SCF result')
        aos=50 if '[cartesian-' in node else 48
        overlap=finite(rows['snapshots'][0]['overlap'],[aos,aos])
        density=finite(rows['snapshots'][0]['density'],[2 if node.endswith('-pbe0-uks]') else 1,aos,aos])
        coeff=rows['snapshots'][0]['coefficients']
        check(len(coeff)==3 and all(math.isfinite(v) for v in coeff),'Coefficient schema differs')
        check(coeff[2]==(0.0 if node.endswith('-pbe-rks]') else -0.25 if node.endswith('-pbe0-uks]') else -0.125),'Exchange coefficient differs from original method')
        arrays=[];records=[];cursor=0
        for j,(row,name,shape) in enumerate(zip(rows['exports'],protocol['per_test']['exports'],protocol['per_test']['export_shapes'],strict=True)):
            check(row['name']==name and row['core_sha256']==core and row['census_file']=='census-'+str(index)+'.jsonl' and row['census_before_lines']==cursor,'Export/core/census binding differs')
            check(row['args']==([6,536870912] if j==1 else [6]) and row['kwargs']==({'range_exchange':False,'combined_two_electron':True} if j==1 else {}),'Export invocation differs')
            arrays.append(finite(row['values'],shape))
            current=row['census_records']
            check(current and all(r['schema']=='generativeqc.direct-force-census.v1' and r['diagnostic_only'] is True and r['valid'] is True and r['cuda_error']==0 for r in current),'Missing/invalid real CUDA work census')
            check(all(any(c['class']<21 and c['angular_order']<=6 and type(c['admitted']) is int and c['admitted']>0 for c in r['classes']) for r in current),'No positive grouped low-order queue work')
            records.extend(current);cursor+=len(current)
        original=[json.loads(line) for line in (modeout/('census-'+str(index)+'.jsonl')).read_text().splitlines()]
        check(original==records,'Actual census file truncated/rebound')
        actuals=[];desireds=[]
        for j,(row,kwargs) in enumerate(zip(rows['comparisons'],protocol['per_test']['allclose_kwargs'],strict=True)):
            shape=[aos,aos] if j==0 else protocol['per_test']['allclose_shapes'][j]
            actuals.append(finite(row['actual'],shape))
            desireds.append(finite(row['desired'],shape) if j<4 else finite(row['desired'],[]))
            check(row['passed'] is True and row['args']==[] and row['kwargs']==kwargs,'Original numerical tolerance changed')
            np.testing.assert_allclose(actuals[-1],desireds[-1],**kwargs)
        check(np.array_equal(actuals[0],overlap) and np.array_equal(actuals[1],arrays[0]) and np.array_equal(actuals[2],arrays[1][2]) and np.array_equal(actuals[3],arrays[2]) and np.array_equal(actuals[4],arrays[0].sum(axis=1)),'Comparison not bound to actual native channel/snapshot')
        check(np.array_equal(desireds[2],desireds[1].sum(axis=0)) and np.array_equal(desireds[3],desireds[1]) and desireds[4]==0,'Independent channel reconstruction differs')
        equals=rows['array_equal_comparisons']
        check(len(equals)==(1 if coeff[2]==0 else 0),'Pure-PBE zero-exchange assertion differs')
        if equals:
            row=equals[0]
            check(row['passed'] is True and row['args']==[] and row['kwargs']=={} and np.array_equal(finite(row['actual'],[6,3]),arrays[0][1]) and finite(row['desired'],[])==0 and np.all(arrays[0][1]==0),'Independent pure-PBE exact zero exchange differs')
        summaries.append({'nodeid':node,'aos':aos,'energy_only_scf':1,'source_exports':3,'full_EF':0,'positive_census_records':len(records)})
    return summaries

def main():
    manifest=read(TASK/(COHORT+'-download-manifest.json'))
    check(manifest['platform_status']=='SUCCEEDED' and manifest['recorded_provenance_errors']==[] and all((TASK/n).stat().st_size==r['bytes'] and sha(TASK/n)==r['sha256'] for n,r in manifest['files'].items()),'Terminal archive/provenance differs')
    out=TASK/COHORT;campaign=read(out/'campaign.json');protocol=read(TASK/'spatial-protocol-v1.json')
    check(campaign['status']=='TERMINAL_SPATIAL_SOURCES_AND_UNFILTERED_SANITIZERS_RECEIVER_PENDING' and sha(out/'campaign.json')==manifest['campaign_sha256'],'Incomplete parent')
    check(campaign['source_revision']==protocol['source_revision']==REVISION and campaign['source_tree']==protocol['source_tree']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse',REVISION+'^{tree}'],text=True).strip(),'Frozen source differs')
    check(campaign['protocol_sha256']==sha(TASK/'spatial-protocol-v1.json'),'Protocol differs')
    check(campaign['allocation_cgroups']=={'cpu.max':'2000000 100000','memory.max':'214748364800'} and campaign['device']['uuid']==campaign['device_after']['uuid'] and 'H100' in campaign['device']['snapshot'],'Device/allocation differs')
    buildpath=TASK/'production-build-v1/build-receipt.json';build=read(buildpath)
    check(campaign['build_receipt_sha256']==sha(buildpath) and campaign['artifacts']==build['artifacts'],'Production package differs')
    for name,digest in campaign['admissions'].items():check(sha(TASK/name)==digest==sha(TASK/(COHORT+'-available-admissions')/name),'Consumed admission differs')
    for name,digest in campaign['scripts'].items():check(sha(TASK/name)==digest==sha(TASK/(COHORT+'-executed-producers')/name),'Executed producer differs')
    for name,digest in protocol['source_files'].items():
        original=out/'executed-source'/name
        check(sha(original)==digest and original.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show',REVISION+':'+name]),'Scientific source differs')
    toolpath=TASK/'native-qualification-v1-available-admissions/external/sanitizer129-staging-v1.json';tool=read(toolpath)
    check(campaign['sanitizer_staging_sha256']==sha(toolpath) and campaign['sanitizer_sha256']==tool['tool_sha256'] and campaign['sanitizer_version']==tool['version'] and tool['component_version']=='12.9.79','Sanitizer identity differs')
    check([r['mode'] for r in campaign['runs']]==protocol['modes'] and all(r['returncode']==0 for r in campaign['runs']),'Missing/failed affected modes')
    summaries={}
    for run in campaign['runs']:
        mode=run['mode'];modeout=out/mode;raw=read(modeout/'observations.json')
        check(raw['device']['uuid']==raw['device_after']['uuid']==campaign['device']['uuid'] and raw['plugin_sha256']==campaign['scripts']['spatial-qz-plugin-v2.py'],'Device/recorder differs')
        summaries[mode]=validate_observation(raw,protocol,modeout,build['artifacts']['libgenerativeqc.so'])
        xml=ET.parse(modeout/'results.xml');check(len(xml.findall('.//testcase'))==6 and not any(xml.findall('.//'+tag) for tag in ['failure','error','skipped']),'JUnit population differs')
        logs=sorted(modeout.glob('sanitizer-*.log'))
        if mode=='normal':check(not logs,'Normal process unexpectedly instrumented')
        else:
            check(logs,'Sanitizer log missing')
            combined='\n'.join(p.read_text() for p in logs)
            errors=[int(v) for v in re.findall(r'ERROR SUMMARY:\s+(\d+) errors?',combined)]
            check(errors and all(v==0 for v in errors),'Unfiltered sanitizer reported errors/missing summary')
            check('--target-processes' in run['command'] and run['command'][run['command'].index('--target-processes')+1]=='all' and '--kernel-name' not in run['command'] and '--launch-skip' not in run['command'],'Sanitizer process/kernel coverage filtered')
            if mode=='racecheck':
                races=re.findall(r'RACECHECK SUMMARY:\s+(\d+) hazards displayed \((\d+) errors, (\d+) warnings\)',combined)
                check(races and all(all(int(v)==0 for v in row) for row in races),'Unfiltered racecheck hazards/warnings/missing summary')
    result={'status':'PASS_CURRENT_SOURCE_GROUPED_DIRECT_SIX_ALL_MODES_AND_POSITIVE_WORK','source_revision':REVISION,'source_tree':protocol['source_tree'],'campaign_sha256':sha(out/'campaign.json'),'protocol_sha256':sha(TASK/'spatial-protocol-v1.json'),'summaries':summaries,'energy_only_scf':18,'source_exports':54,'full_EF':0,'clean_timing_votes':0,'scope':protocol['scope']}
    with (TASK/(COHORT+'-local-replay-v1.json')).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='summaries'}))
if __name__=='__main__':main()
