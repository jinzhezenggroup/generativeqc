"""Receiver replay of current-source original DF tests and every numeric operand."""
from pathlib import Path
import ast, hashlib, json, math, subprocess, traceback
import xml.etree.ElementTree as ET
import numpy as np
TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[1]/'generativeqc-h100-measurement-cd2b'
COHORT='df-molecular-v1'
SOURCE='cd2b4514be0a38c4920934481b1f02d157a694b4'
def check(value,message):
    if not value: raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def numeric_comparison(row):
    a,b=np.asarray(row['actual'],dtype=float),np.asarray(row['desired'],dtype=float)
    check(a.shape==b.shape or a.ndim==0 or b.ndim==0,'unexpected comparison broadcast')
    check(np.isfinite(a).all() and np.isfinite(b).all(),'nonfinite comparison operands')
    np.testing.assert_allclose(a,b,*row['args'],**row['kwargs'])
    return float(np.max(np.abs(a-b))) if a.size and b.size else 0.0

def main():
    out=TASK/COHORT
    manifest_path=TASK/(COHORT+'-download-manifest.json');manifest=read(manifest_path)
    check(manifest['cohort']==COHORT,'wrong cohort')
    check(all((TASK/n).stat().st_size==v['bytes'] and sha(TASK/n)==v['sha256'] for n,v in manifest['files'].items()),'recovered originals differ')
    check(manifest['recorded_provenance_errors']==[] and manifest['platform_status']=='SUCCEEDED','failed platform or recorded provenance gap')
    observed_platform=read(TASK/(COHORT+'-archival-admissions/observed-platform-status.json'))
    check(sha(TASK/(COHORT+'-archival-admissions/observed-platform-status.json'))==manifest['platform_status_sha256'] and observed_platform['success'] is True and observed_platform['data']['status']=='SUCCEEDED','platform response changed')
    campaign=read(out/'campaign.json');protocol_path=TASK/'df-molecular-protocol-v1.json';protocol=read(protocol_path)
    check(observed_platform['data']['name']==campaign['job'],'wrong platform job')
    check(campaign['source_revision']==protocol['source_revision']==SOURCE and sha(out/'campaign.json')==manifest['campaign_sha256'],'source/campaign differs')
    check(campaign['source_tree']==protocol['source_tree']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse',SOURCE+'^{tree}'],text=True).strip(),'Git tree differs')
    check(campaign['protocol_sha256']==sha(protocol_path),'protocol changed')
    build_path=TASK/(COHORT+'-archival-admissions/build-receipt.json');build=read(build_path)
    check(sha(build_path)==campaign['build_receipt_sha256'] and build['source_revision']==SOURCE and build['status']=='PASS_SM90_PRODUCTION33_ARTIFACT_BUILD_ONLY_NOT_SCIENTIFIC_ACCEPTANCE' and build['artifacts']==campaign['artifacts'] and len(build['artifacts'])==33,'production package differs')
    mandatory={COHORT+'/'+n for n in ['campaign.json','observations.json','pytest.stdout','pytest.stderr','junit.xml']}|{COHORT+'/executed-source/'+n for n in protocol['source_files']}|{COHORT+'-executed-producers/'+n for n in campaign['scripts']}|{COHORT+'-archival-producers/archive-terminal-v3.py',COHORT+'-archival-admissions/build-receipt.json',COHORT+'-archival-admissions/observed-platform-status.json'}
    check(mandatory<=set(manifest['files']),'missing required raw/source/producer/admission')
    check(set(campaign['scripts'])=={'run-df-molecular-v3.py','run-df-molecular-v3-job.sh','df-molecular-qz-plugin-v2.py','pilot.py','df-molecular-protocol-v1.json'},'producer inventory differs')
    for name,digest in campaign['scripts'].items():check(sha(TASK/name)==digest==sha(TASK/(COHORT+'-executed-producers')/name),'producer changed')
    for name,digest in protocol['source_files'].items():
        path=out/'executed-source'/name
        check(sha(path)==digest and subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':'+name])==path.read_bytes(),'executed source differs from Git')
    check(campaign['allocation_cgroups']=={'cpu.max':'2000000 100000','memory.max':'214748364800'} and campaign['shared_memory_bytes']==16*(1<<30),'allocation differs')
    check(campaign['status']=='TERMINAL_CURRENT_SOURCE44_DF_TESTS_RECEIVER_REPLAY_PENDING' and len(campaign['runs'])==1 and campaign['runs'][0]['returncode']==0 and 0<campaign['elapsed_seconds']<protocol['maximum_parent_seconds'],'failed or incomplete test process')
    raw = read(out / 'observations.json')
    check(raw['status'] == 'PYTEST_TERMINAL_REPLAY_PENDING' and raw['exitstatus'] == 0 and raw['collection_only'] is False and raw['python'].split()[0] == '3.11.16', 'not an actual successful native pytest process')
    check(raw['pytest_isolation'] == {'noconftest':True,'autoload_disabled':'1','PYTEST_PLUGINS':None,'PYTEST_ADDOPTS':None,'loaded_conftest_files':[]}, 'actual pytest plugins/conftest isolation differs')
    check(raw['oracle_runtime']['numpy']['version'] == '2.2.6' and raw['oracle_runtime']['pyscf']['version'].startswith('2.14.') and raw['gpu4pyscf_imported'] is False and all(row['file'] and row['version'] for row in raw['oracle_runtime'].values()), 'actual CPU oracle runtime differs')
    check(raw['device']['uuid'] == raw['device_after']['uuid'] == campaign['device']['uuid'] == campaign['device_after']['uuid'] and 'H100' in raw['device']['snapshot'], 'physical GPU changed')
    check(raw['plugin_sha256'] == campaign['scripts']['df-molecular-qz-plugin-v2.py'] and raw['actual_core']['sha256'] == build['artifacts']['libgenerativeqc.so'] and len(raw['actual_core']['paths']) == 1, 'actual plugin/core differs')
    check(raw['collected_nodeids'] == protocol['expected_nodeids'] and len(set(raw['collected_nodeids'])) == 44, '44 fixed tests differ')
    expected_modules = {Path(name).stem: digest for name, digest in protocol['source_files'].items() if name in protocol['test_modules']}
    check(len(raw['transforms']) == 4 and {r['module']:r['source_sha256'] for r in raw['transforms']} == expected_modules and all(r['all_other_ast_nodes_equal'] is True for r in raw['transforms']) and sorted(len(r['scheduler_assertion_lines']) for r in raw['transforms']) == [0,1,2,3], 'scientific source/scheduler adaptation differs')
    reports = {(r['nodeid'], r['when']): r for r in raw['reports']}
    check(len(reports) == len(raw['reports']) == 132 and all(reports[(node, phase)]['outcome'] == 'passed' for node in protocol['expected_nodeids'] for phase in ['setup','call','teardown']), 'missing, duplicate, skipped, failed or xfailed test phases')
    junit = ET.parse(out / 'junit.xml')
    cases = junit.findall('.//testcase')
    check(len(cases) == 44 and not junit.findall('.//failure') and not junit.findall('.//error') and not junit.findall('.//skipped'), 'JUnit does not retain44 passes')
    actual_nodes = set()
    grouped = {node:[] for node in protocol['expected_nodeids']}
    check([call['event_id'] for call in raw['executions']] == list(range(236)), 'execute identity/population differs')
    ef_count = 0
    for call in raw['executions']:
        check(call['nodeid'] in protocol['expected_nodeids'] and call['items'], 'undeclared/empty public execution')
        actual_nodes.add(call['nodeid'])
        grouped[call['nodeid']].append(call)
        contract = protocol['observed_call_contracts'][call['nodeid']]
        check([item['index'] for item in call['items']] == list(range(contract['items_per_call'])), 'ragged output identity/population differs')
        for item in call['items']:
            check(item['status'] == 0 and item['succeeded'] is True and item['converged'] is True and item['executed_backend'] == 'cuda', 'failed/unconverged/non-CUDA public item')
            check(math.isfinite(item['energy']) and item['forces'] is not None, 'missing/nonfinite public full E/F')
            forces = np.asarray(item['forces'], dtype=float)
            check(forces.ndim == 2 and forces.shape[1] == 3 and np.isfinite(forces).all(), 'invalid public forces')
            ef_count += 1
    check(actual_nodes == set(protocol['expected_nodeids']), 'test without actual public native execution')
    check(ef_count == 299 and all(len(grouped[node]) == contract['execute_calls'] for node,contract in protocol['observed_call_contracts'].items()), 'truncated or extra multi-stage public execution')
    event_map = {call['event_id']:call for call in raw['executions']}

    def bound_value(node, binding, expected, kind):
        check((binding is None) == (expected is None), 'missing or unexpected native operand binding')
        if expected is None:
            return None
        check(set(binding) == {'event_id','item_index','transform'}, 'unexpected operand binding schema')
        call = event_map[binding['event_id']]
        check(call['nodeid'] == node and grouped[node][expected['call_index']]['event_id'] == binding['event_id'] and binding['item_index'] == expected['item_index'] and binding['transform'] == expected['transform'], 'comparison uses wrong public call/item/transform')
        item = call['items'][binding['item_index']]
        value = item['energy'] if kind == 'energy' else np.asarray(item['forces'], dtype=float)
        if binding['transform'] == 'sum_axis0':
            check(kind == 'forces', 'invalid energy transform')
            value = value.sum(axis=0)
        else:
            check(binding['transform'] == 'identity', 'unknown transform')
        return value

    def exact(actual, bound):
        if bound is not None:
            a,b = np.asarray(actual),np.asarray(bound)
            check(a.shape == b.shape and np.array_equal(a,b), 'comparison operand differs from actual public E/F')

    for node,contract in protocol['observed_call_contracts'].items():
        force_rows = [row for row in raw['comparisons'] if row['nodeid'] == node]
        scalar_rows = [row for row in raw['scalar_comparisons'] if row['nodeid'] == node]
        check(len(force_rows) == len(contract['force_comparisons']) and len(scalar_rows) == len(contract['scalar_comparisons']), 'source-fixed comparison population differs')
        for row,rule in zip(force_rows,contract['force_comparisons'],strict=True):
            check(row['args'] == [] and row['kwargs'] == {'atol':rule['atol'],'rtol':rule['rtol']}, 'original force comparison tolerance changed')
            exact(row['actual'],bound_value(node,row['actual_binding'],rule['actual'],'forces'))
            exact(row['desired'],bound_value(node,row['desired_binding'],rule['desired'],'forces'))
        for row,rule in zip(scalar_rows,contract['scalar_comparisons'],strict=True):
            check(row['tolerance'] == rule['tolerance'], 'original energy tolerance changed')
            exact(row['actual'],bound_value(node,row['actual_binding'],rule['actual'],'energy'))
            exact(row['expected'],bound_value(node,row['expected_binding'],rule['expected'],'energy'))
    comparison_nodes, scalar_nodes = set(), set()
    for row in raw['comparisons']:
        check(row['passed'] is True and row['nodeid'] in actual_nodes, 'failed/undeclared allclose comparison')
        numeric_comparison(row); comparison_nodes.add(row['nodeid'])
    for row in raw['scalar_comparisons']:
        check(row['passed'] is True and row['nodeid'] in actual_nodes and all(math.isfinite(row[k]) for k in ['actual','expected','tolerance']) and row['tolerance'] >= 0 and abs(row['actual'] - row['expected']) <= row['tolerance'], 'failed scalar energy comparison')
        scalar_nodes.add(row['nodeid'])
    check(comparison_nodes == scalar_nodes == actual_nodes, 'every test requires independent replay of force and energy comparisons')
    check(len(raw['comparisons']) == 295 and len(raw['scalar_comparisons']) == 259, 'full comparison census differs')
    expected_traces = []
    for index,budget in enumerate([24,64]):
        node = f'test_dft_df_final_projection_cuda.py::test_projection_scratch_preserves_warm_and_moved_geometry[{budget}]'
        for policy_index,policy in enumerate(['off','auto']):
            for stage_index,stage in enumerate(['cold','warm','moved']):
                expected_traces.append({'nodeid':node,'call_index':3*policy_index+stage_index,'policy':policy,'stage':stage,'budget_mib':budget,'relative_path':f'pytest-tmp/test_projection_scratch_preser{index}/{budget}-{policy}-{stage}.jsonl','expected_completed_reuse':False if policy=='off' else (None if stage=='warm' else True)})
    check(protocol['trace_contracts'] == expected_traces and len(expected_traces) == 12, 'declared per-call trace contracts differ')
    trace_replays = []
    for contract in expected_traces:
        name = COHORT + '/' + contract['relative_path']
        check(name in manifest['files'], 'required per-call force trace absent from inventory')
        path = out / contract['relative_path']
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        response = [row for row in rows if row['operation'] == 'force_response']
        completed = [row for row in response if row['counters'].get('atom_coordinates') == 36]
        check(completed and all(row['valid'] is True and row['cuda_error'] == 0 for row in completed), 'per-call force response empty, invalid or failed')
        fields = ['response_reused_final_fitted_projection','response_final_fitted_projection_reused']
        check(all(type(row['counters'].get(field,0)) is int and row['counters'].get(field,0) >= 0 for row in response for field in fields), 'invalid per-call lease counter')
        admitted = sum(row['counters'].get(fields[0],0) for row in response)
        arithmetic = sum(row['counters'].get(fields[1],0) for row in response)
        complete_hits = sum(row['counters'].get(fields[1],0) for row in completed)
        expected = contract['expected_completed_reuse']
        check(admitted <= 1 and arithmetic <= 1, 'per-call projection lease consumed twice')
        if expected is not None:
            check(complete_hits == int(expected), 'per-call cold/moved/off projection reuse differs')
            if expected is False:
                check(admitted == arithmetic == 0, 'off call consumed a projection')
        call = grouped[contract['nodeid']][contract['call_index']]
        check(len(call['items']) == 1 and len(call['items'][0]['forces']) == 12, 'trace not bound to original12atom public call')
        trace_replays.append({**contract,'event_id':call['event_id'],'trace_sha256':sha(path),'admitted':admitted,'arithmetic':arithmetic,'completed_reuse':complete_hits})
    result={'status':'PASS_CURRENT_SOURCE44_DF_TESTS_FULL_PUBLIC_EF_AND_COMPARISON_REPLAY','source_revision':SOURCE,'source_tree':campaign['source_tree'],'campaign_sha256':sha(out/'campaign.json'),'manifest_sha256':sha(manifest_path),'protocol_sha256':sha(protocol_path),'build_receipt_sha256':sha(build_path),'tests':44,'public_full_ef':ef_count,'force_comparisons':len(raw['comparisons']),'energy_comparisons':len(raw['scalar_comparisons']),'per_call_projection_trace_replays':trace_replays,'scope':protocol['scope']}
    target=TASK/(COHORT+'-local-replay.json');check(not target.exists(),'inspect prior replay');target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
