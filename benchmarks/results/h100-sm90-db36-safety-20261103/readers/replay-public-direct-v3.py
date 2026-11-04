"""Replay new-source public E/F and original independent comparison operands."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1] / 'generativeqc-h100-current-evidence'
COHORT = 'public-direct-v1'
REVISION = 'db3657f08d983791c45b7d949cb9028fe0055e3f'


def check(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_population(raw, kind, protocol):
    nodes = protocol['expected_nodeids'][kind]
    contracts = protocol['comparison_contracts'][kind]
    executions = raw['executions']
    check(all(type(row['event_id']) is int for row in executions) and [row['event_id'] for row in executions] == list(range(protocol['expected_execute_calls'][kind])), 'Public event population differs')
    grouped = {node: [] for node in nodes}
    for row in executions:
        check(row['nodeid'] in grouped, 'Execution from unknown node')
        grouped[row['nodeid']].append(row)
    for field in ['comparisons', 'scalar_comparisons', 'array_equal_comparisons']:
        check(all(row['nodeid'] in grouped for row in raw[field]), 'Comparison from unknown node')
    ef = rejected = independent = 0
    for node in nodes:
        contract, calls = contracts[node], grouped[node]
        check(len(calls) == len(contract['item_atoms']), 'Per-node call population differs')
        for index, call in enumerate(calls):
            atoms = contract['item_atoms'][index]
            check(len(call['items']) == len(atoms) and all(type(item['index']) is int for item in call['items']) and [item['index'] for item in call['items']] == list(range(len(atoms))), 'Per-call item population differs')
            for item_index, (item, nat) in enumerate(zip(call['items'], atoms, strict=True)):
                success = contract['successful_items'][index][item_index]
                check(item['succeeded'] is success and type(item['warm_start_used']) is bool, 'Item success/warm schema differs')
                if not success:
                    failed_energy = item['energy']
                    retained_energy = type(failed_energy) in (float, int) and math.isfinite(failed_energy)
                    if isinstance(failed_energy, dict) and set(failed_energy) == {'nonfinite_float_hex'} and isinstance(failed_energy['nonfinite_float_hex'], str):
                        retained_energy = not math.isfinite(float.fromhex(failed_energy['nonfinite_float_hex']))
                    check(item['status'] != 0 and item['converged'] is False and item['forces'] is None and retained_energy, 'Unexpected ragged failure output')
                    rejected += 1
                    continue
                check(item['status'] == 0 and item['converged'] is True and item['executed_backend'] == 'cuda' and isinstance(item['energy'], (int,float)) and math.isfinite(item['energy']), 'Invalid/non-CUDA public result')
                require_force = contract['require_force'][index][item_index]
                check((item['forces'] is not None) == require_force, 'Per-call force presence differs')
                if require_force:
                    force = np.asarray(item['forces'], dtype=float)
                    check(force.shape == (nat, 3) and np.isfinite(force).all(), 'Truncated/nonfinite full atom vector')
                    ef += 1
        for call_index, item_index in contract['warm_items']:
            check(calls[call_index]['items'][item_index]['warm_start_used'] is True, 'Original warm-state gate failed')
        if 'moved_energy_change' in contract:
            gate = contract['moved_energy_change']
            check(abs(calls[gate['to_call']]['items'][gate['item']]['energy'] - calls[gate['from_call']]['items'][gate['item']]['energy']) > gate['minimum_abs_hartree'], 'Original moved-energy gate failed')
        def bound(binding, expected, field):
            if expected is None:
                check(binding is None, 'Unexpected binding in independent operand')
                return None
            check(isinstance(binding, dict) and set(binding) == {'event_id','item_index','transform'} and type(binding['event_id']) is int and type(binding['item_index']) is int and binding['item_index'] >= 0, 'Invalid/negative operand binding')
            call = calls[expected['call_index']]
            canonical = {'event_id':call['event_id'],'item_index':expected['item_index'],'transform':expected['transform']}
            check(binding == canonical and binding['item_index'] < len(call['items']), 'Cross-node or wrong source-call operand binding')
            value = call['items'][binding['item_index']][field]
            if expected['transform'] == 'sum_axis0':
                check(field == 'forces', 'Invalid energy transform')
                value = np.asarray(value).sum(axis=0)
            else:
                check(expected['transform'] == 'identity', 'Unknown transform')
            return value
        scalars = [row for row in raw['scalar_comparisons'] if row['nodeid'] == node]
        check(len(scalars) == len(contract['scalar']), 'Missing/extra original energy comparison')
        for row, expected in zip(scalars, contract['scalar'], strict=True):
            check(row['passed'] is True and row['tolerance'] == expected['tolerance'] and all(math.isfinite(row[key]) for key in ['actual','expected','tolerance']) and abs(row['actual']-row['expected']) <= expected['tolerance'], 'Original energy tolerance/gate differs')
            for key in ['actual','expected']:
                value = bound(row[key+'_binding'], expected[key+'_binding'], 'energy')
                if value is not None:
                    check(row[key] == value, 'Energy comparison differs from real public item')
        arrays = [row for row in raw['comparisons'] if row['nodeid'] == node]
        check(len(arrays) == len(contract['arrays']), 'Missing/extra force/AO/finalD/W comparison')
        for row, expected in zip(arrays, contract['arrays'], strict=True):
            a,b = np.asarray(row['actual'],dtype=float), np.asarray(row['desired'],dtype=float)
            check(list(a.shape) == expected['actual_shape'] and list(b.shape) == expected['desired_shape'] and np.isfinite(a).all() and np.isfinite(b).all(), 'Original force/AO/D/W operand shape differs')
            check(row['passed'] is True and row['args'] == expected['args'] and row['kwargs'] == expected['kwargs'], 'Original array gate/tolerance changed')
            np.testing.assert_allclose(a,b,*expected['args'],**expected['kwargs'])
            for value,key in [(a,'actual_binding'),(b,'desired_binding')]:
                target = bound(row[key], expected[key], 'forces')
                if target is not None:
                    check(np.array_equal(value,np.asarray(target)), 'Force comparison differs from real full public output')
        equals = [row for row in raw['array_equal_comparisons'] if row['nodeid'] == node]
        check(len(equals) == len(contract['equal_arrays']), 'Missing/extra independent grid equality')
        for row, shape in zip(equals, contract['equal_arrays'], strict=True):
            a,b = np.asarray(row['actual'],dtype=float), np.asarray(row['desired'],dtype=float)
            check(row['passed'] is True and row['args'] == [] and row['kwargs'] == {} and list(a.shape) == list(b.shape) == shape and np.isfinite(a).all() and np.isfinite(b).all() and np.array_equal(a,b), 'Independent full-grid equality differs')
        if 'independent_grid_and_final_state' in node:
            independent += 1
    check(ef == protocol['expected_full_EF'][kind] and rejected == (3 if kind == 'production' else 0), 'Full E/F or rollback population differs')
    check(independent == (4 if kind == 'production' else 0), 'Independent case population differs')
    return {'tests':len(nodes),'public_executions':len(executions),'full_EF':ef,'rejected_ragged_items':rejected,'array_comparisons':len(raw['comparisons']),'scalar_comparisons':len(raw['scalar_comparisons']),'grid_equalities':len(raw['array_equal_comparisons']),'independent_grid_cases':independent}


def main():
    out = TASK / COHORT
    manifest = read(TASK / (COHORT + '-download-manifest.json'))
    check(manifest['platform_status'] == 'SUCCEEDED' and manifest['recorded_provenance_errors'] == [], 'Unsuccessful/unbound terminal archive')
    check(all((TASK / name).stat().st_size == row['bytes'] and sha(TASK / name) == row['sha256'] for name, row in manifest['files'].items()), 'Original bytes differ')
    campaign = read(out / 'campaign.json')
    protocol = read(TASK / 'public-direct-protocol-v2.json')
    check(sha(out / 'campaign.json') == manifest['campaign_sha256'] and campaign['status'] == 'TERMINAL_AFFECTED_PUBLIC_DIRECT_TESTS_RECEIVER_PENDING', 'Parent incomplete')
    check(campaign['source_revision'] == protocol['source_revision'] == REVISION and campaign['source_tree'] == protocol['source_tree'] == subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', REVISION + '^{tree}'], text=True).strip(), 'Frozen source differs')
    check(campaign['protocol_sha256'] == sha(TASK / 'public-direct-protocol-v2.json'), 'Protocol differs')
    check(campaign['allocation_cgroups'] == {'cpu.max': '2000000 100000', 'memory.max': '214748364800'} and campaign['device']['uuid'] == campaign['device_after']['uuid'] and 'H100' in campaign['device']['snapshot'], 'Allocation/device differs')
    build = read(TASK / 'production-build-v1/build-receipt.json')
    native = read(TASK / 'native-qualification-v1/campaign.json')
    check(campaign['build_receipt_sha256'] == sha(TASK / 'production-build-v1/build-receipt.json') and campaign['native_campaign_sha256'] == sha(TASK / 'native-qualification-v1/campaign.json') and campaign['artifacts'] == {'production': build['artifacts'], 'fixture': native['fixture_artifacts']}, 'Package binding differs')
    for name, digest in campaign['admissions'].items():
        check(sha(TASK / name) == digest == sha(TASK / (COHORT + '-available-admissions') / name), 'Consumed admission differs')
    for name, digest in campaign['scripts'].items():
        check(sha(TASK / name) == digest == sha(TASK / (COHORT + '-executed-producers') / name), 'Executed producer differs')
    for name, digest in protocol['source_files'].items():
        original = out / 'executed-source' / name
        check(sha(original) == digest and original.read_bytes() == subprocess.check_output(['git', '-C', str(ROOT), 'show', REVISION + ':' + name]), 'Scientific source differs')
    check([row['kind'] for row in campaign['runs']] == ['production', 'fixture'] and all(row['returncode'] == 0 for row in campaign['runs']), 'Missing or failed public process')
    summaries = {}
    for stage in campaign['runs']:
        kind = stage['kind']
        raw = read(out / (kind + '-observations.json'))
        nodes = protocol['expected_nodeids'][kind]
        check(raw['status'] == 'PYTEST_TERMINAL_REPLAY_PENDING' and raw['exitstatus'] == 0 and raw['collection_only'] is False and raw['collected_nodeids'] == nodes, 'Actual test collection/execution differs')
        check(raw['pytest_isolation'] == {'noconftest': True, 'autoload_disabled': '1', 'PYTEST_PLUGINS': None, 'PYTEST_ADDOPTS': None, 'loaded_conftest_files': []}, 'Pytest isolation differs')
        check(raw['device']['uuid'] == raw['device_after']['uuid'] == campaign['device']['uuid'] and raw['actual_core']['sha256'] == campaign['artifacts'][kind]['libgenerativeqc.so'] == stage['core_sha256'] and len(raw['actual_core']['paths']) == 1 and raw['gpu4pyscf_imported'] is False, 'Actual core/oracle/device differs')
        check(raw['oracle_runtime']['numpy']['version'] == '2.2.6' and raw['oracle_runtime']['pyscf']['version'].startswith('2.14.'), 'Independent oracle runtime differs')
        check(raw['plugin_sha256'] == campaign['scripts']['public-direct-qz-plugin-v2.py'] and len(raw['transforms']) == 1 and raw['transforms'][0]['scheduler_assertion_lines'] == [] and raw['transforms'][0]['all_other_ast_nodes_equal'] is True and raw['transforms'][0]['source_sha256'] == protocol['source_files']['tests/python/test_public_native_cuda_forces.py'], 'Scientific test AST changed')
        reports = {(row['nodeid'], row['when']): row for row in raw['reports']}
        check(len(reports) == len(raw['reports']) == len(nodes) * 3 and all(reports[(node, phase)]['outcome'] == 'passed' for node in nodes for phase in ['setup', 'call', 'teardown']), 'Missing/failed/skipped test phases')
        xml = ET.parse(out / (kind + '.xml'))
        check(len(xml.findall('.//testcase')) == len(nodes) and not any(xml.findall('.//' + tag) for tag in ['failure', 'error', 'skipped']), 'JUnit differs')
        summaries[kind] = validate_population(raw, kind, protocol)
    result = {'status': 'PASS_CURRENT_SOURCE_AFFECTED_PUBLIC_DIRECT_EF_AND_FAULT_TESTS', 'source_revision': REVISION, 'campaign_sha256': sha(out / 'campaign.json'), 'summaries': summaries, 'scope': protocol['scope'] + ' Public fault assertions are source-owned executed tests; this reader recomputes all captured scalar/array operands and authenticates original phase reports. No timing votes.'}
    with (TASK / (COHORT + '-local-replay-v3.json')).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
