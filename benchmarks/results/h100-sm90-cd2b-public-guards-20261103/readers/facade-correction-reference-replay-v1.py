"""Unchanged independent dense reference calibration for corrected facade replay.

Reference math/controls copied from replay-facade-guards-v5.py; this module runs
CPU arithmetic on already measured data and never executes an electronic solver.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import lzma
TASK=Path(__file__).resolve().parent
COHORT='facade-guards-v1'
SOURCE='cd2b4514be0a38c4920934481b1f02d157a694b4'
def load(name):
 spec=importlib.util.spec_from_file_location(name,TASK/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
legacy=load('replay-facade-guards-v5');common=load('cohort-replay-atomic-v1')
check,read,sha,error=legacy.check,legacy.read,legacy.sha,legacy.error

def reference_targets(campaign,manifest,build,policy):
    protocol_path=TASK/'facade-guards-protocol-v1.json'
    reference_protocol_path=TASK/'facade-large-reference-protocol-v1.json'
    out=TASK/COHORT
    protocol,reference_protocol=read(protocol_path),read(reference_protocol_path);refs=TASK/(COHORT+'-reference-inputs')
    check(protocol['source_revision']==reference_protocol['source_revision']==SOURCE and campaign['protocol_sha256']==sha(protocol_path) and campaign['reference_protocol_sha256']==sha(reference_protocol_path)==protocol['large_reference_protocol_sha256'],'protocol/source differs')
    check(protocol['expected_populations']==18 and protocol['expected_complete_public_EF']==142 and protocol['expected_closed_owner_controls']==72 and campaign['execution_policy']==policy and protocol['matrix_preregistration_sha256']==sha(TASK/'screening-protocol-v3.json'),'protocol population/policy differs')
    geometry=read(refs/'native-guard-geometries-v1.json');check(sha(refs/'native-guard-geometries-v1.json')==protocol['geometry_source_sha256'] and protocol['common_observable_gates']==geometry['common_observable_gates'] and protocol['directions']==geometry['directions'],'frozen geometry/directional/observable gates differ')
    check(protocol['native_admission_sha256']==sha(TASK/'native-qualification-v1-local-replay-v2.json') and protocol['explicit_uks_fit_qualification_sha256']==sha(TASK/'uks-fit-v1-local-replay.json'),'current native/fit protocol differs')
    check(campaign['shared_memory_bytes']==16*(1<<30),'actual shared memory differs')
    for name,digest in reference_protocol['source_files'].items():
        path=refs/'source-helper'/name;check(sha(path)==digest and COHORT+'-reference-inputs/source-helper/'+name in manifest['files'],'actual independent reference helper differs')
    check(sha(TASK/'oracle_replication6_solver_v2.py')==reference_protocol['solver_artifact']['sha256'],'explicit independent solver differs')
    runtime=read(out/'reference-runtime.json');check(runtime['status']=='PASS_REFERENCE_RUNTIME_AND_CUTENSOR_ONLY' and sha(out/'reference-runtime.json')==campaign['reference_runtime_sha256'] and runtime['script_sha256']==campaign['scripts']['facade-reference-runtime-v1.py'] and campaign['device']['uuid'] in runtime['device'] and runtime['installed_files_equal_original_wheels']>0 and runtime['max_abs_error']<1e-12,'actual reference runtime differs')
    budget=read(TASK/'public-budget-v9-local-replay.json');check(campaign['admissions']['public-budget-v9-local-replay.json']==sha(TASK/'public-budget-v9-local-replay.json') and budget['source_revision']==SOURCE and budget['status']=='PASS_SMALL_DIRECT_PUBLIC_BUDGET_12_EF_6_CONTROLS_AND_FIVE_TYPED_PLANNING_REFUSALS','supported public budget/refusal proof differs')
    root=Path(build['stages'][0]['command'][build['stages'][0]['command'].index('-B')+1]);selection={'selected_policy':policy}
    fit=read(TASK/'uks-fit-protocol-v1.json');defaults=read(refs/'inputs-default.json')['files']
    oracle_path=refs/'data/oracles.json.xz';check(sha(oracle_path)==protocol['original_large_oracle_sha256'],'independent old large oracle differs')
    old_large=next(row for row in json.loads(lzma.decompress(oracle_path.read_bytes())) if (row['method'],row['atoms'],row['backend'])==('pbe0',96,'cuda'));old_base=next(sample for sample in old_large['samples'] if sample['level']==5)
    terminal_parent_accepted = campaign['status'] == 'TERMINAL_THREE_REFERENCE_EF_AND_EIGHTEEN_FACADE_ATTEMPTS_INDEPENDENT_REPLAY_PENDING' and 'error' not in campaign
    references, targets, all_qualified = [], {}, terminal_parent_accepted
    check(len(campaign['reference_runs']) <= 2, 'extra reference attempts')
    for case, stage in zip(reference_protocol['cases'], campaign['reference_runs']):
        stem = 'reference-' + case['id']
        plan_path, raw_path = out / (stem + '.plan.json'), out / (stem + '.json')
        plan = read(plan_path)
        check(stage['case'] == case['id'] and stage['plan_sha256'] == sha(plan_path) and plan['case'] == case and plan['protocol_sha256'] == sha(reference_protocol_path) and plan['source_revision'] == SOURCE and plan['source_files'] == reference_protocol['source_files'] and plan['solver_sha256'] == reference_protocol['solver_artifact']['sha256'], 'reference plan identity differs')
        row = {'case': case['id'], 'returncode': stage.get('returncode'), 'status': 'FAIL_OR_INCOMPLETE'}
        if raw_path.exists():
            raw = read(raw_path)
            check(('receipt_sha256' not in stage or sha(raw_path) == stage['receipt_sha256']) and raw['case'] == case and raw['plan_sha256'] == sha(plan_path) and raw['source_revision'] == SOURCE and raw['runner_sha256'] == sha(TASK / 'facade-large-reference-endpoint-v1.py'), 'reference raw bytes/source differ')
            complete = stage.get('returncode') == 0 and stage.get('receipt_sha256') == sha(raw_path) and raw['status'] == 'COMPLETE_INDEPENDENT_DECLARED_REFERENCE_EF_REPLAY_PENDING' and [sample['level'] for sample in raw['samples']] == case['levels']
            check(raw['environment']['packages'] == reference_protocol['reference_packages'] and raw['accelerator']['nvidia_smi']['uuid'] == campaign['device']['uuid'] and raw['accelerator']['compute_capability'] == [9, 0], 'reference actual runtime/device differs')
            check(raw['native_execution_blocked_entrypoints'] == ['generativeqc._native.load_library', 'Calculator.__init__', 'PrecisionCalculator.__init__'], 'reference isolation instrumentation differs')
            complete &= raw.get('native_library_mappings_before') == raw.get('native_library_mappings_after') == raw.get('native_blocked_calls') == []
            for sample in raw['samples']:
                check(sample['controls'] == {'energy': 1e-11, 'force_grid_response': True, 'full_density_fock': True, 'orbital_gradient': 1e-7, 'screening': 1e-14} and sample['converged'] is True, 'reference scientific settings differ')
                error(sample, sample, 96)
            if complete:
                if case['id'].endswith('base-calibration'):
                    check(case['geometry_bohr'] == old_large['geometry_bohr'] and case['old_driver_sha256'] == old_large['driver_sha256'], 'base calibration physical identity differs')
                    differences = error(raw['samples'][0], old_base, 96)
                    target_id, target = 'rks-pbe0-water96-base', raw['samples'][0]
                else:
                    differences = error(raw['samples'][0], raw['samples'][1], 96)
                    target_id, target = 'rks-pbe0-water96-moved', raw['samples'][1]
                complete &= all(value <= protocol['common_observable_gates'][name] * 0.25 for name, value in differences.items())
                row.update(errors=differences, status='PASS' if complete else 'FAIL')
                if complete:
                    targets[target_id] = target
            all_qualified &= complete
        else:
            all_qualified = False
        references.append(row)
    all_qualified &= len(references) == 2 and len(targets) == 2
    admission = read(refs / 'guard-cpu-oracles-local-replay.json')
    cpu_root = refs / 'guard-cpu-oracles-v1'
    cpu_campaign = read(cpu_root / 'campaign.json')
    check(sha(refs / 'guard-cpu-oracles-local-replay.json') == protocol['cpu_guard_admission_sha256'] and admission['status'] == 'PASS_28_CPU_EF_CANDIDATES_DUAL_GRID_AND_8_DIRECTIONAL_GATES' and admission['campaign_sha256'] == sha(cpu_root / 'campaign.json'), 'small CPU admission differs')
    lookup = {case['id']: case for case in protocol['cases']}
    for case in protocol['cases']:
        if case['atoms'] == 96:
            continue
        path = cpu_root / (case['id'] + '-grid4.json')
        target = read(path)
        binding = next(row for row in cpu_campaign['runs'] if row['case'] == case['id'] and row['grid_level'] == 4)
        check(sha(path) == binding['receipt_sha256'] and target['case'] == case and target['status'] == 'complete' and target['converged'] is True, 'small complete CPU target differs')
        targets[case['id']] = target
    check(all_qualified,'independent reference calibration failed')
    return protocol,refs,targets,lookup,old_base,root,fit,defaults,references
