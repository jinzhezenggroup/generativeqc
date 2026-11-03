"""Independently replay facade vectors,directions,real-call provenance and refusals."""
from pathlib import Path
import hashlib
import json
import lzma
import math

TASK = Path(__file__).resolve().parent
COHORT = 'facade-guards-v1'
SOURCE = 'cd2b4514be0a38c4920934481b1f02d157a694b4'


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def error(sample, target, atoms):
    for row in [sample, target]:
        check(type(row['energy']) in [int, float] and math.isfinite(row['energy']) and len(row['forces']) == atoms, 'invalid complete E/F')
        check(all(len(force) == 3 and all(type(value) in [int, float] and math.isfinite(value) for value in force) for force in row['forces']), 'invalid dense force vector')
    differences = [a - b for aa, bb in zip(sample['forces'], target['forces'], strict=True) for a, b in zip(aa, bb, strict=True)]
    return {'energy_abs_per_atom_hartree': abs(sample['energy'] - target['energy']) / atoms,
            'force_max_abs_hartree_per_bohr': max(map(abs, differences)),
            'force_rms_hartree_per_bohr': math.sqrt(math.fsum(value * value for value in differences) / (3 * atoms))}


def resolved_native_fock(native):
    """Use the native total or its explicit valid execution-owned census."""
    value = native['fock_builds']
    if value is not None:
        check(type(value) is int and value >= 0, 'invalid execution-owned native Fock total')
        return value
    if native['status'] != 0 or native['converged'] is not True:
        return None
    work = native['precision'] or {}
    names = ['mixed_stage_fock_builds', 'strict_stage_fock_builds', 'post_scf_fock_builds']
    if work.get('operator_work_counters_valid') is True and all(name in work for name in names):
        check(all(type(work[name]) is int and work[name] >= 0 for name in names), 'invalid execution-owned precision census')
        return sum(work[name] for name in names)
    return None


def native_stage_proof(samples, calls, arm, expected_count):
    """Cover every public item with a distinct successful real native return."""
    if len(samples) != expected_count or not samples:
        return {'status': 'FAIL_OR_INCOMPLETE', 'reason': 'missing public stage members', 'witness': []}
    first, last = samples[0]['native_call_range']
    check(all(sample['native_call_range'] == [first, last] for sample in samples), 'stage members have different native-call ranges')
    selected = calls[first:last]
    available = []
    for offset, call in enumerate(selected):
        if call['status'] != 'COMPLETE':
            continue
        for index, item in enumerate(call['returned_statuses']):
            if item['status'] == 0 and item['converged'] is True and item['executed_backend'] == 'cuda' and item['finite_EF_available'] is True:
                error(item, item, len(item['forces']))
                available.append((first + offset, index, item))
    def metadata_matches(sample, native_index, item):
        if arm != 'standard':
            return True
        fields=['iterations','fock_builds','warm_start_used','warm_start_fallback','precision']
        check(all(name in sample and name in item for name in fields),'Standard work/warm witness field missing')
        for name in ['iterations','fock_builds']:
            check(item[name] is None or type(item[name]) is int and item[name]>=0,'invalid known Standard execution work count')
        check(type(item['warm_start_used']) is bool and type(item['warm_start_fallback']) is bool,'invalid Standard warm witness flags')
        return sample['item']==native_index and all(sample[name]==item[name] for name in fields)
    edges = [[index for index, (_, native_index, item) in enumerate(available) if sample['energy'] == item['energy'] and sample['forces'] == item['forces'] and metadata_matches(sample,native_index,item)] for sample in samples]
    assigned = {}

    def augment(public_index, visited):
        for native_index in edges[public_index]:
            if native_index in visited:
                continue
            visited.add(native_index)
            if native_index not in assigned or augment(assigned[native_index], visited):
                assigned[native_index] = public_index
                return True
        return False

    covered = all(augment(index, set()) for index in range(len(samples)))
    witness = [{'public_item': samples[public_index]['item'], 'native_call': available[native_index][0], 'native_item': available[native_index][1]}
               for native_index, public_index in sorted(assigned.items(), key=lambda row: row[1])]
    inventory = bool(selected and selected[0]['system_count'] == expected_count and selected[0]['status'] == 'COMPLETE')
    if arm == 'standard':
        inventory &= len(selected) == 1 and len(selected[0]['returned_statuses']) == expected_count
    else:
        batch = samples[0]['batch_precision_validation']
        check(all(sample['batch_precision_validation'] == batch for sample in samples), 'Fast stage members disagree on fleet provenance')
        audit_calls = batch['audit_calls']
        inventory &= all(type(attempt['index']) is int and 0 <= attempt['index'] < expected_count for attempt in audit_calls)
        inventory &= sum(sample['precision_validation']['solve_count'] for sample in samples) == expected_count + len(audit_calls)
        inventory &= len(selected) == 1 + len(audit_calls)
        if selected and selected[0]['status'] == 'COMPLETE':
            candidate = selected[0]['returned_statuses']
            inventory &= batch['candidate_fock_builds'] == [item['fock_builds'] for item in candidate]
            inventory &= batch['candidate_precision'] == [item['precision'] for item in candidate]
            for sample in samples:
                solves = sample['precision_validation']['solves']
                audits = [(attempt, call) for attempt, call in zip(audit_calls, selected[1:]) if attempt['index'] == sample['item']]
                inventory &= sample['precision_validation']['solve_count'] == len(solves) == 1 + len(audits)
                actual = [candidate[sample['item']]] + [call['returned_statuses'][0] for _, call in audits if call['status'] == 'COMPLETE' and call['system_count'] == 1]
                inventory &= len(actual) == len(solves)
                for solve, native in zip(solves, actual):
                    inventory &= all(solve[name] == native[name] for name in ['iterations', 'executed_backend'])
                    inventory &= solve['fock_builds'] == resolved_native_fock(native)
                inventory &= bool(solves and solves[0]['role'] == 'candidate')
                for (attempt, _), solve in zip(audits, solves[1:]):
                    expected_role = 'fp64_fallback_verification' if solve['role'] == 'fp64_fallback_verification' else 'fp64_audit'
                    inventory &= solve['role'] in ['fp64_audit', 'fp64_fallback', 'fp64_fallback_verification'] and attempt['role'] == expected_role
                known = all(solve['fock_builds'] is not None for solve in solves)
                inventory &= sample['fock_builds'] == (sum(solve['fock_builds'] for solve in solves) if known else None)
            for attempt, call in zip(audit_calls, selected[1:]):
                inventory &= attempt.get('completed') is True and 'error' not in attempt
                inventory &= type(attempt.get('seconds')) in [int, float] and math.isfinite(attempt['seconds']) and attempt['seconds'] >= 0
                inventory &= call['status'] == 'COMPLETE' and call['system_count'] == 1 and len(call['returned_statuses']) == 1
                if call['status'] == 'COMPLETE' and call['system_count'] == 1:
                    native = call['returned_statuses'][0]
                    inventory &= all(attempt.get(name) == native[name] for name in ['status', 'fock_builds', 'iterations', 'warm_start_used'])
    return {'status': 'PASS' if covered and inventory else 'FAIL_OR_INCOMPLETE', 'one_to_one_public_coverage': covered,
            'candidate_audit_inventory_matches': inventory, 'successful_native_returns': len(available), 'witness': witness,
            'scope': 'Matching is per stage; unused successful candidate/audit returns and failed proposals remain visible. Unknown work counts remain null.'}


def emit(receipt):
    path = TASK / (COHORT + '-local-replay.json')
    check(not path.exists(), 'inspect prior facade replay')
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key not in ['rows', 'references']}))


import importlib.util
spec=importlib.util.spec_from_file_location('cohort_replay',TASK/'cohort-replay-atomic-v1.py');common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
def main():
    scripts=['run-facade-guards-v5.py','run-facade-guards-v5-job.sh','facade-guard-endpoint-v1.py','facade-large-reference-endpoint-v1.py','facade-reference-runtime-v1.py','oracle_replication6_solver_v2.py','facade-guards-protocol-v1.json','facade-large-reference-protocol-v1.json','cohort-gates-v4.py','cohort-gates-v3.py','cohort-gates.py','process-ownership-v1.py','run-screening-v3.py','pilot.py']
    campaign,manifest,build,policy=common.authenticate(TASK,COHORT,'gqc-h100-maincd2b-facade-1102-v1',scripts,'TERMINAL_THREE_REFERENCE_EF_AND_EIGHTEEN_FACADE_ATTEMPTS_INDEPENDENT_REPLAY_PENDING')
    out=TASK/COHORT;manifest_path=TASK/(COHORT+'-download-manifest.json');build_path=TASK/'production-build-v1/build-receipt.json'
    protocol_path=TASK/'facade-guards-protocol-v1.json';reference_protocol_path=TASK/'facade-large-reference-protocol-v1.json'
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
    rows, total, refusals, native_requests, native_successes, native_failed_items, directional_count = [], 0, 0, 0, 0, 0, 0
    check(len(campaign['runs']) <= len(protocol['populations']), 'extra facade attempts')
    for population, stage in zip(protocol['populations'], campaign['runs']):
        stem = population['id']
        plan_path, raw_path = out / (stem + '.plan.json'), out / (stem + '.json')
        plan = read(plan_path)
        check(stage['case'] == stem and stage['plan_sha256'] == sha(plan_path) and plan['population'] == population and plan['protocol_sha256'] == sha(protocol_path) and plan['source_revision'] == SOURCE and plan['build_artifacts'] == build['artifacts'] and plan['execution_policy'] == selection['selected_policy'], 'native population/plan differs')
        check(Path(plan['library_root'])==root and Path(plan['source_root'])==root.parent/'source','facade selected another source,core or fixture')
        auxiliary=Path(plan['auxiliary_basis']);relative=auxiliary.relative_to(root.parent.parent/'20261030-h100-production-15ab-2241')
        family=population['family'];mode=population['mode'];case0=next(case for case in plan['cases'] if case['id']==population['owners'][0]['case_ids'][0])
        refined=family=='uks-pbe-water3-cation' and mode=='df'
        expected_aux=fit['new_auxiliary_prior_relative'] if refined else 'inputs-default/'+case0['method']+'-auxiliary.json'
        expected_sha=fit['new_auxiliary_sha256'] if refined else defaults[auxiliary.name]['sha256']
        check(relative.as_posix()==expected_aux and plan['auxiliary_basis_sha256']==expected_sha==sha(refs/relative),'facade actual fitting input differs')
        expected = [(step['stage'], index, name) for step in population['stages'] for index, name in enumerate(step['case_ids'])]
        row = {'case': stem, 'returncode': stage.get('returncode'), 'status': 'FAIL_OR_INCOMPLETE', 'numerical': [], 'directional': []}
        if not raw_path.exists():
            all_qualified = False; rows.append(row); continue
        raw = read(raw_path)
        check(('receipt_sha256' not in stage or sha(raw_path) == stage['receipt_sha256']) and raw['source_revision'] == SOURCE and raw['protocol_sha256'] == sha(protocol_path) and raw['plan_sha256'] == sha(plan_path) and raw['endpoint_sha256'] == plan['endpoint_sha256'] == sha(TASK / 'facade-guard-endpoint-v1.py') and raw['population'] == population, 'raw facade bytes/source differ')
        check(raw['execution_policy'] == selection['selected_policy'] and raw['qz_allocation'] == plan['qz_allocation'] and raw['qz_allocation']['job_name'] == campaign['job'] and raw['accelerator']['nvidia_smi']['uuid'] == campaign['device']['uuid'] and raw['accelerator']['compute_capability'] == [9, 0] and raw['environment']['packages'] == {'cupy': '13.6.0', 'gpu4pyscf': '1.8.1', 'numpy': '2.2.6', 'pyscf': '2.14.0'}, 'actual native runtime/device differs')
        observed = [(sample['stage'], sample['item'], sample['oracle_case']) for sample in raw['samples']]
        check(observed == expected[:len(observed)], 'public E/F order missing,duplicated or changed')
        qualified = stage.get('returncode') == 0 and stage.get('receipt_sha256') == sha(raw_path) and raw['status'] == 'COMPLETE_PUBLIC_FACADE_EF_AND_LIFETIME_INDEPENDENT_REPLAY_PENDING' and observed == expected
        for call in raw['native_calls']:
            count = call['system_count']
            check(type(count) is int and count > 0 and call['requested_properties'] == ['energy', 'forces'] and call['eligible_before'] == [True] * count, 'real native force admission missing')
            native_requests += count
            check(len(call['transport_before']) == count, 'pre-call transport snapshot lost item order')
            if call['status'] == 'COMPLETE':
                check(call['eligible_after'] == [True] * count and call['returned_items'] == len(call['returned_statuses']) == len(call['transport_after']) == count and [item['index'] for item in call['returned_statuses']] == list(range(count)), 'actual native return metadata differs')
                for item in call['returned_statuses']:
                    check(type(item['warm_start_used']) is bool and type(item['warm_start_fallback']) is bool, 'underlying native warm-start provenance missing')
                    successful = item['status'] == 0 and item['converged'] is True
                    native_successes += successful; native_failed_items += not successful
            else:
                qualified = False
        for sample in raw['samples']:
            case = lookup[sample['oracle_case']]
            check(sample['geometry_bohr'] == case['geometry_bohr'] and sample['charge'] == case['charge'] and sample['spin'] == case['spin'] and sample['actual_core_sha256'] == build['artifacts']['libgenerativeqc.so'], 'returned model/geometry/core differs')
            first, last = sample['native_call_range']
            check(type(first) is int and type(last) is int and 0 <= first < last <= len(raw['native_calls']) and all(call['phase'] == sample['stage'] and call['status'] == 'COMPLETE' for call in raw['native_calls'][first:last]), 'public result lacks real native-call provenance')
            if population['arm'] == 'fast':
                validation, batch_validation = sample['precision_validation'], sample['batch_precision_validation']
                check(validation and validation['profile'] == 'fast' and validation['schema_version'] == 2 and validation['timing_scope'] == 'shared_batch_endpoint' and validation['endpoint_seconds'] is None and validation['solve_count'] == len(validation['solves']) and batch_validation and batch_validation['schema_version'] == 1 and batch_validation['profile'] == 'fast' and batch_validation['candidate_calls'] == 1, 'actual Fast precision/audit provenance missing')
            if case['id'] in targets:
                differences = error(sample, targets[case['id']], case['atoms'])
                passed = all(value <= protocol['common_observable_gates'][name] for name, value in differences.items())
                if case['id'] == 'rks-pbe0-water96-base':
                    passed &= all(value <= protocol['common_observable_gates'][name] for name, value in error(sample, old_base, 96).items())
            else:
                differences, passed = None, False
            qualified &= passed
            row['numerical'].append({'stage': sample['stage'], 'item': sample['item'], 'oracle_case': case['id'], 'status': 'PASS' if passed else 'FAIL_OR_UNQUALIFIED_REFERENCE', 'errors': differences, 'warm_start_used': sample['warm_start_used'], 'warm_start_fallback': sample['warm_start_fallback']})
            total += 1
        row['native_stage_proofs'] = []
        for declaration in population['stages']:
            members = [sample for sample in raw['samples'] if sample['stage'] == declaration['stage']]
            proof = native_stage_proof(members, raw['native_calls'], population['arm'], len(declaration['case_ids']))
            row['native_stage_proofs'].append({'stage': declaration['stage'], **proof})
            qualified &= proof['status'] == 'PASS'
        if qualified:
            required_controls = [{'owner': index, 'name': name, 'exception': 'RuntimeError'} for index in range(len(population['owners'])) for name in ['closed-owner-execute', 'closed-owner-clear-warm-starts', 'closed-owner-set-warm-updates']]
            check(raw['negative_controls'] == required_controls and raw['detached_result_count'] == len(expected), 'actual refusal or detached-result checks differ')
            refusals += len(required_controls)
            if population['kind'] == 'small-fast' and population['family'] in protocol['directions']:
                samples = {sample['stage']: sample for sample in raw['samples'] if sample['item'] == 0}
                direction = protocol['directions'][population['family']]
                check(abs(math.fsum(value * value for vector in direction for value in vector) - 1) < 1e-14, 'direction is not normalized')
                analytic = -math.fsum(force * component for ff, dd in zip(samples['base-cold']['forces'], direction, strict=True) for force, component in zip(ff, dd, strict=True))
                for step in [0.001, 0.0005]:
                    central = (samples[f'direction-{step:g}-+1']['energy'] - samples[f'direction-{step:g}--1']['energy']) / (2 * step)
                    absolute = abs(central - analytic)
                    passed = math.isfinite(absolute) and absolute <= protocol['directional_force_gate_hartree_per_bohr']
                    qualified &= passed; directional_count += 1
                    row['directional'].append({'step_bohr': step, 'analytic_gradient': analytic, 'central_energy_derivative': central, 'absolute_error': absolute, 'status': 'PASS' if passed else 'FAIL'})
        row.update(status='PASS' if qualified else 'FAIL_OR_INCOMPLETE', returned_public_EF=len(raw['samples']), error=raw.get('error'))
        all_qualified &= qualified; rows.append(row)
    all_qualified &= len(rows) == 18 and total == 142 and refusals == 72 and directional_count == 8
    emit({'status': 'PASS_PUBLIC_FAST_LARGE_MOVED_AND_RAGGED_18_POPULATIONS_142_PUBLIC_EF_72_REFUSALS_8_DIRECTIONS' if all_qualified else 'FACADE_GUARDS_FAILED_OR_INCOMPLETE_ALL_ORIGINALS_RETAINED',
          'source_revision': SOURCE, 'campaign_sha256': sha(out / 'campaign.json'), 'manifest_sha256': sha(manifest_path), 'protocol_sha256': sha(protocol_path), 'build_receipt_sha256': sha(build_path), 'screening_preregistration_sha256':sha(TASK/'screening-protocol-v3.json'),
          'qualified_populations': sum(row['status'] == 'PASS' for row in rows), 'attempted_populations': len(rows), 'public_returned_EF': total, 'verified_refusal_controls': refusals, 'directional_controls': directional_count,
          'observed_native_item_requests': native_requests, 'observed_successful_native_items': native_successes, 'observed_failed_native_proposals_or_audits': native_failed_items,
          'terminal_parent_accepted': terminal_parent_accepted, 'parent_error': campaign.get('error'),
          'references': references, 'rows': rows, 'scope': protocol['scope']})


def guarded_main():
    result=TASK/(COHORT+'-local-replay.json')
    check(not result.exists(),'inspect existing facade replay before retry')
    try:main()
    except BaseException as error:
        if not result.exists():emit({'status':'FAIL_FACADE_INDEPENDENT_REPLAY_ALL_ORIGINALS_RETAINED','source_revision':SOURCE,'reader':Path(__file__).name,'reader_sha256':sha(Path(__file__)),'error':type(error).__name__+': '+str(error),'scope':'No scientific or lifetime acceptance; all originals retained.'})
        raise
if __name__=='__main__':guarded_main()
