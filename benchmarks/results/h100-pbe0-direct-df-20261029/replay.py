"""Offline retained PBE0 evidence replay. Standard library; no chemistry execution."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import struct
import tarfile


ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(data):
    def reject(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(data, parse_constant=reject)


def metrics(sample, reference, atoms):
    delta = [a-b for row, other in zip(sample['forces'], reference['forces'], strict=True)
             for a, b in zip(row, other, strict=True)]
    assert len(delta) == atoms * 3
    return {'energy_abs_per_atom_hartree': abs(sample['energy']-reference['energy'])/atoms,
            'force_max_abs_hartree_per_bohr': max(map(abs, delta)),
            'force_rms_hartree_per_bohr': math.sqrt(math.fsum(x*x for x in delta)/len(delta))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-root", type=Path, default=ROOT, help="verified restored historical payload directory")
    payload = parser.parse_args().payload_root
    manifest = load((ROOT/'manifest.json').read_bytes())
    for name, row in manifest['files'].items():
        data = ((payload if name == 'evidence.tar.xz' else ROOT)/name).read_bytes()
        assert len(data) == row['bytes'] and digest(data) == row['sha256'], name
    members = {}
    with tarfile.open(fileobj=io.BytesIO((payload/'evidence.tar.xz').read_bytes()), mode='r:xz') as archive:
        for entry in archive:
            name = entry.name
            path = PurePosixPath(name)
            assert entry.isfile() and not path.is_absolute() and '..' not in path.parts
            assert name not in members
            members[name] = archive.extractfile(entry).read()
    witness_data = members.pop('witnesses.json')
    assert digest(witness_data) == manifest['witnesses_sha256']
    witnesses = load(witness_data)
    if 'restore.json' in members:
        restore_data = members.pop('restore.json')
        assert digest(restore_data) == manifest['restore_sha256']
        recipes = load(restore_data)
        # Bounded recipes reconstruct only this producer's JSON duplicates and worker copies.
        for name, recipe in recipes.items():
            assert name.startswith('raw/populations/') and name not in members
            if recipe['kind'] == 'alias':
                data = members[recipe['source']]
            else:
                assert recipe['kind'] == 'producer-json-indent2'
                value = load(members[recipe['source']])
                for part in recipe['keys']: value = value[part]
                data = (json.dumps(value,indent=2)+'\n').encode()
            members[name] = data
    assert members.keys() == witnesses['members'].keys(), 'public member set mismatch'
    for name, data in members.items():
        row = witnesses['members'][name]
        assert len(data) == row['public_bytes'] and digest(data) == row['public_sha256'], name
        if not row['redacted']:
            assert row['original_sha256_witness'] == row['public_sha256']
            assert row['original_bytes_witness'] == row['public_bytes']
        else:
            assert row['original_sha256_witness'] != row['public_sha256']
            assert row['original_bytes_in_publication'] is False
    raw_names = [name for name in members if name.startswith('raw/')]
    assert len(raw_names) == 620
    def read(name):
        return load(members['raw/'+name])
    plan, terminal = read('matrix-plan.json'), read('matrix-terminal.json')
    assert plan['source_revision'] == 'cbd4d8074de35105717c8aac7cd1f86babbe90ec'
    assert plan['driver_sha256'] == '74f61cc0eb9d1afb54e210eefb2620bbf01e91c2e02d55e5fbe65fbfc88c9055'
    assert len(plan['schedule']) == len(terminal['populations']) == 36
    assert plan['retries'] == 0 and plan['on_failure'].startswith('STOP_ALL')
    expected = {(n, mode, arm, repeat) for n in [48,96] for mode in ['direct','df']
                for arm in ['standard','fast','gpu4pyscf'] for repeat in range(3)}
    assert {tuple(r[k] for k in ['atoms','mode','arm','repeat']) for r in plan['schedule']} == expected
    rotation = []
    arms = ['standard','fast','gpu4pyscf']
    for n in [48,96]:
        for mode in ['direct','df']:
            for repeat in range(3):
                rotation.extend((n,mode,arm,repeat) for arm in arms[repeat:]+arms[:repeat])
    assert [tuple(r[k] for k in ['atoms','mode','arm','repeat']) for r in plan['schedule']] == rotation
    for name, row in plan['input_files'].items():
        original = witnesses['members']['raw/'+name]
        assert original['original_sha256_witness'] == row['sha256']
        assert original['original_bytes_witness'] == row['bytes']
    gates = {'energy_abs_per_atom_hartree': 1e-5,
             'force_max_abs_hartree_per_bohr': 1e-4, 'force_rms_hartree_per_bohr': 3e-5}
    for case in plan['cases'].values():
        assert case['gates'] == gates
        assert case['native_library_sha256'] == '5645dacffe534cff08bd5cc042eefc9ab1f5335cced0f16220387eefd61a38df'
        assert case['oracle_archive_sha256'] == 'd61a3e8ba0a8f8a8c08ee4c4b30c8e3e397e67863b8d75e7ba974e90cf0f85a4'
        assert case['auxiliary_archive_sha256'] == '707aaace01581e42ca9b1ba9bc924a9e739e73cc0ae337807012449589780862'
        stability = metrics(case['gpu_l4_reference'], case['gpu_l5_reference'], case['atoms'])
        assert all(stability[k] <= v/4 for k,v in gates.items())
    groups, count, worst = {}, 0, {key: 0.0 for key in gates}
    for slot, wanted in zip(terminal['populations'], plan['schedule'], strict=True):
        assert all(slot[k] == wanted[k] for k in ['atoms','mode','arm','repeat'])
        assert slot['status'] == 'PASS'
        folder = slot['directory']
        control, population, inputs = (read(folder+'/'+name) for name in ['controller-terminal.json','terminal.json','plan.json'])
        assert control['returncode'] == 0 and control['process_reaped'] is True
        assert control['process_group_absent'] is True and control['timed_out'] is False
        # Recorded producer hashes refer to original remote bytes, including redacted paths.
        original_pop = witnesses['members']['raw/'+folder+'/terminal.json']['original_sha256_witness']
        assert control['terminal_sha256'] == slot['terminal_sha256'] == original_pop, 'original remote SHA witness mismatch'
        assert population['source_revision'] == plan['source_revision']
        assert population['status'] == 'PASS_SCOPED_REGISTERED_SOURCE_SETUP_POPULATION'
        assert population['driver_sha256'] == plan['driver_sha256']
        case = plan['cases'][str(wanted['atoms'])]
        assert population['geometry_bohr'] == inputs['geometry_bohr'] == case['geometry_bohr']
        assert inputs['gates'] == gates
        assert population['device']['name'] == 'NVIDIA H100 80GB HBM3'
        assert all(population['controls'][k] == '1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'])
        assert len(population['samples']) == 4
        admission = read(folder+'/aot-admission.json')
        assert admission['architecture'] == 'sm_90'
        artifacts = {r['name']:r for r in admission['artifacts']}
        for name, sha in [('pbe0_rks','7a438d6f745ec99e33e643ed6c4ca21b6b77fa8f3c7bb5c85817f7773f83286a'),
                          ('pbe0_rks_spd','6e1dc94f4f18152b93ce3d97ff903a9f28b3951f911f1ccac7febbe57674b4bd')]:
            assert artifacts[name]['binary_sha256'] == sha
            assert 'cubin' in artifacts[name]['code_kinds'] and artifacts[name]['driver_ptx_jit_required'] is False
        for index, sample in enumerate(population['samples']):
            assert sample['phase'] == ('cold' if index == 0 else 'warm')
            assert sample['converged'] and math.isfinite(sample['seconds']) and sample['seconds'] > 0
            assert len(sample['forces']) == wanted['atoms'] and all(len(row) == 3 for row in sample['forces'])
            values = [sample['energy']]+[v for row in sample['forces'] for v in row]
            assert all(math.isfinite(v) for v in values)
            raw = read(folder+'/raw-endpoint-'+str(index)+'.json')
            assert [struct.pack('>d',float(v)).hex() for v in values] == raw['binary64_big_endian_hex'], 'binary64 full-vector mismatch'
            errors = metrics(sample, case['gpu_l5_reference'], wanted['atoms'])
            assert all(errors[k] <= limit for k,limit in gates.items()), 'independent numeric gate failed'
            for key, value in errors.items(): worst[key] = max(worst[key], value)
            if wanted['arm'] != 'gpu4pyscf':
                assert population['native_build']['library_sha256'] == case['native_library_sha256']
                assert type(sample['fock_builds']) is int and sample['fock_builds'] > 0
                if wanted['arm'] == 'fast': assert sample['precision_validation']['profile'] == 'fast'
            else:
                assert type(sample['effective_potential_builds']) is int and sample['effective_potential_builds'] > 0
            count += 1
        samples = population['samples']
        key = tuple(wanted[k] for k in ['atoms','mode','arm'])
        groups.setdefault(key, []).append({'repeat':wanted['repeat'], 'cold':samples[0]['seconds'],
            'warm_median':statistics.median(s['seconds'] for s in samples[1:])})
    assert count == 144 and len(groups) == 12
    summary = load((ROOT/'summary.json').read_bytes())
    assert summary['compiler_cache_launcher'] == terminal['compiler_cache_launcher']
    assert summary['related_issues'] == ['#2731','#2739']
    assert summary['default_promotion'] == 'NOT_REQUESTED_OR_ESTABLISHED'
    for row in summary['statistics']:
        key = tuple(row[k] for k in ['atoms','mode','arm'])
        processes = groups[key]
        assert row['independent_processes'] == 3
        assert len(processes) == len({p['repeat'] for p in processes}) == 3
        cold, warm = [p['cold'] for p in processes], [p['warm_median'] for p in processes]
        assert row['cold_median_seconds'] == statistics.median(cold)
        assert row['cold_range_seconds'] == [min(cold),max(cold)]
        assert row['median_of_process_warm_medians_seconds'] == statistics.median(warm)
        assert row['process_warm_median_range_seconds'] == [min(warm),max(warm)]
    assert summary['error_maxima'] == worst
    binding = load(members['supplements/source-binding.json'])
    source = binding['source']
    assert source['source_revision'] == plan['source_revision']
    assert source['git_tree'] == 'c8a85362459c4258f9f64ea4f0f94d6b43872d89'
    assert source['derived_git_lf_archive_sha256'] == '21fe3643a42d982a4a18fa18e4319be7d44db45f5f8d50ab1199f8c9c0485026'
    assert source['derived_archive_bytes'] == 159764480
    assert source['qualified_patch_sha256'] == '3ebd93ce481b47058e20990f99f11a1c9d349d6780cc5a63286583bbd3316041'
    assert source['qualified_patch_base'] == 'f5ca5398f25ed7b31285367b5b8fd829d3b024b5'
    assert 'not applied twice' in source['reconstruction']
    assert source['pinned_file_sha256']['benchmarks/readme_defaults.py'] == plan['driver_sha256']
    assert source['original_5852_package_archive_provenance'].startswith('UNAVAILABLE')
    assert binding['file_count'] == 12130
    def platform(name):
        retained = load(members['supplements/'+name])
        assert retained['returncode'] == 0
        response = load(retained['stdout'])
        assert response['success'] is True
        return response['data']
    running, final = platform('platform-running.json'), platform('platform-terminal.json')
    assert running['name'] == final['name'] == plan['job_name'] == 'gqc-pbe0-matrix-1027-fd06-r1'
    assert running['project'] == final['project'] == 'CQ-NexChem'
    assert running['nodes'] == ['qb-prod-gpu511'] and final['status'] == 'SUCCEEDED'
    assert running['priority'] == final['priority'] == 20
    assert running['resource'] == final['resource']
    assert all(running['resource'][k] == v for k,v in {'cpu':20,'memory_gib':200,'gpu':1,'nodes':1,'shared_memory_gib':32}.items())
    postflight = load(members['supplements/postflight-integrity.json'])
    assert postflight['status'] == 'PASS_SELECTED_AUTHENTICATED_COPY_ASSETS_UNCHANGED'
    assert postflight['expected_manifest_sha256'] == postflight['observed_manifest_sha256'] == '2789961124939951bebee6312581224fce599de474bc6134a3f35f0eea1428c6'
    assert postflight['checked_members'] == postflight['passed_members'] == 12136
    assert not postflight['failures']
    if postflight.get('inventory_records_scope') == 'SELECTED_IDENTITIES_ONLY':
        assert len(postflight['members']) == 6
        assert postflight['full_inventory_reconstructible_from_this_minimal_capsule'] is False
        assert postflight['full_original_receipt']['sha256'] == '5c12e073d3d559f4a51d5d5f75a4be8ee9b03000f3fdf8f75761167391c3e26c'
        assert postflight['full_original_receipt']['bytes'] == 5748012
    else:
        assert len(postflight['members']) == 12136
    for row in postflight['members']:
        assert row['status'] == 'PASS'
        if row['type'] == 'file':
            assert all(row['expected_'+key] == row['observed_'+key] for key in ['sha256','bytes','mode'])
    env = load(members['supplements/environment-metadata.json'])
    assert env['provenance'] == 'POSTFLIGHT_FROZEN_ENV_METADATA'
    assert env['selected_files'] == env['selected_files_matching_initial_manifest'] == 7
    assert env['python_runtime_from_released_matrix_workers'].startswith('UNAVAILABLE')
    assert env['package_runtime_from_released_matrix_workers'].startswith('UNAVAILABLE')
    assert summary['historical_RTX5090']['repeat_statistics'] == 'NOT_INFERRED'
    assert summary['original_complete_measured_source_manifest'] == 'UNAVAILABLE'
    assert summary['hardware_only_attribution'] == 'NOT_CLAIMED'
    assert summary['parent_issue_disposition'] == 'REFS_2731_REVIEW_PENDING_NO_CLOSURE'
    print(json.dumps({'status':'PASS_RETAINED_PBE0_144_EF_OFFLINE_REPLAY',
        'raw_members':len(raw_names),'public_members_hashed':len(members)+1,
        'processes':36,'complete_EF':count,'groups':len(groups),'error_maxima':worst,
        'original_redacted_bytes':'UNAVAILABLE; original SHA values are separate witnesses',
        'sdk_acceptance':'SEPARATE','parent_issue':'REFS_2731_REVIEW_PENDING_NO_CLOSURE'},indent=2))


if __name__ == '__main__':
    main()
