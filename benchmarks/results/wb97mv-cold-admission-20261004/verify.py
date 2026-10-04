"""Verify every complete endpoint pair before summarizing a frozen AO experiment."""
import hashlib
import lzma
import json
import os
import sys
from pathlib import Path
from statistics import median
import numpy as np
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
atoms = int(sys.argv[1])
expected = manifest['source']

def read_payload(name):
    """Check stored and decoded bytes before trusting retained measurements."""
    entry = manifest['files'][name]
    stored = (root / name).read_bytes()
    if len(stored) != entry['bytes'] or hashlib.sha256(stored).hexdigest() != entry['sha256']:
        raise ValueError('Stored report size or SHA-256 mismatch')
    payload = lzma.decompress(stored)
    if len(payload) != entry['decoded_bytes'] or hashlib.sha256(payload).hexdigest() != entry['decoded_sha256']:
        raise ValueError('Decoded report size or SHA-256 mismatch')
    return payload


support = json.loads(read_payload('support.json.xz'))
qualification = support['qualification']
for key in ('source_identity', 'library_sha256', 'native_tests'):
    if qualification[key] != expected[key]:
        raise ValueError('Qualification identity mismatch')
if support['run_outcome'] != {'exit_code': 0, 'slurm_job': manifest['slurm_job']}:
    raise ValueError('Allocation did not complete successfully')
if json.loads(support['run_receipts']['source-identity.json']) != expected:
    raise ValueError('Deployed source/library receipt mismatch')
if not qualification['full_native_dft'] or qualification['admission_cases'] != 4:
    raise ValueError('Native/admission qualification incomplete')
if support['qualification_native_outcomes'] != {'ordinary_eigen': True, 'seed_gates': True, 'full_native_dft': True}:
    raise ValueError('Native programs did not all pass')
if set(qualification['modes']) != {'sparse', 'zero'}:
    raise ValueError('Incomplete sparse/fallback qualification')
for mode in qualification['modes'].values():
    if mode != {'endpoint_cases': 7, 'successful_calls': 66, 'xc_submissions': 467}:
        raise ValueError('Qualification work mismatch')
for name, log in support['qualification_logs'].items():
    if hashlib.sha256(log.encode()).hexdigest() != qualification['hashes']['results/' + name]:
        raise ValueError('Qualification log differs: ' + name)
for name, digest in support['runtime_script_sha256'].items():
    if hashlib.sha256(support['runtime_scripts'][name].encode()).hexdigest() != digest:
        raise ValueError('Retained runtime script differs: ' + name)

summary = {'atoms': atoms, 'variants': {}, 'input_identity': expected}
for mode in ('none', 'lda16'):
    payload = read_payload(f'matched{atoms}-{mode}.json.xz')
    d = json.loads(payload)
    if d['atoms'] != atoms or d['aos'] != 8 * atoms:
        raise ValueError('Report scale differs from the requested water/def2-SVP case')
    if not (d['status'] == 'measured' and d['stage'] == 'complete' and d['accepted']):
        raise ValueError("d['status'] == 'measured' and d['stage'] == 'complete' and d['accepted']")
    if not d['native_build']['library_sha256'] == expected['library_sha256']:
        raise ValueError("d['native_build']['library_sha256'] == expected['library_sha256']")
    if not d['native_build']['probe']['source_identity'] == expected['source_identity']:
        raise ValueError("d['native_build']['probe']['source_identity'] == expected['source_identity']")
    ns = [d['native_cold'], d['native_priming'], *d['native_samples']]
    seed = ns[0]['semilocal_seed']
    if not seed['requested'] == mode:
        raise ValueError("seed['requested'] == mode")
    if not (seed['target_method'] == 'wb97m-v' and seed['target_grid'] == [48, 16, 32]):
        raise ValueError("seed['target_method'] == 'wb97m-v' and seed['target_grid'] == [48, 16, 32]")
    if not (seed['target_energy_tolerance'], seed['target_density_tolerance']) == (1e-11, 1e-09):
        raise ValueError("(seed['target_energy_tolerance'], seed['target_density_tolerance']) == (1e-11, 1e-09)")
    if mode != 'none':
        if not seed['source_method'] == {'lda16': 'lda-rks', 'pbe16': 'pbe-rks'}[mode]:
            raise ValueError("seed['source_method'] == {'lda16': 'lda-rks', 'pbe16': 'pbe-rks'}[mode]")
        if not (seed['source_backend'] == 'cuda' and seed['source_grid'] == [16, 8, 16]):
            raise ValueError("seed['source_backend'] == 'cuda' and seed['source_grid'] == [16, 8, 16]")
        if not (seed['source_energy_tolerance'], seed['source_density_tolerance']) == (1e-06, 0.0001):
            raise ValueError("(seed['source_energy_tolerance'], seed['source_density_tolerance']) == (1e-06, 0.0001)")
        if not (seed['source_max_iterations'] == 64 and seed['source_diis_history'] == 8):
            raise ValueError("seed['source_max_iterations'] == 64 and seed['source_diis_history'] == 8")
        if not seed['source_scf_ao_work']['requested'] == seed['source_scf_ao_work']['selected'] == 0:
            raise ValueError("seed['source_scf_ao_work']['requested'] == seed['source_scf_ao_work']['selected'] == 0")
    if not all((s['semilocal_seed'] == seed for s in ns)):
        raise ValueError("all((s['semilocal_seed'] == seed for s in ns))")
    if seed['requested'] != 'none':
        if not (seed['selected'] and seed['source_converged']):
            raise ValueError("seed['selected'] and seed['source_converged']")
        if not all(c['warm_start_used'] and not c['warm_start_fallback'] for c in ns[0]['convergence']):
            raise ValueError('Target did not use the admitted preliminary density')
        if not seed['source_scf_ao_work']['xc_evaluations'] == seed['source_iterations']:
            raise ValueError("seed['source_scf_ao_work']['xc_evaluations'] == seed['source_iterations']")
        if not seed['complete_source_seconds'] <= d['native_prepare_seconds']:
            raise ValueError("seed['complete_source_seconds'] <= d['native_prepare_seconds']")
        phases = [seed[k] for k in ('source_construct_seconds', 'source_prepare_seconds', 'source_solve_seconds', 'export_seconds', 'import_seconds', 'source_destroy_seconds', 'source_bookkeeping_seconds')]
        if not all((np.isfinite(t) and t >= 0 for t in phases)):
            raise ValueError('all((np.isfinite(t) and t >= 0 for t in phases))')
        if not abs(sum(phases) - seed['complete_source_seconds']) <= 1e-09:
            raise ValueError("abs(sum(phases) - seed['complete_source_seconds']) <= 1e-09")
        if not (seed['source_fock_builds'] is None or seed['source_fock_builds'] > 0):
            raise ValueError("seed['source_fock_builds'] is None or seed['source_fock_builds'] > 0")
    elif not (seed['complete_source_seconds'] is None and (not seed['selected'])):
        raise ValueError("seed['complete_source_seconds'] is None and (not seed['selected'])")
    if not not seed['source_reference_density_used']:
        raise ValueError("not seed['source_reference_density_used']")
    rs = [d['reference_cold'], d['reference_priming'], *d['reference_samples']]
    if not len(ns) == len(rs) == 5:
        raise ValueError('len(ns) == len(rs) == 5')
    errors = {}
    for key, gate in (('energies_hartree', 1e-08), ('forces_hartree_per_bohr', 1e-07)):
        values = []
        for n, r in zip(ns, rs, strict=True):
            a, b = (np.asarray(n[key]), np.asarray(r[key]))
            if a.shape != ((1,) if key == 'energies_hartree' else (1, atoms, 3)):
                raise ValueError('Endpoint array shape differs from its molecular case')
            if not (a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError('a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()')
            values.append(float(np.max(np.abs(a - b))))
        if not max(values) <= gate:
            raise ValueError((key, values))
        errors[key] = values
    for s in ns + rs:
        if not all((x['converged'] for x in s['convergence'])):
            raise ValueError("all((x['converged'] for x in s['convergence']))")
    if not all((x['iterations'] == 1 for s in ns[1:] + rs[1:] for x in s['convergence'])):
        raise ValueError("all((x['iterations'] == 1 for s in ns[1:] + rs[1:] for x in s['convergence']))")
    if not all((s['reference_xc_backend'] and all((x['on_gpu'] for x in s['reference_xc_backend'])) for s in rs)):
        raise ValueError("all((s['reference_xc_backend'] and all((x['on_gpu'] for x in s['reference_xc_backend'])) for s in rs))")
    if True:
        for s in ns:
            work = s['force_work']['active_ao_maps']
            if not work['discovery_density_contractions'] == 0:
                raise ValueError("work['discovery_density_contractions'] == 0")
            if not work['point_ao_square_sum'] <= work['dense_point_ao_square_sum']:
                raise ValueError("work['point_ao_square_sum'] <= work['dense_point_ao_square_sum']")
        if not all((s['force_work']['active_ao_maps']['discoveries'] == 0 for s in ns[1:])):
            raise ValueError("all((s['force_work']['active_ao_maps']['discoveries'] == 0 for s in ns[1:]))")
    work = ns[-1]['force_work'].get('active_ao_maps')
    for s in ns:
        scf = s['native_scf_ao_work']
        if not scf['requested'] == scf['selected'] == 1:
            raise ValueError("scf['requested'] == scf['selected'] == 1")
        if not scf['xc_evaluations'] > 0:
            raise ValueError("scf['xc_evaluations'] > 0")
        if not scf['point_ao_square_sum'] <= scf['dense_point_ao_square_sum']:
            raise ValueError("scf['point_ao_square_sum'] <= scf['dense_point_ao_square_sum']")
        for key in ('tiles', 'empty_tiles', 'active_sum', 'max_active', 'host_peak_bytes', 'discovery_seconds'):
            if not scf[key] == ns[0]['native_scf_ao_work'][key]:
                raise ValueError("scf[key] == ns[0]['native_scf_ao_work'][key]")
    for engine in ('native', 'reference'):
        total = d[f'{engine}_complete_cold']
        if not total['seconds'] == total['prepare_seconds'] + total['first_execute_seconds']:
            raise ValueError("total['seconds'] == total['prepare_seconds'] + total['first_execute_seconds']")
        if not total['prepare_seconds'] == d[f'{engine}_prepare_seconds']:
            raise ValueError("total['prepare_seconds'] == d[f'{engine}_prepare_seconds']")
        if not total['first_execute_seconds'] == d[f'{engine}_cold']['seconds']:
            raise ValueError("total['first_execute_seconds'] == d[f'{engine}_cold']['seconds']")
    scf = ns[-1]['native_scf_ao_work']
    nv, rv = ([s['seconds'] for s in d['native_samples']], [s['seconds'] for s in d['reference_samples']])
    summary['variants'][mode] = {'raw_sha256': hashlib.sha256(payload).hexdigest(), 'all_pair_errors': errors, 'native_warm_seconds': nv, 'reference_warm_seconds': rv, 'native_median_seconds': median(nv), 'reference_median_seconds': median(rv), 'native_over_reference': median(nv) / median(rv), 'execute_only_cold_seconds': {e: d[f'{e}_cold']['seconds'] for e in ('native', 'reference')}, 'reported_prepare_seconds': {e: d[f'{e}_prepare_seconds'] for e in ('native', 'reference')}, 'force_warm_ao_work': work, 'force_gm2_fraction': None if work is None else work['point_ao_square_sum'] / work['dense_point_ao_square_sum'], 'all_reference_xc_on_gpu': True, 'complete_cold': {e: d[f'{e}_complete_cold'] for e in ('native', 'reference')}, 'semilocal_seed': seed, 'native_scf_cold_work': ns[0]['native_scf_ao_work'], 'native_scf_warm_work': scf, 'scf_gm2_fraction': scf['point_ao_square_sum'] / scf['dense_point_ao_square_sum'], 'cold_iterations': {e: [x['iterations'] for x in d[f'{e}_cold']['convergence']] for e in ('native', 'reference')}}
# Verification is read-only; never rewrite scientific evidence.
for mode, v in summary['variants'].items():
    print(mode, 'native', v['native_median_seconds'], 'reference', v['reference_median_seconds'], 'ratio', v['native_over_reference'], 'GM2', v['force_gm2_fraction'])
print('Every endpoint, source lifecycle, actual work and provenance gate passed.')
