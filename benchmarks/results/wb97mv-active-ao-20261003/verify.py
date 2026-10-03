"""Verify every complete endpoint pair before summarizing a frozen AO experiment."""
import gzip
import hashlib
import json
import sys
from pathlib import Path
from statistics import median
import numpy as np

root = Path(__file__).resolve().parent
atoms = int(sys.argv[1])
manifest = json.loads((root/'manifest.json').read_text())
expected = manifest['source']
summary = {'atoms': atoms, 'variants': {}, 'input_identity': expected}
for mode in ('dense', 'sparse'):
    path = root / f'matched{atoms}-{mode}.json.gz'
    stored = path.read_bytes()
    entry = manifest['files'][path.name]
    assert len(stored) == entry['bytes'] and hashlib.sha256(stored).hexdigest() == entry['sha256']
    payload = gzip.decompress(stored)
    assert len(payload) == entry['decoded_bytes'] and hashlib.sha256(payload).hexdigest() == entry['decoded_sha256']
    d = json.loads(payload)
    assert d['status'] == 'measured' and d['stage'] == 'complete' and d['accepted']
    assert d['native_build']['library_sha256'] == expected['library_sha256']
    assert d['native_build']['probe']['source_identity'] == expected['source_identity']
    ns = [d['native_cold'], d['native_priming'], *d['native_samples']]
    rs = [d['reference_cold'], d['reference_priming'], *d['reference_samples']]
    assert len(ns) == len(rs) == 5
    errors = {}
    for key, gate in (('energies_hartree', 1e-8), ('forces_hartree_per_bohr', 1e-7)):
        values = []
        for n, r in zip(ns, rs, strict=True):
            a, b = np.asarray(n[key]), np.asarray(r[key])
            assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            values.append(float(np.max(np.abs(a-b))))
        assert max(values) <= gate, (key, values)
        errors[key] = values
    for s in ns + rs:
        assert all(x['converged'] for x in s['convergence'])
    assert all(x['iterations'] == 1 for s in ns[1:]+rs[1:] for x in s['convergence'])
    assert all(s['reference_xc_backend'] and all(x['on_gpu'] for x in s['reference_xc_backend']) for s in rs)
    if mode == 'sparse':
        for s in ns:
            work = s['force_work']['active_ao_maps']
            assert work['discovery_density_contractions'] == 0
            assert work['point_ao_square_sum'] <= work['dense_point_ao_square_sum']
        assert all(s['force_work']['active_ao_maps']['discoveries'] == 0 for s in ns[1:])
    # The disabled force caller does not expose cache counters. Preserve absent
    # evidence rather than filling it with zero or an estimated dense count.
    work = ns[-1]['force_work'].get('active_ao_maps')
    for s in ns:
        scf = s['native_scf_ao_work']
        assert scf['requested'] == scf['selected'] == (1 if mode == 'sparse' else 0)
        assert scf['xc_evaluations'] > 0
        assert scf['point_ao_square_sum'] <= scf['dense_point_ao_square_sum']
        for key in ('tiles','empty_tiles','active_sum','max_active','host_peak_bytes','discovery_seconds'):
            assert scf[key] == ns[0]['native_scf_ao_work'][key]
    for engine in ('native','reference'):
        total = d[f'{engine}_complete_cold']
        assert total['seconds'] == total['prepare_seconds'] + total['first_execute_seconds']
        assert total['prepare_seconds'] == d[f'{engine}_prepare_seconds']
        assert total['first_execute_seconds'] == d[f'{engine}_cold']['seconds']
    scf = ns[-1]['native_scf_ao_work']
    nv, rv = [s['seconds'] for s in d['native_samples']], [s['seconds'] for s in d['reference_samples']]
    summary['variants'][mode] = {
        'raw_sha256': hashlib.sha256(payload).hexdigest(), 'all_pair_errors': errors,
        'native_warm_seconds': nv, 'reference_warm_seconds': rv,
        'native_median_seconds': median(nv), 'reference_median_seconds': median(rv),
        'native_over_reference': median(nv)/median(rv),
        'execute_only_cold_seconds': {e: d[f'{e}_cold']['seconds'] for e in ('native','reference')},
        'reported_prepare_seconds': {e: d[f'{e}_prepare_seconds'] for e in ('native','reference')},
        'force_warm_ao_work': work,
        'force_gm2_fraction': None if work is None else work['point_ao_square_sum']/work['dense_point_ao_square_sum'],
        'all_reference_xc_on_gpu': True,
        'complete_cold': {e:d[f'{e}_complete_cold'] for e in ('native','reference')},
        'native_scf_cold_work': ns[0]['native_scf_ao_work'],
        'native_scf_warm_work': scf,
        'scf_gm2_fraction': scf['point_ao_square_sum']/scf['dense_point_ao_square_sum'],
        'cold_iterations': {e:[x['iterations'] for x in d[f'{e}_cold']['convergence']] for e in ('native','reference')},
    }
# Read-only verification preserves the retained scientific records.
for mode,v in summary['variants'].items():
    print(mode, 'native',v['native_median_seconds'],'reference',v['reference_median_seconds'], 'ratio',v['native_over_reference'],'GM2',v['force_gm2_fraction'])
print('Every E/F pair, source/library identity, actual AO selection, reference XC backend and complete-cold total passed.')
