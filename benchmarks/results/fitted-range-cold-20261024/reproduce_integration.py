"""Recheck the current-parent bridge without pooling historical timing cohorts.

Run from the repository root with PYTHONPATH=python:. This offline reader uses
only retained independent references, full observable receipts and manifests.
Single native/GPU4 observations are a screen, not a statistical speedup claim.
"""
import copy
import hashlib
import json
import lzma
from pathlib import Path
import xml.etree.ElementTree as ET

from benchmarks import readme_defaults_report as reader
from benchmarks.readme_defaults_qualification import _comparison, qualify

ROOT = Path(__file__).resolve().parent
if not __debug__:
    raise SystemExit('Evidence verification requires assertions; do not use -O.')


def unpack(path):
    return json.loads(lzma.decompress(path.read_bytes()))


def digest(raw):
    return hashlib.sha256(raw.encode()).hexdigest()


packet = unpack(ROOT / 'integration.json.xz')
assert packet['schema'] == 'generativeqc.fitted-range-parent-integration.v1'
support = packet['support']
for name, raw in support.items():
    assert digest(raw) == packet['support_sha256'][name], name
manifest = json.loads(support['stage37-source-inputs.json'])
verification = json.loads(support['stage37-source-verification.json'])
assert manifest['source_revision'] == verification['source_revision'] == packet['source_revision']
assert not verification['mismatches']
assert verification['files_verified'] == len(manifest['files'])
assert packet['library_sha256'] in support['stage37-library-sha256.txt']
assert packet['library_sha256'] in support['stage37-package-sha256.txt']
launcher = json.loads(support['stage37-launcher-verification.json'])
assert launcher['compile_commands'] > 0 and not launcher['uncached_commands']
tests = {}
for name, expected in (('direct', 24), ('df', 40), ('checkpoint', 61),
                       ('range', 40), ('fitted', 6), ('failure', 1)):
    suites = ET.fromstring(support[f'stage37-{name}-tests.xml'])
    cases = list(suites.iter('testcase'))
    assert len(cases) == expected, (name, len(cases))
    assert not any(case.find(tag) is not None for case in cases
                   for tag in ('failure', 'error', 'skipped')), name
    tests[name] = expected

historical = unpack(ROOT / 'endpoints.json.xz')
provenance = unpack(ROOT / 'provenance.json.xz')
refs = ROOT.parent / 'readme-final-20261022'
oracles = reader.load_records([refs / 'oracles.json.xz'],
                             schema='generativeqc.readme-defaults-oracle.v1')
oracles.append(json.loads(provenance['support']['stage29-larger-oracle/oracle.json']))
reader.SIZES = (*reader.SIZES, 108)
reuse = json.loads(support['stage37-endpoint-qualification.reference-reuse.json'])
methods = ('pbe', 'pbe0', 'b3lyp', 'r2scan', 'r2scan-3c', 'm06-2x', 'wb97m-v')
expected = {
    f'stage37-seven-{mode}/{method}-{mode}-{arm}.json'
    for mode in ('direct', 'df') for method in methods
    for arm in ('standard', 'gpu4pyscf')
}
expected.update(f'stage37-larger/wb97m-v-df-{arm}.json'
                for arm in ('standard', 'gpu4pyscf'))
expected.update(('stage37-moved-df/wb97m-v-df-standard.json',
                 'stage37-fast-df/wb97m-v-df-fast.json'))
assert set(packet['records']) == expected
records, qualification = {}, {}
for name, raw in packet['records'].items():
    assert digest(raw) == packet['record_sha256'][name], name
    record = json.loads(raw)
    assert record['status'] == 'complete', name
    assert record['source_revision'] == packet['source_revision'], name
    assert record['runner_sha256'] == manifest['files']['benchmarks/readme_defaults.py'], name
    if record['arm'] != 'gpu4pyscf':
        assert record['native_build']['library_sha256'] == packet['library_sha256'], name
    view = copy.deepcopy(record)
    moved = []
    if 'moved_geometry_bohr' in record:
        reference = historical['moved_reference']
        assert record['moved_geometry_bohr'] == reference['geometry_bohr'], name
        for key in ('method', 'mode', 'orbital_basis', 'basis_representation',
                    'charge', 'spin', 'auxiliary_input_sha256'):
            assert record.get(key) == reference.get(key), (name, key)
        assert [s['phase'] for s in record['samples']] == ['cold', 'warm', 'moved', 'moved-warm']
        moved = [_comparison(s, reference['samples'][0], record['atoms'])
                 for s in record['samples'][2:]]
        assert all(check['passed'] for check in moved), name
        view['samples'] = view['samples'][:2]
        view['warm_repeats'] = 1
    else:
        assert [s['phase'] for s in record['samples']] == ['cold', 'warm'], name
    q = qualify([view], oracles, reference_reuse=reuse)[0]
    assert q['status'] == 'qualified', (name, q)
    if moved:
        q['moved'] = moved
    records[name], qualification[name] = record, q

assert len(records) == 32
assert sum(len(r['samples']) for r in records.values()) == packet['complete_calls'] == 66
screen = {}
groups = [(f'stage37-seven-{mode}', method, mode) for mode in ('direct', 'df')
          for method in methods] + [('stage37-larger', 'wb97m-v', 'df')]
for cohort, method, mode in groups:
    pair = {arm: records[f'{cohort}/{method}-{mode}-{arm}.json']
            for arm in ('standard', 'gpu4pyscf')}
    native, gpu4 = pair['standard'], pair['gpu4pyscf']
    for key in ('geometry_bohr', 'orbital_basis', 'basis_representation', 'charge',
                'spin', 'auxiliary_input_sha256', 'slurm_job_id', 'thread_controls'):
        assert native.get(key) == gpu4.get(key), (cohort, method, key)
    assert native['environment']['packages'] == gpu4['environment']['packages']
    assert native['environment']['runtime']['cuda_visible_devices'] == gpu4['environment']['runtime']['cuda_visible_devices']
    assert {k: v for k, v in native['accelerator'].items() if k != 'nvidia_smi'} == {
        k: v for k, v in gpu4['accelerator'].items() if k != 'nvidia_smi'}
    screen[f'{cohort}/{method}'] = {
        arm: [{k: s.get(k) for k in ('phase', 'seconds', 'iterations', 'fock_builds')}
              for s in record['samples']]
        for arm, record in pair.items()
    }

print(json.dumps(dict(records=32, qualified_calls=66, tests=tests,
                     timing_scope='single current-source screen; no paired performance gate',
                     screen=screen, qualification=qualification), indent=2))
