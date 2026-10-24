"""Recheck current-source occupied-XC endpoints without reusing historical timing."""
from __future__ import annotations

import copy
import hashlib
import json
import lzma
import math
from pathlib import Path
from statistics import median

from benchmarks.readme_defaults_qualification import _comparison, qualify
from benchmarks.readme_defaults_report import load_records

ROOT = Path(__file__).resolve().parent


def read(path: Path) -> dict:
    """Decode a retained lossless evidence bundle."""
    return json.loads(lzma.decompress(path.read_bytes()))


def timing(before: list[float], after: list[float]) -> dict:
    """Each fresh process contributes one cold or median warm observation."""
    b, a = median(before), median(after)
    bm, am = median(abs(x - b) for x in before), median(abs(x - a) for x in after)
    floor = max(0.02, 2 * (bm / b + am / a))
    return dict(before=before, after=after, before_median=b, after_median=a,
                before_mad=bm, after_mad=am, gain=1 - a / b,
                gain_floor=floor, passes_gain_gate=1 - a / b > floor)


def verify_profile(actual: dict, expected: dict) -> None:
    """Keep identity exact while allowing summation roundoff across Python versions."""
    actual = actual.copy()
    expected = expected.copy()
    actual_ms = actual.pop('gpu_ms')
    expected_ms = expected.pop('gpu_ms')
    assert actual == expected
    assert math.isfinite(actual_ms) and math.isfinite(expected_ms)
    # Python 3.12 changed float sum() accuracy. This is below one nanosecond
    # in the diagnostic millisecond total, not a scientific acceptance gate.
    assert math.isclose(actual_ms, expected_ms, rel_tol=0.0, abs_tol=1e-10)


def main() -> None:
    """Authenticate the exact input manifest and every independently gated call."""
    data = read(ROOT / 'receipts.json.xz')
    delta = data['source_manifest']
    files = read(ROOT / delta['base'])['files']
    for name in delta['removed']:
        del files[name]
    files.update(delta['changed'])
    manifest = dict(source_revision=data['source_revision'], files=dict(sorted(files.items())))
    assert len(files) == delta['file_count']
    assert hashlib.sha256((json.dumps(manifest, indent=2) + '\n').encode()).hexdigest() == delta['sha256']
    refs = ROOT.parent / 'readme-final-20261022'
    oracles = load_records([refs / 'oracles.json.xz'], schema='generativeqc.readme-defaults-oracle.v1')
    reuse = json.loads((refs / 'reference-reuse.json').read_text())
    moved_refs = {
        'df': read(ROOT.parent / 'df-md-charge-splits-20261023/integration.json.xz')['cohorts']['stage7-pr-moved-matrix']['r2scan-3c-df-moved-gpu4pyscf.json'],
        'direct': data['cohorts']['stage11-moved-reference']['r2scan-3c-direct-gpu4pyscf.json'],
    }
    result = dict(qualified_calls=0, records={}, moved={}, timing={}, profiles=data['profiles'])
    for name, receipt in data['profile_receipts'].items():
        rows = receipt['xc_operations']
        actual = {
            'xc_calls': len(rows),
            'occupied_generations': [row['counters']['occupied_xc_frame_generation'] for row in rows],
            'gpu_ms': sum(row['regions'][0]['gpu_ms'] for row in rows),
        }
        assert all(row['valid'] and not row['cuda_error'] for row in rows)
        verify_profile(actual, data['profiles'][name])
    for cohort, records in data['cohorts'].items():
        for name, record in records.items():
            assert record['status'] == 'complete'
            if record['arm'] == 'gpu4pyscf':
                continue
            assert record['source_revision'] == data['source_revision']
            view = copy.deepcopy(record)
            if 'moved_geometry_bohr' in record:
                reference = moved_refs[record['mode']]
                assert record['moved_geometry_bohr'] == reference['geometry_bohr']
                assert [s['phase'] for s in record['samples']] == ['cold', 'warm', 'moved', 'moved-warm']
                for key in ('method', 'mode', 'orbital_basis', 'basis_representation', 'charge', 'spin', 'auxiliary_input_sha256'):
                    assert record.get(key) == reference.get(key)
                errors = [_comparison(s, reference['samples'][0], record['atoms']) for s in record['samples'][2:]]
                assert all(e['passed'] for e in errors), (name, errors)
                result['moved'][name] = errors
                view['samples'] = view['samples'][:2]
                view['warm_repeats'] = 1
            q = qualify([view], oracles, reference_reuse=reuse)[0]
            assert q['status'] == 'qualified', (name, q)
            result['records'][cohort + '/' + name] = q
            result['qualified_calls'] += len(record['samples'])
    assert result['qualified_calls'] == 168
    for mode in ('df', 'direct'):
        pairs = data['cohorts']['stage11-pairs-' + mode]
        for phase in ('cold', 'warm'):
            values = [[median(s['seconds'] for s in pairs[f'r2scan-3c-{mode}-occupied-{variant}-{i}.json']['samples'] if s['phase'] == phase) for i in range(5)] for variant in (0, 1)]
            result['timing'][mode + '/' + phase] = timing(*values)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
