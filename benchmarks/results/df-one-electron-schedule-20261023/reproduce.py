"""Authenticate source inputs and independently recheck every retained endpoint."""
from __future__ import annotations

import copy
import hashlib
import json
import lzma
from pathlib import Path
from statistics import median

from benchmarks.readme_defaults_qualification import _comparison, qualify
from benchmarks.readme_defaults_report import load_records

ROOT = Path(__file__).resolve().parent


def read(path: Path) -> dict:
    """Read a lossless retained JSON bundle."""
    return json.loads(lzma.decompress(path.read_bytes()))


def timing(before: list[float], after: list[float]) -> dict:
    """Use one cold or median warm observation per fresh process."""
    b, a = median(before), median(after)
    bm, am = median(abs(x - b) for x in before), median(abs(x - a) for x in after)
    floor = max(0.02, 2 * (bm / b + am / a))
    return dict(before=before, after=after, before_median=b, after_median=a,
                before_mad=bm, after_mad=am, gain=1 - a / b,
                gain_floor=floor, passes_gain_gate=1 - a / b > floor)


def main() -> None:
    """Preserve separate pilot, diagnostic, changed-geometry and paired cohorts."""
    bundle = read(ROOT / 'receipts.json.xz')
    delta = bundle['source_manifest']
    files = read(ROOT / delta['base'])['files']
    for name in delta['removed']:
        del files[name]
    files.update(delta['changed'])
    manifest = dict(source_revision=bundle['source_revision'], files=dict(sorted(files.items())))
    assert len(files) == delta['file_count']
    raw = (json.dumps(manifest, indent=2) + '\n').encode()
    assert hashlib.sha256(raw).hexdigest() == delta['sha256']
    refs = ROOT.parent / 'readme-final-20261022'
    oracles = load_records([refs / 'oracles.json.xz'], schema='generativeqc.readme-defaults-oracle.v1')
    reuse = json.loads((refs / 'reference-reuse.json').read_text())
    pbe = read(ROOT.parent / 'df-raw-coulomb-20261023/stage2-moved-reference.json.xz')['pbe-df-96-gpu4pyscf.json']
    three_c = read(ROOT.parent / 'df-md-charge-splits-20261023/integration.json.xz')['cohorts']['stage7-pr-moved-matrix']['r2scan-3c-df-moved-gpu4pyscf.json']
    result = dict(qualified_calls=0, records={}, moved={}, timing={})
    for cohort, records in bundle['cohorts'].items():
        for name, record in records.items():
            assert record['status'] == 'complete'
            expected = bundle['baseline_revision'] if cohort == 'stage10-pairs' and '-before-' in name else bundle['source_revision']
            assert record['source_revision'] == expected
            view = copy.deepcopy(record)
            if 'moved_geometry_bohr' in record:
                reference = pbe if record['method'] == 'pbe' else three_c
                assert [s['phase'] for s in record['samples']] == ['cold', 'warm', 'moved', 'moved-warm']
                assert record['moved_geometry_bohr'] == reference['geometry_bohr']
                for key in ('method', 'mode', 'orbital_basis', 'basis_representation', 'charge', 'spin', 'auxiliary_input_sha256'):
                    assert record[key] == reference[key]
                errors = [_comparison(s, reference['samples'][0], record['atoms']) for s in record['samples'][2:]]
                assert all(e['passed'] for e in errors), (name, errors)
                result['moved'][name] = errors
                view['samples'] = view['samples'][:2]
                view['warm_repeats'] = 1
            q = qualify([view], oracles, reference_reuse=reuse)[0]
            assert q['status'] == 'qualified', (name, q)
            result['records'][cohort + '/' + name] = q
            result['qualified_calls'] += len(record['samples'])
    assert result['qualified_calls'] == 128
    pairs = bundle['cohorts']['stage10-pairs']
    for method in ('pbe', 'r2scan-3c'):
        for phase in ('cold', 'warm'):
            values = [[median(s['seconds'] for s in pairs[f'{method}-{variant}-{i}.json']['samples'] if s['phase'] == phase) for i in range(5)] for variant in ('before', 'after')]
            result['timing'][method + '/' + phase] = timing(*values)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
