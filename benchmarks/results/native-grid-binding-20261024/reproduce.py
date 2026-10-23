"""Reproduce numerical and paired timing gates without using a GPU."""
from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path
from statistics import median

from benchmarks.readme_defaults_qualification import qualify
from benchmarks.readme_defaults_report import load_records

ROOT = Path(__file__).resolve().parent


def read(path: Path) -> dict:
    """Decode immutable retained JSON."""
    return json.loads(lzma.decompress(path.read_bytes()) if path.suffix == '.xz' else path.read_bytes())


def integration(generation: str = "") -> int:
    """Check the separately frozen master integration without pooling timings."""
    data = read(ROOT/f'integration{generation}.json.xz'); delta = data['source_manifest']
    files = read(ROOT/delta['base'])['files']
    for name in delta['removed']: del files[name]
    files.update(delta['changed'])
    manifest = dict(source_revision=data['source_revision'],files=dict(sorted(files.items())))
    assert len(files)==delta['file_count']
    assert hashlib.sha256((json.dumps(manifest,indent=2)+'\n').encode()).hexdigest()==delta['sha256']
    refs=ROOT.parent/'readme-final-20261022'
    oracles=load_records([refs/'oracles.json.xz'],schema='generativeqc.readme-defaults-oracle.v1')
    reuse=read(refs/'reference-reuse.json'); count=0; domains=set()
    for name,raw in data['records'].items():
        record=json.loads(raw)
        assert record['source_revision']==data['source_revision']
        assert record['native_build']['library_sha256'] in data['support'][f'grid-binding-pr-integration{generation}-package-sha256.txt']
        verdict=qualify([record],oracles,reference_reuse=reuse)[0]
        assert verdict['status']=='qualified',(name,verdict)
        assert len(record['samples'])==2
        domains.add((record['arm'],record['mode'],record['method']))
        count+=len(record['samples'])
    assert domains=={(arm,mode,method) for arm in ('standard','fast') for mode in ('df','direct') for method in ('pbe','pbe0','r2scan-3c')}
    assert count==24
    return count


def main() -> None:
    """Authenticate measured inputs, qualify all samples, and recompute MAD gates."""
    if not __debug__:
        raise SystemExit('Evidence verification requires assertions; do not use -O.')
    data = read(ROOT/'receipts.json.xz')
    delta = data['source_manifest']
    files = read(ROOT/delta['base'])['files']
    for name in delta['removed']:
        del files[name]
    files.update(delta['changed'])
    manifest = dict(source_revision=data['source_revision'], files=dict(sorted(files.items())))
    assert len(files) == delta['file_count']
    assert hashlib.sha256((json.dumps(manifest,indent=2)+'\n').encode()).hexdigest() == delta['sha256']
    refs = ROOT.parent/'readme-final-20261022'
    oracles = load_records([refs/'oracles.json.xz'],schema='generativeqc.readme-defaults-oracle.v1')
    reuse = read(refs/'reference-reuse.json')
    calls, timing, identities = 0, {}, set()
    runner_hash = hashlib.sha256(data['support']['grid-binding-pr-pairs.py'].encode()).hexdigest()
    for mode, cohort in data['cohorts'].items():
        records = {name:json.loads(raw) for name,raw in cohort.items()}
        assert len(records) == 20
        for name, record in records.items():
            assert record['source_revision'] == data['source_revision']
            control = record['native_grid_binding_control']
            assert control['same_source_and_binary'] and control['runner_sha256'] == runner_hash
            identities.add(record['native_build']['library_sha256'])
            verdict = qualify([record],oracles,reference_reuse=reuse)[0]
            assert verdict['status'] == 'qualified', (name,verdict)
            calls += len(record['samples'])
        for method in ('pbe','pbe0'):
            for pair in range(5):
                b, a = [records[f'{method}-{mode}-consumer-{v}-{pair}.json'] for v in (0,1)]
                assert not b['native_grid_binding_control']['consumer_binding']
                assert a['native_grid_binding_control']['consumer_binding']
                for before, after in zip(b['samples'],a['samples'],strict=True):
                    assert before['settings'] == after['settings']
                    assert (before['iterations'],before['fock_builds']) == (after['iterations'],after['fock_builds'])
            for phase in ('cold','warm'):
                values = [[median(s['seconds'] for s in records[f'{method}-{mode}-consumer-{v}-{i}.json']['samples'] if s['phase']==phase) for i in range(5)] for v in (0,1)]
                b,a=values;bm,am=median(b),median(a)
                bd,ad=median(abs(x-bm) for x in b),median(abs(x-am) for x in a)
                gain=1-am/bm;floor=max(.02,2*(bd/bm+ad/am))
                timing[f'{method}/{mode}/{phase}']=dict(baseline=b,candidate=a,baseline_median=bm,candidate_median=am,gain=gain,floor=floor,passes=gain>floor)
    assert calls == 160 and len(identities) == 1
    assert next(iter(identities)) in data['support']['grid-binding-pr-package-sha256.txt']
    print(json.dumps(dict(calls=calls,integration_calls=integration(),integration_v2_calls=integration("-v2"),timing=timing),indent=2))


if __name__ == '__main__':
    main()
