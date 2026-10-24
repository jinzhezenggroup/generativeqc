"""Authenticate every range receipt, independent observable gate and paired population.

Run from a checkout with PYTHONPATH=python:. No GPU, native library or external
archive is needed. Original instrumented and single observations stay separate
from the five alternating fresh-process native/GPU4 comparisons.
"""
from collections import Counter
import copy
import gzip
import hashlib
import json
import lzma
from pathlib import Path
from statistics import median

from benchmarks import readme_defaults_report as reader
from benchmarks.readme_defaults_qualification import _comparison, qualify

ROOT = Path(__file__).resolve().parent
if not __debug__:
    raise SystemExit('Evidence verification requires assertions; do not use -O.')


def unpack(path):
    return json.loads(lzma.decompress(path.read_bytes()))


def digest(raw):
    return hashlib.sha256(raw.encode()).hexdigest()


# Storage-only compaction preserves every original byte of historical evidence.
for item in json.loads((ROOT / "history-storage.json").read_text())["files"]:
    stored = (ROOT.parents[2] / item["stored_path"]).read_bytes()
    assert len(stored) == item["stored_bytes"]
    assert hashlib.sha256(stored).hexdigest() == item["stored_sha256"]
    raw = gzip.decompress(stored)
    assert len(raw) == item["bytes"]
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]


packet = unpack(ROOT/'endpoints.json.xz')
provenance = unpack(ROOT/'provenance.json.xz')
previous = unpack(ROOT.parent/'xc-bounded-cross-20261024/provenance.json.xz')
base_raw = previous['source_manifests']['guarded']
base = json.loads(base_raw)
assert base['source_revision'] == provenance['base_revision']
manifests = {}
for name, delta in provenance['source_manifest_deltas'].items():
    assert digest(base_raw) == delta['base_manifest_sha256']
    files = dict(base['files'])
    for p in delta['removed']:
        del files[p]
    files.update(delta['changed'])
    value = dict(source_revision=delta['source_revision'],files=dict(sorted(files.items())))
    assert digest(json.dumps(value,indent=2)+'\n') == delta['manifest_sha256']
    manifests[name] = value
for revision, patch in provenance['patches'].items():
    assert digest(patch) == provenance['patch_sha256'][revision]
for name, raw in provenance['support'].items():
    assert digest(raw) == provenance['support_sha256'][name], name
support = provenance['support']
for stage in (26,27,28,29,33):
    manifest = manifests[f'stage{stage}-source-inputs.json']
    verified = json.loads(support[f'stage{stage}-source-verification.json'])
    assert verified['source_revision'] == manifest['source_revision']
    assert not verified['mismatches'] and verified['files_verified'] == len(manifest['files'])
for stage in (26,27):
    built = manifests[f'stage{stage}-source-inputs.json']
    tested = manifests[f'stage{stage}-test-source-inputs-v2.json']
    assert set(built['files']) == set(tested['files'])
    changed = {p for p in built['files'] if built['files'][p] != tested['files'][p]}
    assert changed == {'tests/python/test_range_resident_ks_cuda.py'}
    receipt = json.loads(support[f'stage{stage}-test-source-verification-v2.json'])
    assert receipt['production_inputs_unchanged'] and not receipt['mismatches']
    assert receipt['build_revision'] == built['source_revision']
    assert receipt['test_revision'] == tested['source_revision']

identities = {
    'ebe88719525b70aaf00b9c62267e165778c0adeb': '5f31db0ea769f548accc73d860f9a457cdf7459339b76c2c92e45b7c1ce94a11',
    '1ccc53419acd4c22d9496178a65fffd83f26d6b1': '4555f81fe4e277565cec634d642b41af99aa8c99ae108b0997ea78dec47e43db',
    '9029ad14236f7179547e3bab309974d6ea0f4da3': 'bd3cff2e92d8b4631ed65ba599ce44754dd82c6714de5ae3e468be5d8a4106ad',
    '4be44f9015d22dbf4425f6552a419d38add01b9e': '91111e95998c8049252ff8a09b8d6ba2beed59961b6051b3141bc2dc02e01f05',
    '68ad0382a4d56a3357ea998af7550a641c9ce6e5': 'e1cd4abc88363e9615a34ad5c36726241ffe1b6b5ebc878d0c15fd43c9c64abd',
    '7dcfdc50eb2c0fcb47635e2094c9a00918f44e54': 'a7fe54048d25483a9fc26094a11355546f99960577a0abf6a410e3ecacf965d2',
}
final_revision = '7dcfdc50eb2c0fcb47635e2094c9a00918f44e54'
for stage in (26,27,28,29,33):
    revision = manifests[f'stage{stage}-source-inputs.json']['source_revision']
    assert identities[revision] in support[f'stage{stage}-library-sha256.txt']

refs = ROOT.parent/'readme-final-20261022'
oracles = reader.load_records([refs/'oracles.json.xz'],schema='generativeqc.readme-defaults-oracle.v1')
reuse = json.loads(support['stage33-reference-reuse.json'])
# Larger qualification extends only the archive reader's population, never
# the production scientific settings. Its fresh two-grid oracle is retained.
reader.SIZES = (*reader.SIZES,108)
oracles.append(json.loads(support['stage29-larger-oracle/oracle.json']))
records = {}
qualified = {}
for path, raw in packet['records'].items():
    assert digest(raw) == packet['raw_record_sha256'][path],path
    record = json.loads(raw)
    assert record['status'] == 'complete' and record['method'] == 'wb97m-v'
    assert record['mode'] == 'df' and record['source_revision'] in identities
    if record['arm'] != 'gpu4pyscf':
        assert record['native_build']['library_sha256'] == identities[record['source_revision']]
    stage = int(path.split('-',1)[0].removeprefix('stage'))
    if stage == 29:
        assert record['source_revision'] == final_revision
    elif stage == 33:
        assert record['source_revision'] == 'ebe88719525b70aaf00b9c62267e165778c0adeb'
    else:
        assert record['arm'] == 'standard' and record['atoms'] == 96
    manifest = base if record['source_revision'] == base['source_revision'] else manifests[f'stage{stage}-source-inputs.json']
    assert record['runner_sha256'] == manifest['files']['benchmarks/readme_defaults.py']
    view = copy.deepcopy(record)
    if path.startswith(('stage29-moved-df/','stage33-moved-df/')) and record['arm'] == 'standard':
        ref = packet['moved_reference']
        assert record['moved_geometry_bohr'] == ref['geometry_bohr']
        for key in ('method','mode','orbital_basis','basis_representation','charge','spin','auxiliary_input_sha256'):
            assert record.get(key) == ref.get(key),(path,key)
        assert [s['phase'] for s in record['samples']] == ['cold','warm','moved','moved-warm']
        assert all(_comparison(s,ref['samples'][0],96)['passed'] for s in record['samples'][2:])
        view['samples'] = view['samples'][:2]
        view['warm_repeats'] = 1
    q = qualify([view],oracles,reference_reuse=reuse)[0]
    assert q['status'] == 'qualified',(path,q)
    records[path] = record
    qualified[path] = q
assert len(records) == 23
assert sum(len(r['samples']) for r in records.values()) == packet['complete_calls'] == 50
expected_paths = {
    'stage33-moved-df/wb97m-v-df-standard.json',
    *{f'stage33-larger/wb97m-v-df-{arm}.json' for arm in ('standard','gpu4pyscf')},
    *{f'stage33-gpu4-pairs-df/wb97m-v-df-pair-0-{arm}.json' for arm in ('standard','fast','gpu4pyscf')},
    'stage26-screen-df/wb97m-v-df-range-baseline.json',
    'stage26-screen-df/wb97m-v-df-range-cached.json',
    'stage27-screen-df/wb97m-v-df-range-plain.json',
    'stage27-screen-df/wb97m-v-df-range-cached.json',
    'stage28-screen-df/wb97m-v-df-range-cached.json',
    'stage29-moved-df/wb97m-v-df-standard.json',
    'stage29-moved-df/wb97m-v-df-gpu4.json',
    *{f'stage29-gpu4-pairs-df/wb97m-v-df-pair-{i}-{arm}.json' for i in range(5) for arm in ('standard','gpu4pyscf')},
}
assert set(records) == expected_paths,'Missing or additional receipt population'
assert Counter((p.split('/')[0],r['arm'],r['atoms']) for p,r in records.items()) == Counter({
    ('stage33-larger','standard',108):1,('stage33-larger','gpu4pyscf',108):1,
    ('stage33-gpu4-pairs-df','standard',96):1,('stage33-gpu4-pairs-df','fast',96):1,
    ('stage33-gpu4-pairs-df','gpu4pyscf',96):1,('stage33-moved-df','standard',96):1,
    ('stage26-screen-df','standard',96):2,('stage27-screen-df','standard',96):2,
    ('stage28-screen-df','standard',96):1,
    ('stage29-gpu4-pairs-df','standard',96):5,('stage29-gpu4-pairs-df','gpu4pyscf',96):5,
    ('stage29-moved-df','standard',96):1,('stage29-moved-df','gpu4pyscf',96):1,
})

# The capacity fix has a separate current-source bridge. Never pool these
# single observations with the historical five-pair performance population.
capacity_bridge = {}
for cohort, atoms in (('stage33-larger',108),('stage33-gpu4-pairs-df',96)):
    group = {r['arm']:r for p,r in records.items() if p.startswith(cohort+'/')}
    gpu4 = group['gpu4pyscf']
    for arm, r in group.items():
        for key in ('geometry_bohr','orbital_basis','basis_representation','charge','spin',
                    'auxiliary_input_sha256','slurm_job_id','thread_controls'):
            assert r.get(key) == gpu4.get(key),(cohort,arm,key)
        assert r['environment']['packages'] == gpu4['environment']['packages']
        assert r['environment']['runtime']['cuda_visible_devices'] == gpu4['environment']['runtime']['cuda_visible_devices']
        assert [s['phase'] for s in r['samples']] == ['cold','warm']
        assert all(_comparison(x,y,atoms)['passed'] for x,y in zip(r['samples'],gpu4['samples'],strict=True))
    capacity_bridge[cohort] = {arm:{s['phase']:s['seconds'] for s in r['samples']} for arm,r in group.items()}

pairs = {}
for pair in range(5):
    for arm in ('standard','gpu4pyscf'):
        r = records[f'stage29-gpu4-pairs-df/wb97m-v-df-pair-{pair}-{arm}.json']
        c = r['gpu4_pair_control']
        assert c['pair'] == pair and r['arm'] == arm
        assert c['native_validation_gemm'] == c['native_range_resident_values'] == '1'
        assert c['order'] == ('standard gpu4pyscf' if pair%2 == 0 else 'gpu4pyscf standard')
        assert c['runner_sha256'] == digest(support['stage29-gpu4-pairs.py'])
        assert [s['phase'] for s in r['samples']] == ['cold','warm']
        pairs[pair,arm] = r
    native,gpu4 = (pairs[pair,arm] for arm in ('standard','gpu4pyscf'))
    for key in ('geometry_bohr','orbital_basis','basis_representation','charge','spin','auxiliary_input_sha256','slurm_job_id','thread_controls','gpu4_pair_control'):
        assert native.get(key) == gpu4.get(key),(pair,key)
    assert native['environment']['runtime']['cuda_visible_devices'] == gpu4['environment']['runtime']['cuda_visible_devices']
    assert native['environment']['packages'] == gpu4['environment']['packages']
    assert {k:v for k,v in native['accelerator'].items() if k!='nvidia_smi'} == {k:v for k,v in gpu4['accelerator'].items() if k!='nvidia_smi'}
    assert all(_comparison(x,y,96)['passed'] for x,y in zip(native['samples'],gpu4['samples'],strict=True))

timing = {}
for phase in ('cold','warm'):
    values = {arm:[median(s['seconds'] for s in pairs[pair,arm]['samples'] if s['phase']==phase) for pair in range(5)] for arm in ('standard','gpu4pyscf')}
    medians = {arm:median(v) for arm,v in values.items()}
    mads = {arm:median(abs(x-medians[arm]) for x in v) for arm,v in values.items()}
    gain = 1-medians['standard']/medians['gpu4pyscf']
    gate = max(.02,2*sum(mads[arm]/medians[arm] for arm in values))
    timing[phase] = dict(values=values,medians=medians,mads=mads,gain=gain,gate=gate,significant_gain=gain>gate)

# Reconstruct the trace tree. Root response scopes are disjoint; child times
# are never summed into endpoint time. Counters are additive event observations.
trace = [json.loads(s) for s in support['stage29-moved-df/wb-df-96-progress.jsonl'].splitlines()]
nodes = {r['id']:dict(r,children=[],values=[]) for r in trace if r['event']=='BEGIN'}
assert len(nodes) == sum(r['event']=='BEGIN' for r in trace)
for r in trace:
    if r['event']=='VALUE':
        nodes[r['id']]['values'].append((r['key'],r['value']))
    elif r['event']=='END':
        assert 'end' not in nodes[r['id']]
        nodes[r['id']]['end'] = r
for node in nodes.values():
    assert node['end']['status'] in {'stream_complete','returned','accepted','completed'},node
    if node['parent']!=-1:
        nodes[node['parent']]['children'].append(node['id'])


def counters(index):
    result = Counter()
    for key,value in nodes[index]['values']:
        result[key] += value
    for child in nodes[index]['children']:
        result.update(counters(child))
    return result


force_scopes = [n for n in nodes.values() if n['name']=='force_response' and n['parent']==-1]
assert len(force_scopes)==8
work=[]
for phase,group in zip(('cold','warm','moved','moved-warm'),range(0,8,2),strict=True):
    for index,node in enumerate(force_scopes[group:group+2]):
        c = counters(node['id'])
        assert c['response_fitted_panel_calls']==0
        if index==0:
            assert c['response_occupied_rank']==160
            assert c['response_final_fitted_projection_reused']==1
            assert c['response_occupied_projection_blas_calls']==1
            assert c['response_occupied_projection_products']==3616
        work.append(dict(phase=phase,response_index=index,scope_id=node['id'],elapsed_ms=node['end']['elapsed_ms'],counters=dict(c)))
assert set(packet['failed_records']) == {'stage29-larger/wb97m-v-df-standard.json'}
for path,raw in packet['failed_records'].items():
    assert digest(raw)==packet['failed_record_sha256'][path]
    record=json.loads(raw)
    assert record['status']=='failed' and record['source_revision']==final_revision
    assert record['atoms']==108 and record['method']=='wb97m-v' and record['arm']=='standard'
    # Failure occurred before the endpoint driver published its binary field.
    # Keep that missing receipt explicit; the retained Slurm recipe selects the
    # immutable Stage29 package but is not a completed endpoint attestation.
    assert 'native_build' not in record
    assert 'stage29-library/libgenerativeqc.so' in support['local-stage29/stage29-larger.sh']
    assert not record['samples']
    assert 'stationary CUDA byte budget is not representable' in record['error']
print(json.dumps(dict(qualified_calls=50,records=23,failed_records=1,capacity_bridge=capacity_bridge,timing=timing,work=work,qualification=qualified),indent=2))
