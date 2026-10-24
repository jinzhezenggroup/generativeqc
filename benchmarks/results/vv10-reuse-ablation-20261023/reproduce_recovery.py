"""Restore complete receipts and independently replay the selected mixed dataset."""
from exact_pack import bounded_xz, checked_path, relative, require, restore

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys


def logical_bytes(path):
    """Read one explicitly compressed or plain logical metadata member."""
    path = Path(path)
    candidates = [candidate for candidate in (path, Path(str(path) + '.xz')) if candidate.is_file()]
    require(len(candidates) == 1, 'missing/ambiguous logical metadata member')
    selected = candidates[0]
    require(not selected.is_symlink(), 'symlink metadata member')
    raw = selected.read_bytes()
    return bounded_xz(raw) if selected.name.endswith('.xz') else raw


def load(path):
    return json.loads(logical_bytes(path))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore_tools(publication, destination, manifest):
    source = manifest['offline_source_pack']
    name = relative(source['file'])
    require(name in {row['path'] for row in manifest['files'] if row['role'] == 'reproduction'}, 'offline source pack is not declared')
    raw = (Path(publication) / name).read_bytes()
    require(len(raw) == source['packed_bytes'] and hashlib.sha256(raw).hexdigest() == source['sha256'], 'offline source pack changed')
    target = Path(destination) / 'postprocessing'
    restore(raw, target)
    return target


def restore_collection(collection_directory, destination):
    source, destination = Path(collection_directory), Path(destination)
    require(not destination.exists(), 'fresh restoration directory required')
    collection = load(source / 'collection.json')
    require(collection['schema'] == 'generativeqc.vv10-mixed-protocol-collection.v1', 'wrong collection schema')
    require(hashlib.sha256(logical_bytes(source / 'raw-inventory.json')).hexdigest() == collection['inventory_sha256'], 'inventory changed')
    require(hashlib.sha256(logical_bytes(source / 'numerical-replay.json')).hexdigest() == collection['numerical_replay_sha256'], 'stored replay changed')
    seen = set()
    for pack in collection['packs']:
        name, mount = relative(pack['file']), relative(pack['mount'])
        require(name not in seen, 'duplicate pack')
        seen.add(name)
        raw = (source / name).read_bytes()
        require(len(raw) == pack['packed_bytes'] and hashlib.sha256(raw).hexdigest() == pack['sha256'], 'pack changed')
        restore(raw, destination / mount)
    inventory = load(source / 'raw-inventory.json')
    for alias in collection['aliases']:
        require(alias['mount'] in inventory and alias['source_mount'] in inventory, 'unlisted alias mount')
        canonical = checked_path((destination / alias['source_mount']).resolve(), alias['source_path'])
        target = checked_path((destination / alias['mount']).resolve(), alias['path'])
        require(canonical.stat().st_size == alias['bytes'] and sha(canonical) == alias['sha256'], 'canonical whole-file alias bytes changed')
        require(not target.exists(), 'alias refuses existing destination')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(canonical.read_bytes())
        if os.name == 'posix':
            target.chmod(alias['mode'])
    for mount, catalog in inventory.items():
        root = destination / relative(mount)
        expected = set()
        for row in catalog['included']:
            name = relative(row['path'])
            require(name not in expected, 'duplicate original inventory member')
            expected.add(name)
            path = root / name
            require(path.is_file() and not path.is_symlink(), 'missing/restored symlink member')
            require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'original raw bytes differ')
            if os.name == 'posix':
                require(stat.S_IMODE(path.stat().st_mode) == row['mode'], 'original POSIX mode differs')
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        require(actual == expected, 'restoration does not conserve original raw inventory')
    return collection


def verify_provenance(restored, publication, helper_root=None):
    from collect_recovery import verify_terminal_bindings
    restored, publication = Path(restored), Path(publication)
    helper_root = Path(helper_root) if helper_root is not None else publication
    recovery, sdk, control = [restored / mount for mount in ('recovery', 'sdk', 'control')]
    verify_terminal_bindings(recovery)
    spec = load(recovery / 'spec.json')
    expected_sdk = spec['expected_sdk']
    for name, expected in expected_sdk['manifests'].items():
        require(sha(sdk / relative(name)) == expected, 'SDK manifest differs: ' + name)
    source = load(sdk / 'source-manifest.json')
    require(source['revision'] == spec['source_revision'] == expected_sdk['source_revision'], 'source revision mismatch')
    require(len(source['files']) == expected_sdk['source_entries'], 'source entry inventory mismatch')
    source_rows = {row['path']: row for row in source['files']}
    require(len(source_rows) == len(source['files']), 'duplicate source identity')
    owner = load(control / 'global-campaign-claim-v1/owner.json')
    failed = load(control / 'global-campaign-claim-v1/failed.json')
    require(failed['owner'] == owner and not (control / 'global-campaign-claim-v1/complete.json').exists(),
            'original failed campaign was reinterpreted as complete')
    attempt = control / ('campaign-attempt-' + owner['claim_nonce'])
    require(load(attempt / 'receipts/owner.json') == owner, 'old attempt belongs to a different owner')
    require(load(attempt / 'receipts/sdk.json') == expected_sdk, 'legacy starting SDK integrity differs')
    post_failure = control / 'post-failure-20261024-0248'
    require(load(post_failure / 'sdk-post-failure.json') == expected_sdk, 'legacy post-failure SDK integrity differs')
    audit = load(post_failure / 'audit.json')
    require(audit['integrity_status'] == 'PASS' and audit['scientific_status'] == 'INCOMPLETE',
            'legacy source audit must preserve scientific incompleteness')
    require(load(post_failure / 'observer.json')['failed_sha256']
            == sha(control / 'global-campaign-claim-v1/failed.json'), 'post-failure observation belongs to another failure')
    for row in load(sdk / 'receipts/driver-manifest.json'):
        candidate = sdk / relative(row['path'])
        require(candidate.stat().st_size == row['bytes'] and sha(candidate) == row['sha256'], 'old producer bytes differ')
    for row in load(control / 'campaign-code-manifest.json'):
        require(sha(control / relative(row['path'])) == row['sha256'], 'old controller bytes differ')
    require(sha(control / 'global_claim.py') == owner['implementation_sha256'], 'old claim implementation differs')
    for name, key in (('retired-original-job.json', 'retirement_sha256'),
                      ('atomic-controls-v1/validation.json', 'claim_control_sha256'),
                      ('campaign-code-manifest.json', 'code_manifest_sha256')):
        require(sha(control / name) == owner['context'][key], 'control receipt binding differs')
    for name, expected in load(control / 'retired-original-job.json')['receipts'].items():
        require(sha(restored / 'retirement-local' / relative(name)) == expected, 'retired Job receipt differs')
    for row in spec['harness_files']:
        require(sha(recovery / 'harness' / relative(row['path'])) == row['sha256'], 'executed recovery harness changed')
        if row['path'] in {'recovery_contract.py', 'recovery_replay.py'}:
            require(sha(helper_root / row['path']) == row['sha256'], 'offline replay source differs from executed frozen code')

    def old_path(absolute):
        root = PurePosixPath(spec['old_control_root'])
        return control / relative(str(PurePosixPath(absolute).relative_to(root)))

    for binding in spec['selected_old_cases'] + spec['references']:
        require(sha(old_path(binding['source_path'])) == sha(recovery / binding['path']) == binding['sha256'],
                'copied selected evidence differs from original raw')
    for predecessor in spec['predecessors']:
        path = PurePosixPath(predecessor['path'])
        if path.name.endswith('.tar.gz'):
            # This derived container is indexed; its complete constituent raw
            # records are restored separately. It is not a scientific input.
            continue
        if path.is_relative_to(PurePosixPath(spec['old_control_root'])):
            candidate = old_path(predecessor['path'])
        else:
            require(path.parent.name == 'diagnostic-lease' and path.name == 'failed.json', 'unknown predecessor namespace')
            candidate = restored / 'diagnostic/diagnostic-lease/failed.json'
        require(candidate.stat().st_size == predecessor['bytes'] and sha(candidate) == predecessor['sha256'],
                'retained failed predecessor differs from admitted recovery spec')
    solver = source_rows['benchmarks/readme_defaults_accuracy.py']['sha256']
    adapter = sha(control / 'metadata-entry.py')
    for path in sorted((attempt / 'output').glob('reference-*.json')):
        row = load(path)
        producer = publication / 'reference-cpu-producer.py' if row['backend'] == 'cpu' else sdk / 'reference.py'
        require(row['source_revision'] == source['revision'] and row['solver_sha256'] == solver
                and row['driver_sha256'] == sha(producer), 'independent oracle source/producer differs')
        if row['backend'] == 'cuda':
            require(row['execution_adapter']['sha256'] == adapter, 'oracle execution adapter differs')
            require(row['allocation']['immutable_id'] == owner['context']['allocation_id']
                    and row['allocation']['created_at'] == owner['context']['allocation_created_at'], 'oracle allocation differs')
    device = load(attempt / 'receipts/device.json')
    for path in sorted((attempt / 'output').glob('native-*.json')):
        row = load(path)
        require(row['source_revision'] == source['revision'] and row['driver_sha256'] == sha(sdk / 'measure.py'), 'legacy native producer differs')
        require(row['execution_adapter']['sha256'] == adapter, 'legacy native execution adapter differs')
        require(row['environment']['host']['node'] == device['hostname'] == owner['hostname'], 'legacy device lifecycle differs')
        require(row['device_pci_bus_id'] == device['pci'], 'legacy device PCI differs')
        for sample in row['samples']:
            require(sample['native_build']['library_sha256'] == expected_sdk['core_sha256']
                    and sample['native_build']['source_identity'] == expected_sdk['source_identity'], 'legacy actual library differs')
    return expected_sdk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    publication = Path(__file__).resolve().parent
    require(not args.output.resolve().is_relative_to(publication), 'restore outside publication')
    manifest = load(publication / 'publication.json')
    names = set()
    for entry in manifest['files']:
        name = relative(entry['path'])
        require(name not in names, 'duplicate publication member')
        names.add(name)
        path = publication / name
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'publication member changed')
    require({row['file'] for row in manifest['receipt_packs']} <= names, 'unlisted raw pack')
    note = manifest['delivery_note']
    note_bytes = (publication.parents[2] / relative(note['path'])).read_bytes()
    require(len(note_bytes) == note['bytes'] and hashlib.sha256(note_bytes).hexdigest() == note['sha256'], 'delivered Agent Note differs')
    collection = restore_collection(publication, args.output)
    helpers = restore_tools(publication, args.output, manifest)
    sys.path.insert(0, str(helpers))
    from recovery_summary import summarize, timing_table, work_table
    retirement = manifest['retirement_pack']
    require(manifest['receipt_packs'] == collection['packs'] + [retirement], 'publication/collection raw inventories differ')
    require(retirement['file'] in names, 'unlisted retirement raw pack')
    restore((publication / retirement['file']).read_bytes(), args.output / 'retirement-local')
    verify_provenance(args.output, publication, helpers)
    frozen = args.output / 'recovery/harness/recovery_replay.py'
    result = subprocess.run([sys.executable, str(frozen), '--root', str(args.output / 'recovery')],
                            check=True, capture_output=True, text=True)
    require(result.stdout.strip() == collection['numerical'], 'frozen source replay differs')
    summary = summarize(args.output / 'recovery', load(args.output / 'sdk/receipts/input-audit.json'))
    require(summary['samples'] == json.loads(bounded_xz((publication / 'samples.json.xz').read_bytes())), 'sample/work index differs from full raw')
    evidence = json.loads(bounded_xz((publication / 'validation.json.xz').read_bytes()))
    require('timings' not in evidence and set(evidence['record_parts']) == {'timings'}, 'invalid timing companion layout')
    parts = evidence.pop('record_parts')['timings']
    require(len(parts) == 1, 'one complete timing-list companion required')
    timing_part = parts[0]
    raw = (publication / relative(timing_part['path'])).read_bytes()
    require(len(raw) == timing_part['bytes'] and hashlib.sha256(raw).hexdigest() == timing_part['sha256'], 'timing companion changed')
    evidence['timings'] = json.loads(bounded_xz(raw))
    for key in ('inputs_hash', 'block_errors', 'timings'):
        require(summary[key] == evidence[key], 'numerical/timing summary differs: ' + key)
    require(timing_table(summary).encode() == (publication / 'timings.md').read_bytes(), 'separate timing tables differ')
    require(work_table(summary).encode() == (publication / 'work.md').read_bytes(), 'actual work tables differ')
    require(summary['selected_replay'] == load(publication / 'numerical-replay.json'), 'terminal pure-data replay differs')
    print('PASS: exact full raw bytes/modes; old80 + new48 and pilot64; protocols/hardware separate; original protocol INCOMPLETE')


if __name__ == '__main__':
    main()
