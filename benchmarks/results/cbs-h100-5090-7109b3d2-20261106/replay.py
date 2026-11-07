"""CPU-only reconstruction and numerical replay of two frozen CBS cohorts."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import subprocess
import struct
import sys
import tempfile
import tarfile

SOURCE = '7109b3d208d23352f266e2bf7440fdfcc71df974'
NATIVE = '94d7b0c9b4275ea7300ca9d7d237103072c0992fdc2606eba5aa229d73129b75'
INPUT = '4370d6c4b1bf4033818f31a9fcc622d3045a33bf'
HISTORICAL = 'benchmarks/results/cbs-routine-standard-20261025/records.json.xz'
INPUT_SHA = '03b2722d016c13daaf6025a021de18e0ac333ebbb6efe94943c57748a57f5d3a'
DICTIONARY = {
    'revision': SOURCE,
    'path': 'benchmarks/results/h100-cbs-5090-cohort-20261105/records-01.json.xz',
    'sha256': '19155ae9345aedd015152fbf4150f3510cf1303bb4f23ddaa3b4a4c42da09747',
}


def restore_json(node, previous):
    """Decode storage bits only; historical native values never certify a gate."""
    tag = node[0]
    if tag == 'r':
        return copy.deepcopy(previous)
    if tag == 'v':
        return node[1]
    if tag == 'f':
        require(type(previous) is float, 'Float delta has no float dictionary value')
        bits = struct.unpack('>Q', struct.pack('>d', previous))[0] ^ int(node[1], 16)
        return struct.unpack('>d', struct.pack('>Q', bits))[0]
    if tag == 'p':
        require(isinstance(previous, dict), 'Object delta has no object dictionary')
        result = copy.deepcopy(previous)
        for key, value in node[1].items():
            result[key] = restore_json(value, previous[key])
        return result
    if tag == 'd':
        require(isinstance(previous, dict), 'Ordered object delta has no dictionary')
        return {key: restore_json(value, previous.get(key)) for key, value in node[1]}
    if tag == 'l':
        require(isinstance(previous, list), 'List delta has no list dictionary')
        result = copy.deepcopy(previous)
        for index, value in node[1]:
            result[index] = restore_json(value, previous[index])
        return result
    raise ValueError('Unknown storage-delta opcode')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(repository, *arguments):
    return subprocess.check_output(['git', '-C', str(repository), *arguments])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path)
    parser.add_argument('--recover', type=Path)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    repository = args.repository or package.parents[2]
    manifest = json.loads((package / 'publication.json').read_text(encoding='utf-8'))
    require(manifest['source'] == {'revision': SOURCE, 'dirty': False}, 'Publication source differs')
    for row in manifest['files']:
        relative = PurePosixPath(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in row['path'], 'Unsafe publication path')
        raw = (package / row['path']).read_bytes()
        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'Publication member differs: ' + row['path'])
    require(manifest['members_file'] == 'members.json.xz', 'Member inventory reference differs')
    members = json.loads(lzma.decompress((package / manifest['members_file']).read_bytes()))
    with tempfile.TemporaryDirectory(prefix='cbs-dual-replay-') as temporary:
        restored = args.recover or Path(temporary) / 'restored'
        require(not restored.exists(), 'Recovery destination must not exist')
        restored.mkdir(parents=True)
        names = set()
        dictionary_raw = git(repository, 'show', SOURCE + ':' + DICTIONARY['path'])
        require(digest(dictionary_raw) == DICTIONARY['sha256'], 'Storage dictionary differs')
        dictionary = {}
        for item in json.loads(lzma.decompress(dictionary_raw)):
            if 'text' not in item:
                continue
            raw = item['text'].encode('utf-8')
            require(len(raw) == item['bytes'] and digest(raw) == item['sha256'], 'Historical dictionary member differs')
            dictionary[item['path']] = raw
        recovered_bytes = {}
        for capsule in manifest['capsules']:
            payload = json.loads(lzma.decompress((package / capsule).read_bytes()))
            require(payload['dictionary'] == DICTIONARY, 'Storage dictionary binding differs')
            for item in payload['files']:
                name = item['path']
                relative = PurePosixPath(name)
                require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name and str(relative) == name, 'Unsafe capsule path')
                require(name not in names, 'Duplicate capsule member')
                names.add(name)
                require(sum(key in item for key in ['text', 'base64', 'json_delta']) == 1, 'Ambiguous capsule encoding')
                if 'json_delta' in item:
                    require(('dictionary_member' in item) != ('member_reference' in item), 'Ambiguous delta reference')
                    previous = dictionary[item['dictionary_member']] if 'dictionary_member' in item else recovered_bytes[item['member_reference']]
                    value = restore_json(item['json_delta'], json.loads(previous))
                    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
                else:
                    raw = item['text'].encode('utf-8') if 'text' in item else base64.b64decode(item['base64'], validate=True)
                identity = members[name]
                require(len(raw) == identity['bytes'] and digest(raw) == identity['sha256'], 'Capsule member differs: ' + name)
                path = restored / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                recovered_bytes[name] = raw
        recovery = manifest['git_recovery']
        require(recovery == {'source-manifest.json': {'revision': SOURCE, 'format': 'git-archive-file-sha256-map.v1'}}, 'Git recovery recipe differs')
        inventory = {}
        # Reproduce the authenticated source transport's line-ending conversion,
        # independently of the reader's local Git configuration.
        archive_raw = git(repository, '-c', 'core.autocrlf=true', 'archive', '--format=tar', SOURCE)
        with tarfile.open(fileobj=io.BytesIO(archive_raw), mode='r:') as archive:
            for member in archive:
                if member.isfile():
                    inventory[member.name] = digest(archive.extractfile(member).read())
        raw = (json.dumps(inventory, indent=2) + '\n').encode('utf-8')
        identity = members['source-manifest.json']
        require(len(raw) == identity['bytes'] and digest(raw) == identity['sha256'], 'Git-recovered full source inventory differs')
        (restored / 'source-manifest.json').write_bytes(raw)
        names.add('source-manifest.json')
        require(names == set(members), 'Capsule inventory differs')
        historical_raw = git(repository, 'show', INPUT + ':' + HISTORICAL)
        require(digest(historical_raw) == INPUT_SHA, 'Frozen independent references differ')
        (restored / 'common/historical-records.json.xz').write_bytes(historical_raw)
        historical = json.loads(lzma.decompress(historical_raw))
        inputs = json.loads((restored / 'common/inputs.json').read_text())
        expected = {}
        for group in historical['reference'].values():
            for name, row in group['cases'].items():
                expected[row['protocol']['model'] + '--' + name] = dict(kind='validation', name=name, protocol=row['protocol'])
        for run in historical['qm7_runs'].values():
            for carbon, protocol in run['protocols'].items():
                expected['qm7-C' + carbon] = dict(kind='timing', carbon=int(carbon), protocol=protocol)
        require(len(expected) == 20 and inputs['cohort'] == expected, 'Scientific population/input changed')
        require(inputs['measured_source'] == SOURCE, 'Measured source differs')
        basis = json.loads((restored / 'common/basis-manifest.json').read_text())
        require(set(basis) == set(expected), 'Prepared basis population differs')
        for key, entry in expected.items():
            for role in ['orbital', 'auxiliary']:
                rows = basis[key][role]
                require([row['identity'] for row in rows] == entry['protocol'][role + '_identities'], 'Prepared basis identities differ')
                for row in rows:
                    require(digest((restored / 'common' / row['path']).read_bytes()) == row['sha256'], 'Prepared basis bytes differ')
        require(git(repository, 'rev-parse', SOURCE + '^{tree}').decode().strip() == '33dff27444736e511f6679a37d64f46eaea42b09', 'Source tree differs')
        code_files = {p.name: digest(p.read_bytes()) for p in (restored / 'common').glob('*.py')}
        for gpu, model, arch in [('h100', 'H100', '90'), ('rtx5090', 'RTX 5090', '120')]:
            root = restored / gpu
            build = json.loads((root / 'build-receipts/build.json').read_text())
            supervisor = json.loads((root / 'measurements/supervisor.json').read_text())
            require(build['status'] == 'BUILD_PASS_SCIENCE_NOT_RUN' and build['architecture'] == arch, 'Build not accepted')
            require(build['source']['revision'] == SOURCE and build['native_source_identity'] == build['loaded_native_source_identity'] == NATIVE, 'Native source binding differs')
            require(all(stage['returncode'] == 0 for stage in build['stages']), 'Failed build stage')
            require(supervisor['status'] == 'COMPLETED_VERIFIED_COHORT' and supervisor['aggregate_verification_returncode'] == 0, 'Incomplete scientific cohort')
            require(supervisor['automatic_retries'] == 0 and len(supervisor['rows']) == 20 and {r['name'] for r in supervisor['rows']} == set(expected), 'Dispatch population changed')
            require(all(r['status'] == 'NUMERICAL_PASS' and r['native']['returncode'] == r['verification']['returncode'] == 0 for r in supervisor['rows']), 'Failed case retained')
            require(supervisor['producer_sha256'] == code_files, 'Producer identity differs')
            device = supervisor['device_before']
            require(device == supervisor['device_after'] and model in device['model'], 'Device identity differs')
            require(supervisor['library_sha256'] == build['library_sha256'], 'Binary identity differs')
            configure = next(stage['command'] for stage in build['stages'] if stage['name'] == 'configure')
            require('-DCMAKE_CUDA_ARCHITECTURES=' + arch in configure and '-DGENERATIVEQC_AOT_PROFILE=portable_cuda' in configure and '-DGENERATIVEQC_CUDA_FAST_COMPILE=OFF' in configure, 'Build architecture/profile changed')
            require((root / 'build-receipts/cache-version.stdout').read_text().strip() == 'sccache 0.16.0', 'Compiler-cache prerequisite differs')
            if gpu == 'h100':
                preflight_raw = (root / 'measurements/preflight.json').read_bytes()
                preflight = json.loads(preflight_raw)
                require(preflight == {key: supervisor[key] for key in ['platform_job_name', 'hostname', 'source', 'library_sha256', 'native_source_identity', 'device_before', 'producer_sha256', 'input_sha256']}, 'Executed preflight differs')
                release_raw = (root / 'platform-release.json').read_bytes()
                release = json.loads(release_raw)
                raw_status = (root / 'platform-running.json').read_bytes()
                raw_logs = (root / 'platform-preflight-logs.txt').read_bytes()
                require(release['preflight'] == preflight and release['status_sha256'] == digest(raw_status) and release['logs_sha256'] == digest(raw_logs) and supervisor['platform_release_sha256'] == digest(release_raw), 'External allocation release binding differs')
                status = json.loads(raw_status)
                require(status['success'] and status['data']['status'] == 'RUNNING' and status['data']['name'] == supervisor['platform_job_name'], 'Actual H100 running allocation absent')
                require(status['data']['resource'] == dict(cpu=10, memory_gib=200, gpu=1, nodes=1, shared_memory_gib=8, gpu_type='NVIDIA H100 (80GB)'), 'Actual H100 allocation differs')
                require(('CBS_DUAL_PREFLIGHT_SHA256 ' + digest(preflight_raw)).encode() in raw_logs, 'Platform log does not authenticate preflight')
                terminal = json.loads((root / 'platform-terminal.json').read_text())
                instances = json.loads((root / 'platform-terminal-instances.json').read_text())
                require(terminal['success'] and terminal['data']['status'] == 'SUCCEEDED' and terminal['data']['name'] == supervisor['platform_job_name'], 'H100 terminal state is not successful')
                require(instances['success'] and instances['data']['name'] == supervisor['platform_job_name'] and len(instances['data']['items']) == 1 and instances['data']['items'][0]['status'] == 'instance_success', 'H100 worker terminal state differs')
            else:
                require(supervisor['slurm_job_id'] == '16608', 'Actual Slurm job differs')
                allocation = (root / 'measurements/allocation.txt').read_text()
                terminal = (root / 'measure-slurm-terminal.txt').read_text()
                require('JobId=16608' in allocation and 'JobState=RUNNING' in allocation and 'NumNodes=1' in allocation, 'Slurm running allocation absent')
                require('JobId=16608' in terminal and 'JobState=COMPLETED' in terminal and 'ExitCode=0:0' in terminal and 'Requeue=0' in terminal and 'Restarts=0' in terminal, 'Slurm successful terminal allocation absent')
            counts = [0, 0, 0]
            process_ids = set()
            for key in expected:
                row = json.loads((root / 'measurements' / (key + '.json')).read_text())
                require(row['source'] == build['source'] and row['native_source_identity'] == NATIVE and row['library_sha256'] == build['library_sha256'], 'Point source/core differs')
                require(row['device_identity'] == device and row['driver_sha256'] == code_files['native.py'], 'Point producer/device differs')
                require(row['basis_manifest_sha256'] == digest((restored / 'common/basis-manifest.json').read_bytes()), 'Prepared basis manifest differs')
                require(type(row['process_id']) is int and row['process_id'] > 0 and row['process_id'] not in process_ids, 'Native process identity missing or repeated')
                process_ids.add(row['process_id'])
                for call in row['samples'] + [row['startup']] + ([row['energy_only']] if 'energy_only' in row else []):
                    for value in [call] + [part['result'] for part in call['components']]:
                        require(value['converged'] is True and value['executed_backend'] == 'cuda', 'Native constituent used another backend or did not converge')
                counts[0] += len(row['samples']); counts[1] += int('startup' in row); counts[2] += int('energy_only' in row)
            require(counts == [73, 20, 13], 'Complete call population differs')
            # This copied verifier uses NumPy and retained independent energies;
            # it never imports the measured solver, CUDA or a compiler.
            subprocess.run([sys.executable, '-B', str(restored / 'common/verify.py'), '--records', str(root / 'measurements'), '--output', str(root / 'measurements/aggregate-verification.json')], check=True)
            subprocess.run([sys.executable, '-B', str(restored / 'verify-supplemental.py'), '--repository', str(repository), '--inputs', str(restored / 'common/inputs.json'), '--records', str(root / 'measurements'), '--output', str(root / 'recomputed-supplemental.json')], check=True)
        spec = importlib.util.spec_from_file_location('cbs_descriptive_summary', restored / 'summarize.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        actual = module.summarize(restored)
        expected_summary = json.loads(lzma.decompress((package / 'summary.json.xz').read_bytes()))
        require(actual == expected_summary, 'Descriptive tables differ from retained raw observations')
        print('PASS: both GPUs, same frozen source/protocol, 146 primary + 40 startup E+F and 26 energy-only returns; CPU-only replay')


if __name__ == '__main__':
    main()
