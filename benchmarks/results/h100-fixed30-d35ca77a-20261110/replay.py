"""Offline readback of the H100 fixed30 cohort on upstream master d35ca77a4.

Authenticates the stored receipt stream and every member, reconstructs the original
task layout in a temporary directory, materializes the pinned benchmark helpers
from local Git objects of the measured revision, reruns the independent
qualification/timing reduction and checks it against the sealed reduction and
the published compact summary. No CUDA, compiler, solver or network use.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import lzma
import os
import subprocess
import sys
import tarfile
import tempfile

BUNDLE = Path(__file__).resolve().parent


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def compact(value):
    keys = ['status', 'scope', 'source_revision', 'native_source_identity', 'core_sha256', 'device_uuid',
            'campaign_sha256', 'protocol_sha256', 'attempted_populations', 'qualified_populations', 'returned_EF',
            'science_pass', 'summary', 'default_force_integration', 'limitation']

    def timing(record):
        return {phase: record['timing'][phase]['median'] for phase in ['cold', 'warm']} if record else None

    def work(record):
        if not record or not record['work']:
            return None
        cold = record['work'][0]
        return {k: cold.get(k) for k in ['phase', 'iterations', 'fock_builds', 'effective_potential_builds', 'initial_guess']}

    return {**{k: value[k] for k in keys},
            'cases': [{'id': c['id'], 'method': c['method'], 'atoms': c['atoms'], 'mode': c['route'],
                       'science_pass': c['science_pass'], 'phase': c['phase'],
                       'pair_timings': [{'pair_index': p['pair_index'], 'qualified': p['qualified'],
                                         'native': timing(p['native']), 'reference': timing(p['reference']),
                                         'cold_native_work': work(p['native']),
                                         'cold_reference_work': work(p['reference'])} for p in c['pairs']]}
                      for c in value['cases']],
            'failures': value['failures']}


def pinned_source(repository, revision, isolated, env):
    isolated.mkdir()
    subprocess.run(['git', 'init', '-q', str(isolated)], env=env, check=True, capture_output=True)
    inventory = subprocess.check_output(['git', '-C', str(repository), 'ls-tree', '-r', '--name-only', revision, '--',
                                         'python', 'data', 'benchmarks', 'manifests', 'cmake/GenerativeQCSourceIdentity.json'],
                                        env=env, text=True).splitlines()
    names = [n for n in inventory if n.startswith(('python/', 'data/', 'manifests/')) or n == 'cmake/GenerativeQCSourceIdentity.json'
             or (n.startswith('benchmarks/') and '/results/' not in n and n.endswith('.py'))]
    check(names, 'Empty pinned source inventory')
    archive = subprocess.check_output(['git', '-C', str(repository), 'archive', revision, '--', *names], env=env)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        members = stream.getmembers()
        check(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                  for m in members), 'Unsafe pinned helper member')
        stream.extractall(isolated, filter='data')
    return isolated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=BUNDLE.parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    check(not sys.flags.optimize and (args.output is None or not args.output.exists()), 'Use unoptimized Python and preserve output')
    cohort = read(BUNDLE / 'cohort.json')['records']
    packed = b''
    for part in cohort['parts']:
        check(Path(part['path']).parts[0] == 'records' and '..' not in Path(part['path']).parts, 'Unsafe part path')
        data = (BUNDLE / part['path']).read_bytes()
        check(len(data) == part['bytes'] and sha(data) == part['sha256'], 'Stored part differs: ' + part['path'])
        packed += data
    check(sha(packed) == cohort['xz_sha256'], 'Joined XZ stream differs')
    raw = lzma.decompress(packed)
    check(len(raw) == cohort['decoded_bytes'] and sha(raw) == cohort['decoded_sha256'], 'Decoded container differs')
    payload = json.loads(raw)
    check(payload['schema'] == 'generativeqc.h100-fixed30-receipts.v1', 'Unknown receipt container')
    entries, blobs = payload['files'], payload['blobs']
    check(len(entries) == cohort['files'] and len(blobs) == cohort['distinct_blobs'] and
          {e['sha256'] for e in entries} == set(blobs), 'Member/blob population differs')
    values = {}
    for digest, blob in blobs.items():
        data = blob['data'].encode('utf8') if blob['encoding'] == 'utf8' else base64.b64decode(blob['data'], validate=True)
        check(blob['encoding'] in ['utf8', 'base64'] and len(data) == blob['bytes'] and sha(data) == digest, 'Receipt blob differs')
        values[digest] = data
    with tempfile.TemporaryDirectory(prefix='gqc-h100-fixed30-d35-') as name:
        task = Path(name) / 'task'
        for entry in entries:
            parts = entry['path'].split('/')
            check(not entry['path'].startswith('/') and all(p not in ['', '.', '..'] for p in parts) and
                  '\\' not in entry['path'] and ':' not in entry['path'], 'Unsafe receipt path')
            target = task / entry['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(values[entry['sha256']])
        build = read(task / 'build-receipts-v1/build.json')
        revision = build['source']['revision']
        common = subprocess.check_output(['git', '-C', str(args.repository), 'rev-parse', '--path-format=absolute', '--git-common-dir'], text=True).strip()
        env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_ALLOW_PROTOCOL': '', 'GIT_TERMINAL_PROMPT': '0',
               'GIT_ALTERNATE_OBJECT_DIRECTORIES': str(Path(common) / 'objects'),
               'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}
        helpers = pinned_source(args.repository, revision, Path(name) / 'helpers', env)
        output = Path(name) / 'analysis.json'
        result = subprocess.run([sys.executable, str(task / 'analyze-fixed30-d35-v1.py'), '--task', str(task),
                                 '--repository', str(helpers), '--output', str(output),
                                 '--reference-catalog', str(task / 'reference-installation-history-v1.json')],
                                env=env, capture_output=True, text=True, timeout=600)
        check(result.returncode == 0, 'Offline reduction failed\n' + result.stderr[-3000:])
        value = read(output)
        check(value == read(task / 'fixed30-analysis-d35-v1.json'), 'Offline reduction differs from sealed reduction')
        summary = compact(value)
        check(summary == read(BUNDLE / 'summary.json'), 'Published summary differs from full raw reduction')
        if args.output:
            with args.output.open('x') as stream:
                json.dump(summary, stream, indent=2, allow_nan=False)
                stream.write('\n')
        print(json.dumps({'status': 'PASS_ALL_RECEIPT_BYTES_AND_OFFLINE_REDUCTION', 'files': len(entries),
                          **{k: summary[k] for k in ['status', 'source_revision', 'qualified_populations', 'returned_EF',
                                                     'science_pass', 'summary']}}))


if __name__ == '__main__':
    main()
