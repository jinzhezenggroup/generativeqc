"""Recover a terminal native packet, preserving every original attempt."""
from pathlib import Path
import hashlib
import json
import tarfile

TASK = Path(__file__).resolve().parent
COHORT = 'spatial-racecheck-v2'


def check(value, message):
    if not value:
        raise ValueError(message)


def main():
    manifest = json.loads((TASK / (COHORT + '-download-manifest.json')).read_text())
    archive = TASK / manifest['archive']
    check(archive.parent == TASK and archive.stat().st_size == manifest['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['sha256'], 'Archive bytes differ')
    allowed = {COHORT, COHORT + '-executed-producers', COHORT + '-available-admissions', COHORT + '-archival-admissions', COHORT + '-archival-producers'}
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        check(len(members) == len(manifest['files']) and {m.name for m in members} == set(manifest['files']), 'Duplicate or missing member')
        for member in members:
            path = Path(member.name)
            check(member.isfile() and not path.is_absolute() and '..' not in path.parts and path.parts[0] in allowed and path.name != '.env' and not (TASK / path).exists() and member.size == manifest['files'][member.name]['bytes'], 'Unsafe or existing extraction target')
        stream.extractall(TASK, filter='data')
    check(all((TASK / name).stat().st_size == row['bytes'] and hashlib.sha256((TASK / name).read_bytes()).hexdigest() == row['sha256'] for name, row in manifest['files'].items()), 'Recovered raw differs')
    status_path = TASK / (COHORT + '-archival-admissions/spatial-racecheck-terminal-status-v2.json')
    status = json.loads(status_path.read_text())
    check(hashlib.sha256(status_path.read_bytes()).hexdigest() == manifest['platform_status_sha256'] and status['success'] is True and status['data']['name'] == 'gqc-h100-finaldb36-spatial-race-1103-v2' and status['data']['status'] == manifest['platform_status'], 'Exact platform response differs')
    print(json.dumps({'status': 'RECOVERED_EXACT_RACECHECK_CONTINUATION_ONLY', 'platform_status': manifest['platform_status'], 'members': len(members)}))


if __name__ == '__main__':
    main()
