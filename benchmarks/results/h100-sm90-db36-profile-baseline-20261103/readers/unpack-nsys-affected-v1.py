"""Recover authenticated small profile metadata/SQLite, retaining identical files."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile

TASK = Path(__file__).resolve().parent


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    check(Path(args.manifest).name == args.manifest, 'Flat manifest name required')
    manifest = json.loads((TASK / args.manifest).read_text())
    archive_binding = manifest['archive']
    if isinstance(archive_binding, str):
        archive_binding = {'name': archive_binding, 'bytes': manifest['bytes'], 'sha256': manifest['sha256']}
    archive = TASK / archive_binding['name']
    check(archive.parent == TASK and archive.stat().st_size == archive_binding['bytes'] and sha(archive) == archive_binding['sha256'], 'Archive bytes differ')
    roots = {'nsys-v1', 'nsys-v1-executed-producers', 'nsys-v1-unverified-producers', 'nsys-v1-available-producers', 'nsys-v1-archival-admissions', 'nsys-v1-archival-producers'}
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        check(len(members) == len(manifest['files']) and {m.name for m in members} == set(manifest['files']), 'Duplicate/missing member')
        preserved = []
        for member in members:
            name = Path(member.name)
            binding = manifest['files'][member.name]
            check(member.isfile() and not name.is_absolute() and '..' not in name.parts and name.parts[0] in roots and name.name != '.env' and member.size == binding['bytes'], 'Unsafe member')
            target = TASK / name
            if target.exists():
                check(target.is_file() and not target.is_symlink() and target.stat().st_size == binding['bytes'] and sha(target) == binding['sha256'], 'Existing original differs')
                preserved.append(member.name)
        for member in members:
            if member.name not in preserved:
                stream.extract(member, TASK, filter='data')
    check(all((TASK / name).stat().st_size == binding['bytes'] and sha(TASK / name) == binding['sha256'] for name, binding in manifest['files'].items()), 'Recovered byte differs')
    print(json.dumps({'status': 'RECOVERED_EXACT_SMALL_PROFILE_ORIGINALS_ONLY', 'members': len(members), 'existing_identical_preserved': len(preserved)}))


if __name__ == '__main__':
    main()
