"""Verify/restore exact Phase89 metadata without overwriting existing differences."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def local_path(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe snapshot path: {relative}')
    current = ROOT
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Refuse symlink in snapshot path: {current}')
    return current


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_entry(entry, data):
    if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
        raise ValueError(f'Snapshot hash mismatch: {entry["path"]}')
    destination = local_path(entry['path'])
    if destination.exists() and (not destination.is_file() or sha(destination.read_bytes()) != entry['sha256']):
        raise ValueError(f'Existing file differs; preserve and reconcile: {destination}')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'metadata_manifest.json').read_text())
    archive = local_path(manifest['archive']['path'])
    if sha(archive.read_bytes()) != manifest['archive']['sha256']:
        raise ValueError('Metadata archive checksum mismatch')
    bundled = {e['path']: e for e in manifest['files'] if e['storage'] == 'archive'}
    if len({e['path'] for e in manifest['files']}) != len(manifest['files']):
        raise ValueError('Duplicate metadata manifest entries')
    pending = set()
    for entry in manifest['files']:
        if entry['storage'] == 'git':
            verify_entry(entry, local_path(entry['path']).read_bytes())
        elif entry['storage'] != 'archive':
            raise ValueError('Unknown storage type')
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in bundled or member.name in seen:
                raise ValueError(f'Unexpected archive member: {member.name}')
            seen.add(member.name)
            data = tar.extractfile(member).read()
            destination = verify_entry(bundled[member.name], data)
            if not destination.exists():
                pending.add(member.name)
    if seen != set(bundled):
        raise ValueError('Archive membership does not match manifest')
    # The entire bundle and all existing destinations must pass before writes.
    if not args.check_only:
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                if member.name not in pending:
                    continue
                destination = local_path(member.name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = tar.extractfile(member).read()
                verify_entry(bundled[member.name], data)
                with destination.open('xb') as handle:
                    handle.write(data)
                destination.chmod(bundled[member.name]['mode'])
    print(json.dumps(dict(verified_files=len(manifest['files']), missing_files=len(pending),
        restored_files=0 if args.check_only else len(pending), inference_launched=False,
        complete_route_replay=False, note='Metadata validity does not establish destination runtime parity.')))


if __name__ == '__main__':
    main()
