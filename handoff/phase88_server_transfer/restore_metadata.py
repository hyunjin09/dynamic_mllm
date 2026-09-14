"""Restore the exact compressed Phase88 metadata; never replace different data."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((Path(__file__).parent / 'metadata_manifest.json').read_text())
    pending = []
    for entry in manifest['files']:
        source = root / entry['git_storage']
        destination = root / entry['path']
        for path in (source, destination):
            if not path.resolve().is_relative_to(root):
                raise ValueError(f'Path leaves repository: {path}')
        payload = source.read_bytes()
        if entry['encoding'] == 'gzip':
            payload = gzip.decompress(payload)
        elif entry['encoding'] != 'identity':
            raise ValueError(f'Unknown encoding: {entry["encoding"]}')
        if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError(f'Snapshot hash mismatch: {source}')
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Existing metadata differs; preserve it and reconcile: {destination}')
        else:
            pending.append((destination, payload))
    # Validate the entire snapshot before writing anything.
    if not args.check_only:
        for destination, payload in pending:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(payload)
    print(json.dumps({'verified_files': len(manifest['files']),
                      'missing_files': len(pending), 'restored_files': 0 if args.check_only else len(pending),
                      'raw_uid_records_included': False}))


if __name__ == '__main__':
    main()
