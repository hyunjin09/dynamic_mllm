"""Verify explicitly mapped external assets; never copy, download or launch work."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def checked_path(value, roots, depth=0):
    path = Path(os.path.abspath(value))
    base = next((root for root in roots if path.is_relative_to(root)), None)
    if base is None or depth > 32:
        raise ValueError(f'Path outside explicitly allowed roots: {path}')
    current = base
    for part in path.relative_to(base).parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return path
        if stat.S_ISLNK(mode):
            target = Path(os.readlink(current))
            if not target.is_absolute():
                target = current.parent / target
            return checked_path(target / path.relative_to(current), roots, depth+1)
    return path


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', required=True)
    parser.add_argument('--model-root', required=True)
    parser.add_argument('--allowed-root', action='append', default=[],
                        help='Repeat only roots permitted by the destination ACCESS_POLICY.md')
    args = parser.parse_args()
    manifest = json.loads((HERE/'external_assets.json').read_text())
    if file_hash(HERE/'external_files.jsonl.gz') != manifest['inventory_sha256']:
        raise ValueError('External inventory checksum mismatch')
    roots = [ROOT] + [Path(os.path.abspath(p)) for p in args.allowed_root]
    mapping = dict(project=ROOT, package=checked_path(args.package_root, roots), model=checked_path(args.model_root, roots))
    verified = 0
    failures = []
    with gzip.open(HERE/'external_files.jsonl.gz', 'rt') as handle:
        for line in handle:
            entry = json.loads(line)
            relative = PurePosixPath(entry['relative_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe inventory relative path')
            path = checked_path(mapping[entry['root']] / relative, roots)
            reason = None
            if not path.is_file():
                reason = 'missing'
            elif path.stat().st_size != entry['bytes']:
                reason = 'size_mismatch'
            elif file_hash(path) != entry['sha256']:
                reason = 'hash_mismatch'
            if reason:
                failures.append(dict(root=entry['root'], path=str(relative), reason=reason))
            else:
                verified += 1
    if verified + len(failures) != manifest['files']:
        raise ValueError('External inventory count mismatch')
    print(json.dumps(dict(verified_files=verified, failed_files=len(failures), failures=failures), indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
