"""Build this stopped-server handoff once; never mutate original evidence."""
import collections
from concurrent.futures import ThreadPoolExecutor
import datetime
import gzip
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / 'analysis/3b_greedy_route_corpus_replay_audit'
V2 = ROOT / 'analysis/3b_greedy_route_corpus_replay_audit_v2'
sys.path.insert(0, str(V2/'concurrency_v1'))
from runner import read_overlay, validate_resume
from common import PACKAGE, MODEL, allowed_path, file_hash, digest, rows, read_json, stem


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')


def main():
    assert not (HERE/'metadata_manifest.json').exists(), 'Preserve the existing snapshot'
    queue = subprocess.check_output(['squeue','-h','-j','2994,2957','-o','%i|%T'], text=True)
    assert not queue.strip(), 'Source replay/report must be stopped'
    c, overlay = read_overlay()
    index = list(rows(V2/'source_inventory/prepared_index.jsonl'))
    source_map = {r['uid']: r for r in index}
    snapshot = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_checkout=str(ROOT), source_package=str(PACKAGE), source_model=str(MODEL),
        git_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        inference_contract_sha256=c['contract_sha256'], concurrency_overlay_sha256=overlay['overlay_sha256'],
        status='STOPPED_BY_USER_INCOMPLETE', restart_authorized=False, total_samples=10000, total_routes=3907717,
        dense_stage=read_json(V2/'dense/stage_complete.json'), slurm_accounting=subprocess.check_output(
            ['sacct','-j','2939,2940,2941,2942,2954,2955,2956,2957,2994',
             '--format=JobID,JobName,State,ExitCode,Elapsed,Start,End','-P'],text=True))
    regular_files = []
    external = {}

    def add_external(kind, relative, path, role, expected=None):
        key = (kind, str(relative))
        path = allowed_path(path)
        assert path.is_file(), path
        entry = dict(root=kind, relative_path=str(relative), source_path=str(path), role=role,
                     bytes=path.stat().st_size, sha256=expected)
        if key in external:
            assert external[key]['bytes'] == entry['bytes']
            if expected and external[key]['sha256']:
                assert external[key]['sha256'] == expected
            elif expected:
                external[key]['sha256'] = expected
        else:
            external[key] = entry

    for base in [V1,V2]:
        for path in sorted(base.rglob('*')):
            rel = path.relative_to(base)
            if any(p in ['cache','tmp','__pycache__'] for p in rel.parts) or '.tmp.' in path.name or path.name == 'coordinator.lock':
                continue
            assert not path.is_symlink(), path
            if not path.is_file():
                continue
            if (base == V1 and rel.parts[:2] == ('source_inventory','by_sample')) or (base == V2 and rel.parts[:2] == ('replay','by_sample')):
                add_external('project', path.relative_to(ROOT), path,
                             'prepared_source_shard' if base==V1 else 'partial_replay_checkpoint')
            else:
                regular_files.append(path)

    # Validate every durable route record against its frozen source before
    # advertising an exact stopped frontier. This does not interpret partial labels.
    counters = collections.Counter()
    by_dataset = collections.defaultdict(collections.Counter)
    job_counts = collections.Counter()
    for path in sorted((V2/'replay/by_sample').glob('*.json')):
        payload = read_json(path)
        assert path.name == stem(payload['sample_uid']) + '.json'
        item = source_map[payload['sample_uid']]
        source = {r['route_key']: r for r in rows(item['path'])}
        records = validate_resume(payload, source, c['contract_sha256'], item['uid'])
        dense = read_json(V2/'dense/by_sample'/f"{stem(item['uid'])}.json")
        assert next(r for r in records if r['route_key']==item['dense_route_key']) == dense
        errors = sum(r['replay_status']!='COMPLETE' for r in records)
        counts = dict(checkpoint_files=1, saved_records=len(records), error_records=errors,
                      complete_samples=int(payload['complete']), partial_samples=int(not payload['complete']))
        counters.update(counts)
        by_dataset[item['dataset']].update(counts)
        job_counts.update(str(r['slurm_job_id']) for r in records)
    snapshot.update(route_checkpoint_counts=dict(counters), route_counts_by_dataset=dict(by_dataset),
                    route_record_jobs=dict(job_counts), untouched_samples=10000-counters['checkpoint_files'],
                    remaining_routes=3907717-counters['saved_records'], record_identity_hash_validation_passed=True,
                    additional_dense_records_available=10000-counters['checkpoint_files'],
                    unique_route_results_available=counters['saved_records']+10000-counters['checkpoint_files'],
                    remaining_new_route_inferences=3907717-counters['saved_records']-10000+counters['checkpoint_files'],
                    final_report_complete=False, geometry_readiness='NOT ASSESSED: full replay and interpretation pending')

    for item in index:
        add_external('project', Path(item['path']).relative_to(ROOT), item['path'], 'prepared_source_shard', item['sha256'])
    import csv
    for entry in csv.DictReader((V2/'source_inventory/file_inventory.csv').open()):
        add_external('package', entry['relative_path'], PACKAGE/entry['relative_path'], entry['role'], entry['sha256'])
    for path in [PACKAGE/'README.md', PACKAGE/'00_METADATA/PACKAGE_SUMMARY.json', PACKAGE/'00_METADATA/PATH_MAP.json']:
        add_external('package', path.relative_to(PACKAGE), path, 'package_documentation')
    for sample in rows(V2/'source_inventory/relocated_samples.jsonl'):
        path = allowed_path(sample['local_image_path'])
        add_external('package', path.relative_to(PACKAGE), path, 'input_image', sample['image_content_sha256'])
    runtime_files = []
    for name, expected in c['runtime_source_hashes'].items():
        path = Path(name)
        if path.is_relative_to(PACKAGE):
            relative = path.relative_to(PACKAGE)
            add_external('package', relative, path, 'frozen_executor_source', expected)
            destination = HERE/'runtime_snapshot'/path.relative_to(PACKAGE/'07_RESUME_TOOLS')
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert not destination.exists()
            shutil.copyfile(allowed_path(path), destination)
            runtime_files.append(dict(source=str(path), git_path=str(destination.relative_to(ROOT)), sha256=expected))
    for name, expected in c['small_model_hashes'].items():
        add_external('model', name, MODEL/name, 'model_processor_metadata', expected)
    for name in c['weight_sizes']:
        add_external('model', name, MODEL/name, 'model_weight_shard')

    def verify(entry):
        actual = file_hash(entry['source_path'])
        if entry['sha256']:
            assert actual == entry['sha256'], entry['source_path']
        return dict(entry, sha256=actual, source_verified_at_packaging=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(verify, sorted(external.values(), key=lambda e:(e['root'],e['relative_path']))))
    with (HERE/'external_files.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            for entry in verified:
                zipped.write((json.dumps(entry,sort_keys=True)+'\n').encode())
    totals = collections.defaultdict(lambda:dict(files=0, bytes=0))
    for entry in verified:
        total = totals[entry['root']+'/'+entry['role']]
        total['files'] += 1
        total['bytes'] += entry['bytes']
    save(HERE/'external_assets.json', dict(source_roots=dict(project=str(ROOT),package=str(PACKAGE),model=str(MODEL)),
         totals=dict(totals), files=len(verified), total_bytes=sum(e['bytes'] for e in verified),
         inventory_sha256=file_hash(HERE/'external_files.jsonl.gz'), transferred=False))
    (HERE/'transfer_lists').mkdir(exist_ok=True)
    for kind in ['project','package','model']:
        (HERE/'transfer_lists'/f'{kind}.txt').write_text('\n'.join(e['relative_path'] for e in verified if e['root']==kind)+'\n')
    save(HERE/'progress_snapshot.json', snapshot)
    save(HERE/'runtime_snapshot_manifest.json',dict(files=runtime_files, destination_installation='Reference only; canonical package paths are not automatically rewritten.'))
    import torch
    save(HERE/'environment_snapshot.json', dict(python=sys.version, torch=torch.__version__, torch_cuda=torch.version.cuda,
         distributions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
         gpu='8 x NVIDIA H100 80GB HBM3', driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines()[0],
         uv=subprocess.check_output(['uv','--version'],text=True).strip(), source_policy_only='Slurm GPU, local CPU by default',
         copy_machine_policies=False, destination_policy_precedence=True))

    entries = []
    archive_path = HERE/'metadata.tar.gz'
    with archive_path.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as tar:
                for path in regular_files:
                    relative = str(path.relative_to(ROOT))
                    data = path.read_bytes()
                    storage = 'git' if path.suffix in ['.py','.md','.sh','.slurm'] else 'archive'
                    mode = path.stat().st_mode & 0o777
                    entry = dict(path=relative, bytes=len(data), sha256=file_hash(path), mode=mode, storage=storage)
                    entries.append(entry)
                    if storage == 'archive':
                        info = tarfile.TarInfo(relative)
                        info.size=len(data);info.mode=mode;info.mtime=0
                        tar.addfile(info,io.BytesIO(data))
    save(HERE/'metadata_manifest.json',dict(schema='phase89_stopped_server_snapshot_v1',
         archive=dict(path=str(archive_path.relative_to(ROOT)),sha256=file_hash(archive_path),bytes=archive_path.stat().st_size),
         files=entries, raw_partial_replay_included=False, dense_label_records_included=True))
    (HERE/'included_git_paths.txt').write_text('\n'.join(e['path'] for e in entries if e['storage']=='git')+'\n')
    print(json.dumps(dict(metadata_files=len(entries), metadata_archive_bytes=archive_path.stat().st_size,
                         external_files=len(verified), external_bytes=sum(e['bytes'] for e in verified), stopped_frontier=dict(counters))))


if __name__ == '__main__':
    main()
