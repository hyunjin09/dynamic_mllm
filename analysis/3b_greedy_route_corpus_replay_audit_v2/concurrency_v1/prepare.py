"""Freeze scheduling and pilot references without touching an active replay."""
from runner import *


def main():
    assert not (HERE / 'contract.json').exists(), 'Preserve existing frozen overlay'
    c = read_execution_contract()
    index = list(rows(OUT / 'source_inventory/prepared_index.jsonl'))
    weights = {}
    complete = []
    for item in index:
        uid = item['uid']
        dense = read_json(OUT / 'dense/by_sample' / f'{stem(uid)}.json')
        path = OUT / 'replay/by_sample' / f'{stem(uid)}.json'
        payload = read_json(path) if path.exists() else None
        remaining = item['routes'] - (len(payload['records']) if payload else 1)
        weights[uid] = max(0, remaining) * max(0.01, dense['runtime_seconds'])
        if payload and payload['complete']:
            complete.append((item, path, dense))
    assignments, loads = assign(index, weights)
    # Freeze24 distinct completed samples covering available datasets and input
    # cost quantiles. Full32-anchor and sparse gates additionally cover all4 datasets.
    by_dataset = collections.defaultdict(list)
    for item, path, dense in complete:
        by_dataset[item['dataset']].append((item, path, dense))
    selected = []
    datasets = sorted(by_dataset)
    for dataset in datasets:
        values = sorted(by_dataset[dataset], key=lambda v: (v[2]['visual_tokens'], v[0]['uid']))
        count = WORKERS // len(datasets) + (datasets.index(dataset) < WORKERS % len(datasets))
        assert len(values) >= count
        selected.extend(values[round(i*(len(values)-1)/(count-1))] for i in range(count))
    assert len({v[0]['uid'] for v in selected}) == WORKERS
    # Spread the cost quantiles over GPUs and replicas deterministically.
    selected.sort(key=lambda v: (v[2]['visual_tokens'], v[0]['uid']))
    pilots = []
    for worker, (item, path, dense) in enumerate(selected):
        reference = HERE / 'pilot_references' / path.name
        atomic_json(reference, read_json(path), immutable=True)
        pilots.append(dict(item=item, reference_path=str(reference), reference_sha256=file_hash(reference),
            visual_tokens=dense['visual_tokens'], worker=worker, gpu=worker % 8))
    schedule = dict(assignments=assignments, weights=weights, estimated_loads=loads, pilots=pilots,
        assignment_rule='24 disjoint LPT shards; estimate remaining routes times current dense inference seconds; original source ranks retained',
        pilot_datasets=datasets, total_samples=len(index), total_routes=sum(r['routes'] for r in index))
    atomic_json(HERE / 'schedule.json', schedule, immutable=True)
    overlay = dict(schema='three_processes_per_gpu_scheduling_overlay_v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous_job=2956,
        inference_contract_sha256=c['contract_sha256'], processes_per_gpu=3, gpus=8, workers=24,
        cpu_threads_per_worker=4, requested_cpus=96, requested_memory_gib=384,
        schedule_sha256=file_hash(HERE / 'schedule.json'),
        code_hashes={str(p): file_hash(p) for p in sorted(HERE.glob('*')) if p.is_file() and p.suffix in ['.py', '.slurm']},
        canonical_record_contract='Unchanged inference hash; new records add overlay hash and worker/GPU IDs; saved records reused byte-equivalently',
        gate='Exact32 HF/custom anchors, four sparse probes twice on24workers matching prior sparse outputs, then24 complete saved samples exactly',
        stop_rule='Any parity, hash, duplicate, runtime or stale-worker failure stops the job and blocks its report',
        authorization='User requested3 processes/GPU, brief sample monitoring and measured ETA; original V2 completion remains authorized')
    overlay['overlay_sha256'] = digest(overlay)
    atomic_json(HERE / 'contract.json', overlay, immutable=True)
    print(json.dumps(dict(overlay_sha256=overlay['overlay_sha256'], workers=WORKERS,
        samples=len(index), estimated_worker_seconds_min=min(loads), estimated_worker_seconds_max=max(loads),
        pilot_datasets=datasets, pilot_visual_tokens=[p['visual_tokens'] for p in pilots])))


if __name__ == '__main__':
    main()
