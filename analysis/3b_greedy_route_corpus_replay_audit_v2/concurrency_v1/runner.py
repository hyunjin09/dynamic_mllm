"""Three replicas/GPU scheduling overlay; frozen inference and records are reused."""
import argparse
import collections
import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'code'))
from common import *
from execution import read_execution_contract
from replay import Engine, gate_passes, replay_record, require_dense_record, validate_resume

WORKERS = 24


def assign(items, weights):
    """Disjoint LPT assignment; source ranks and source order are never rewritten."""
    result = [[] for _ in range(WORKERS)]
    loads = [0.0] * WORKERS
    for item in sorted(items, key=lambda x: (-weights[x['uid']], x['uid'])):
        worker = min(range(WORKERS), key=lambda k: (loads[k], len(result[k]), k))
        result[worker].append(item['uid'])
        loads[worker] += weights[item['uid']]
    assert len({uid for part in result for uid in part}) == len(items)
    return result, loads


def read_overlay():
    c = read_execution_contract()
    overlay = read_json(HERE / 'contract.json')
    assert digest({k: v for k, v in overlay.items() if k != 'overlay_sha256'}) == overlay['overlay_sha256']
    assert overlay['inference_contract_sha256'] == c['contract_sha256']
    for path, expected in overlay['code_hashes'].items():
        assert file_hash(path) == expected, path
    assert file_hash(HERE / 'schedule.json') == overlay['schedule_sha256']
    for item in read_json(HERE / 'schedule.json')['pilots']:
        assert file_hash(item['reference_path']) == item['reference_sha256']
    return c, overlay


def same_result(actual, reference):
    return (actual['generated_ids'] == reference['current_generated_ids']
            and actual['answer'] == reference['current_answer']
            and actual['score'] == reference['current_score']
            and actual['correctness'] == reference['current_correctness']
            and actual['action_trace'] == reference['action_trace']
            and actual['visual_tokens'] == reference['visual_tokens']
            and actual['text_tokens'] == reference['text_tokens'])


def barrier(path, failures, seconds=1800):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if list(failures.glob('worker*_failure.json')):
            raise RuntimeError('Peer failed; preserving evidence and stopping')
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(1)


def preflight(engine, c, overlay, samples, worker, target):
    anchors = list(rows(OUT / 'replay_gate/frozen_anchor_rows.jsonl'))
    records = []
    for i, anchor in enumerate(anchors):
        if i % WORKERS != worker:
            continue
        sample = samples[anchor['uid']]
        inputs, prepared = engine.inputs(sample)
        hf = engine.native(sample, inputs)
        custom = engine.binary(sample, inputs, '1' * LAYERS, prepared)
        records.append(dict(uid=sample['uid'], passed=all(hf[k] == custom[k] for k in hf), native=hf, custom=custom))
        del inputs, prepared
    write_jsonl(target / f'anchors_worker{worker}.jsonl', records)
    probes = []
    for probe in rows(OUT / 'replay_gate/frozen_sparse_probes.jsonl'):
        sample = samples[probe['uid']]
        inputs, prepared = engine.inputs(sample)
        first = engine.binary(sample, inputs, probe['source_record']['mask_key'], prepared)
        second = engine.binary(sample, inputs, probe['source_record']['mask_key'], prepared)
        probes.append(dict(uid=sample['uid'], route_key=probe['route_key'], first=first, second=second, passed=first == second))
        del inputs, prepared
    old = read_json(OUT / 'replay_gate/job_2956_routes/sparse_rank0.json')
    sparse = dict(passed=all(r['passed'] for r in probes) and probes == old['outputs'], outputs=probes)
    atomic_json(target / f'sparse_worker{worker}.json', sparse)
    atomic_json(target / f'gate_worker{worker}_complete.json', dict(passed=sparse['passed']))
    barrier(target / 'gate_decision.json', target)
    assert read_json(target / 'gate_decision.json')['passed'], '24-worker parity gate failed'


def pilot(engine, c, samples, item, worker, target):
    source_item = item['item']
    source = {r['route_key']: r for r in rows(source_item['path'])}
    assert file_hash(source_item['path']) == source_item['sha256']
    payload = read_json(item['reference_path'])
    records = validate_resume(payload, source, c['contract_sha256'], source_item['uid'])
    assert payload['complete'] and all(r['replay_status'] == 'COMPLETE' for r in records)
    barrier(target / 'pilot_start.json', target)
    started = time.monotonic()
    inputs, prepared = engine.inputs(samples[source_item['uid']])
    exact = 0
    for reference in records:
        actual = engine.binary(samples[source_item['uid']], inputs, reference['mask_key'], prepared)
        if not same_result(actual, reference):
            atomic_json(target / f'pilot_worker{worker}_mismatch.json', dict(reference=reference, actual=actual))
            raise RuntimeError('Concurrent pilot differs from saved current-server replay')
        exact += 1
    elapsed = time.monotonic() - started
    atomic_json(target / f'pilot_worker{worker}.json', dict(passed=True, worker=worker, gpu=worker % 8,
        uid=source_item['uid'], routes=exact, elapsed_seconds=elapsed,
        old_serial_inference_seconds=sum(r['runtime_seconds'] for r in records),
        max_gpu_memory_reserved=engine.torch.cuda.max_memory_reserved(engine.device)))
    del inputs, prepared
    barrier(target / 'pilot_decision.json', target)
    assert read_json(target / 'pilot_decision.json')['passed']


def worker_main(worker, job):
    c, overlay = read_overlay()
    target = HERE / f'job_{job}'
    schedule = read_json(HERE / 'schedule.json')
    index = {r['uid']: r for r in rows(OUT / 'source_inventory/prepared_index.jsonl')}
    samples = {r['uid']: r for r in rows(OUT / 'source_inventory/relocated_samples.jsonl')}
    dense_stage = read_json(OUT / 'dense/stage_complete.json')
    assert dense_stage['passed'] and dense_stage['contract_sha256'] == c['contract_sha256']
    assert file_hash(OUT / 'dense/current_dense_manifest.jsonl') == dense_stage['manifest_sha256']
    engine = Engine(c, worker % 8)
    preflight(engine, c, overlay, samples, worker, target)
    pilot(engine, c, samples, schedule['pilots'][worker], worker, target)
    started = time.monotonic()
    snapshot = read_json(target / 'resume_snapshot.json')
    stats = {str(k): dict(samples=0, success=0, reused=0) for k in range(8)}
    for uid in schedule['assignments'][worker]:
        prior = snapshot[uid]
        bucket = stats[str(index[uid]['rank'])]
        bucket['samples'] += prior['complete']
        bucket['reused'] += prior['records']
    new_samples = 0
    completed_cost = 0.0

    def progress(current_uid=None, complete=False):
        result = dict(worker=worker, gpu=worker % 8, job_id=job, contract_sha256=c['contract_sha256'],
            overlay_sha256=overlay['overlay_sha256'], by_source_rank=stats, new_samples=new_samples,
            completed_estimated_seconds=completed_cost, current_uid=current_uid,
            elapsed_seconds=time.monotonic()-started, updated_unix=time.time(), complete=complete)
        atomic_json(target / f'worker{worker}_progress.json', result)
        if complete:
            atomic_json(target / f'worker{worker}_complete.json', dict(result, passed=True))

    progress()
    for uid in schedule['assignments'][worker]:
        item = index[uid]
        rank = item['rank']
        bucket = stats[str(rank)]
        assert file_hash(item['path']) == item['sha256']
        source_list = list(rows(item['path']))
        source = {r['route_key']: r for r in source_list}
        assert len(source) == item['routes']
        dense = require_dense_record(read_json(OUT / 'dense/by_sample' / f'{stem(uid)}.json'), item, source, c)
        path = OUT / 'replay/by_sample' / f'{stem(uid)}.json'
        existing = read_json(path) if path.exists() else dict(sample_uid=uid, contract_sha256=c['contract_sha256'], records=[dense], complete=False)
        records = validate_resume(existing, source, c['contract_sha256'], uid)
        assert all(r['replay_status'] == 'COMPLETE' for r in records), 'Preserved failure requires diagnosis'
        assert next(r for r in records if r['route_key'] == item['dense_route_key']) == dense
        assert len(records) >= snapshot[uid]['records']
        bucket['reused'] += len(records) - snapshot[uid]['records']
        if existing['complete']:
            assert snapshot[uid]['complete']
            progress()
            continue
        done = {r['route_key'] for r in records}
        sample = samples[uid]
        inputs, prepared = engine.inputs(sample)
        progress(uid)
        for source_item in source_list:
            if source_item['route_key'] in done:
                continue
            record = replay_record(engine, c, sample, source_item, source[item['dense_route_key']]['source_record'],
                                   inputs, prepared, rank, job, 'routes')
            record.update(concurrency_overlay_sha256=overlay['overlay_sha256'], worker_id=worker, gpu_index=worker % 8)
            record['record_sha256'] = digest({k: v for k, v in record.items() if k != 'record_sha256'})
            records.append(record)
            done.add(record['route_key'])
            bucket['success'] += record['replay_status'] == 'COMPLETE'
            if len(records) % c['checkpoint_routes'] == 0 or record['replay_status'] == 'ERROR':
                atomic_json(path, dict(existing, records=records, complete=False))
                progress(uid)
            if record['replay_status'] == 'ERROR':
                raise RuntimeError('Replay error preserved; stopping peers')
        assert done == set(source)
        atomic_json(path, dict(existing, records=records, complete=True))
        bucket['samples'] += 1
        new_samples += 1
        completed_cost += schedule['weights'][uid]
        progress()
        print('sample_complete', worker, uid, new_samples, sum(v['success'] for v in stats.values()), flush=True)
        del inputs, prepared
    progress(complete=True)


def aggregate(progress, c, overlay, job, total_by_rank, complete=False):
    combined = {k: dict(samples=0, success=0, reused=0) for k in range(8)}
    for record in progress:
        assert record['job_id'] == job and record['overlay_sha256'] == overlay['overlay_sha256']
        for rank, values in record['by_source_rank'].items():
            for name in ['samples', 'success', 'reused']:
                combined[int(rank)][name] += values[name]
    if complete:
        assert len(progress) == WORKERS and all(r['complete'] for r in progress)
        for rank, values in combined.items():
            assert values['samples'] == total_by_rank[rank]['samples']
            assert values['success'] + values['reused'] == total_by_rank[rank]['routes']
    for rank, values in combined.items():
        result = dict(values, total_samples=total_by_rank[rank]['samples'], job_id=job, stage='routes',
            contract_sha256=c['contract_sha256'], concurrency_overlay_sha256=overlay['overlay_sha256'],
            elapsed_seconds=max((r['elapsed_seconds'] for r in progress), default=0),
            aggregation='24 disjoint workers grouped by original source rank')
        atomic_json(OUT / 'replay' / f'rank{rank}_progress.json', result)
        if complete:
            atomic_json(OUT / 'replay' / f'rank{rank}_complete.json', dict(result, passed=True))
    return combined


def coordinate():
    assert os.environ.get('SLURM_JOB_ID')
    job = os.environ['SLURM_JOB_ID']
    c, overlay = read_overlay()
    target = HERE / f'job_{job}'
    target.mkdir(parents=True, exist_ok=True)
    lock = (HERE / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not list(target.glob('worker*')), 'Never reuse job gate or worker evidence'
    old_state = subprocess.check_output(['sacct', '-n', '-X', '-j', str(overlay['previous_job']), '--format=State', '-P'], text=True).strip()
    assert old_state and not any(s in old_state for s in ['RUNNING', 'PENDING', 'COMPLETING']), old_state
    # Preserve legacy progress before replacing only the aggregate status files.
    for path in sorted((OUT / 'replay').glob('rank*_progress.json')):
        atomic_json(target / 'prior_progress' / path.name, read_json(path), immutable=True)
    atomic_json(target / 'previous_global_gate.json', read_json(OUT / 'replay_gate/gate_result.json'), immutable=True)
    index = list(rows(OUT / 'source_inventory/prepared_index.jsonl'))
    snapshot = {}
    for item in index:
        path = OUT / 'replay/by_sample' / f"{stem(item['uid'])}.json"
        payload = read_json(path) if path.exists() else None
        snapshot[item['uid']] = dict(records=len(payload['records']) if payload else 0,
            complete=bool(payload and payload['complete']))
    atomic_json(target / 'resume_snapshot.json', snapshot, immutable=True)
    total_by_rank = {k: dict(samples=sum(r['rank'] == k for r in index), routes=sum(r['routes'] for r in index if r['rank'] == k)) for k in range(8)}
    processes = []
    streams = []
    try:
        for worker in range(WORKERS):
            log = (target / f'worker{worker}.log').open('w')
            streams.append(log)
            processes.append(subprocess.Popen([sys.executable, '-B', str(HERE / 'runner.py'), '--worker', str(worker)], stdout=log, stderr=subprocess.STDOUT))
        start = time.monotonic()
        while True:
            for worker, process in enumerate(processes):
                if process.poll() not in [None, 0]:
                    raise RuntimeError(f'Worker {worker} failed: exit {process.returncode}')
            if not (target / 'gate_decision.json').exists() and len(list(target.glob('gate_worker*_complete.json'))) == WORKERS:
                anchors = list(rows(OUT / 'replay_gate/frozen_anchor_rows.jsonl'))
                ar = [r for k in range(WORKERS) for r in rows(target / f'anchors_worker{k}.jsonl')]
                sr = [read_json(target / f'sparse_worker{k}.json') for k in range(WORKERS)]
                gate = dict(passed=gate_passes(ar, sr, anchors, WORKERS), anchors=32, exact_matches=sum(r['passed'] for r in ar),
                    world_size=WORKERS, contract_sha256=c['contract_sha256'], concurrency_overlay_sha256=overlay['overlay_sha256'], job_id=job)
                atomic_json(target / 'gate_decision.json', gate)
                assert gate['passed'], 'Concurrent gate failed'
                atomic_json(target / 'pilot_start.json', dict(started_unix=time.time()))
                print('gate_passed', json.dumps(gate), flush=True)
            if not (target / 'pilot_decision.json').exists() and len(list(target.glob('pilot_worker[0-9]*.json'))) == WORKERS:
                pilots = [read_json(target / f'pilot_worker{k}.json') for k in range(WORKERS)]
                wall = time.time() - read_json(target / 'pilot_start.json')['started_unix']
                decision = dict(passed=all(r['passed'] for r in pilots), samples=WORKERS,
                    routes=sum(r['routes'] for r in pilots), wall_seconds=wall,
                    aggregate_routes_per_second=sum(r['routes'] for r in pilots)/wall,
                    historical_serial_gpu_seconds=sum(r['old_serial_inference_seconds'] for r in pilots),
                    paired_speedup_vs_historical_one_per_gpu=sum(r['old_serial_inference_seconds'] for r in pilots)/(8*wall),
                    timing_caveat='Pilot uses completed samples; later datasets and sequence costs may differ.',
                    inference_contract_sha256=c['contract_sha256'], concurrency_overlay_sha256=overlay['overlay_sha256'])
                atomic_json(target / 'pilot_decision.json', decision)
                assert decision['passed']
                atomic_json(OUT / 'replay_gate/gate_result.json', dict(read_json(target / 'gate_decision.json'), saved_sample_pilot_passed=True))
                print('pilot_passed', json.dumps(decision), flush=True)
            progress = [read_json(p) for p in sorted(target.glob('worker*_progress.json'))]
            if progress:
                aggregate(progress, c, overlay, job, total_by_rank)
                atomic_json(target / 'status.json', dict(job_id=job, updated_unix=time.time(), workers=progress,
                    new_routes=sum(sum(v['success'] for v in r['by_source_rank'].values()) for r in progress),
                    new_samples=sum(r['new_samples'] for r in progress)))
                for record in progress:
                    if not record['complete'] and time.time()-record['updated_unix'] > 900:
                        raise RuntimeError(f"Worker {record['worker']} stale for over900s")
            if all(p.poll() == 0 for p in processes):
                aggregate(progress, c, overlay, job, total_by_rank, complete=True)
                atomic_json(target / 'complete.json', dict(passed=True, elapsed_seconds=time.monotonic()-start, workers=WORKERS,
                    routes=sum(x['routes'] for x in total_by_rank.values()), inference_contract_sha256=c['contract_sha256'],
                    concurrency_overlay_sha256=overlay['overlay_sha256']))
                break
            if not (target / 'pilot_decision.json').exists() and time.monotonic()-start > 3600:
                raise TimeoutError('Concurrent preflight/pilot exceeded1hour')
            time.sleep(5)
    except BaseException:
        atomic_json(target / 'coordinator_failure.json', dict(traceback=traceback.format_exc()))
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        raise
    finally:
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.worker is None:
        coordinate()
    else:
        assert 0 <= args.worker < WORKERS
        try:
            worker_main(args.worker, os.environ['SLURM_JOB_ID'])
        except BaseException:
            atomic_json(HERE / f"job_{os.environ['SLURM_JOB_ID']}" / f'worker{args.worker}_failure.json', dict(traceback=traceback.format_exc()))
            raise
