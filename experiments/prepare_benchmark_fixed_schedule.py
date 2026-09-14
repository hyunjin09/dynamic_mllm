"""Freeze outcome-blind calibration/test identities for the fixed schedule plan."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import io
import json

import pyarrow.parquet as pq
from PIL import Image

from dense_failure_stage2.benchmark_fixed_schedule import FAMILIES, SEED, content_groups, family, group_prefix, stable_key
from experiments.run_full_benchmark_end_to_end_eval import _load_source_population, atomic_csv, atomic_json, atomic_jsonl, file_sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/benchmark_calibrated_fixed_rw_schedule'
DATA = (ROOT/'datasets').resolve()
EXT = DATA.parent/'outputs/benchmark_calibrated_fixed_rw_schedule_v1'


def image_identity(path):
    payload = Path(path).read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert('RGB')
        pixel = sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
    return sha256(payload).hexdigest(), pixel


def train_row(benchmark, uid, question, answers, path, identity, source):
    answer = Counter(answers).most_common(1)[0][0]
    return dict(uid=uid, sample_id=uid, benchmark=benchmark, benchmark_family=benchmark,
                prompt=question+'\nAnswer the question using a single word or phrase.', question=question,
                answer=answer, all_answer_norms=answers, image_path=str(path), image_paths=[str(path)],
                local_image_paths=[str(path)], data_root=str(path.parent), image_count=1,
                native_image_id=identity, group_keys=[f'native:{benchmark}:{identity}'],
                max_pixels=None, max_image_tokens=0, max_new_tokens=16,
                metric_name='relaxed_accuracy' if benchmark=='chartqa' else 'textvqa_evalai_consensus',
                correctness_threshold=1. if benchmark=='chartqa' else .5, source_split='train',
                source_dataset=source, prospective_pool='train', source_annotation='official_train')


def main():
    if (OUT/'splits/split_registry.jsonl').exists():
        raise RuntimeError('Split already frozen; do not recreate it')
    for folder in ['splits','calibration','schedules','heldout','metrics','efficiency','figures','summaries','parity']:
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'work').exists():
        (OUT/'work').symlink_to(EXT, target_is_directory=True)
    config = json.loads((ROOT/'configs/full_benchmark_end_to_end_eval_v1.json').read_text())
    config = {k:config[k] for k in ['model','generation','datasets','evaluation','backend_settings']}
    external = _load_source_population(config, verify_images=False)
    for row in external:
        row['benchmark_family'] = family(row['benchmark'])
        row['prospective_pool'] = 'existing_evaluation'
        row['group_keys'] = []
        if row['benchmark'].startswith('mmmu_pro'):
            row['group_keys'].append('native:mmmupro:'+str(row['source_id']))
        elif row['benchmark'].startswith('pope'):
            row['group_keys'].append('native:pope:'+str(row['image_source']))
        elif row['benchmark']=='textvqa':
            row['group_keys'].append('native:textvqa:'+str(row['image_id']))
    paths = sorted({p for r in external for p in r['local_image_paths']})
    with ThreadPoolExecutor(16) as pool:
        identities = dict(zip(paths, pool.map(image_identity, paths)))
    for row in external:
        assert [identities[p][0] for p in row['local_image_paths']] == row['image_content_sha256s']
        row['group_keys'] += [prefix+identities[p][i] for p in row['local_image_paths'] for i,prefix in [(0,'bytes:'),(1,'rgb:')]]
    external_keys = {k for r in external for k in r['group_keys']}
    train = []
    chart_root = DATA/'stage2_scale_sources/chartqa/ChartQA Dataset/train'
    for annotation in ['human','augmented']:
        for i,item in enumerate(json.loads((chart_root/f'train_{annotation}.json').read_text())):
            path = chart_root/'png'/item['imgname']
            if not path.is_file():
                continue
            uid = f'cal_chartqa:{annotation}:{i}:{item["imgname"]}'
            row = train_row('chartqa',uid,str(item['query']),[str(item['label'])],path,str(item['imgname']),'vis-nlp/ChartQA@044eabfc306abfe9340c5741f0093aefc5973d06')
            row['source_annotation'] = annotation
            train.append(row)
    # Available official train shards are the fixed source frame; no old model labels used.
    text_root = DATA/'stage2_scale_sources/textvqa/data'
    for path in sorted(text_root.glob('train-*.parquet')):
        metadata = pq.read_table(path, columns=['image_id','question_id','question','answers']).to_pylist()
        for index,item in enumerate(metadata):
            row = train_row('textvqa',f'cal_textvqa:{item["question_id"]}',str(item['question']),[str(a) for a in item['answers']],EXT/'calibration_images'/f'{item["image_id"]}.jpg',str(item['image_id']),'lmms-lab-encoder/textvqa@9c0699cd19768ac5ab97568f6b3cbac4c0062884')
            row['source_parquet'] = str(path)
            row['source_row'] = index
            train.append(row)
    # Materialize the train image pool once, retaining source bytes unchanged.
    needed = defaultdict(dict)
    for row in train:
        if row['benchmark']=='textvqa':
            needed[row['source_parquet']][row['source_row']] = row['local_image_paths'][0]
    for path, indices in needed.items():
        table = pq.read_table(path,columns=['image'])
        for index,destination in indices.items():
            target = Path(destination)
            payload = table.column('image')[index].as_py()['bytes']
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                assert target.read_bytes()==payload
            else:
                target.write_bytes(payload)
    paths = sorted({p for r in train for p in r['local_image_paths']})
    with ThreadPoolExecutor(16) as pool:
        identities = dict(zip(paths,pool.map(image_identity,paths)))
    eligible = []
    exclusions = Counter()
    for row in train:
        digest,pixel = identities[row['local_image_paths'][0]]
        row['image_content_sha256s'] = [digest]
        row['image_content_sha256'] = digest
        row['group_keys'] += ['bytes:'+digest,'rgb:'+pixel]
        if set(row['group_keys']) & external_keys:
            exclusions[row['benchmark']] += 1
        else:
            eligible.append(row)
    groups = content_groups(external+eligible)
    assert all(len({r['benchmark_family'] for r in g})==1 for g in groups), 'cross-family content needs explicit reconciliation'
    frozen = []
    support = []
    for benchmark in FAMILIES:
        pool = [g for g in groups if g[0]['benchmark_family']==benchmark and (g[0]['prospective_pool']=='train' if benchmark in ['chartqa','textvqa'] else True)]
        pool = sorted(pool,key=lambda g:stable_key(g[0]['image_group_id']))
        oversized = [g for g in pool if len(g)>256]
        pool = [g for g in pool if len(g)<=256]
        calibration = group_prefix(pool,256)
        assert calibration, benchmark
        cal_groups = {r['image_group_id'] for r in calibration}
        ordered_groups = [g for g in pool if g[0]['image_group_id'] in cal_groups]
        cal_ids = {r['uid'] for r in calibration}
        for gi,group in enumerate(ordered_groups):
            for row in sorted(group,key=lambda r:r['uid']):
                row['split'] = 'CAL'
                row['calibration_group_order'] = gi
                row['calibration_half'] = 'A' if gi%2==0 else 'B'
                row['nested_sizes'] = [n for n in [32,64,128,256] if row['uid'] in {x['uid'] for x in group_prefix(ordered_groups,n)}]
                frozen.append(row)
        test = [r for r in external if r['benchmark_family']==benchmark and r['image_group_id'] not in cal_groups]
        for row in test:
            row['split']='TEST'
            frozen.append(row)
        support.append(dict(benchmark=benchmark, calibration_uids=len(calibration),calibration_groups=len(cal_groups),test_uids=len(test),test_groups=len({r['image_group_id'] for r in test}),calibration_source='available official train image/shard frame' if benchmark in ['chartqa','textvqa'] else 'deterministic whole-group partition of existing labeled evaluation',train_overlap_exclusions=exclusions[benchmark],oversized_groups_excluded_from_cal=len(oversized)))
        assert not cal_ids & {r['uid'] for r in test}
    cal_keys = {k for r in frozen if r['split']=='CAL' for k in r['group_keys']}
    test_keys = {k for r in frozen if r['split']=='TEST' for k in r['group_keys']}
    assert not cal_keys & test_keys
    assert len({r['uid'] for r in frozen})==len(frozen)
    atomic_jsonl(OUT/'splits/split_registry.jsonl',frozen)
    atomic_csv(OUT/'splits/benchmark_support.csv',support)
    config.update(seed=SEED,world_size=4,random_schedules=20,bootstrap_draws=5000,selection_bootstrap_draws=2000,stage1_enabled=False,training_enabled=False,write_selection_layers=list(range(27)),read_selection_layers=list(range(28)),primary_metric='native scorer correctness under inherited thresholds; native raw score reported separately',split_sha256=file_sha256(OUT/'splits/split_registry.jsonl'))
    atomic_json(OUT/'execution_config.json',config)
    (OUT/'splits/leakage_check.md').write_text('# Frozen split audit\n\nPASS: no UID, image-byte, RGB-pixel, native-image, or connected content-group overlap between CAL and TEST, including MMMU Standard/Vision native question counterparts and all POPE variants. All external image byte hashes match the frozen source manifests. No prior intervention outcomes or correctness strata were used to select the pool. ChartQA/TextVQA retain their full existing evaluation sets. Their train pool is limited to locally available, previously prospectively acquired images/shards; representativeness of this frame is a limitation.\n\nMMMU-Pro and POPE have no matching separate train/dev split in the pinned evaluation contract, so their test results will exclude complete selected calibration groups. Current official dataset cards corroborate the test-only/task-variant configurations: https://huggingface.co/datasets/MMMU/MMMU_Pro/blob/main/README.md and https://huggingface.co/datasets/lmms-lab/POPE/blob/main/README.md . No dataset revision was changed.\n\nNested pools use whole-group prefixes up to N=32/64/128/256, so actual sizes may fall below the target. Split-half assignment alternates frozen groups and is independent of labels.\n')
    atomic_json(OUT/'splits/complete.json',dict(passed=True,states=len(frozen),support=support,split_sha256=config['split_sha256']))
    print(json.dumps(support,indent=2),flush=True)


if __name__=='__main__':
    main()
