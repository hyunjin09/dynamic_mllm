"""Identity, atomic evidence, and access checks for the immutable 3B audit."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import os
import stat

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
PACKAGE = Path('/data/research/datasets/Sparse_Visual_Contextualization/Qwen2.5-VL-3B-Instruct')
MODEL = Path('/data/research/models/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3')
CORPUS = PACKAGE / '02_GREEDY_SEARCH/vqa_10k'
FINAL = CORPUS / 'final_phase1_phase2'
PAIRS = PACKAGE / '03_PAIRED_DATASETS/vqa_correctness_first_v31'
REVISION = MODEL.name
WORLD_SIZE = 8
LAYERS = 36
DOMAIN = 'binary_contextualized_visual_on_off_36_v1'


def allowed_path(value, depth=0):
    """Check each symlink target before dereferencing; never inspect source paths."""
    p = Path(os.path.abspath(value))
    roots = [ROOT, Path('/data/research/datasets'), Path('/data/research/models')]
    base = next((r for r in roots if p.is_relative_to(r)), None)
    if base is None or depth > 32:
        raise ValueError(f'Path outside access policy or symlink loop: {p}')
    cur = base
    for part in p.relative_to(base).parts:
        cur = cur / part
        try:
            mode = cur.lstat().st_mode
        except FileNotFoundError:
            return p
        if stat.S_ISLNK(mode):
            target = Path(os.readlink(cur))
            target = target if target.is_absolute() else cur.parent / target
            return allowed_path(target / p.relative_to(cur), depth + 1)
    return p


def file_hash(path):
    h = hashlib.sha256()
    with allowed_path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(allowed_path(path).read_text())


def rows(path):
    with allowed_path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def output_path(path):
    p = allowed_path(path)
    if not p.is_relative_to(OUT):
        raise ValueError(f'Audit writes must stay in new analysis root: {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def atomic_json(path, value, *, immutable=False):
    p = output_path(path)
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if immutable and p.exists():
        if p.read_text() != data:
            raise ValueError(f'Preserve conflicting existing artifact: {p}')
        return
    tmp = p.with_name(p.name + f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def write_text(path, text):
    output_path(path).write_text(text)


def write_jsonl(path, records):
    p = output_path(path)
    tmp = p.with_name(p.name + f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def write_csv(path, records, fields=None):
    records = list(records)
    fields = fields or (list(records[0]) if records else ['no_rows'])
    with output_path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def stem(uid):
    return hashlib.sha256(uid.encode()).hexdigest()


def route_key(uid, mask):
    if len(mask) != LAYERS or set(mask) - {'0', '1'}:
        raise ValueError(f'Invalid binary route: {mask}')
    return digest({'sample_uid': uid, 'mask_key': mask, 'domain': DOMAIN})


def validate_route(r):
    mask = r['mask_key']
    key = route_key(r['uid'], mask)
    if r['visual_on_mask'] != [int(v) for v in mask]:
        raise ValueError('mask/sequence mismatch')
    expected_id = r['uid'] + ':mask:' + hashlib.sha256(f"{r['uid']}:{mask}".encode()).hexdigest()[:16]
    if r['route_id'] != expected_id:
        raise ValueError('source route ID mismatch')
    if r['num_visual_on_layers'] != mask.count('1'):
        raise ValueError('route budget mismatch')
    if r['is_all_on'] != (mask == '1' * LAYERS) or r['is_all_off'] != (mask == '0' * LAYERS):
        raise ValueError('anchor flag mismatch')
    return key


def old_label(r):
    return r['result_correct'] if type(r.get('result_correct')) is bool else None


def transition(old, current):
    if type(old) is not bool or type(current) is not bool:
        return 'UNKNOWN'
    return ('C' if old else 'W') + '_TO_' + ('C' if current else 'W')


def eligibility(c, w):
    return {'C_W': c >= 1 and w >= 1, 'C_C': c >= 2, 'W_W': w >= 2,
            'C_C_plus_C_W': c >= 2 and w >= 1, 'W_W_plus_C_W': w >= 2 and c >= 1,
            'full_pairwise': c >= 2 and w >= 2}


def distribution(values):
    import numpy as np
    a = np.asarray(values)
    return dict(mean=float(a.mean()), std=float(a.std()), min=int(a.min()),
                **{name: float(np.percentile(a, q)) for name, q in [('p10',10),('p25',25),('median',50),('p75',75),('p90',90),('p95',95),('p99',99)]}, max=int(a.max()))


def read_contract():
    c = read_json(OUT / 'frozen_contract.json')
    assert digest({k:v for k,v in c.items() if k != 'contract_sha256'}) == c['contract_sha256']
    for path, expected in c['code_hashes'].items():
        assert file_hash(path) == expected, f'Bound code drift: {path}'
    for path, expected in c['runtime_source_hashes'].items():
        assert file_hash(path) == expected, f'Runtime source drift: {path}'
    for path, expected in c['small_model_hashes'].items():
        assert file_hash(MODEL / path) == expected, f'Model metadata drift: {path}'
    for name, size in c['weight_sizes'].items():
        assert allowed_path(MODEL / name).stat().st_size == size
    return c
