"""Outcome-independent grouping and fixed, calibration-only schedule selection."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import math
import numpy as np

FAMILIES = ('chartqa', 'textvqa', 'mmmupro', 'pope')
SEED = 2026091401


def family(benchmark):
    return 'mmmupro' if benchmark.startswith('mmmu_pro') else 'pope' if benchmark.startswith('pope_') else benchmark


def stable_key(value):
    return sha256(f'{SEED}:{value}'.encode()).hexdigest()


def content_groups(rows):
    """Transitive union over every image and native identity, including variants."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in row['group_keys']:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[find(i)].append(row)
    for group in members.values():
        group_id = 'group:' + sha256('|'.join(sorted({k for r in group for k in r['group_keys']})).encode()).hexdigest()
        for row in group:
            row['image_group_id'] = group_id
    return list(members.values())


def group_prefix(groups, maximum):
    """Nested whole-group prefix; no truncation or outcome-dependent backfill."""
    selected = []
    for group in groups:
        if len(selected) + len(group) > maximum:
            break
        selected.extend(group)
    return selected


def schedule(read_layer=None, write_layer=None):
    if read_layer is not None and read_layer not in range(28):
        raise ValueError('READ layer outside 0..27')
    if write_layer is not None and write_layer not in range(27):
        raise ValueError('WRITE layer27 is a control, not selectable')
    actions = ['FULL'] * 28
    if read_layer is not None:
        actions[read_layer] = 'WRITE_ONLY'
    if write_layer is not None:
        actions[write_layer] = 'IGNORE' if read_layer == write_layer else 'READ_ONLY'
    return actions


def layer_effects(rows, bit, weights=None):
    """Rows contain one UID's complete frozen dense/single-bit measurements."""
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, float)
    if len(w) != len(rows) or w.sum() <= 0:
        raise ValueError('empty calibration support')
    output = []
    for layer in range(28):
        branch = [r['branches'][f'{bit}{layer}'] for r in rows]
        dense = np.asarray([r['dense']['correct'] for r in rows], bool)
        correct = np.asarray([r['correct'] for r in branch], bool)
        gain = correct.astype(float) - dense.astype(float)
        q = np.asarray([r['q'] - source['dense']['q'] if r.get('q') is not None and source['dense'].get('q') is not None else np.nan for r, source in zip(branch, rows)])
        valid = np.isfinite(q)
        output.append(dict(bit=bit, layer=layer, n=len(rows), weight_sum=float(w.sum()), w_to_c=float(w[(~dense)&correct].sum()), c_to_w=float(w[dense&(~correct)].sum()), net=float(w@gain), gain=float(w@gain/w.sum()), mean_q=float(w@q/w.sum()) if valid.all() else None, q_valid=int(valid.sum()), selectable=bit=='R' or layer<27))
    return output


def ranked_layers(effects):
    eligible = [r for r in effects if r['selectable']]
    # MeanQ is usable only if every candidate has valid q on the same full pool.
    use_q = all(r['mean_q'] is not None and math.isfinite(r['mean_q']) for r in eligible)
    return sorted(eligible, key=lambda r: (-r['gain'], -r['mean_q'] if use_q else 0, r['c_to_w'], r['layer']))


def select_layer(effects):
    ranked = ranked_layers(effects)
    return int(ranked[0]['layer']) if ranked[0]['gain'] > 0 else None


def bootstrap_deltas(rows, differences, draws=5000, seed=SEED):
    """Paired cluster bootstrap, preserving all rows in each resampled group."""
    ids = sorted({r['image_group_id'] for r in rows})
    index = {g:i for i,g in enumerate(ids)}
    group = np.array([index[r['image_group_id']] for r in rows])
    counts = np.bincount(group, minlength=len(ids))
    values = np.asarray(differences, float)
    if values.ndim == 1:
        values = values[:,None]
    sums = np.stack([np.bincount(group, weights=values[:,j], minlength=len(ids)) for j in range(values.shape[1])],axis=1)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws,values.shape[1]))
    for start in range(0,draws,100):
        mult = rng.multinomial(len(ids), np.ones(len(ids))/len(ids), size=min(100,draws-start))
        samples[start:start+len(mult)] = (mult@sums)/(mult@counts)[:,None]
    return samples
