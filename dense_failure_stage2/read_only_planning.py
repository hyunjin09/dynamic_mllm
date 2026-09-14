"""Deterministic binary search, unique-route accounting, and planning summaries.

MCTS retains the prefix-tree UCB1 structure of corrective_search.run_sequential_mcts;
this analysis specializes its alphabet to ON/OFF, uses raw q reward, and budgets
unique terminals rather than binary-success-stopped iterations.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations, product
import math
import random

from dense_failure_stage2.corrective_search import _stable_digest

ALGORITHMS = ('greedy', 'beam2', 'beam4', 'beam8', 'random_uniform', 'random_sparse', 'binary_mcts')
BUDGETS = (1, 4, 8, 16, 32, 64, 128, 256, 512)
SEEDS = (2026091201, 2026091202, 2026091203)


def route_hash(uid, trigger, suffix):
    if not uid or not 0 <= trigger < 28 or len(suffix) != 28-trigger or set(suffix)-{'0','1'}:
        raise ValueError('invalid binary route identity')
    return sha256(f'{uid}\0{trigger}\0{suffix}'.encode()).hexdigest()


def actions(trigger, suffix):
    route_hash('validation', trigger, suffix)
    return ('FULL',)*trigger + tuple('FULL' if b == '1' else 'WRITE_ONLY' for b in suffix)


def enumeration(t, distance=2):
    yield '1'*t
    for m in range(1, min(distance,t)+1):
        for positions in combinations(range(t), m):
            bits = ['1']*t
            for p in positions:
                bits[p] = '0'
            yield ''.join(bits)


def beam(t, width):
    """Each expansion is scored by its actual prefix plus an all-ON completion."""
    yield '1'*t
    prefixes = ['']
    for depth in range(t):
        candidates = []
        for prefix in prefixes:
            for bit in ('1','0'):
                child = prefix+bit
                q = yield child+'1'*(t-len(child))
                candidates.append((float(q), len(candidates), child))
        candidates.sort(key=lambda x:(-x[0], x[1]))
        prefixes = [p for _,_,p in candidates[:width]]


def random_routes(t, seed, uid, sparse=False):
    rng = random.Random(int(_stable_digest(seed, uid, 'sparse' if sparse else 'uniform')[:16],16))
    seen = {'1'*t}
    yield '1'*t
    support = 1+sum(math.comb(t,m) for m in range(1,min(4,t)+1)) if sparse else 2**t
    while len(seen)<support:
        if sparse:
            m = rng.randint(1,min(4,t))
            pos = set(rng.sample(range(t),m))
            route = ''.join('0' if i in pos else '1' for i in range(t))
        else:
            route = ''.join(rng.choice('10') for _ in range(t))
        if route not in seen:
            seen.add(route)
            yield route


def mcts(t, seed, uid):
    visits, sums = Counter(), Counter()
    children = {}
    terminals = set()
    known = Counter()
    rng = random.Random(int(_stable_digest(seed,uid,'binary_mcts')[:16],16))
    dense = '1'*t
    q = yield dense
    terminals.add(dense)
    # Dense is shared budget candidate 1, not an artificial UCB tree visit.
    for d in range(t+1):
        known[dense[:d]] += 1
    iteration = 0
    while len(terminals)<2**t:
        iteration += 1
        prefix = ''
        path = [prefix]
        while len(prefix)<t:
            available = [b for b in '10' if known[prefix+b]<2**(t-len(prefix)-1)]
            if not available:
                raise RuntimeError('MCTS exhausted prefix accounting')
            order = sorted(available,key=lambda b:_stable_digest(seed,uid,'expand',prefix,b))
            used = children.setdefault(prefix,set())
            untried = [b for b in order if b not in used]
            if untried:
                b = untried[0]
                used.add(b)
                prefix += b
                path.append(prefix)
                break
            def ucb(b):
                p=prefix+b
                n=visits[p]
                return (math.inf if not n else sums[p]/n+math.sqrt(2*math.log(max(1,visits[prefix]))/n),
                        _stable_digest(seed,uid,'ucb',p))
            prefix += max(order,key=ucb)
            path.append(prefix)
        # Random binary rollout restricted to unexhausted children; prevents
        # duplicate iterations from silently consuming an unbounded budget.
        route=prefix
        while len(route)<t:
            available=[b for b in '10' if known[route+b]<2**(t-len(route)-1)]
            route += rng.choice(available)
        q=float((yield route))
        if not math.isfinite(q):
            raise ValueError('nonfinite MCTS q')
        terminals.add(route)
        for d in range(t+1):
            known[route[:d]] += 1
        for p in path:
            visits[p]+=1
            sums[p]+=q


class SearchSession:
    """Pause/resume the same algorithm at unique evaluation checkpoints."""
    def __init__(self, algorithm, t, uid, seed=SEEDS[0]):
        self.algorithm, self.uid, self.seed = algorithm, uid, seed
        if algorithm=='greedy' or algorithm.startswith('beam'):
            self.generator=beam(t,1 if algorithm=='greedy' else int(algorithm[4:]))
        elif algorithm=='binary_mcts':
            self.generator=mcts(t,seed,uid)
        elif algorithm=='enumeration':
            self.generator=enumeration(t)
        else:
            self.generator=random_routes(t,seed,uid,algorithm=='random_sparse')
        self.rows=[]
        self.seen={}
        self.started=False
        self.feedback=None
        self.exhausted=False
        self.requests=0

    def advance(self, budget, evaluate):
        while len(self.rows)<budget and not self.exhausted:
            try:
                route=self.generator.send(self.feedback) if self.started else next(self.generator)
                self.started=True
            except StopIteration:
                self.exhausted=True
                break
            self.requests+=1
            if route in self.seen:
                row=self.seen[route]
            else:
                row=dict(evaluate(route))
                row.update(suffix=route, evaluation_rank=len(self.rows)+1,
                           search_algorithm=self.algorithm, seed=self.seed)
                if not math.isfinite(float(row['q'])):
                    raise ValueError('nonfinite route q')
                self.rows.append(row)
                self.seen[route]=row
            self.feedback=row['q']
        return self.rows


def curve(rows, budget):
    selected=rows[:budget]
    if not selected:
        raise ValueError('empty search history')
    winner=max(selected,key=lambda x:(float(x['q']),-int(x['evaluation_rank'])))
    any_correct=any(r['correct'] for r in selected)
    first=next((int(r['evaluation_rank']) for r in selected if r['correct']),None)
    return dict(budget=budget,actual_unique_routes=len(selected),any_correct=bool(any_correct),
                selected_correct=bool(winner['correct']), selected_suffix=winner['suffix'],
                selected_q=winner['q'],first_rescue_rank=first,
                outcome='SUCCESS' if winner['correct'] else 'RANKING_FAILURE' if any_correct else 'GENERATION_FAILURE')


def complexity(suffix, trigger):
    off=[i+trigger for i,b in enumerate(suffix) if b=='0']
    return dict(read_off_count=len(off),hamming_distance=len(off),first_read_off_layer=min(off) if off else None,
                last_read_off_layer=max(off) if off else None,first_off_delay=min(off)-trigger if off else None,
                off_span=max(off)-min(off)+1 if off else 0,
                switches=sum(a!=b for a,b in zip(suffix,suffix[1:])))
