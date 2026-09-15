"""CPU census and prospective freeze; never run inference or modify the package."""
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
import importlib.metadata
import json
import os
import time

from common import *


def inventory():
    paths = {PACKAGE/'README.md', PACKAGE/'00_METADATA/PACKAGE_SUMMARY.json', PACKAGE/'00_METADATA/PATH_MAP.json'}
    for directory in [FINAL, PAIRS, CORPUS/'config', CORPUS/'replay_gate_v1', CORPUS/'manifests']:
        paths.update(p for p in directory.iterdir() if p.is_file())
    paths.update(p for p in (CORPUS/'phase1_aggregate').iterdir() if p.is_file() and p.suffix == '.json')
    for phase in ['phase1', 'phase2']:
        for shard in sorted((CORPUS/'raw'/phase).iterdir()):
            paths.update(p for p in shard.iterdir() if p.is_file())
            # Raw files are lineage examples only; canonical counting uses FINAL.
            for benchmark in ['chartqa','docvqa','gqa','textvqa']:
                found = next(iter(sorted((shard/'samples').glob(benchmark+'__*.json'))), None)
                if found:
                    paths.add(found)
    result = []
    for i,p in enumerate(sorted(paths)):
        result.append(dict(relative_path=str(p.relative_to(PACKAGE)), file_type=p.suffix,
                           size_bytes=p.stat().st_size, sha256=file_hash(p),
                           role='raw_lineage' if '/raw/' in str(p) else 'canonical_metadata_or_population',
                           canonical_or_raw='raw' if '/raw/' in str(p) else 'canonical'))
        if i % 20 == 0:
            print('Inventory', i, len(paths), flush=True)
    write_csv(OUT/'source_inventory/file_inventory.csv', result)
    by_path = {r['relative_path']:r for r in result}
    # Validate saved checksum files without accessing their source-server paths.
    checks = []
    for directory in [FINAL, PAIRS]:
        for line in (directory/'checksums.sha256').read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            name = name.lstrip('*')
            p = directory/Path(name).name
            rel = str(p.relative_to(PACKAGE))
            if rel not in by_path:
                raise ValueError(f'Checksum target outside inventoried files: {line}')
            passed = by_path[rel]['sha256'] == expected
            checks.append(dict(path=rel, passed=passed))
            assert passed, rel
    atomic_json(OUT/'source_inventory/checksums.json', checks)
    return result


def prepare():
    if (OUT/'frozen_contract.json').exists():
        raise RuntimeError('Contract already frozen; use the existing prepared corpus, never regenerate it')
    started = time.time()
    source_inventory = inventory()
    manifests = list(rows(CORPUS/'manifests/all_samples.jsonl'))
    sample_map = {r['uid']:r for r in manifests}
    assert len(sample_map) == len(manifests)
    samples = []
    for r in manifests:
        p = PACKAGE/'01_SOURCE_DATA/vqa_train_10k/images'/r['benchmark']/Path(r['local_image_path']).name
        p = allowed_path(p)
        assert p.is_file(), p
        assert file_hash(p) == r['image_content_sha256'], p
        samples.append(dict(r, local_image_path_source_server=r['local_image_path'],
                            local_image_path=str(p), image_path=str(p), image_paths=[str(p)]))
    sample_map = {r['uid']:r for r in samples}
    write_jsonl(OUT/'source_inventory/relocated_samples.jsonl', samples)
    old_index = {r['uid']:r for r in rows(FINAL/'sample_index.jsonl')}
    seen = set()
    census_rows, dense_rows, prepared_index = [], [], []
    label_counts = Counter()
    breakdown = Counter()
    phase_counts = Counter()
    provenance = Counter()
    old_categories = defaultdict(lambda: {'routes':0,'samples':set()})
    raw_count = duplicate_count = 0
    visual_token_sum = 0
    for uid, group in groupby(rows(FINAL/'evaluated_mask_candidates.jsonl'), lambda r:r['uid']):
        assert uid not in seen, 'Noncontiguous canonical UID block; preserve source and stop'
        seen.add(uid)
        sample = sample_map[uid]
        unique = {}
        for r in group:
            raw_count += 1
            assert r['benchmark'] == sample['benchmark'] and r['sample_id'] == sample['sample_id']
            key = validate_route(r)
            if key in unique:
                duplicate_count += 1
                assert digest(unique[key]['source_record']) == digest(r), 'Conflicting duplicate route'
                unique[key]['source_occurrences'] += 1
                continue
            r2 = dict(source_record=r, route_key=key, source_occurrences=1)
            unique[key] = r2
        rs = [unique[k] for k in sorted(unique)]
        counts = Counter(old_label(r['source_record']) for r in rs)
        n,c,w,unknown = len(rs),counts[True],counts[False],counts[None]
        dense = [r for r in rs if r['source_record']['mask_key'] == '1'*LAYERS]
        if len(dense) != 1:
            atomic_json(OUT/'dense/ambiguity.json', dict(uid=uid,candidates=len(dense)))
            raise ValueError(f'Dense anchor ambiguity: {uid}')
        d = dense[0]
        old_d = old_label(d['source_record'])
        idx = old_index[uid]
        assert idx['combined_candidates'] == n
        assert idx['combined_correct_candidates'] == c
        assert idx['all_on_route_id'] == d['source_record']['route_id']
        assert idx['all_on_correct'] == old_d
        dense_rows.append(dict(sample_uid=uid,dense_route_key=d['route_key'],source_route_id=d['source_record']['route_id'],old_dense_answer=d['source_record']['prediction'],old_dense_correctness=old_d))
        composition = 'UNKNOWN' if unknown else 'ALL_C' if c==n else 'ALL_W' if w==n else 'MIXED_CW'
        entry = dict(sample_uid=uid,dataset=sample['benchmark'],image_group=sample['image_content_sha256'],routes=n,saved_correct=c,saved_wrong=w,saved_unknown=unknown,saved_composition=composition,old_dense_correctness=old_d,**eligibility(c,w))
        census_rows.append(entry)
        p = OUT/'source_inventory/by_sample'/f'{stem(uid)}.jsonl'
        write_jsonl(p, rs)
        prepared_index.append(dict(uid=uid, dataset=sample['benchmark'],path=str(p),sha256=file_hash(p),routes=n,dense_route_key=d['route_key']))
        for item in rs:
            r = item['source_record']
            lab = old_label(r)
            label_counts[(sample['benchmark'],str(lab))] += 1
            phase_counts[r['observed_phase']] += 1
            families = sorted({o['family'] for o in r.get('phase1_origins',[])+r.get('phase2_origins',[])})
            for family in families:
                provenance[(sample['benchmark'],family)] += 1
            breakdown[(sample['benchmark'],r['observed_phase'],len(r['mask_key']),LAYERS-r['num_visual_on_layers'])] += 1
            cat = transition(old_d,lab)
            cell = old_categories[(sample['benchmark'],cat)]
            cell['routes'] += 1
            cell['samples'].add(uid)
            # Phase2 omits token-count telemetry. The unchanged input geometry is
            # available from this exact sample's Phase1 all-on anchor.
            visual_token_sum += r.get('visual_tokens', d['source_record']['visual_tokens'])
        if len(seen)%250 == 0:
            print('Census',len(seen),'samples',raw_count,'records',flush=True)
            atomic_json(OUT/'census/preparation_progress.json',dict(samples=len(seen),records=raw_count,elapsed_seconds=time.time()-started))
    assert seen == set(sample_map) == set(old_index)
    lineage=[]
    for inv in source_inventory:
        if inv['canonical_or_raw']!='raw' or '/samples/' not in inv['relative_path']:
            continue
        raw=read_json(PACKAGE/inv['relative_path'])
        uid=raw['sample']['uid']
        final={r['source_record']['route_id']:r['source_record'] for r in rows(OUT/'source_inventory/by_sample'/f'{stem(uid)}.jsonl')}
        key='candidate_executions' if 'candidate_executions' in raw else 'new_candidate_executions'
        for r in raw[key]:
            merged=final[r['route_id']]
            for field in ['visual_on_mask','generated_ids','prediction','score','result_correct']:
                assert r[field]==merged[field], (uid,r['route_id'],field)
        lineage.append(dict(path=inv['relative_path'],uid=uid,raw_field=key,matched_routes=len(raw[key]),passed=True))
    atomic_json(OUT/'source_inventory/raw_lineage_checks.json',lineage)
    # Sample assignment uses greedy least-route-load bin packing, frozen before replay.
    loads = [0]*WORLD_SIZE
    for item in sorted(prepared_index,key=lambda r:(-r['routes'],r['uid'])):
        rank = min(range(WORLD_SIZE), key=lambda k:(loads[k],k))
        item['rank'] = rank
        loads[rank] += item['routes']
    prepared_index.sort(key=lambda r:r['uid'])
    write_jsonl(OUT/'source_inventory/prepared_index.jsonl', prepared_index)
    write_jsonl(OUT/'dense/dense_route_manifest_original.jsonl', dense_rows)
    write_csv(OUT/'census/routes_per_sample.csv', census_rows)
    write_csv(OUT/'census/saved_sample_composition.csv', census_rows)
    write_csv(OUT/'census/saved_label_summary.csv',[dict(dataset=b,label=lab,routes=n) for (b,lab),n in sorted(label_counts.items())])
    write_csv(OUT/'census/original_route_categories.csv',[dict(dataset=b,category=cat,routes=v['routes'],samples=len(v['samples'])) for (b,cat),v in sorted(old_categories.items())])
    write_csv(OUT/'census/route_provenance.csv',[dict(dataset=b,family=f,routes=n,overlap_allowed=True) for (b,f),n in sorted(provenance.items())])
    write_csv(OUT/'census/route_composition.csv',[dict(dataset=b,phase=p,route_length=l,interventions=k,routes=n) for (b,p,l,k),n in sorted(breakdown.items())])
    per_dataset=[]
    for b in sorted({r['dataset'] for r in census_rows}):
        subset=[r for r in census_rows if r['dataset']==b]
        per_dataset.append(dict(dataset=b,samples=len(subset),routes=sum(r['routes'] for r in subset),correct=sum(r['saved_correct'] for r in subset),wrong=sum(r['saved_wrong'] for r in subset),mixed=sum(r['saved_composition']=='MIXED_CW' for r in subset),**distribution([r['routes'] for r in subset])))
    write_csv(OUT/'census/dataset_breakdown.csv', per_dataset)
    hist={name:sum(lo<=r['routes']<=hi for r in census_rows) for name,lo,hi in [('1',1,1),('2-4',2,4),('5-9',5,9),('10-19',10,19),('20-49',20,49),('50+',50,10**9)]}
    global_census=dict(unique_samples=len(seen),unique_image_groups=len({r['image_group'] for r in census_rows}),unique_routes=sum(loads),raw_records=raw_count,duplicate_records=duplicate_count,duplicate_canonical_routes=duplicate_count,routes_per_sample=distribution([r['routes'] for r in census_rows]),histogram=hist,saved_correct=sum(r['saved_correct'] for r in census_rows),saved_wrong=sum(r['saved_wrong'] for r in census_rows),saved_unknown=sum(r['saved_unknown'] for r in census_rows),sample_composition=dict(Counter(r['saved_composition'] for r in census_rows)),geometry_eligibility={k:sum(r[k] for r in census_rows) for k in eligibility(0,0)},dense_audit=dict(exactly_one=len(seen),zero=0,multiple=0),phase_counts=dict(phase_counts),rank_route_loads=loads,macro_saved_correct_fraction=sum(r['correct']/r['routes'] for r in per_dataset)/len(per_dataset))
    assert global_census['unique_routes']==read_json(FINAL/'summary.json')['combined_unique_evaluated_candidates']
    atomic_json(OUT/'census/global_census.json',global_census)
    config=read_json(MODEL/'config.json');text_config=config.get('text_config',config)
    assert text_config['num_hidden_layers']==LAYERS
    d=text_config['hidden_size'];text_bytes=sum(loads)*LAYERS*d*2
    storage=dict(layers=LAYERS,hidden_size=d,routes=sum(loads),last_question_bf16_bytes=text_bytes,visual_route_token_sum=visual_token_sum,visual_bytes={dtype:visual_token_sum*LAYERS*d*width for dtype,width in [('BF16',2),('FP16',2),('FP32',4)]},visual_tokens_source='saved per-route telemetry when present; Phase2 uses same-sample Phase1 dense input geometry; confirm current counts during replay',hidden_capture='deferred',reason='label replay first; no validated last-question-token capture path; raw visual capture not authorized')
    atomic_json(OUT/'storage/prospective_estimate.json',storage)
    write_text(OUT/'storage/last_question_token_storage_estimate.md',f'Actual model depth {LAYERS}, hidden size {d}.\n\n{sum(loads):,} routes × {LAYERS} × {d} × 2 = {text_bytes:,} bytes ({text_bytes/2**30:.2f} GiB) before metadata. The plan’s 28-layer example does not apply to this 3B model.\n')
    write_text(OUT/'storage/visual_hidden_storage_estimate.md','Prospective estimate from saved visual token counts, without inference or raw hidden extraction.\n\n'+json.dumps(storage['visual_bytes'],indent=2)+'\n')
    write_text(OUT/'storage/hidden_capture_decision.md','Defer last-question-token capture and all raw visual tensors. Storage is available, but label/runtime validation is the primary action and exact question-token capture is not validated. A future authorized geometry phase can capture only the filtered corpus; this costs a further pass. No hidden-state geometry or router training is authorized.\n')
    anchors=list(rows(CORPUS/'replay_gate_v1/generation_anchor_rows.jsonl'))
    assert len({r['uid'] for r in anchors})==len(anchors)==32
    assert {r['uid'] for r in anchors} <= seen
    write_jsonl(OUT/'replay_gate/frozen_anchor_rows.jsonl',anchors)
    # One deterministic sparse route for each benchmark, repeated on every worker.
    probes=[]
    for b in sorted({r['benchmark'] for r in anchors}):
        anchor=min((r for r in anchors if r['benchmark']==b),key=lambda r:r['uid'])
        rs=list(rows(OUT/'source_inventory/by_sample'/f"{stem(anchor['uid'])}.jsonl"))
        sparse=min((r for r in rs if 0<r['source_record']['num_visual_on_layers']<LAYERS),key=lambda r:r['route_key'])
        probes.append(dict(uid=anchor['uid'],**sparse))
    write_jsonl(OUT/'replay_gate/frozen_sparse_probes.jsonl',probes)
    write_text(OUT/'source_inventory/route_schema.md',SCHEMA)
    write_text(OUT/'replay_gate/replay_contract.md',REPLAY_CONTRACT)
    write_text(OUT/'protocol.md',PROTOCOL)
    runtime=PACKAGE/'07_RESUME_TOOLS/dvr_qwen'
    runtime_files=list(runtime.glob('*.py'))+[runtime/'scripts/cache_preference_gt_router_features.py']
    installed=ROOT/'.venv/lib/python3.12/site-packages/transformers'
    runtime_files += [installed/'generation/utils.py',installed/'models/qwen2_5_vl/modeling_qwen2_5_vl.py']
    model_files={p.name:file_hash(p) for p in MODEL.iterdir() if p.name.endswith(('.json','.txt'))}
    index=read_json(MODEL/'model.safetensors.index.json')
    weight_sizes={p:allowed_path(MODEL/p).stat().st_size for p in sorted(set(index['weight_map'].values()))}
    assert all(v>0 for v in weight_sizes.values())
    contract=dict(schema='current_server_3b_replay_audit_v1',created_unix=time.time(),model_revision=REVISION,layers=LAYERS,hidden_size=d,model_path=str(MODEL),package=str(PACKAGE),domain=DOMAIN,route_key_definition='sha256 of canonical JSON {sample_uid,mask_key,domain}; full36-bit mask at layers0..35; immutable source route_id also retained',world_size=WORLD_SIZE,source_inventory=source_inventory,prepared_index_sha256=file_hash(OUT/'source_inventory/prepared_index.jsonl'),relocated_samples_sha256=file_hash(OUT/'source_inventory/relocated_samples.jsonl'),anchor_sha256=file_hash(OUT/'replay_gate/frozen_anchor_rows.jsonl'),sparse_probe_sha256=file_hash(OUT/'replay_gate/frozen_sparse_probes.jsonl'),checkpoint_routes=16,attempts_per_route=1,hidden_capture=False,code_hashes={str(p):file_hash(p) for p in sorted((OUT/'code').glob('*')) if p.is_file()},runtime_source_hashes={str(p):file_hash(p) for p in sorted(runtime_files)},small_model_hashes=model_files,weight_sizes=weight_sizes,installed_versions={x:importlib.metadata.version(x) for x in ['torch','transformers','qwen-vl-utils','numpy','Pillow']},generation=dict(do_sample=False,num_beams=1,eos_token_ids=[151645],repetition_penalty=1.05,dtype='bfloat16',attention='sdpa',processor_use_fast=False,max_new_tokens='source row16 or32',position_convention='unchanged packaged binary executor; no Phase88 repair'),source_runtime=read_json(FINAL/'summary.json')['phase1_runtime'],gate_rule='32 frozen anchors exact current HF/binary IDs, answers and scores; old/current matches reported descriptively. Four sparse masks repeat exactly twice on all8 ranks and agree across ranks. Hard stop on any failure.',readiness_rule='all unique routes terminal; zero errors required for READY; exactly one successful current dense per UID; each benchmark has at least one full-pairwise eligible sample; exact identities; text storage fits observed free space. No claim of statistical adequacy.',stability_rule='No arbitrary RS-A/B/C thresholds: report exact transitions; exact zero drift => RS-A, otherwise qualitative category requires main-agent review after completion',plan_sha256=file_hash(ROOT/'plans/3b_greedy_route_corpus_audit_replay_filtering_plan.md'),protocol_sha256=file_hash(OUT/'protocol.md'))
    contract['contract_sha256']=digest(contract)
    atomic_json(OUT/'frozen_contract.json',contract,immutable=True)
    atomic_json(OUT/'source_inventory/prepared_complete.json',dict(passed=True,samples=len(seen),routes=sum(loads),contract_sha256=contract['contract_sha256'],elapsed_seconds=time.time()-started))
    write_text(OUT/'summaries/corpus_audit_summary.md','Canonical 3B corpus census; GPU replay pending.\n\n'+json.dumps(global_census,indent=2)+'\n\nDataset-specific counts and distributions: census/dataset_breakdown.csv. Canonical labels remain historical.\n')
    print(json.dumps(global_census,indent=2),flush=True)


SCHEMA='''Canonical schema measured from final_phase1_phase2/evaluated_mask_candidates.jsonl.

uid → sample_uid; benchmark → dataset; sample_id retained; image/question/answer and
content SHA256 join exactly to manifests/all_samples.jsonl by uid. result_correct
is the saved Boolean label; score is the native score; prediction/generated_ids
are saved output. mask_key and visual_on_mask must encode the same36-bit route;
num_visual_on_layers and is_all_on/is_all_off are checked. source route_id is
uid:mask:sha256(uid+":"+mask_key)[:16]. Full36-bit sequence and binary contextualized
domain define identity; there is no trigger/start layer. Route key additionally
uses full SHA256 and never conflates samples. Provenance includes observed_phase,
phase1_origins, phase2_origins, phase2_requested and permutation_final_orders.
Origin families can overlap. All original fields and occurrence counts are retained
in by_sample files. Model/config references are package-level, not assumed per row.
Phase2 rows omit visual_tokens/text_tokens telemetry; prospective storage uses
the same sample's Phase1 dense counts, marked as derived input-geometry estimates.
Current replay records measure both token counts independently.
Final merge contains unique Phase1 candidates plus only newly evaluated Phase2
candidates; reused requests are provenance, not extra primary routes.
Raw summaries and stratified raw examples are inventoried for lineage only.
Missing/invalid labels remain UNKNOWN. Dense is the unique all-one mask and is
cross-checked with sample_index; any ambiguity stops preparation.
'''

REPLAY_CONTRACT='''Model: Qwen2.5-VL-3B-Instruct revision66285546d2b821cf421d4f5eb2576359d3770cd3,
36 decoder layers, hidden size2048. Model/tokenizer/processor use the same local
snapshot; small files hash-bound, weights inventoried but not redundantly hashed.
Source phase1 runtime records Torch2.9.1+cu128, SDPA, slow processor, repetition
penalty1.05, EOS151645. Source Transformers/Python and exact collector Git revision:
NOT RECORDED in the canonical runtime summary. Config transformers_version4.41.2
describes saved config provenance and is not evidence of collector runtime.
Current project environment is separately frozen; version drift is reported.
The packaged dvr_qwen binary executor runs visual-on joint rows and visual-off
compact text rows while preserving carried visual state and heterogeneous caches.
This is binary contextualized routing, not Phase88's four-action API.
Generation uses unmodified packaged binary_dvrc_greedy_generate and native HF
generate with greedy decoding, one beam, repetition1.05, EOS151645, row16/32 limit.
Input builder is the package's cache_preference_gt_router_features.build_processor_inputs:
one user message, image then exact saved prompt, snapshot chat template with
generation prompt, qwen-vl-utils image processing, native mm_token_type_ids.
DocVQA max_pixels1605632 (1913 rows) or802816 (87); other families native.
Exact saved image SHA256 is verified before preparation and each worker's sample.
Packaged eval_metrics.score_prediction: GQA normalized exact match threshold1;
ChartQA relaxed_accuracy threshold1; DocVQA ANLS threshold.5;
TextVQA EvalAI consensus threshold.5. Original all_answer_norms retained.
Anchor gate reconstructs all32 frozen source UIDs; requires CURRENT native/binary
tokens, answers and score equality. Historical current-source and earlier source
output matches are diagnostics, not interchangeable gate requirements. The gate
does not prove sparse-route parity. Four sparse source masks repeated on each of
eight GPUs test exact repeatability and cross-rank agreement before full replay.
No numerical patch or acceptance relaxation occurs automatically on failure.
'''

PROTOCOL='''User authorized plans/3b_greedy_route_corpus_audit_replay_filtering_plan.md,
eight GPUs via Slurm, submission only this turn. Preserve the canonical package.
One new derived corpus; source identities and labels remain immutable.
CPU inventory/census and relocation precede GPU gate, then deterministic unique
route replay. Gates fail closed. No geometry, raw visual capture, training,
pair-supervision rebuild, or automatic next experiment.
Gate rules and identities are in frozen_contract.json. Resume validates every
completed record's canonical hash, source hash, contract and route key; successful
routes are never replayed. Checkpoint atomically every16 routes and at sample end.
At most one attempt per uncompleted route per authorized run; errors retain UID,
route identity, type/message and attempt count. An interrupted uncommitted batch
may be retried; durable complete records are never duplicated. Error records are
not silently retried. Missing dense or any replay error prevents READY.
Per-benchmark reports and equal-family macro views accompany pooled counts.
Dense itself belongs to the enumerated route set and is replayed exactly once.
Readiness is a technical corpus check, not a claim of statistical adequacy.
Nonzero replay drift requires qualitative RS-category review against actual
rates; no post-outcome numerical threshold is invented by the batch reporter.
The job stops after derived reports with analysis_ready_for_interpretation;
the main agent must review RS/readiness conclusions when the user returns.
'''

if __name__=='__main__':
    prepare()
