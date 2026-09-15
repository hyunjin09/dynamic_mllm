"""CPU-only exact complete-cohort reporting after the gated replay."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache
import json
import shutil
import time

from common import *
from replay import validate_resume


def pair_status(pair, current):
    a=current.get(pair['chosen_mask_key']);b=current.get(pair['rejected_mask_key'])
    if a is None or b is None or a['replay_status']!='COMPLETE' or b['replay_status']!='COMPLETE':
        return 'ROUTE_MISSING_OR_REPLAY_ERROR',False
    old_a=pair['chosen_correct'];old_b=pair['rejected_correct']
    ca=a['current_correctness'];cb=b['current_correctness']
    if ca==old_a and cb==old_b:
        status='BOTH_LABELS_STABLE'
    elif old_a!=old_b and ca==old_b and cb==old_a:
        status='PREFERRED_REJECTED_LABELS_REVERSED'
    else:
        status='ONE_OR_BOTH_LABELS_CHANGED'
    if pair['pair_type']=='correctness':
        valid=ca and not cb
    elif pair['pair_type']=='efficiency':
        valid=ca and cb and pair['chosen_budget']<pair['rejected_budget']
    else:
        raise ValueError('Unrecognized pair objective')
    return status,valid


def hamming_histograms(records):
    import numpy as np
    masks=np.array([int(r['mask_key'],2) for r in records],dtype=np.uint64)
    labs=np.array([int(r['current_correctness']) if r['replay_status']=='COMPLETE' else -1 for r in records])
    left,right=np.triu_indices(len(records),1)
    distances=np.bitwise_count(np.bitwise_xor(masks[left],masks[right]))
    out={}
    for name,select in [('C_C',(labs[left]==1)&(labs[right]==1)),('W_W',(labs[left]==0)&(labs[right]==0)),('C_W',((labs[left]==0)&(labs[right]==1))|((labs[left]==1)&(labs[right]==0)))]:
        out[name]=np.bincount(distances[select],minlength=LAYERS+1).tolist()
    return out


def finish():
    c=read_contract()
    for rank in range(WORLD_SIZE):
        report=read_json(OUT/'replay'/f'rank{rank}_complete.json')
        assert report['passed'] and report['contract_sha256']==c['contract_sha256']
    gate=read_json(OUT/'replay_gate/gate_result.json')
    assert gate['passed'] and gate['contract_sha256']==c['contract_sha256']
    assert file_hash(OUT/'source_inventory/prepared_index.jsonl')==c['prepared_index_sha256']
    index=list(rows(OUT/'source_inventory/prepared_index.jsonl'))
    samples={r['uid']:r for r in rows(OUT/'source_inventory/relocated_samples.jsonl')}
    current_counts=Counter();transitions=Counter();by_dataset=Counter();by_phase=Counter();by_intervention=Counter();by_family=Counter()
    categories=defaultdict(lambda:dict(routes=0,samples=set()))
    sample_rows=[];dense_rows=[];pair_estimates=[];action_counts=Counter()
    diversity=[];hamming_global={k:[0]*(LAYERS+1) for k in ['C_C','W_W','C_W']}
    total_visual_tokens=0
    paths=['replay/current_replay_routes.jsonl','replay/replay_errors.jsonl','filtering/current_route_manifest.jsonl','geometry_readiness/w_to_c_samples.jsonl','geometry_readiness/c_to_c_samples.jsonl']
    with ExitStack() as stack:
        handles={p:stack.enter_context(output_path(OUT/p).open('w')) for p in paths}
        def emit(name,r):handles[name].write(json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
        for si,item in enumerate(index):
            uid=item['uid'];sample=samples[uid]
            assert file_hash(item['path'])==item['sha256']
            source={r['route_key']:r for r in rows(item['path'])}
            payload=read_json(OUT/'replay/by_sample'/f'{stem(uid)}.json')
            rs=validate_resume(payload,source,c['contract_sha256'],uid)
            assert payload['complete'] and len(rs)==item['routes']
            dense=[r for r in rs if r['route_key']==item['dense_route_key']]
            assert len(dense)==1
            dense=dense[0];dc=dense['current_correctness'] if dense['replay_status']=='COMPLETE' else None
            cr=[r for r in rs if r['replay_status']=='COMPLETE' and r['current_correctness']]
            wr=[r for r in rs if r['replay_status']=='COMPLETE' and not r['current_correctness']]
            errors=[r for r in rs if r['replay_status']=='ERROR']
            n_c,n_w=len(cr),len(wr)
            for r in rs:
                emit('replay/current_replay_routes.jsonl',r)
                current_counts[r['replay_status']]+=1
                if r['replay_status']=='ERROR':emit('replay/replay_errors.jsonl',r)
                cat=transition(dc,r['current_correctness'])
                derived=dict(r,current_dense_correctness=dc,current_route_category=cat)
                derived['replay_record_sha256']=derived.pop('record_sha256')
                derived['record_sha256']=digest(derived)
                emit('filtering/current_route_manifest.jsonl',derived)
                cell=categories[(r['dataset'],cat)];cell['routes']+=1;cell['samples'].add(uid)
                tr=r['label_transition'];transitions[tr]+=1;by_dataset[(r['dataset'],tr)]+=1
                by_phase[(r['phase'],tr)]+=1;by_intervention[(r['interventions'],tr)]+=1
                for family in r['provenance']:by_family[(family,tr)]+=1
                action_counts[(r['dataset'],r['interventions'],str(r['current_correctness']))]+=1
                if r['replay_status']=='COMPLETE':total_visual_tokens+=r['visual_tokens']
            changed=sum(r['label_transition'] in ['C_TO_W','W_TO_C'] for r in rs)
            known=sum(r['label_transition']!='UNKNOWN' for r in rs)
            entry=dict(sample_uid=uid,dataset=sample['benchmark'],current_dense_correctness=dc,
                       routes=len(rs),current_correct=n_c,current_wrong=n_w,errors=len(errors),
                       C_TO_W=sum(r['label_transition']=='C_TO_W' for r in rs),W_TO_C=sum(r['label_transition']=='W_TO_C' for r in rs),
                       fraction_changed=changed/known if known else None,known_label_pairs=known,
                       unique_correct_action_sequences=len({r['mask_key'] for r in cr}),unique_wrong_action_sequences=len({r['mask_key'] for r in wr}),
                       **eligibility(n_c,n_w))
            sample_rows.append(entry)
            if dc is False and n_c:emit('geometry_readiness/w_to_c_samples.jsonl',entry)
            if dc is True and n_c:emit('geometry_readiness/c_to_c_samples.jsonl',entry)
            dense_rows.append(dict(sample_uid=uid,dataset=sample['benchmark'],dense_route_key=dense['route_key'],current_dense_answer=dense['current_answer'],current_dense_correctness=dc,old_dense_correctness=dense['old_correctness'],dense_label_transition=transition(dense['old_correctness'],dc),replay_status=dense['replay_status']))
            pair_estimates.append(dict(sample_uid=uid,dataset=sample['benchmark'],C_C=n_c*(n_c-1)//2,W_W=n_w*(n_w-1)//2,C_W=n_c*n_w))
            hist=hamming_histograms(rs)
            for category,counts in hist.items():
                for distance,count in enumerate(counts):
                    hamming_global[category][distance]+=count
                    if count:diversity.append(dict(sample_uid=uid,dataset=sample['benchmark'],pair_category=category,hamming_distance=distance,pairs=count))
            if (si+1)%250==0:print('aggregate',si+1,len(index),flush=True)
    assert current_counts['COMPLETE']+current_counts['ERROR']==read_json(OUT/'census/global_census.json')['unique_routes']
    write_jsonl(OUT/'dense/current_dense_manifest.jsonl',dense_rows)
    write_csv(OUT/'dense/dense_transition_summary.csv',[dict(dataset=b,transition=t,samples=n) for (b,t),n in sorted(Counter((r['dataset'],r['dense_label_transition']) for r in dense_rows).items())])
    write_csv(OUT/'replay/sample_replay_stability.csv',sample_rows)
    write_csv(OUT/'geometry_readiness/geometry_eligibility.csv',sample_rows)
    write_csv(OUT/'geometry_readiness/pair_count_estimates.csv',pair_estimates)
    write_csv(OUT/'geometry_readiness/route_action_diversity.csv',diversity)
    write_csv(OUT/'geometry_readiness/action_count_distribution.csv',[dict(dataset=b,interventions=k,current_correctness=lab,routes=n) for (b,k,lab),n in sorted(action_counts.items())])
    write_csv(OUT/'filtering/current_route_categories.csv',[dict(dataset=b,category=cat,routes=v['routes'],samples=len(v['samples'])) for (b,cat),v in sorted(categories.items())])
    total=sum(transitions.values())
    write_csv(OUT/'replay/replay_transition_matrix.csv',[dict(transition=t,routes=n,percent=100*n/total) for t,n in sorted(transitions.items())])
    for filename,counts,axis in [('dataset_transition_matrix.csv',by_dataset,'dataset'),('phase_transition_matrix.csv',by_phase,'phase'),('intervention_transition_matrix.csv',by_intervention,'interventions'),('provenance_transition_matrix.csv',by_family,'provenance')]:
        totals=Counter()
        for (cell,tr),n in counts.items():totals[cell]+=n
        write_csv(OUT/'replay'/filename,[{axis:cell,'transition':tr,'routes':n,'percent':100*n/totals[cell]} for (cell,tr),n in sorted(counts.items())])
    per_dataset=[]
    for b in sorted({r['dataset'] for r in sample_rows}):
        subset=[r for r in sample_rows if r['dataset']==b]
        per_dataset.append(dict(dataset=b,samples=len(subset),routes=sum(r['routes'] for r in subset),
            routes_per_sample=sum(r['routes'] for r in subset)/len(subset),
            current_correct_routes=sum(r['current_correct'] for r in subset),current_wrong_routes=sum(r['current_wrong'] for r in subset),
            dense_correct=sum(r['current_dense_correctness'] is True for r in subset),dense_wrong=sum(r['current_dense_correctness'] is False for r in subset),
            mixed=sum(r['C_W'] for r in subset),full_pairwise=sum(r['full_pairwise'] for r in subset),
            changed_route_fraction=(sum(r['C_TO_W']+r['W_TO_C'] for r in subset)/sum(r['known_label_pairs'] for r in subset)) if sum(r['known_label_pairs'] for r in subset) else None))
    write_csv(OUT/'summaries/current_dataset_summary.csv',per_dataset)
    @lru_cache(maxsize=32)
    def current_for(uid):
        payload=read_json(OUT/'replay/by_sample'/f'{stem(uid)}.json')
        return {r['mask_key']:r for r in payload['records']}
    pair_counts=Counter();pair_total=0;pair_valid=0
    import csv
    with output_path(OUT/'filtering/v31_pair_replay_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['pair_id','sample_uid','dataset','split','pair_type','status','current_preference_valid'])
        writer.writeheader()
        for split in ['train','validation']:
            for pair in rows(PAIRS/f'{split}_preference_pairs.jsonl'):
                current=current_for(pair['uid'])
                for side in ['chosen','rejected']:
                    record=current.get(pair[f'{side}_mask_key'])
                    if record:
                        assert record['source_route_id']==pair[f'{side}_route_id']
                        assert record['old_correctness']==pair[f'{side}_correct']
                status,valid=pair_status(pair,current)
                writer.writerow(dict(pair_id=pair['pair_id'],sample_uid=pair['uid'],dataset=pair['benchmark'],split=split,pair_type=pair['pair_type'],status=status,current_preference_valid=valid))
                pair_counts[(split,pair['benchmark'],pair['pair_type'],status)]+=1
                pair_total+=1;pair_valid+=int(valid)
    write_csv(OUT/'filtering/v31_pair_summary.csv',[dict(split=s,dataset=b,pair_type=p,status=t,pairs=n) for (s,b,p,t),n in sorted(pair_counts.items())])
    prospective=read_json(OUT/'storage/prospective_estimate.json')
    current_text=current_counts['COMPLETE']*LAYERS*c['hidden_size']*2
    current_visual={dtype:total_visual_tokens*LAYERS*c['hidden_size']*width for dtype,width in [('BF16',2),('FP16',2),('FP32',4)]}
    storage=dict(last_question_bf16_bytes=current_text,visual_bytes=current_visual,visual_tokens_source='current successful replay token counts',free_bytes=shutil.disk_usage(OUT).free,hidden_capture='deferred')
    atomic_json(OUT/'storage/current_estimate.json',storage)
    write_text(OUT/'storage/last_question_token_storage_estimate.md',f"Current replay-valid routes × actual {LAYERS} layers × {c['hidden_size']} dimensions ×2 = {current_text:,} BF16 bytes ({current_text/2**30:.2f} GiB), excluding indexing. No hidden tensors were collected.\n")
    write_text(OUT/'storage/visual_hidden_storage_estimate.md','Exact raw storage projection from current successful route visual-token counts (every layer, all tokens):\n\n'+json.dumps(current_visual,indent=2)+'\n\nNo raw visual capture authorized or performed. Estimates omit serialization/index overhead.\n')
    blockers=[]
    if current_counts['ERROR']:blockers.append('Replay errors')
    if any(r['current_dense_correctness'] is None for r in sample_rows):blockers.append('Missing valid current dense anchor')
    if any(r['full_pairwise']==0 for r in per_dataset):blockers.append('At least one benchmark lacks a full-pairwise eligible sample')
    if current_text>storage['free_bytes']:blockers.append('Current free space below projected last-token raw storage')
    eligibility_summary={k:dict(samples=sum(r[k] for r in sample_rows),routes=sum(r['current_correct']+r['current_wrong'] for r in sample_rows if r[k])) for k in eligibility(0,0)}
    geometry_breakdown=[]
    for b in sorted({r['dataset'] for r in sample_rows}):
        for dc in [True,False,None]:
            subset=[r for r in sample_rows if r['dataset']==b and r['current_dense_correctness'] is dc]
            for k in eligibility(0,0):
                selected=[r for r in subset if r[k]]
                geometry_breakdown.append(dict(dataset=b,dense_class=str(dc),eligibility=k,samples=len(selected),routes=sum(r['current_correct']+r['current_wrong'] for r in selected)))
    write_csv(OUT/'geometry_readiness/eligibility_breakdown.csv',geometry_breakdown)
    summary=dict(samples=len(index),routes=total,replay=dict(current_counts),transitions=dict(transitions),
        dense_correct=sum(r['current_dense_correctness'] is True for r in sample_rows),dense_wrong=sum(r['current_dense_correctness'] is False for r in sample_rows),
        eligibility=eligibility_summary,pairs={k:dict(total=sum(r[k] for r in pair_estimates),**distribution([r[k] for r in pair_estimates])) for k in ['C_C','W_W','C_W']},
        v31_pairs=pair_total,v31_valid_pairs=pair_valid,hamming_histograms=hamming_global,
        macro_changed_route_fraction=sum(r['changed_route_fraction'] for r in per_dataset)/len(per_dataset),
        technical_readiness=not blockers,blockers=blockers,
        RS_category='RS-A' if not transitions['C_TO_W'] and not transitions['W_TO_C'] and not transitions['UNKNOWN'] else 'PENDING_QUALITATIVE_REVIEW',
        review_status='analysis_ready_for_interpretation',contract_sha256=c['contract_sha256'])
    atomic_json(OUT/'summaries/complete_replay_summary.json',summary)
    write_text(OUT/'summaries/replay_filtering_summary.md','Complete current-server replay evidence.\n\n'+json.dumps(summary,indent=2)+'\n\nDataset/phase/intervention/provenance transitions are under replay/. The source runtime differs; old labels were preserved. Nonzero drift requires qualitative RS-A/B/C review of actual rates and eligibility impact, without invented numerical thresholds.\n')
    write_text(OUT/'filtering/filtering_report.md',f'{total:,} unique route records retain old/current labels and identity. Current dense anchors define categories. V3.1 is unchanged; {pair_valid:,}/{pair_total:,} pairs retain their declared preference semantics. See detailed CSV and source-bound manifests.\n')
    write_text(OUT/'summaries/geometry_readiness_summary.md','Technical readiness: '+str(not blockers)+'; blockers: '+json.dumps(blockers)+'\n\n'+json.dumps(dict(eligibility=eligibility_summary,pairs=summary['pairs'],storage=storage),indent=2)+'\n\nHamming histograms are exact unordered within-sample counts, separate C-C/W-W/C-W. Pair combinatorics do not establish practical hidden-geometry cost; future work must consider balanced sampling at the measured scale. No hidden-state geometry was analyzed. Last-token extraction remains deferred and must be validated in the next authorized phase. Technical eligibility does not establish statistical adequacy.\n')
    recommendation='Routing Trajectory Representation Geometry' if not blockers else 'Resolve the listed current-corpus blockers'
    write_text(OUT/'summaries/next_phase_recommendation.md',f'One unexecuted next-phase recommendation, subject to final evidence review: {recommendation}.\n\nUse the measured eligibility counts, route-Hamming distribution and storage estimates. No geometry, finetuning, preference rebuilding or next phase was executed.\n')
    # Preserve original inventory, then establish it still matches before completion.
    for row in c['source_inventory']:
        assert file_hash(PACKAGE/row['relative_path'])==row['sha256'], 'Canonical package changed during replay'
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.suffix not in ['.pyc'] and 'logs' not in p.parts and p.name not in ['artifact_manifest.json','final_verification.json'] and '.tmp.' not in p.name:
            files.append(dict(path=str(p.relative_to(OUT)),sha256=file_hash(p),bytes=p.stat().st_size))
    atomic_json(OUT/'artifact_manifest.json',dict(contract_sha256=c['contract_sha256'],files=files))
    atomic_json(OUT/'final_verification.json',dict(passed=True,artifact_manifest_sha256=file_hash(OUT/'artifact_manifest.json'),status='analysis_ready_for_interpretation',scientific_review_pending=True,source_unchanged=True))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    finish()
