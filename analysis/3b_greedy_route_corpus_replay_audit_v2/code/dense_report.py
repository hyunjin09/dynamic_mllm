"""Certify complete current Dense replay before allowing routed execution."""
from common import *
from execution import read_execution_contract
from replay import require_dense_record


def main():
    c=read_execution_contract()
    for rank in range(WORLD_SIZE):
        done=read_json(OUT/'dense'/f'rank{rank}_complete.json')
        assert done['passed'] and done['contract_sha256']==c['contract_sha256']
    index=list(rows(OUT/'source_inventory/prepared_index.jsonl'));manifest=[];counts=Counter()
    for item in index:
        assert file_hash(item['path'])==item['sha256']
        source={r['route_key']:r for r in rows(item['path'])}
        record=require_dense_record(read_json(OUT/'dense/by_sample'/f"{stem(item['uid'])}.json"),item,source,c)
        manifest.append(dict(sample_uid=item['uid'],dataset=item['dataset'],dense_route_key=record['route_key'],
            old_dense_answer=record['old_answer'],old_dense_correctness=record['old_correctness'],
            current_dense_answer=record['current_answer'],current_dense_correctness=record['current_correctness'],
            dense_transition=record['label_transition'],replay_status=record['replay_status'],
            replay_record_sha256=record['record_sha256'],contract_sha256=c['contract_sha256']))
        counts[('ALL',record['label_transition'])]+=1
        counts[(item['dataset'],record['label_transition'])]+=1
    assert len(manifest)==len({r['sample_uid'] for r in manifest})==10000
    write_jsonl(OUT/'dense/current_dense_manifest.jsonl',manifest)
    datasets=['ALL']+sorted({r['dataset'] for r in manifest})
    write_csv(OUT/'transitions/dense_transition_matrix.csv',[
        dict(dataset=b,transition=t,samples=counts[(b,t)],fraction=counts[(b,t)]/sum(counts[(b,x)] for x in ['C_TO_C','C_TO_W','W_TO_C','W_TO_W']))
        for b in datasets for t in ['C_TO_C','C_TO_W','W_TO_C','W_TO_W']])
    atomic_json(OUT/'dense/stage_complete.json',dict(passed=True,samples=len(manifest),contract_sha256=c['contract_sha256'],
        manifest_sha256=file_hash(OUT/'dense/current_dense_manifest.jsonl'),dense_correct=sum(r['current_dense_correctness'] for r in manifest),
        dense_wrong=sum(not r['current_dense_correctness'] for r in manifest)))
    print('Dense manifest certified for10000 samples; routed replay admitted.',flush=True)


if __name__=='__main__':main()
