"""Verify and reuse the completed census without regenerating any frozen population."""
import csv
import time
from common import *

PREVIOUS = ROOT / 'analysis/3b_greedy_route_corpus_replay_audit'


def main():
    assert not (OUT/'frozen_contract.json').exists(), 'Preserve the V2 contract'
    old = read_json(PREVIOUS/'frozen_contract.json')
    assert digest({k:v for k,v in old.items() if k!='contract_sha256'}) == old['contract_sha256']
    for section in ['code_hashes', 'runtime_source_hashes']:
        for path, expected in old[section].items():
            assert file_hash(path)==expected, path
    for path, expected in old['small_model_hashes'].items():
        assert file_hash(MODEL/path)==expected, path
    for item in old['source_inventory']:
        path=PACKAGE/item['relative_path']
        assert path.stat().st_size==item['size_bytes'] and file_hash(path)==item['sha256'], str(path)
    print('Verified original contract, code, runtime, model metadata and 84 source inventory files.', flush=True)
    for name, key in [('source_inventory/prepared_index.jsonl','prepared_index_sha256'),
                      ('source_inventory/relocated_samples.jsonl','relocated_samples_sha256'),
                      ('replay_gate/frozen_anchor_rows.jsonl','anchor_sha256'),
                      ('replay_gate/frozen_sparse_probes.jsonl','sparse_probe_sha256')]:
        assert file_hash(PREVIOUS/name)==old[key], name
    index=list(rows(PREVIOUS/'source_inventory/prepared_index.jsonl'))
    for i, item in enumerate(index):
        assert file_hash(item['path'])==item['sha256'], item['uid']
        if i%2000==0: print('Verified prepared source shards',i,flush=True)
    samples=list(rows(PREVIOUS/'source_inventory/relocated_samples.jsonl'))
    for sample in samples:
        assert file_hash(sample['local_image_path'])==sample['image_content_sha256'],sample['uid']
    copied={}
    names=['source_inventory/file_inventory.csv','source_inventory/route_schema.md',
           'source_inventory/raw_lineage_checks.json','source_inventory/checksums.json',
           'source_inventory/prepared_index.jsonl','source_inventory/relocated_samples.jsonl',
           'replay_gate/frozen_anchor_rows.jsonl','replay_gate/frozen_sparse_probes.jsonl']
    names += [str(p.relative_to(PREVIOUS)) for p in sorted((PREVIOUS/'census').glob('*')) if p.suffix in ['.json','.csv']]
    for name in names:
        data=allowed_path(PREVIOUS/name).read_bytes()
        output_path(OUT/name).write_bytes(data)
        copied[name]=file_hash(OUT/name)
    dense=list(rows(PREVIOUS/'dense/dense_route_manifest_original.jsonl'))
    assert len(dense)==len(index)==len(samples)==10000
    assert {r['sample_uid'] for r in dense}=={r['uid'] for r in index}
    write_jsonl(OUT/'dense/original_dense_manifest.jsonl',dense)
    copied['dense/original_dense_manifest.jsonl']=file_hash(OUT/'dense/original_dense_manifest.jsonl')
    prior_gate=read_json(PREVIOUS/'replay_gate/gate_result.json')
    assert prior_gate['job_id']=='2939' and not prior_gate['passed']
    atomic_json(OUT/'parity/prior_job_2939_gate.json',prior_gate,immutable=True)
    write_csv(OUT/'parity/current_hf_vs_custom_full.csv',[
        dict(sample_uid=r['uid'],slurm_job_id='2939',evidence_origin='preserved_v1_current_server_gate',
             hf_ids=json.dumps(r['current_hf']['generated_ids']),custom_ids=json.dumps(r['current_binary']['generated_ids']),
             hf_answer=r['current_hf']['answer'],custom_answer=r['current_binary']['answer'],
             exact_tokens=r['exact_token_match'],passed=r['passed']) for r in prior_gate['anchor_rows']])
    write_text(OUT/'parity/gate_a_report.md',
        'Gate A: FAIL, carried forward from job2939 on the unchanged current runtime.\n\n'
        '31/32 exact token matches. textvqa:textvqa_19417 emitted HF22 versus custom23.\n'
        'This is current implementation disagreement, not acceptance of historical drift.\n'
        'V2 first runs one focused diagnostic; no new full replay is admitted.\n')
    write_text(OUT/'protocol.md',PROTOCOL)
    write_text(OUT/'replay_gate/replay_contract.md',
        'V2 diagnostic contract: exact original snapshot, inputs, scoring, BF16, SDPA,\n'
        'greedy decoding, repetition1.05 and EOS151645. Full baseline uses row token limit.\n'
        'Instrumented trace limits generation to2 tokens to locate the known first\n'
        'decision mismatch; it must reproduce the baseline prefix. Source Torch2.9.1+cu128;\n'
        'source Transformers/Python NOT RECORDED. Current versions are frozen below.\n'
        'Original32 anchors and four sparse probes are unchanged. Diagnostic changes\n'
        'no model weights, execution outputs, canonical package or gate criteria.\n\n'+json.dumps(old['generation'],indent=2)+'\n')
    contract={k:v for k,v in old.items() if k not in ['contract_sha256','code_hashes','created_unix','plan_sha256','protocol_sha256','gate_rule','schema']}
    contract.update(schema='current_server_3b_replay_audit_v2_diagnostic',created_unix=time.time(),
        parent_contract_path=str(PREVIOUS/'frozen_contract.json'),parent_contract_sha256=old['contract_sha256'],
        inherited_artifact_hashes=copied,diagnostic_uid='textvqa:textvqa_19417',diagnostic_gpus=1,
        stage='FOCUSED_PARITY_DIAGNOSTIC_ONLY',full_replay_authorized_by_gate=False,
        code_hashes={str(p):file_hash(p) for p in sorted((OUT/'code').glob('*')) if p.is_file()},
        plan_sha256=file_hash(ROOT/'plans/3b_greedy_route_corpus_audit_replay_filtering_plan_v2.md'),
        protocol_sha256=file_hash(OUT/'protocol.md'),
        gate_rule='Identical complete generated IDs on all32 fixed anchors. No corpus replay while Gate A fails.')
    for rel in ['integrations/sdpa_attention.py','cache_utils.py','masking_utils.py']:
        path=ROOT/'.venv/lib/python3.12/site-packages/transformers'/rel
        contract['runtime_source_hashes'][str(path)]=file_hash(path)
    contract['contract_sha256']=digest(contract)
    atomic_json(OUT/'frozen_contract.json',contract,immutable=True)
    atomic_json(OUT/'source_inventory/reuse_verification.json',dict(passed=True,source_files=len(old['source_inventory']),
        prepared_shards=len(index),images=len(samples),routes=sum(r['routes'] for r in index),
        parent_contract_sha256=old['contract_sha256'],contract_sha256=contract['contract_sha256'],
        population_regenerated=False,source_mutated=False))
    print('V2 verification complete',contract['contract_sha256'],flush=True)


PROTOCOL='''Execute plans/3b_greedy_route_corpus_audit_replay_filtering_plan_v2.md.
Canonical source and original audit evidence are immutable. Reuse verified
census, route identities and source shards; do not regenerate a population.
Gate A already failed on the current server. First trace only textvqa_19417
on one Slurm GPU. No automatic numerical patch, environment upgrade or bypass.
Instrumentation records checksums/norms/differences; no raw visual tensor corpus
or geometry analysis. Native and custom traces use the same loaded weights and
input tensors. Raw diagnostic evidence is scoped to the Slurm job.
Inspect the first measured divergence, then implement only a supported repair
in this V2 tree; preserve this diagnostic contract before a versioned execution
contract is created. Repeat all32 anchors after repair. Full replay on8 GPUs
is permitted only after that gate succeeds. Dense/FULL for every sample comes
before routed replay; reuse each already-completed dense route exactly once.
Old/current dense and routed labels remain separate. Current labels alone
define cohorts. No geometry, finetuning, V3.1 rebuild or threshold relaxation.
'''


if __name__=='__main__':
    main()
