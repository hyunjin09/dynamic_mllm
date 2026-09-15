"""Validate the versioned replay contract, preserving the diagnostic freeze."""
from common import *


def read_execution_contract():
    diagnostic=read_contract()
    c=read_json(OUT/'contracts/replay_v1.json')
    assert digest({k:v for k,v in c.items() if k!='contract_sha256'})==c['contract_sha256']
    assert c['diagnostic_contract_sha256']==diagnostic['contract_sha256']
    for path,expected in c['code_hashes'].items():assert file_hash(path)==expected,path
    for path,expected in c['prerequisite_hashes'].items():assert file_hash(path)==expected,path
    gate=read_json(c['passed_repair_gate'])
    assert gate['passed'] and gate['anchors']==gate['exact_anchor_matches']==32
    assert gate['sparse_repeat_passed']
    for path,key in [('source_inventory/prepared_index.jsonl','prepared_index_sha256'),
                     ('source_inventory/relocated_samples.jsonl','relocated_samples_sha256'),
                     ('replay_gate/frozen_anchor_rows.jsonl','anchor_sha256'),
                     ('replay_gate/frozen_sparse_probes.jsonl','sparse_probe_sha256')]:
        assert file_hash(OUT/path)==c[key],path
    return c
