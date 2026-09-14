"""Mechanical final evidence validation and hash manifest for Phase87."""
from pathlib import Path
from experiments.run_write_harm_structure_learnability import *
from experiments.analyze_write_harm_structure_learnability import require_parity


def main():
    c=require_parity();assert read_json(OUT/'features/complete.json')['states']==15185;assert read_json(OUT/'learnability/complete.json')['passed'];gate=read_json(OUT/'learnability/local_decision_gate.json');external=[]
    for task in read_jsonl(OUT/'work/local_training_tasks.jsonl'):
        path=OUT/f"work/local_fits/task_{task['task_index']:04d}.json";r=read_json(path);assert r['status'] in ['complete','unestimable_sparse_matched_roles'];external.append(dict(path=str(path.resolve()),sha256=file_sha256(path)))
        if r['status']=='complete':
            norm=path.with_name(path.stem+'_normalization.pt');assert norm.is_file();external.append(dict(path=str(norm.resolve()),sha256=file_sha256(norm)))
    if gate['category']=='W-LOCAL-WEAK':
        from experiments.run_write_harm_propagation import check
        check();assert read_json(OUT/'propagation/complete.json')['passed'];assert read_json(OUT/'propagation/extraction_complete.json')['uids']==1413
        for task in read_jsonl(OUT/'work/propagation_tasks.jsonl'):
            path=OUT/f"work/propagation_fits/task_{task['task_index']:04d}.json";assert read_json(path)['status']=='complete';external.append(dict(path=str(path.resolve()),sha256=file_sha256(path)))
            norm=path.with_name(path.stem+'_normalization.pt');assert norm.is_file();external.append(dict(path=str(norm.resolve()),sha256=file_sha256(norm)))
        raw=read_jsonl(OUT/'propagation/rollout_manifest.jsonl');assert len(raw)==1413 and len({r['uid'] for r in raw})==1413
        for r in raw:external.append(dict(path=r['raw_file'],sha256=r['raw_sha256'],verified_by='complete finalize propagation pass: full raw-file SHA256 readback'))
    else:assert read_json(OUT/'generalization/complete.json')['passed']
    required=['write_structure_summary.md','write_mechanism_summary.md','write_learnability_summary.md','read_vs_write_characterization.md','next_research_direction.md']
    if gate['category']=='W-LOCAL-WEAK':required.append('write_propagation_summary.md')
    assert all((OUT/'summaries'/p).is_file() for p in required)
    figures=['write_harm_by_layer','write_harm_span_length','write_flip_neighborhood','write_update_norm','write_visual_diversity_change','write_query_alignment_change','write_feature_effect_sizes','write_learnability_comparison','read_vs_write_summary']
    if gate['category']=='W-LOCAL-WEAK':
        figures+=['write_horizon_spearman','write_horizon_harmful_auroc','write_text_vs_visual_divergence_by_horizon']
        interpretation=read_json(OUT/'propagation/interpretation_decision.json');assert interpretation['primary_frozen_category']==read_json(OUT/'propagation/propagation_decision.json')['category']
    assert all((OUT/f'figures/{name}.png').is_file() for name in figures)
    files=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p.name in ['artifact_manifest.json','final_verification.json']:continue
        files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=file_sha256(p)))
    code=[ROOT/p for p in ['experiments/run_write_harm_structure_learnability.py','experiments/analyze_write_harm_structure_learnability.py','experiments/run_write_harm_propagation.py','experiments/analyze_write_harm_propagation.py','experiments/continue_write_harm_propagation.py','experiments/reconcile_write_harm_propagation.py','experiments/summarize_write_harm_structure_learnability.py','experiments/finalize_write_harm_structure_learnability.py','dense_failure_stage2/write_harm_learnability.py','dense_failure_stage2/write_bootstrap.py','tests/test_write_harm_structure_learnability.py','tests/test_write_harm_propagation.py','tests/test_write_bootstrap.py']]
    atomic_json(OUT/'artifact_manifest.json',{'primary_contract_sha256':c['contract_sha256'],'files':files,'external_execution_and_raw_evidence':external,'code':{str(p.relative_to(ROOT)):file_sha256(p) for p in code}})
    atomic_json(OUT/'final_verification.json',{'passed':True,'artifact_files':len(files),'external_records':len(external),'figures':len(figures),'local_tasks':len(read_jsonl(OUT/'work/local_training_tasks.jsonl')),'propagation_tasks':len(read_jsonl(OUT/'work/propagation_tasks.jsonl')) if gate['category']=='W-LOCAL-WEAK' else 0,'dense_states':15185,'uids':1413,'image_groups':1385,'manifest_sha256':file_sha256(OUT/'artifact_manifest.json')});print(read_json(OUT/'final_verification.json'),flush=True)

if __name__=='__main__':main()
