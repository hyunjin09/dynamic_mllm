Canonical schema measured from final_phase1_phase2/evaluated_mask_candidates.jsonl.

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
