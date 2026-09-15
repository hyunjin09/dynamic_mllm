Model: Qwen2.5-VL-3B-Instruct revision66285546d2b821cf421d4f5eb2576359d3770cd3,
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
