# Full-benchmark exhaustive audit protocol

- Analysis type: deterministic offline reconstruction; no model inference, training, search, or threshold change.
- Frozen source: `analysis/dense_failure_stage2/full_benchmark_eval/paired/all_paired.jsonl` (67ed80e4badde4e32a2a01db4da43ee78eb0f010202a21127a77b83c498a77ed).
- Phase-69 contract: `63379eeff80fd5b046cb980cdfccea7fab232e15eb5b14924a24b60393327e83`.
- Population: exactly 19,960 unique paired UIDs: ChartQA 2,500; TextVQA 5,000; MMMU-Pro 3,460; POPE 9,000.
- Frozen gate: five-head mean, strict score > P90 threshold `0.9061332901863008`.
- Git HEAD at audit execution: `6c07e0aa1f0f1469c399b0b21caed9fa7f6f3ef2`; worktree was dirty and preserved. Status SHA-256: `a3f44b2f4ee682123646515c1e4ff3aef5c4233271b63172538e1380b52aa639`.
- Immediate means first non-FULL at the trigger layer (delay 0); delayed means delay > 0.
- Regression labels overlap: R1 one non-FULL; R2 more than one; R3 delay 0; R4 delay > 0; R5 any READ_ONLY; R6 any WRITE_ONLY; R7 IGNORE count exceeds all READ_ONLY+WRITE_ONLY occurrences; R8 more than one non-FULL action type.
- Fixed bottleneck rule order: INACTIVE when no triggers; TREATMENT_QUALITY_LIMITED when >=20 W interventions yield zero rescue; PRESERVATION_LIMITED when C→W >= 2×max(1,W→C); INTERVENTION_LIMITED when P(nonFULL|trigger,W)<0.10; ADMISSION_LIMITED when P(trigger|W)<0.05; otherwise MIXED.
- These are descriptive accounting rules. An answer change following a non-FULL trace does not causally identify one action as the cause.
