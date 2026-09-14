# Phase 59: Stage-2 Data-Scale Search Memory

## Current Objective
Freeze an approximately 4,000-record GQA/ChartQA/TextVQA training pool disjoint from every existing Stage-1/Stage-2 train, validation, and test identity; generate current-runtime dense, frozen Stage-1 trigger, exhaustive-single, and cap-200 MCTS labels once; then build provenance-separated expanded corpora and stop before training.

## Active Constraints
- Follow `plans/stage2_data_scale_single_plus_mcts_search_plan.md`; this phase is data expansion and label generation only.
- Preserve the Qwen2.5-VL runtime, LMMS task scoring, Shared Random-4 checkpoint/threshold/trigger rule, four-action executor, and Phase-56 cap-200 MCTS semantics.
- Freeze candidate identities before observing current dense outcomes, triggers, or fixability. Enforce UID and image-content-group disjointness from the complete current 7,999-record Stage-1 population and all existing Stage-2 bases.
- Keep `SINGLE_FIXABLE`, `MCTS_ONLY_FIXABLE`, `UNRESOLVED`, and `PRESERVATION_C` separate; search only new triggered Dense-W rows and never validation/test labels.
- Four RTX 6000 Ada GPUs are occupied by low-utilization jobs. The user explicitly permits sharing; recheck live headroom before every GPU launch and pass a bounded memory/executor smoke first.

## Current State
- Done: Read the complete plan, Phase-58/56/54/45 memory, promoted lessons, environment policy, and live GPU state.
- Done: Confirmed the transferred source has exactly the original 8,000 images and cannot supply any disjoint new candidate.
- Done: Independent review returned `revise`: task quotas alone do not control source-composition drift; require outcome-blind source-metadata stratification, immutable native row keys/revisions, deterministic backfill, content-hash rejection, and explicit distribution reporting.
- Done: Acquired pinned canonical source subsets and froze exactly 4,000 outcome-blind candidates (2,000 GQA / 1,000 ChartQA / 1,000 TextVQA), each with a unique SHA-256 group and zero overlap with all 8,000 legacy candidates.
- Done: Froze contract `d85b5e9b...c4d94`, passed the shared-memory smoke, and completed dense generation for 4,000/4,000 candidates with zero skips on four direct GPUs.
- Done: Applied the frozen Shared Random-4 gate to all 4,000 rows and searched exactly the 257 triggered Dense-W samples; separately replayed all 1,691 triggered Dense-C rows as FULL preservation supervision.
- Done: Exact search outcomes are 75 `SINGLE_FIXABLE`, 33 additional `MCTS_ONLY_FIXABLE`, and 149 `UNRESOLVED`; all 2,707 retained routes replay exactly and all 1,799 new state shards pass provenance/hash validation.
- Done: Built expanded corpora A/B/C with 1,730/8,578/508 route records over 1,730/773/242 bases, plus 1,123 combined unresolved records. No training or validation/test search/evaluation ran.
- Blocked: no. Phase stop condition reached.
- Most recent useful observation: the new canonical-source pool has a large gate distribution shift (`P(trigger|C)=0.5404` versus `0.0122` old) and lower `P(trigger|W)=0.2951` versus `0.5878` old; bounded fixability among triggered W remains within the prospectively frozen ±0.10 tolerance (0.4202 new versus 0.4822 old).

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Existing source image root contains exactly 8,000 files | `datasets/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713/images` | Rules out invalid reuse as the new disjoint pool | confirmed |
| Existing completed Stage-1 population has 7,999 records and 7,476 content groups | `analysis/dense_failure_stage1/current_dense_8k/` | Defines the exclusion population | confirmed |
| LMMS source repos expose GQA balanced train and TextVQA train; the official ChartQA repository exposes train assets | installed LMMS task YAMLs and source repository metadata | Provides canonical new training sources | confirmed |
| Phase-56 search produced 698 single, 209 MCTS-only, and 974 unresolved under cap 200 | `analysis/dense_failure_stage2/full_corrective_labels/` | Supplies frozen search semantics and old corpora | confirmed |
| Current shared-GPU usage is about 17.3 GiB/GPU of 49.1 GiB | live `nvidia-smi` at phase start | Makes shared execution plausible but still smoke-gated | confirmed |
| Frozen gate coverage and V1 intervention differ sharply by dataset | Phase-54/58 metrics | Task quotas alone cannot make the later comparison a pure scale-only claim | confirmed |
| New candidate manifest has 4,000 unique UIDs/groups and zero legacy overlap | `analysis/dense_failure_stage2/data_scale_search/manifests/candidate_audit.json` | Freezes the label-blind population before any dense/trigger/fixability observation | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Reuse the transferred 8K source as a new pool | Every available image belongs to the existing Stage-1 identity population | supported | source file count and Phase-45 manifest | Download genuinely new train identities and freeze before any labels | Do not resample or relabel existing 8K rows as scale-up data |
| First candidate-freeze invariant assumed the portable provenance file had 8K rows | Portable provenance has 10K rows; current authoritative candidates have 8K | supported | line counts and regression test | Join portable source metadata onto the exact current 8K UID set | Do not use all 10K portable rows as the exclusion/reference population |
| First byte freeze treated source IDs as image groups and rejected duplicated SHA groups late | Different native IDs can contain identical bytes | supported | fail-closed duplicate-group check | Select/deduplicate on SHA-256 group and backfill deterministically | Do not equate native image IDs with content groups |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Metadata-stratified canonical train sources, 2K GQA / 1K ChartQA / 1K TextVQA | Matches task mixture, reduces measurable source shift, and supplies new identities | Executes the named scale-up plan without evaluation leakage | high | selected after independent review |
| Unstratified canonical 50/25/25 pool | Matches task counts but not source/question composition | Supplies new identities with a weaker comparison | high | rejected in favor of metadata stratification |
| Use only locally transferred records | No download | Cannot satisfy new-population disjointness | low | rejected |
| Change the dataset mixture to whichever source is cheapest | Easier acquisition | Confounds task coverage with data scale | medium | rejected unless source availability blocks the matched mixture |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: complete one larger, executor-consistent Stage-2 label corpus and stop; the action is complete and the next bottleneck is an explicitly authorized matched training comparison.
- Relevant memory item used: Phase 58 showed narrow correction and no GQA intervention, while Phase 56 showed strongly task-dependent bounded support.
- Confirmed observation: the new pool contributes 75 single-fixable and 33 MCTS-only bases, including 44 new bounded-fixable GQA bases, while preserving all 1,691 triggered Dense-C rows.
- Unverified interpretation: the larger single corpus may improve learned correction; route-level oracle support does not establish that it will generalize.
- Diagnosis: the data-scale action succeeded; the frozen gate's canonical-source trigger shift is supported, while the causal source of V1's narrow behavior remains unknown.
- Evidence path if diagnosis is not unknown: `analysis/dense_failure_stage2/data_scale_search/metrics/old_vs_new_yield.csv` and `dataset_breakdown.csv`.
- Viable alternatives considered: later Scaled-Single (A+B) versus the same matched setup with MCTS Corpus C added.
- Chosen action: stop. If separately authorized, train Scaled-Single first under the frozen V1 architecture/Stage-1/optimizer/loss/validation contract, then test `+MCTS` as a controlled second experiment.
- Strongest objection: the 4K canonical-source pool differs sharply in dense outcome and gate-trigger composition from the historical source, so later performance cannot be attributed to sample count alone.
- How this differs from failed attempts: the new corpora add disjoint, current-runtime, exact-replay supervision without changing the gate, executor, search budget, or held-out populations.
- Automatic execution authorized: no further research action is authorized.
- Authorization basis: the requested data-scale plan is complete.
- Stop condition: reached; wait before any Stage-2 training or evaluation.

## Latest Research-Action Result
- Action taken: froze and executed the 4,000-candidate canonical-source dense/gate/single+MCTS scale-up once on four shared direct GPUs, then aggregated provenance-separated corpora.
- Result: 4,000 dense rows (3,129 C / 871 W), 257 triggered W, 1,691 triggered C, 75 single-fixable, 33 MCTS-only, and 149 bounded-unresolved. New bounded fixability is 108/257 = 0.4202; expanded A/B/C contain 1,730/8,578/508 routes over 1,730/773/242 bases.
- Evidence saved: `analysis/dense_failure_stage2/data_scale_search/`, especially `artifact_manifest.json`, `metrics/`, `combined_corpora/corpus_manifest.json`, and `summaries/data_scale_search_summary.md`.
- Failure or issue: no accepted-row failure. Two orchestration-only CLI defects stopped before GPU/scientific output (missing explicit worker rank and an unbound local aggregation config); exact wrapper invocations completed the frozen implementation without changing the scientific contract. The accepted artifact manifest passes all 1,837 hashes, and 39 focused/regression tests pass.
- Lesson learned: outcome-blind canonical-source expansion produced useful new corrective supervision, including 44 GQA bases, but also exposed a severe frozen-gate source shift. Treat this as canonical-source scale-up evidence, not a pure count-only replication.
- Next implication: preserve A/B/C provenance; if separately authorized, compare Scaled-Single A+B first and then add C with every other training/evaluation choice fixed.

## Post-completion Gate-Shift Interpretation
- Deliberation mode: standard; one cheap read-only score-distribution diagnostic.
- Confirmed observation: with the same bound gate/checkpoint/normalization/threshold, new correct ChartQA and TextVQA trigger at 0.8654 and 0.8522 versus 0.0088 and 0.0138 in old train. Their median maximum scores are 1.0000 and 0.9984, and most false triggers begin at L0. New maximum-score failure AUROC is 0.3469 ChartQA, 0.7453 GQA, 0.6846 TextVQA, and 0.4056 overall versus 0.9900/0.8906/0.9873/0.9529 old.
- Diagnosis: supported canonical-source/OOD gate generalization failure; suspected source/image-format shortcut reliance because new correct ChartQA/TextVQA visual-token regimes shift toward old wrong regimes. Exact causal feature remains unknown.
- Evidence: `analysis/dense_failure_stage2/data_scale_search/diagnostics/posthoc_gate_shift_diagnostic.md`.
- Chosen action and stop: retain the accepted labels/corpora but do not interpret Phase-59 trigger counts as calibrated failure risk on the new source; no retraining, retuning, or additional experiment is authorized by this interpretation request.
