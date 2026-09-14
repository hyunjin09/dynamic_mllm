# Phase 56: Full Corrective Label Generation Memory

## Current Objective
Generate replay-valid corrective supervision for all 1,881 frozen train triggered Dense-W samples and FULL-suffix preservation supervision for the 39 train triggered Dense-C samples, while keeping single, MCTS, and preservation sources separate. Stop before Stage-2 training.

## Active Constraints
- Follow `plans/full_corrective_label_generation_plan.md`; do not change Stage 1, its threshold/trigger map, split membership, current LMMS labels, model/runtime, executor semantics, or four-action space.
- Preserve the exact FULL prefix before each frozen trigger layer; run exhaustive singles first and cap MCTS at exactly 200 only for single-unresolved W rows.
- Reuse Phase-55 pilot rows only when transcript, route replay, tensor, model, code, and source provenance are compatible with the cap-200 contract.
- Search no validation/test sample. Run no Stage-2 training, downstream routing evaluation, or external evaluation.
- Use four direct GPUs on this non-Slurm server after a contract-bound implementation smoke passes.

## Current State
- Done: Read the active plan, Phase-55 memory/evidence, Phase-54 population handoff, current runtime/compute policy, and relevant promoted state.
- Done: Independently audited the Phase-55 transcript at cap 200. One GQA rescue first appeared at iteration 294 and must be excluded; the reusable pilot becomes 35 SINGLE_FIXABLE, 21 MCTS_ONLY_FIXABLE, and 64 UNRESOLVED with 42 canonical retained MCTS routes.
- Done: Required read-only independent review ranked provenance-safe cap-200 import above rerunning all 120 and confirmed every canonical cap-200 route has a replay-valid pilot shard.
- Done: Froze accepted contract `6489a0b39af3efb22bcb527d3b43c474dc7bcb01cc453445aab45f420eb9905b`, passed the 8/8 implementation smoke, imported 120/120 pilot rows, and completed 1,800/1,800 fresh W/C rows once across four ranks with zero failures.
- Done: Aggregated 698 SINGLE_FIXABLE, 209 MCTS_ONLY_FIXABLE, and 974 UNRESOLVED W rows; replay-audited 8,109 routes and 208,280 routed states; and passed the 977-file final artifact audit.
- Stopped: No Stage-2 training, Stage-1 change, validation/test search, downstream routing evaluation, or external evaluation ran.
- Blocked: No.
- Most recent useful observation: Full bounded correctability is 0.4822, close to the cap-200 pilot's 0.4667, but varies from 0.1986 GQA to 0.6448 TextVQA and declines substantially with later trigger depth.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Frozen Phase-54 train handoff contains 1,881 triggered W and 39 triggered C | `analysis/dense_failure_stage1/trigger_map/` | Defines the complete authorized population | confirmed |
| Phase-55 pilot contract and all 106 artifact hashes pass | `analysis/dense_failure_stage2/corrective_search_pilot/artifact_manifest.json` | Permits evidence reuse if cap semantics match | confirmed |
| Cap-200 transcript gives 35/21/64; `gqa:gqa_gh_09367372` succeeds only at iteration 294 | Phase-55 raw sample/search rows; independent review packet | Requires exclusion of one post-cap rescue and its tensors | confirmed |
| All 42 canonical cap-200 retained MCTS routes have replay-valid state rows | Phase-55 state index/shards; independent reviewer | Avoids an unnecessary 120-sample rerun | confirmed |
| Focused/inherited tests pass | `tests/test_full_corrective_label_generation.py`; `tests/test_corrective_search_pilot.py`; `tests/test_stage1_trigger_map.py` | Establishes pure contract, cap, completion, corpus, and upstream regression invariants | confirmed |
| Full frozen W classes are 698 single / 209 MCTS-only / 974 unresolved | `analysis/dense_failure_stage2/full_corrective_labels/metrics/overall_fixability.csv` | Establishes full-cohort bounded oracle support | confirmed |
| All 8,109 retained routes replay exactly into 946 shards and 208,280 state rows | `analysis/dense_failure_stage2/full_corrective_labels/artifact_manifest.json` | Makes corpora A/B/C provenance-valid for future training | confirmed |
| GQA/ChartQA/TextVQA support is 0.1986/0.4854/0.6448 | `analysis/dense_failure_stage2/full_corrective_labels/metrics/dataset_breakdown.csv` | Requires dataset-specific reporting and limits general claims | confirmed |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Treat all 22 Phase-55 MCTS rescues as reusable | One rescue first occurs after the new cap | supported | Phase-55 search transcript at iteration 294 | Reinterpret the full transcript prospectively at cap 200 and exclude all post-cap routes/states | Do not import Phase-55 cap-300 classes unchanged |
| First Phase-56 smoke contract | Fresh route serialization raised `KeyError: trigger_layer` after inference | supported: imported and fresh manifests use different frozen trigger-field aliases | `analysis/dense_failure_stage2/full_corrective_labels_invalid_trigger_alias_0931226b/INVALID_RUN.md` | Accept both aliases through one tested accessor, quarantine the invalid attempt, and refreeze before scientific execution | Do not use contract `0931226b...491dd` or its partial smoke outputs |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Import cap-200-equivalent pilot evidence, execute only remaining rows | Exact canonical routes and states exist | Completes full corpus without redundant pilot GPU work | high | completed |
| Rerun all 120 pilot rows at cap 200 | Avoids tensor import | Fallback if any provenance assertion fails | high | rejected unless import validation fails |
| Import cap-300 pilot unchanged | Cheapest | Violates authorized cap | low | rejected |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: scale the validated corrective search while retaining exact route/state provenance and avoiding post-cap leakage.
- Relevant memory item used: Phase 55 found meaningful single and MCTS-only rescue and selected a 200-iteration cap because the 200→300 gain was only 0.83 percentage points.
- Confirmed observation: The cap-200 canonical pilot subset is fully represented by replay-valid Phase-55 state shards.
- Unverified interpretation: Full-cohort fixability may differ from the balanced 120-sample pilot projection.
- Diagnosis: supported
- Evidence path if diagnosis is not unknown: Phase-55 raw search rows and state shards; independent review outcome recorded in the Phase-56 frozen contract.
- Viable alternatives considered: provenance-safe import plus remaining execution; rerun all pilot rows; invalid cap-300 import.
- Chosen action: freeze a new contract, import exact cap-200 pilot tensor subsets into new shards, pass an 8-row 4W+4C smoke, execute 1,761 remaining W searches and 39 C preservation replays on four GPUs, then aggregate/audit the three corpora.
- Strongest objection: Phase-55 cap-300 retention could have omitted a canonical pre-200 route; the transcript/shard audit resolved this by identifying all 42 canonical routes in replay-valid shards.
- How this differs from failed attempts: post-200 evidence is explicitly excluded and every imported/new shard is rebound to the Phase-56 contract.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to read and perform `plans/full_corrective_label_generation_plan.md`.
- Stop condition: full W/C coverage, exact route replay, three separated corpora, metrics/figures/decision summary, artifact hashes, and compact state update are complete.

## Latest Research-Action Result
- Action taken: Imported the exact cap-200 Phase-55 subset, ran 1,761 remaining triggered-W searches and 39 triggered-C preservation replays on four direct GPUs, then built and audited three separate Stage-2 corpora.
- Result: All 1,881 W rows end in exactly one class: 698 SINGLE_FIXABLE, 209 MCTS_ONLY_FIXABLE, and 974 UNRESOLVED. Corpus A/B/C contain 39/7,628/442 routes and 400/199,193/8,687 state rows. All 8,109 routes replay exactly; the 977-file artifact manifest passes.
- Evidence saved: `analysis/dense_failure_stage2/full_corrective_labels/decision_summary.md`, all required manifests/routes/states/metrics/corpora/figures, and `artifact_manifest.json` (SHA-256 `a26d31e1d118e811ef107152213edcef3cd48ca3ed83ec17b18447ff490ff4d4`).
- Failure or issue: The first implementation-only smoke contract failed closed on a trigger-field alias, was quarantined, repaired with a regression test, and replaced before accepted execution. The repository-wide test collector still has unrelated historical import collisions; the relevant scoped suite passes 57/57.
- Lesson learned: The balanced pilot estimated total bounded support well, but full support is strongly task- and trigger-depth-dependent. Simple corrective support is substantial; MCTS adds a smaller distinct cohort worth preserving separately.
- Next implication: Stop. A separately authorized Stage-2 V1 may train on preservation + single corpora; adding MCTS corpus C is a later controlled V2, not an automatic continuation.
