# Phase 10: Query-Conditioned Visual Refinement Memory

## Current Objective

Execute one bounded frozen-model GQA falsification experiment testing whether a
single fixed-budget post-question refinement of already encoded visual tokens
has target-question-specific answer value beyond equal-compute replay.

## Active Constraints

- Preserve the pinned Qwen2.5-VL-7B-Instruct model and all v2-v4 closures.
- No training, vision re-encoding, crops, new visual evidence, external models,
  routing, probes, or TextVQA replication.
- Freeze 100 new GQA images with exactly two questions each before outcomes.
- Use at most three architecture-selected replay layers, one replay step, the
  validated accepted-reference likelihood, and equal replay compute for
  unconditioned, target-question, and paired-other-question variants.
- Stop before scientific evaluation if replay validity, deterministic parity,
  common-padding identity, answer isolation, or numerical-stability gates fail.

## Current State

- Done: operator review/freeze, 100-image manifest, implementation/tests,
  12-image preflight, 200-question sweep, frozen aggregation, and decision.
- In progress: none.
- Blocked: no execution blocker; the direction is scientifically closed under
  the frozen kill rule.
- Most recent useful observation: layer 4's conditioning raw mean was positive,
  but the robust center was near zero and the paired-other-question contrast
  failed; layers 12/20 did not support conditioning value.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Same-image visual states and WRITE are bitwise identical under common right-padding at the seven validated layers. | `reports/v4_common_padding_preflight.md`; `outputs/v4_discovery/preflight/v4_common_padding_preflight_v1.json` | Target and paired-other replay can begin from exactly the same visual evidence and shape. | confirmed |
| Decoder layer is pre-RMSNorm attention plus residual and row-wise post-attention RMSNorm/MLP plus residual; eager attention consumes additive masks and shared MRoPE embeddings. | pinned Transformers source; `outputs/stage_a/architecture_causal_graph.md` | A frozen visual-row replay can be defined without a trainable module or invented positional system. | confirmed |
| Cached pre-layer injection, dense suffix, FULL parity, READ identity, and WRITE identity are validated. | `interventions/four_state.py`; `interventions/prompt_cache.py`; Stage A/v4 preflight artifacts | Refined visual rows can be inserted at a validated causal boundary and scored through an unchanged suffix. | confirmed |
| v4 local-action oracle headroom was small despite query-associated action disagreement. | `reports/v4_cost_utility_reanalysis.md` | The experiment must test a new question-to-visual capability, not another keep/drop route. | confirmed discovery |
| Frozen replay preflight passed every gate exactly. | `reports/query_refinement_preflight.md` | Scientific failure is interpretable rather than an instrumentation failure. | confirmed |
| No anchor passed target conditioning plus target specificity; behavior regressed by one net answer per anchor. | `reports/query_refinement_gqa_discovery.md`; `outputs/query_refinement/analysis_v1/layer_summaries.json` | Triggers the frozen kill rule and closes TextVQA/confirmation. | confirmed discovery |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Privileged region/crop refinement proposal | Region selection would add privileged evidence and confound target conditioning. | supported design flaw | `reports/dynamic_mllm_v2_v4_synthesis.md`; `workspace/dynamic_mllm_next_direction.md` | Use every already encoded visual row and vary only replay access to target versus paired question. | Crops, detectors, OCR, re-encoding, or answer-derived token selection |
| Parallel execution on `node03` | Three jobs exposed zero usable CUDA devices before model/sample loading. | supported cluster-node failure | `runs/query_refinement_preflight_v1/slurm.log`; `outputs/query_refinement/shards_v1/shard_{01,02,03}/failure.json` | Stop node-local retries; complete unchanged work on validated `node02`. | Further `node03` retries in this action |
| Frozen target-question replay | Layer 4 raw mean was positive but heavy-tail-sensitive; specificity failed; layers 12/20 failed. | supported scientific failure under this operator | `outputs/query_refinement/analysis_v1/layer_summaries.json` | Stop rather than search layers, depth, or subgroups. | TextVQA replication or post-hoc layer/type selection |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Output-boundary native visual replay | Runs the frozen layer once from native `H_l`, retains only visual post-layer rows, replaces those rows in stock `H_{l+1}`, and resumes at `l+1`. | Clean target-versus-unconditioned and target-versus-other contrasts without applying the same block twice. | medium | tested; stopped |
| Full-prompt block replay with discarded text outputs | Uses an unmodified layer call and equal shape. | Simpler implementation but less literal text-state isolation. | medium | runner-up |
| Sequential same-block visual replay | Adds a true second state update at the same layer. | Controls generic extra visual depth, but duplicates the layer/MLP off manifold. | medium | rejected after review |
| Manual visual-query-only attention | Computes only needed visual queries. | Lower replay FLOPs but adds grouped-KV/MRoPE implementation risk. | medium | rejected |

## Next-Step Decision

- Deliberation mode: deep.
- Active objective and bottleneck: interpret the completed frozen result and
  decide whether the confirmation gate opens.
- Relevant memory item used: common padding restores exact visual identity;
  local keep/drop actions lack practical headroom.
- Confirmed observation: all validity gates passed, but zero anchors passed the
  frozen joint conditioning/specificity and behavior rule.
- Unverified interpretation: a substantially different trained refinement
  architecture might succeed; this experiment cannot decide that question.
- Diagnosis: supported failure of this frozen operator, unknown for the broader
  architecture family.
- Viable alternatives considered: honor the stop rule or post-hoc select layer
  4/relation categories; the latter is invalid under the frozen anti-search
  constraint.
- Chosen action: close with `STOP_QUERY_REFINEMENT_DIRECTION`; do not open
  TextVQA, confirmation, deeper replay, or training.
- Strongest objection: layer 4's conditioning CI is entirely positive. It does
  not reverse the decision because the mean misses `0.05`, median/trimmed means
  are near zero, the top 5% drive the effect, and target-versus-other crosses
  zero.
- How this differs from failed attempts: it adds a question-to-existing-visual
  path without new pixels, privileged regions, learned capacity, or route
  selection.
- Automatic execution authorized: yes, for this one bounded experiment only.
- Authorization basis: explicit user task `Test Query-Conditioned Visual
  Refinement with a Frozen MLLM`.
- Stop condition: reached; zero of three anchors passed, versus two required.

## Latest Research-Action Result

- Action taken: froze, validated, executed, and analyzed the one authorized
  100-image GQA query-refinement experiment.
- Result: full integrity; `STOP_QUERY_REFINEMENT_DIRECTION`.
- Evidence saved: `outputs/query_refinement/`,
  `reports/query_refinement_preflight.md`,
  `reports/query_refinement_gqa_discovery.md`, and
  `reports/query_refinement_gqa_decision.md`.
- Failure or issue: scientifically, target specificity and robust practical
  value failed; operationally, node03 CUDA allocation failed before outcomes
  and was bypassed by validated-node serialization.
- Lesson learned: adding a clean question-to-visual edge to a frozen native
  block is technically feasible, but its answer-likelihood effects are still
  heavy-tailed and do not prefer the target question robustly.
- Next implication: stop; any different architecture/training direction would
  require a new explicit strategic amendment.
