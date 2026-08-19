# Phase 17: P13 Multimodal Input Isolation Memory

## Current Objective

Execute the bounded P13 input-information isolation experiment: compare the
same direct exact-valid-set predictor with question-only, image-only, and
image+question inputs, then determine whether native visual context improves
complete-mask routing rather than only probability calibration.

## Active Constraints

- Preserve P11/P12 decisions and all P10-P12 artifacts.
- Reuse the frozen GQA/TextVQA/ChartQA max-50 route sets, image-group split,
  P11 0.3 ALL-ON weighting, 300/150 smoke identities, two epochs, seed,
  optimizer, checkpoint rule, direct 28-bit head, and exact valid-set NLL.
- Use frozen Qwen2.5-VL projected visual-token rows available before decoder
  layer 0; do not use answers, route outcomes, generated text, future decoder
  states, an external vision model, or a learned vision backbone.
- Freeze the visual-feature cache and modality permutations before training
  outcomes are inspected.
- Run the 60-record Qwen execution only if image+question passes the prospective
  prediction-level admission gate.
- Do not full-train, retune decoding, change the objective, reuse the P12 head,
  or run on node04.

## Current State

- Done: P11 Outcome C; P12 Outcome B; all authorized P13 implementation,
  visual caching, preflight, three matched smokes, and modality-shuffle
  diagnostics.
- P13 decision: Outcome B — more input information changes probabilities but
  not routes.
- Execution status: the prospective prediction gate failed, so the frozen
  60-record Qwen execution was not run.
- Blocked: full P13 training and any route-utility objective pivot are not
  authorized or scientifically admitted by this result.
- Most recent useful observation: Image+Question lowers aligned set-NLL to
  `14.4944` from Question-only `14.8699`, and image shuffling worsens it to
  `14.8748`, but Image+Question decodes ALL-ON on `150/150` records.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| P11 aligned/shuffled set-NLL is 14.8699/15.5089 but top-1 is 98% ALL-ON | `reports/binary_polar_p11_results.md` | Question information exists but is not deployable | confirmed |
| P12 selected top-1 is 150/150 and 60/60 ALL-ON | `reports/binary_polar_p12_results.md` | Output restructuring did not repair collapse | confirmed |
| `build_binary_inputs` exposes projected visual rows before decoder layer 0 | `binary_policy/executor/inputs.py` | Supplies a native, deterministic, pre-routing visual feature | confirmed |
| `PolarLayerEncoder` already has a separate image projection path | `binary_policy/predictor.py` | Allows minimal fusion without a deep multimodal module | confirmed |
| Frozen native visual cache passes shape, finite-value, leakage, repeat, and checksum gates | `outputs/binary_polar/p13/visual_features_v1/cache_audit_v1.json` | The modality test used valid pre-routing features | confirmed |
| Image+Question has lower aligned NLL than Question-only and both frozen shuffles | `outputs/binary_polar/p13/conditioning_diagnostic_v1.json` | Visual information changes valid-set probability mass | confirmed |
| Image+Question selects one mask, ALL-ON, for all 150 validation records | `outputs/binary_polar/p13/conditioning_diagnostic_v1.json` | Probability improvement does not survive decoding | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| P11 direct question-only head | probability signal but nearly constant decoded policy | supported decoded-policy collapse; missing image context unknown | P11 report | Isolate input modality with the same direct head | Do not scale question-only training |
| P12 structured head | stronger constant ALL-ON collapse | supported bounded structured-head failure | P12 report | Return to direct head and change only inputs | Do not stack output modules |
| P13 native multimodal input | NLL improves but selected IQ checkpoint is 100% ALL-ON | supported decoded-policy collapse despite visual probability signal | P13 report and diagnostic | Missing visual input is not sufficient to repair direct route generation | Do not full-train or substitute epoch 2 post hoc |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Native projected visual-token rows | Maximum native pre-routing visual information; no second tower | Whether image context improves route prediction | medium | selected |
| Mean-pooled projected visual row | Smaller cache and predictor cost | Same question with a lossy image summary | low | rejected: false-negative risk |
| Pre-projector vision representation | May retain vision detail | Alternate visual representation | medium | rejected: less native to decoder and adds projection ambiguity |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: P13 is complete; the remaining bottleneck is
  that input-conditioned probability signal does not yield a useful complete
  mask under the direct valid-set generation objective.
- Relevant memory item used: P11, P12, and now P13 all preserve probability
  conditioning while selected masks remain essentially constant ALL-ON.
- Confirmed observation: native visual rows improve set-NLL and aligned images
  beat shuffled images overall, but IQ Hit@1 equals constant ALL-ON and route
  diversity is zero.
- Interpretation: visual input supplies probability-level information, but it
  is not sufficient to repair decoded route generation under the bounded
  matched setup.
- Diagnosis: supported for this frozen smoke; the broader cause remains
  unknown.
- Viable alternatives considered: none within the authorized P13 action. A
  candidate-route utility/validity objective is a strategic pivot requiring
  explicit approval.
- Chosen action: stop after Outcome B and preserve the failed admission gate.
- Strongest objection: two epochs may underoptimize the multimodal model, but
  scaling is not justified when the prospectively selected checkpoint is
  exactly constant and the later diverse checkpoint degrades every frozen
  primary route metric.
- How this differs from failed attempts: P13 isolated input information with
  native image tokens rather than changing output structure.
- Automatic execution authorized: no further research action.
- Authorization basis: P13's stop rule after Outcome A/B/C.
- Stop condition: reached.

## Latest Research-Action Result

- Action taken: cached frozen native projected visual rows; trained matched
  Question, Image, and Image+Question two-epoch direct-head smokes; ran the
  frozen four-condition modality-shuffle diagnostic.
- Result: Outcome B. IQ aligned set-NLL `14.4944`; question-only `14.8699`;
  IQ question-shuffled `14.5019`; image-shuffled `14.8748`; both-shuffled
  `14.9335`. IQ decoded ALL-ON for `150/150`, with Hit@1 `57.33%` and nearest
  Hamming `3.693`.
- Evidence saved: `reports/binary_polar_p13_results.md`,
  `outputs/binary_polar/p13/analysis_manifest_v1.json`, and all paths indexed
  there.
- Failure or issue: the execution-admission conditions for nonconstant routes,
  route diversity, and decoded improvement failed.
- Lesson learned: adding native visual context changes route-set probability
  mass but does not repair the P11/P12 probability-to-route gap.
- Next implication: do not execute the 60-record subset, do not full-train P13,
  and do not implement the optional objective pivot without approval.
