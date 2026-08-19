# Phase 16: P12 Structured Binary Head Memory

## Current Objective

Execute the bounded P12 architecture-isolation experiment: replace only the
direct factorized 28-bit output head with canonical maximal-run boundary and
segment-operation heads, then test whether the existing question-conditioned
probability signal yields useful nonconstant executable masks.

## Active Constraints

- Preserve the P11 GQA/TextVQA/ChartQA labels, image-group split, deterministic
  max-50 route sets, 300/150 smoke identities, 60-record execution manifest,
  route weights, frozen Qwen3 encoder, POLAR layer encoder, optimizer, seed,
  checkpoint rule, BF16 contract, and Qwen2.5-VL executor.
- Canonical representation is lossless maximal contiguous runs; do not impose
  POLAR `Kmax=4`, beam search, threshold tuning, or a compute/segment penalty.
- Boundary layer 0 is forced at decode; remaining boundaries use threshold
  `0.5`; operation at each predicted start uses deterministic argmax.
- Run only the two-epoch structured exact-set-NLL smoke and its frozen
  diagnostics/execution. Do not full-train or stack another architecture.
- Do not use node04.

## Current State

- Done: P11 Outcome C and P12 A–G.
- In progress: none; P12 is closed as Outcome B.
- Blocked: none known.
- Most recent useful observation: P12 preserves aligned-question probability
  signal but decodes ALL-ON on 100% of selected validation and execution
  records.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Exact aligned/shuffled set-NLL is `14.8699/15.5089` | `outputs/binary_polar/p11/question/exact_set_nll_conditioning_v1.json` | Establishes probability-level signal P12 must preserve | confirmed |
| P11 selected exact checkpoint is 98% ALL-ON and produces no corrections on 60 executions | `reports/binary_polar_p11_results.md` | Fixes the structured-head comparison baseline | confirmed |
| The same max-50 masks and P11 0.3 ALL-ON weighting are checksum-frozen | `configs/binary_polar_p11_weighted_smoke_v1.yaml` | Prevents a data/loss confound | confirmed |
| Canonical round trip is 237,802/237,802, but median route size is 14 segments | `outputs/binary_polar/p12/segment_geometry_v1.json` | Validates the representation while showing weak segment compression | confirmed |
| P12 aligned/shuffled set-NLL is 17.4918/18.1939, yet both decode ALL-ON | `outputs/binary_polar/p12/structured_conditioning_v1.json` | Signal survives in probability but not top-1 | confirmed |
| P12 executes 60/60 ALL-ON with 50% accuracy and W→C/C→W=0/0 | `outputs/binary_polar/p12/structured_execution_v1.json` | Fails the useful routing gate | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| P11 direct factorized head | Question signal remained in probability mass but top-1 stayed nearly ALL-ON | supported decoded-policy collapse; factorization as unique cause unverified | P11 report | Test one lossless structured representation without other changes | Do not full-train or add multiple new mechanisms |
| P12 canonical structured head | Selected checkpoint decoded ALL-ON for all 150 validation and 60 execution records | supported bounded structured-head collapse; insufficient evidence that any longer training would repair it | P12 report | Close this structured pivot under its frozen gate | Do not full-train P12 or substitute epoch 2 post hoc |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Canonical maximal-run structured head | Encodes complete masks as coherent runs while preserving all supervision | Whether output representation explains the probability-to-decoding gap | medium | rejected by Outcome B |
| Full direct-head training | More optimization could alter P11 result | Scale question | high | rejected by P11 gate |
| Additional structured modules | Could add expressiveness | Broader architecture search | high | prohibited in P12 |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: determine whether a lossless structured
  output inductive bias converts the confirmed input signal into useful masks;
  the first bottleneck is exact probability and round-trip validity.
- Relevant memory item used: P11 probability-level conditioning did not survive
  threshold decoding as useful routes.
- Confirmed observation: direct exact set-NLL is 98% ALL-ON on smoke validation.
- Unverified interpretation: independent bit factorization is the primary
  probability-to-decoding bottleneck.
- Diagnosis: supported top-1 collapse; representation-specific cause unknown.
- Viable alternatives considered: execute the approved P12 structured smoke,
  scale P11, or add broader modules. Only P12 isolates the unresolved cause
  within the approved budget.
- Chosen action: completed P12 A–G exactly once; stop the structured-head pivot.
- Strongest objection: maximal-run labels can still have many transitions, and
  a boundary-plus-operation likelihood may trade one factorization for another
  without improving utility-aligned decoding.
- How this differs from failed attempts: it changes only the output
  representation and complete-route probability; data, features, weights,
  optimization, and executor remain fixed.
- Automatic execution authorized: yes
- Authorization basis: explicit request to read and perform `plans/p12.md`.
- Stop condition: save the geometry, round-trip, training, shuffle, execution,
  and Outcome A/B/C evidence; do not start full training.

## Latest Research-Action Result

- Action taken: audited 237,802 routes, implemented/tested canonical exact-set
  training, ran the matched two-epoch smoke, aligned/shuffled audit, and all 60
  frozen executions.
- Result: Outcome B. Probability conditioning survives, but the selected head
  is a constant ALL-ON policy and produces no behavioral or compute gain.
- Evidence saved: `reports/binary_polar_p12_results.md`,
  `outputs/binary_polar/p12/analysis_manifest_v1.json`, and the P12 output tree.
- Failure or issue: complete-mask collapse; diagnosis is supported for the
  bounded canonical head. The effect of substantially longer training remains
  unknown and is not an authorized repair.
- Lesson learned: switching from direct bits to maximal-run boundaries is not
  sufficient to translate the observed question-conditioned probability mass
  into useful top-1 routes; route labels themselves also exhibit little
  low-segment structure.
- Next implication: do not full-train or stack another structured head. A new
  next research action, if desired, must reconsider feature/supervision
  deployability rather than silently extending P12.
