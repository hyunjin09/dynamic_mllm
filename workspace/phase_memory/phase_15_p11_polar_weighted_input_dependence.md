# Phase 15: P11 POLAR-Weighted Input Dependence Memory

## Current Objective

Execute the bounded P11 A–E diagnostic to determine whether the frozen
POLAR-compatible ALL-ON down-weighting reveals question-dependent routing beyond
global and dataset-only priors. Full predictor training remains out of scope.

## Active Constraints

- Preserve the P10 question-only Qwen3 encoder, POLAR-style predictor, direct
  factorized 28-bit head, regenerated GQA/TextVQA/ChartQA labels, 7K/1K split,
  deterministic max-50 route sets, optimization settings, initialization,
  checkpoint rule, and inference protocol.
- Change only route weights: when ALL-ON and a cheaper valid route coexist,
  assign ALL-ON relative weight `0.3`, all other valid routes `1.0`, then
  normalize within the input for both objectives.
- Run only P11 label geometry, bias-only baselines, two-epoch matched smoke,
  held-out input-dependence diagnostics, and a prespecified 60–120-record real
  execution subset.
- Do not use node04. Do not launch full training or redesign the factorized head.
- Preserve all P10 artifacts unchanged.

## Current State

- Done: P10 matched smoke and constant-policy audit; all P11 A–E implementation,
  label geometry, bias baselines, matched weighted smoke, aligned/shuffled
  diagnostics, frozen 60-record execution, independent review, and report.
- In progress: none.
- Blocked: full matched training is not justified and remains unauthorized;
  the proposed structured-head pivot requires explicit approval.
- Most recent useful observation: aligned questions improve exact set-NLL over
  shuffled questions, but 57/60 executed masks remain ALL-ON and the other
  three masks remain wrong.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Exact set-NLL P10 validation Hit@1 was `0.5733`, but all 18 executed masks were ALL-ON | `reports/binary_polar_p10_smoke_results.md` | Establishes that objective superiority alone is not input-conditioned routing evidence | confirmed |
| Constant ALL-ON reproduced exact set-NLL Hit@1; constant ALL-OFF reproduced BCE Hit@1/Hamming | `outputs/binary_polar/p10_smoke/constant_policy_audit_v1.json` | Requires global and dataset-only prior baselines in P11 | confirmed |
| Exact set-NLL implementation and BF16 validation path pass the refreshed P10 gate | `outputs/binary_polar/preflight/repair_v2/p10_readiness_gate_v2.json` | Allows P11 to focus on weighting/input dependence rather than executor repair | confirmed |
| Validation ALL-ON coverage is `0.5812`; ALL-ON plus cheaper-valid prevalence is `0.5789` | `outputs/binary_polar/p11/label_geometry_v1.json` | Confirms a strong objective shortcut opportunity | confirmed |
| Exact aligned/shuffled set-NLL is `14.8699/15.5089` | `outputs/binary_polar/p11/question/exact_set_nll_conditioning_v1.json` | Shows question-associated probability signal | confirmed, bounded smoke |
| Exact execution is 50% like FULL; 57/60 ALL-ON; three non-FULL masks are uncached and wrong | `outputs/binary_polar/p11/execution/exact_set_nll_v1.json` | Fails the useful decoded-routing gate | confirmed, bounded execution |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Equal-weight P10 smoke | Both objectives decoded a single constant mask | supported smoke constant-policy collapse | P10 report and constant-policy audit | Test POLAR weighting against explicit bias-only and shuffled-input baselines before scaling | Do not infer route learning from Hit@1 alone or launch full training from P10 |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| P11 A–E bounded diagnostic | Directly tests the prespecified nearest POLAR-compatible remedy and input dependence | Whether full matched training is scientifically justified | medium | completed: Outcome C |
| Full matched training | Could test scale effects | Final objective comparison | high | rejected by P11 gate |
| Structured route head | Could model route dependencies | Factorization limitation | medium | proposal only; explicit approval required |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: determine whether POLAR-weighted exact
  set-NLL learns useful question-dependent nonconstant routing; the bottleneck
  is attribution against input-independent priors, not implementation validity.
- Relevant memory item used: P10's exact-set advantage was entirely compatible
  with a constant ALL-ON shortcut.
- Confirmed observation: the P10 models collapsed to constant ALL-OFF and
  ALL-ON masks under equal weighting.
- Unverified interpretation: down-weighting ALL-ON where cheaper valid masks
  exist may expose question-conditioned signal.
- Diagnosis: supported constant-mode behavior; its cause remains unknown.
- Viable alternatives considered: execute the frozen P11 diagnostic, launch
  full training, or redesign the head. Only P11 is both authorized and capable
  of resolving the current attribution problem at bounded cost.
- Chosen action: implement and execute P11 A–E exactly as frozen.
- Strongest objection: a 300/150 two-epoch smoke may underestimate conditional
  learning even if the full training set contains usable signal; P11 is an
  admission diagnostic, so a negative result cannot prove the architecture is
  universally incapable.
- How this differs from failed attempts: it adds only the fixed POLAR weighting
  and directly compares question inputs with global, dataset-conditioned, and
  shuffled-input controls.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to read and perform `plans/p11.md`.
- Stop condition: reached with Outcome C; full training and architecture-pivot
  execution were not started.

## Latest Research-Action Result

- Action taken: implemented and executed the complete bounded P11 A–E protocol.
- Result: Outcome C. Exact set-NLL contains aligned-question signal but its
  factorized top-1 behavior remains nearly constant and behaviorally useless.
- Evidence saved: `reports/binary_polar_p11_results.md` and
  `outputs/binary_polar/p11/`.
- Failure or issue: the selected exact checkpoint predicts ALL-ON on 98% of
  route-validation inputs and 95% of the execution subset; its three non-FULL
  masks are uncached and remain wrong. Diagnosis is supported decoded-policy
  collapse; factorization as the unique cause remains unverified.
- Lesson learned: common set-NLL and aligned/shuffled probability differences
  are insufficient admission evidence without nonconstant, useful execution.
- Next implication: do not full-train. The separate architecture proposal may
  be considered only with explicit approval.
