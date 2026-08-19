# v4 Strategy Transition

## Decision context

The v3 harmfulness-confirmation direction is closed. The independent,
outcome-blind redesign extracted 28,000 exactly reconstructing READ/WRITE
residual pairs from 4,000 new calibration images, so the negative decision is
not an intervention-extraction failure. It is a specificity-control failure:

- the paired real-residual donor null requires a global caliper of `3.09375`,
  driven by persistent rare-shape geometry;
- fixed 32-row, exact-native-shape, and native-row covariance models all fail
  the frozen coverage or `0.50` cross-validated native-fidelity gate, including
  the rank-1,024 extension.

Those gates will not be weakened, and the planned maximum-over-21 v3 held-out
confirmation, Stage D, and READ-specific harmfulness search remain closed.

## What remains scientifically supported

1. The pinned four-action intervention is valid in its documented stock-eager
   execution domain: branches share one cached dense prestate, modify one layer,
   and use an unchanged dense suffix and identical answer scoring.
2. READ and WRITE reconstruct at their validated hooks and through the suffix;
   instrumented FULL parity and deterministic four-state behavior pass.
3. The inspected v2/v3 discovery data contain a heterogeneous four-action value
   landscape poorly recovered by fixed per-layer schedules. This is discovery
   evidence, not confirmation.
4. The v2 TextVQA layer-0 reference-support effect replicated but did not beat
   either frozen structured null (`Outcome B`). It is not a confirmed
   answer-misaligned READ effect.
5. Under common right-padding, the existing same-image diagnostic produced
   bitwise-identical visual states and WRITE at all seven nonterminal layers.
6. Large outcome-uninspected multi-question pools exist. The reserved audit has
   800 groups per task; the broader metadata audit found 9,800 eligible GQA and
   1,243 eligible TextVQA multi-question images before reservation.

## Why v4 is a distinct question

V3 asked whether the best FULL-relative suppression, after a 21-way search,
was more answer-aligned than generic matched residual perturbations. That claim
requires valid structured specificity nulls, and those nulls failed.

V4 does not ask whether suppression is beneficial, harmful, or unusually good
relative to arbitrary perturbations. It holds the image and visual computation
fixed and asks whether different questions assign different downstream values
to the same exact four actions. The key evidence is within-image disagreement,
sign reversal, action-pattern variance, transfer regret, and the gap between an
image-only action oracle and a question-conditioned oracle. Consequently, the
v3 null failure blocks harmfulness attribution but does not by itself answer
the query-dependence question.

V4 still requires strict numerical and semantic controls. Without common
padding, BF16 shape effects can imitate query dependence; without epsilon-aware
ties, argmax labels can disagree mechanically; without paraphrase,
different-evidence, difficulty, and answer-length controls, prompt format can
be mistaken for semantic dependence.

## Candidate choice and independent challenge

Three bounded choices were considered:

| Candidate | Resolution | Cost | Main weakness |
|---|---|---:|---|
| Two-task GQA/TextVQA discovery | Tests task breadth immediately | medium-high | TextVQA has weaker semantic controls and may add answer-format confounding |
| GQA-first discovery | Cleanest test using semantic programs, object IDs, types, equivalents, and dense same-image groups | medium | Initial evidence is task-specific |
| Stop dynamic-policy direction | Avoids further compute | none | Leaves a distinct and technically feasible query-dependence question unanswered |

The provisional two-task choice was revised after the mandatory independent
review. The review correctly identified that a GQA-first experiment is the
minimum discriminator and that the one-image common-padding result must first
generalize across a small prospective range. TextVQA is therefore reserved for
later independent replication rather than consumed in the first discovery.

## Evidence that would falsify v4

The direction stops if any of the following occurs under the frozen protocol:

- common right-padding fails to give exact same-image visual-state and WRITE
  equality across the prospective length/shape range;
- within-image best-action disagreement is explained by numerical ties;
- conditional sign reversals, transfer regret, and the image-only oracle gap
  collapse under medians, trimmed means, or image-level uncertainty;
- difficulty, question type, answer length, or answer format explains the
  apparent differences;
- paraphrases are as unstable as questions linked to different visual evidence;
- results are dominated by a few images or one retrospectively selected layer;
- a new image-disjoint confirmation fails the frozen query-dependence endpoint.

A failure will not be rescued by changing the layer grid, task, action
definition, metric, or threshold after outcomes are visible.

## Single minimum next experiment

After explicit execution approval, freeze 120 outcome-uninspected GQA images
with exactly two primary natural questions each. Use layers
`[0,4,8,12,16,20,24]`, all four actions at one layer at a time, the identical
dense suffix, and per-token accepted-reference likelihood. Before opening any
terminal action value, run a 12-image common-padding preflight spanning prompt
length and visual-token count; require bitwise visual-state/WRITE identity,
FULL parity, valid answer spans, and deterministic scoring.

If that gate passes, collect the 6,720 core branch scores plus 840 frozen
paraphrase-control scores and evaluate
epsilon-aware within-image best-action disagreement, conditional READ/WRITE
sign reversals, FULL-relative four-action variance, bidirectional cross-query
transfer regret, and the image-only versus image+query oracle gap. The image is
the statistical unit. Thirty images receive a prospectively verified official
paraphrase control, and different-evidence pairs use only unambiguous GQA
semantic-program/scene-graph object links. No router, probe, held-out
confirmation, harmfulness claim, or acceleration claim is authorized.

PROCEED_TO_V4_DISCOVERY
