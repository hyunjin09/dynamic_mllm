---
name: research-control
description: >
  Use when a research task requires choosing what to do next, interpreting an
  experiment, responding to a meaningful failure, comparing competing research
  actions, or deciding whether to continue, diagnose, stop, or propose a pivot.
  Do not use for fully specified implementation, routine debugging, mechanical
  edits, or deterministic result parsing.
---

# Research Control

## Purpose

Choose the smallest defensible next research action while avoiding:

- the first merely plausible recommendation,
- unsupported failure explanations,
- premature method abandonment,
- unnecessary diagnostics or broad sweeps,
- repeated near-identical attempts,
- unapproved strategic pivots,
- unbounded chains of experiments.

This skill decides **what research action to take**. After the action is chosen,
use the relevant engineering skill to implement, test, debug, and review it,
and use the existing infrastructure to execute it.

The goal is not to force a positive result or identify every root cause. The
goal is to improve the next decision with proportionate evidence and cost.

## 1. Choose the Lightest Adequate Mode

### FAST

Use when the task is fully specified, there is one reasonable local action, and
no scientific interpretation is required.

Process:

```text
minimal action -> verification -> state update
```

Do not generate alternative research directions.

### STANDARD

Use for:

- next-step selection,
- result interpretation,
- a first meaningful research failure,
- an abstract request with multiple plausible implementations,
- comparison with a user-suggested alternative.

Process:

```text
validate enough
-> separate observation from interpretation
-> compare viable actions when more than one exists
-> challenge the provisional choice once
-> execute at most one authorized research action
```

Budget:

- up to 3 viable candidates,
- one challenge pass,
- at most one cheap diagnostic,
- one research action before stopping.

Do not invent weak alternatives merely to satisfy the comparison format.

### DEEP

Use only when at least one condition is material:

- the same or equivalent failure has repeated,
- the next action is expensive,
- the decision changes the main method, objective, evaluation, or claim,
- result validity is materially uncertain,
- plausible causes lead to different strategic actions,
- the candidate ranking remains unresolved after one challenge.

DEEP does not mean exhaustive root-cause search. Consider only explanations and
checks that could change the decision, one diagnostic at a time.

## 2. Decision Procedure

### A. Fix the decision target

State:

- active objective or claim,
- current bottleneck,
- fixed scope, budget, and evaluation constraints.

### B. Validate only what matters

Before interpreting a result, check conditions that could invalidate the
conclusion, such as:

- intended code and configuration were used,
- intended mechanism was active,
- checkpoint, data, evaluator, and sample count were correct,
- the run was not dominated by an execution failure.

If validity fails, treat it as an implementation or execution issue rather than
a failure of the research hypothesis. If validity cannot be established at
reasonable cost, mark the result inconclusive.

### C. Separate facts from explanations

- **Observation:** directly measured or seen.
- **Interpretation:** a possible explanation.
- **Decision:** the next action justified by the evidence and uncertainty.

Never present an interpretation as an observation.

### D. Calibrate failure diagnosis

Default diagnosis:

```text
unknown
```

Allowed levels:

- `unknown`: no cause-specific evidence,
- `suspected`: indirect evidence exists,
- `supported`: a diagnostic distinguishes the explanation from an important
  alternative.

A `suspected` or `supported` diagnosis must cite a concrete log, metric, trace,
comparison, file, or diagnostic output. Plausible reasoning alone is not
evidence.

Permissions:

- `unknown`: may justify uncertainty or a robust next step, not redesign,
- `suspected`: may justify one cheap diagnostic or reversible local adaptation,
- `supported`: may justify a targeted local correction,
- no diagnosis alone authorizes a strategic pivot.

### E. Diagnose only when it changes the choice

After a first meaningful failure:

1. record the direct observation,
2. perform a quick validity check,
3. ask whether knowing the cause would materially change the next action,
4. if no, proceed under recorded uncertainty,
5. if yes, run at most one cheap decision-changing diagnostic.

An unresolved diagnosis does not block progress.

Escalate to deeper diagnosis only when the failure repeats, was expensive, may
invalidate downstream work, or precedes a strategic decision.

### F. Compare real candidates

When more than one viable action exists, compare up to 3 candidates using the
same criteria:

- relevance to the active objective,
- uncertainty or risk reduced,
- decision enabled by the result,
- cost,
- ambiguity of possible outcomes,
- consistency with approved scope.

For each candidate, capture only:

```text
what it resolves | what decision it enables | cost | main weakness
```

Do not choose primarily because an action is easy, novel, available, likely to
produce a positive metric, or creates the appearance of progress.

### G. Challenge once before commitment

For STANDARD or DEEP decisions, test the provisional choice with:

- strongest argument against it,
- strongest case for the runner-up,
- condition under which the runner-up is better,
- unverified assumption most likely to reverse the ranking.

Report confidence as `high / medium / low`.

If the ranking remains unresolved, prefer one cheap discriminating check or ask
the user. Do not call an unresolved option "best."

### H. Select and bound the action

Choose the smallest action that either:

- remains useful under the main unresolved explanations, or
- distinguishes explanations that would lead to different decisions.

A request to analyze and execute the next step authorizes exactly one research
action. Implementation repair required to complete that action is allowed, but
do not recursively choose another experiment afterward.

Stop before executing any unapproved:

- new method family or major module,
- new objective or loss,
- backbone replacement used as a pivot,
- primary dataset, metric, or evaluation change,
- main-claim change,
- expensive experiment outside the approved budget,
- new direction after a negative result.

These may be recommended with tradeoffs, but require user approval.


## 3. Conditional Independent Review

The internal challenge pass is the default. A subagent review is an escalation,
not a routine step.

Spawn exactly one `research_reviewer` only when at least one trigger remains
material after the internal challenge:

- the candidate ranking is `narrow` or `unresolved`,
- the decision is expensive or can materially change the main method,
  evaluation, or claim,
- the recommendation depends on one unverified assumption and alternative
  choices have materially different costs,
- the same class of recommendation was previously corrected because of a
  candidate, criterion, or evidence omission,
- the user explicitly requests an independent second opinion.

Do not spawn a reviewer merely because:

- a first experiment failed,
- a result is negative,
- more than one candidate exists,
- the agent has low confidence but one cheap diagnostic clearly dominates,
- the task is fully specified implementation or routine debugging.

### Review packet

Give the reviewer only:

```text
decision objective and fixed constraints
confirmed observations with evidence paths
unresolved assumptions
viable candidates and estimated costs
provisional ranking and strongest objection
specific question the reviewer must resolve
```

Do not ask the reviewer to invent a new research program. Do not provide hidden
offline-eval rubrics during normal research work.

### Reviewer output contract

Require:

```text
verdict: stable / revise / unresolved
independent candidate ranking
strongest objection to the provisional choice
omitted viable candidate, if any
evidence or assumption driving disagreement
cheapest discriminator if unresolved
confidence: high / medium / low
```

### Reconciliation

The parent agent owns the final decision.

- `stable`: proceed if all other execution boundaries are satisfied.
- `revise`: verify the reviewer’s evidence, update the candidate comparison,
  and change the recommendation only when the correction is supported.
- `unresolved`: use at most one cheap discriminator if it can settle the
  decision; otherwise ask the user.

A reviewer cannot authorize a strategic pivot or an additional research action.

After reconciliation, execute at most the one research action already permitted
by this skill.

## 4. Learn from Failure Without Detouring

A failure has improved the process only if it does at least one of the
following:

- prevents an identical retry,
- rules out a candidate,
- changes a validation or implementation rule,
- selects a better next action,
- justifies stopping.

Do not require complete root-cause analysis after every failure.

Do not repeat the same failed action unless the next record states what changed
materially and why the previous failure should not recur.

If the same or equivalent failure occurs twice, stop local trial-and-error and
use DEEP mode for a focused diagnostic or decision review.

## 5. User-Suggested Alternatives

When the user suggests an alternative:

1. add it to the same candidate set,
2. evaluate it using the existing criteria,
3. do not accept it merely because the user proposed it,
4. disagree when the evidence still favors the original choice.

If the recommendation changes, state whether the cause was:

- new evidence,
- a new constraint,
- an omitted candidate,
- an omitted criterion,
- incorrect weighting,
- premature commitment.

If already-available evidence showed the alternative was better, record that
as a decision-process error in phase memory.

## 5. Compact Phase-Memory Update

For STANDARD or DEEP decisions, add only:

```md
- Deliberation mode:
- Active objective and bottleneck:
- Confirmed observation / unverified interpretation:
- Diagnosis: unknown / suspected / supported; evidence path if not unknown
- Viable alternatives considered:
- Chosen action and strongest objection:
- How this differs from failed attempts:
- Authorization and stop condition:
```

Omit inapplicable fields. Reference raw logs by path. Do not write a long memo.

## Verification

Before executing a research next step, confirm:

- [ ] The mode is proportionate to ambiguity and cost.
- [ ] The result is valid enough for the intended interpretation.
- [ ] Observation and explanation are separated.
- [ ] Any diagnosis is evidence-gated.
- [ ] Real alternatives were compared only when they existed.
- [ ] The provisional choice survived one meaningful challenge when needed.
- [ ] The action is informative or robust under unresolved causes.
- [ ] Scope, budget, and user authority are respected.
- [ ] Automatic execution covers one research action only.

## Cross-Phase Lesson Promotion

Keep provisional or one-off explanations in phase memory.

Promote a lesson to `workspace/decision_log.md` only when it is supported by a
direct diagnostic, repeated across attempts, or likely to affect later phases.
Before a new phase starts, read relevant promoted lessons.

Do not promote an ambiguous post-hoc explanation into a permanent rule.
