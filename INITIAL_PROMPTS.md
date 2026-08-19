# Initial Prompt Guide

`AGENTS.md` is the persistent project-level instruction file. Codex loads it
when a session starts, so do not paste a long master prompt into every session.

Use one of the following short prompts.

## 1. New Research Project

```text
Read `AGENTS.md`, `ACCESS_POLICY.md`, and the project-local `research-control`
skill. Confirm that globally installed skill metadata is discoverable, but do
not read every engineering `SKILL.md` up front. Load only the engineering skill
or skills that become relevant to the concrete task.

Initialize the compact research state for this project from the information
below. Do not invent missing scientific decisions; mark them unresolved.

Project objective:
<goal>

Current idea or hypothesis:
<hypothesis>

Known evidence:
<evidence>

Constraints:
<compute, dataset, deadline, scope>

Immediate task:
<first concrete task>

After initialization, perform only the immediate task. Update research_plan,
workflow_state, and the active phase memory at the research-action boundary.
```

## 2. Resume an Existing Project

```text
Read AGENTS.md and resume from workspace/research_plan.md,
workspace/workflow_state.md, workspace/decision_log.md, and the active phase
memory.

Summarize the current objective, latest confirmed evidence, unresolved
bottleneck, and currently authorized action in a few lines. Then perform:

<task>
```

## 3. Fully Specified Implementation

```text
Implement the following approved change and verify it:

<exact implementation request>

This is an implementation task, not a request to reconsider the research
direction. Use the relevant engineering skills and existing infra. Update
phase memory only when the research action is complete.
```

## 4. Analyze a Completed Experiment and Execute One Next Step

```text
The current experiment has completed.

Validate the result, update the active claim and phase memory, and choose the
smallest defensible next research action. Do not accept the first plausible
plan. Compare real alternatives only when they exist and challenge the
provisional choice once.

Use the independent research_reviewer only if the research-control triggers
remain material after the internal challenge. Execute at most one research
action, then save evidence, update state, report, and stop.
```

## 5. Ask for a Plan Only

```text
Analyze the current state and recommend the next research step, but do not
implement or run it.

Use research-control. State confirmed observations separately from
interpretations, compare viable alternatives, challenge the provisional choice,
and report confidence and the main condition that could reverse the ranking.
```

## 6. Explicit Independent Review

```text
Before committing to the next research action, invoke the project-scoped
research_reviewer with a compact decision packet. Reconcile its evidence with
your own analysis. Do not automatically follow the reviewer. If the decision
remains unresolved, report the disagreement and stop.
```

## Recommended First Prompt

For most projects, use this once:

```text
Read the project instructions and initialize or resume the compact research
state. My research objective and current status are below.

Objective:
...

Current status and completed work:
...

Important results or failures:
...

Constraints:
...

Immediate task:
...

Work as a bounded research executor. Keep implementation fast when the task is
clear. For planning, result interpretation, or failure response, use
research-control. Use a reviewer only when its escalation triggers apply.
```

After the first session, normal task prompts can be short because project state
is persisted in `workspace/`.
