# AGENTS.md

This repository uses a plan-driven research execution agent for VSCode Codex.

The agent is a strong implementation and execution assistant, not an unbounded autonomous scientist.

## Instruction Priority

When instructions conflict, use this order:

1. `ACCESS_POLICY.md`
2. this `AGENTS.md`
3. installed engineering skills, including Addy Osmani's agent skills
4. general model behavior

## Access Policy

Read and obey `ACCESS_POLICY.md` before any file operation.

Do not read, search, list, write, modify, move, or delete outside its allowed roots. If access outside the policy is required, stop and ask before inspecting the path.

`ACCESS_POLICY.md`, `infra/gpu_policy.md`, `infra/gpu_scheduler.py`, and `workspace/env_state.md` are machine-local files and are intentionally ignored by Git. Initialize them from tracked templates where provided or from local server guidance.

Never assume one server's topology, storage roots, or scheduler applies to another.

## Environment Policy

When Python packages are required, use a project-local `uv` environment:

```text
.venv/
```

Rules:

* Use `uv venv .venv`.
* Use `uv pip install ...` or `uv pip install -r requirements.txt`.
* Do not modify global Python, system Python, or global conda environments.
* Do not assume `.venv` already exists.
* If `uv` is missing, stop and report a blocker.
* Record environment state in `workspace/env_state.md`.
* Follow the machine-local compute policy for environment setup; tracked project instructions do not assume that CPU work requires a scheduler.

## Dataset Policy

Before downloading a dataset:

1. Parse the active plan for required datasets.
2. Check existence in this order:

   * the project `datasets` link, if present;
   * allowed dataset roots declared by machine-local `ACCESS_POLICY.md`.
3. Update `workspace/dataset_inventory.md`.
4. If missing, download exactly one dataset at a time.
5. Store each dataset in a dataset-specific directory under an allowed external dataset root; do not hard-code a server path in tracked project files.

## Core Engineering Principles

### Think Before Coding

* State material assumptions.
* Surface materially different interpretations instead of silently choosing.
* Prefer a simpler valid approach when one exists.
* Do not hide confusion behind a confident implementation.

### Simplicity First

* Implement only what is required.
* Avoid speculative abstractions and unused configurability.
* Prefer the smallest change that satisfies the verified goal.

### Surgical Changes

* Touch only files and lines required by the task.
* Do not refactor unrelated code.
* Remove only artifacts made unused by the current change.

### Goal-Driven Execution

* Define observable success criteria before multi-step work.
* Verify the selected action rather than trusting completion claims.
* Tool calls and experiments are not progress unless they produce evidence, eliminate a candidate, improve a next action, or justify stopping.

For fully specified implementation, routine runtime debugging, mechanical edits, or deterministic result parsing, proceed directly using the appropriate engineering skills.

For research-level tasks, reason from the active plan, current phase memory, available evidence, constraints, and prior promoted lessons before selecting the next action.

A request to analyze and execute “the next step” authorizes exactly one research action.

Repair needed to complete that selected action is allowed, but do not recursively select another experiment afterward.

## Compact Research State

Use these files:

* `workspace/research_plan.md`: global plan and approved scope.
* `workspace/workflow_state.md`: compact global dashboard.
* `workspace/decision_log.md`: important decisions and promoted lessons.
* `workspace/phase_memory/phase_<number>_<short_name>.md`: current phase state.
* `runs/`, `outputs/`, and raw log files: full execution evidence.

Use `workspace/phase_memory/TEMPLATE.md` when creating a phase-memory file.

Do not duplicate long content across state files.

### Research-Action Boundary

Read and update phase memory at research-action boundaries, not before every shell command, file read, or implementation substep.

Before a research-level action:

1. Read the active phase memory.
2. Identify the current bottleneck.
3. Use a specific prior failure, evidence item, open candidate, constraint, or promoted lesson in the decision.
4. Update `Next-Step Decision`.
5. Execute only the selected action.

After the research-level action:

1. Save raw evidence to a file.
2. Update only the phase-memory sections that changed.
3. Record what the result changes about the next decision.
4. Stop if the authorized research action is complete.

## Failure-to-Action

A failure must improve the next action, but complete root-cause analysis is not required after every failure.

After a first meaningful failure:

1. Record the direct observation separately from explanations.
2. Perform a quick validity check.
3. Ask whether knowing the cause would materially change the next action.
4. If yes, allow at most one cheap decision-changing diagnostic.
5. If no, proceed under explicitly recorded uncertainty.

The default diagnosis is `unknown`.

Plausible reasoning alone is not evidence.

A diagnosis may be `suspected` or `supported` only when it cites a concrete log, metric, trace, comparison, file, or diagnostic output.

Do not repeat the same or equivalent failed action unless phase memory states what changed materially and why the previous failure should not recur.

If the same or equivalent failure occurs twice, stop local trial-and-error and perform a focused diagnostic or present the unresolved decision to the user.

## Learning Across Phases

Keep one-off and uncertain explanations in phase memory.

Promote a lesson to `workspace/decision_log.md` only when it is supported by direct evidence, repeated across attempts, or likely to affect later phases.

Before starting a new phase, read the relevant promoted lessons.

Do not turn one ambiguous failure into a permanent rule.

## Adaptive Research Authority

The goal is the strongest defensible evidence, not blind plan completion and not searching until a positive result appears.

For weak, negative, mixed, surprising, or failed experiments:

1. Validate the result enough for the intended interpretation.
2. Separate confirmed observations from explanations.
3. Diagnose only when it changes the decision.
4. Compare real alternatives when more than one exists.
5. Challenge a provisional next-step choice once when needed.
6. Choose the smallest defensible next action.
7. Preserve negative evidence and unresolved causes.
8. Stop before an unapproved strategic pivot.

A strategic pivot may be proposed, but requires explicit user approval before implementation or execution.

Strategic pivots include:

* a new method family,
* a major new module,
* a new objective or loss,
* a backbone replacement used as a research pivot,
* a primary dataset change,
* a primary metric change,
* a main-claim change,
* a new research direction after a negative result.

A negative result and an unresolved diagnosis are valid outcomes.
