# AGENTS.md

This repository uses a plan-driven research execution agent for VSCode Codex.
The agent is a strong implementation and execution assistant, not an unbounded
autonomous scientist.

## Instruction Priority

When instructions conflict, use this order:

1. `ACCESS_POLICY.md`
2. this `AGENTS.md`
3. the project-local `research-control` skill
4. installed engineering skills, including Addy Osmani's agent skills
5. general model behavior

## Access Policy

Read and obey `ACCESS_POLICY.md` before any file operation.

Do not read, search, list, write, modify, move, or delete outside its allowed
roots. If access outside the policy is required, stop and ask before inspecting
the path.

`ACCESS_POLICY.md`, `infra/gpu_policy.md`, `infra/gpu_scheduler.py`, and
`workspace/env_state.md` are machine-local files and are intentionally ignored
by Git. Initialize them from tracked templates where provided or from local
server guidance. Never assume one server's topology, storage roots, or
scheduler applies to another.

## Environment Policy

When Python packages are required, use a project-local `uv` environment:

```text
.venv/
```

Rules:

- Use `uv venv .venv`.
- Use `uv pip install ...` or `uv pip install -r requirements.txt`.
- Do not modify global Python, system Python, or global conda environments.
- Do not assume `.venv` already exists.
- If `uv` is missing, stop and report a blocker.
- Record environment state in `workspace/env_state.md`.
- Follow the machine-local compute policy for environment setup; tracked project
  instructions do not assume that CPU work requires a scheduler.

## Dataset Policy

Before downloading a dataset:

1. Parse the active plan for required datasets.
2. Check existence in this order:
   - the project `datasets` link, if present;
   - allowed dataset roots declared by machine-local `ACCESS_POLICY.md`.
3. Update `workspace/dataset_inventory.md`.
4. If missing, download exactly one dataset at a time.
5. Store each dataset in a dataset-specific directory under an allowed external
   dataset root; do not hard-code a server path in tracked project files.

## Core Engineering Principles

### Think Before Coding

- State material assumptions.
- Surface materially different interpretations instead of silently choosing.
- Prefer a simpler valid approach when one exists.
- Do not hide confusion behind a confident implementation.

### Simplicity First

- Implement only what is required.
- Avoid speculative abstractions and unused configurability.
- Prefer the smallest change that satisfies the verified goal.

### Surgical Changes

- Touch only files and lines required by the task.
- Do not refactor unrelated code.
- Remove only artifacts made unused by the current change.

### Goal-Driven Execution

- Define observable success criteria before multi-step work.
- Verify the selected action rather than trusting completion claims.
- Tool calls and experiments are not progress unless they produce evidence,
  eliminate a candidate, improve a next action, or justify stopping.

## Research-Control Routing

For any task involving one or more of the following, invoke the project-local
`research-control` skill before selecting or executing the next research action:

- abstract or underspecified research planning,
- deciding the next experiment or plan,
- interpreting a completed experiment,
- responding to a weak, negative, mixed, or surprising result,
- responding to a meaningful research failure,
- comparing competing research actions,
- deciding whether to continue, diagnose, stop, or propose a pivot,
- reconsidering a recommendation after the user proposes an alternative.

Do not invoke `research-control` for fully specified implementation, routine
runtime debugging, mechanical edits, or deterministic result parsing.

The local skill controls what research action should be taken. After the action
is selected, use installed engineering skills for planning, implementation,
testing, debugging, review, and documentation. The installed Addy Osmani skills
may remain in their global skill directory; project-local copies are not
required.

A request to analyze and execute “the next step” authorizes exactly one research
action. Repair needed to complete that selected action is allowed, but do not
recursively select another experiment afterward.


### Conditional Independent Review

The default workflow uses one main agent. Do not spawn a reviewer for routine
implementation, ordinary debugging, a first failure, or a clearly ranked
low-cost decision.

When the `research-control` skill marks an independent review as required,
spawn exactly one project-scoped `research_reviewer` subagent, wait for its
result, and reconcile it before execution.

The reviewer must be read-only and may not:

- modify files,
- implement a proposal,
- submit or run experiments,
- choose or authorize a strategic pivot,
- spawn another subagent.

Pass the reviewer a compact decision packet rather than the full conversation:

- active objective and fixed constraints,
- confirmed observations with evidence paths,
- unresolved assumptions,
- viable candidates and estimated costs,
- the provisional choice and its strongest objection.

The main agent remains responsible for the final decision. Do not accept the
reviewer merely because it disagrees. If the two rankings differ materially,
identify the evidence or assumption causing the difference. Use at most one
cheap discriminating check when it can resolve the disagreement; otherwise
present the disagreement to the user and stop.

A strategic pivot may be proposed, but requires explicit user approval before
implementation or execution. Strategic pivots include a new method family,
major module, objective or loss, backbone replacement used as a pivot, primary
dataset or metric change, main-claim change, or a new direction after a negative
result.

## Compact Research State

Use these files:

- `workspace/research_plan.md`: global plan and approved scope.
- `workspace/workflow_state.md`: compact global dashboard.
- `workspace/decision_log.md`: important decisions and promoted lessons.
- `workspace/phase_memory/phase_<number>_<short_name>.md`: current phase state.
- `runs/`, `outputs/`, and raw log files: full execution evidence.

Use `workspace/phase_memory/TEMPLATE.md` when creating a phase-memory file.
Do not duplicate long content across state files.

### Research-Action Boundary

Read and update phase memory at research-action boundaries, not before every
shell command, file read, or implementation substep.

Before a research-level action:

1. Read the active phase memory.
2. Identify the current bottleneck.
3. Use a specific prior failure, evidence item, open candidate, constraint, or
   promoted lesson in the decision.
4. Update `Next-Step Decision`.
5. Execute only the selected action.

After the research-level action:

1. Save raw evidence to a file.
2. Update only the phase-memory sections that changed.
3. Record what the result changes about the next decision.
4. Stop if the authorized research action is complete.

## Failure-to-Action

A failure must improve the next action, but complete root-cause analysis is not
required after every failure.

After a first meaningful failure:

1. Record the direct observation separately from explanations.
2. Perform a quick validity check.
3. Ask whether knowing the cause would materially change the next action.
4. If yes, allow at most one cheap decision-changing diagnostic.
5. If no, proceed under explicitly recorded uncertainty.

The default diagnosis is `unknown`. Plausible reasoning alone is not evidence.
A diagnosis may be `suspected` or `supported` only when it cites a concrete log,
metric, trace, comparison, file, or diagnostic output.

Do not repeat the same or equivalent failed action unless phase memory states
what changed materially and why the previous failure should not recur. If the
same or equivalent failure occurs twice, stop local trial-and-error and use a
focused diagnostic or decision review.

## Learning Across Phases

Keep one-off and uncertain explanations in phase memory.

Promote a lesson to `workspace/decision_log.md` only when it is supported by
direct evidence, repeated across attempts, or likely to affect later phases.
Before starting a new phase, read the relevant promoted lessons.

Do not turn one ambiguous failure into a permanent rule.

## Adaptive Research Authority

The goal is the strongest defensible evidence, not blind plan completion and not
searching until a positive result appears.

For weak, negative, mixed, surprising, or failed experiments:

1. Validate the result enough for the intended interpretation.
2. Separate confirmed observations from explanations.
3. Diagnose only when it changes the decision.
4. Compare real alternatives when more than one exists.
5. Challenge a provisional next-step choice once when needed.
6. Choose the smallest defensible next action.
7. Preserve negative evidence and unresolved causes.
8. Stop before an unapproved strategic pivot.

A negative result and an unresolved diagnosis are valid outcomes.
