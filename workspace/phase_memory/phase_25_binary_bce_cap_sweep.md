# Phase 25: Binary BCE Absolute-Cap Sweep

## Current Objective

Run the approved four-way Image+Question duplicated-BCE comparison using all
selected valid routes with at most 24, 22, 20, or 18 VISUAL_ON layers, followed
by the unchanged 22,307-record external execution evaluation.

## Active Constraints

- Source is the frozen pre-Pareto GQA/TextVQA/ChartQA max-50 manifest.
- All four runs use the identical CAP=18-eligible train/validation UIDs.
- Only the absolute ON cap differs; architecture, seed, initialization,
  optimization, threshold decoder, evaluator, and checkpoint rule are fixed.
- Ten epochs, no early stopping, one GPU per run, concurrent execution.
- GPU nodes are restricted to node02, node06, and node07; node03 and node04 are excluded.
- No new MCTS, Pareto filtering, NLL, architecture change, or follow-on experiment.

## Current State

- Done: plan and inherited contracts inspected; cap transform unit tests pass.
- In progress: freeze manifests, geometry/oracles, configs, and readiness gate.
- Blocked: none.
- Most recent useful observation: the parent manifest contains all selected
  route masks and their exact ON counts, so cap filtering needs no new inference.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Frozen parent max-50 manifest SHA-256 is `3620a347...` | `outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl` | Fixes supervision provenance | confirmed |
| Prior Pareto BCE best train Hit@1 was 18.27% | `reports/binary_pareto_training_fit_analysis.md` | Strong objection: cap filtering may not repair fitting | confirmed |
| Existing full10 trainer implements matched duplicated BCE and checkpoint rule | `experiments/train_binary_polar_full10.py` | Allows a supervision-only comparison | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Strict Pareto supervision | Very sparse/ALL-OFF-heavy predictions and weak train fit | supported training-fit limitation under strict labels | Phase 24 report | Preserve multiple cap-valid alternatives in this approved test | Do not add Pareto filtering |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| CAP 24/22/20/18 matched sweep | Intermediate label density may avoid both unfiltered and strict-Pareto extremes | Whether an absolute supervision budget yields a useful execution frontier | high | testing |

## Next-Step Decision

- Deliberation mode: fast
- Active objective and bottleneck: execute the already fixed four-cap experiment; first freeze exact matched manifests and verify the existing train/eval contract.
- Relevant memory item used: Phase 24 showed strict Pareto underfit and became ALL-OFF-heavy.
- Confirmed observation: all required route geometry is present in the frozen parent manifest.
- Unverified interpretation: an intermediate cap may improve the learned accuracy-compute frontier.
- Diagnosis: unknown
- Chosen action: freeze and audit cap manifests, then run four matched ten-epoch train/eval pipelines in parallel.
- How this differs from failed attempts: keeps every cap-valid selected route and changes only maximum ON count.
- Automatic execution authorized: yes
- Authorization basis: explicit user request to perform `plans/cap_training.md`.
- Stop condition: finish the frozen external comparison and assign exactly one plan outcome, or stop on a technical integrity failure.

## Latest Research-Action Result

- Action taken: in progress.
- Result: pending.
- Evidence saved: pending under `outputs/binary_cap_sweep_v1/`.
- Failure or issue: none.
- Lesson learned: pending.
- Next implication: pending.
