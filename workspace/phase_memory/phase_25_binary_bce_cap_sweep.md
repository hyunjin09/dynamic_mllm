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
- Done: manifests, geometry/oracles, four configs, static readiness, matched
  initialization hashes, and corrected one-node/one-GPU Slurm contracts.
- Done: all four models completed the fixed ten training epochs.
- Done: CAP18, CAP22, and CAP24 completed all 22,307 external records with
  integrity PASS.
- In progress: CAP20 external execution is at 6,897/22,307 records (31%); its
  training and checkpoint selection are complete.
- Blocked: none.
- Most recent useful observation: completed caps span sharply different
  behavior—CAP22 selected an effectively ALL-ON checkpoint, CAP24 reduces mean
  ON to 15.26 with an 8.78-point overall accuracy loss, and CAP18 reduces mean
  ON to 9.79 with a 17.65-point loss. CAP20 remains required before assigning
  the frozen phase outcome.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Frozen parent max-50 manifest SHA-256 is `3620a347...` | `outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl` | Fixes supervision provenance | confirmed |
| Prior Pareto BCE best train Hit@1 was 18.27% | `reports/binary_pareto_training_fit_analysis.md` | Strong objection: cap filtering may not repair fitting | confirmed |
| Existing full10 trainer implements matched duplicated BCE and checkpoint rule | `experiments/train_binary_polar_full10.py` | Allows a supervision-only comparison | confirmed |
| Interim completed-cap execution checkpoint | `outputs/binary_cap_sweep_v1/interim_results_20260820.json` | Prevents partial results from being mistaken for the final four-cap comparison | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Strict Pareto supervision | Very sparse/ALL-OFF-heavy predictions and weak train fit | supported training-fit limitation under strict labels | Phase 24 report | Preserve multiple cap-valid alternatives in this approved test | Do not add Pareto filtering |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| CAP 24/22/20/18 matched sweep | Intermediate label density may avoid both unfiltered and strict-Pareto extremes | Whether an absolute supervision budget yields a useful execution frontier | high | testing |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: finish the already frozen four-cap external comparison; CAP20 is the sole incomplete condition.
- Relevant memory item used: Phase 24 showed strict Pareto underfit and became ALL-OFF-heavy.
- Confirmed observation: CAP18/22/24 completed with integrity PASS and have materially different accuracy/compute behavior.
- Unverified interpretation: CAP20 may or may not occupy a useful intermediate frontier point.
- Diagnosis: unknown
- Chosen action: let the unchanged CAP20 pipeline complete, then aggregate all four caps once.
- Strongest objection: three completed caps already look unfavorable, but concluding now would omit the prospectively selected CAP20 condition most likely to lie between CAP22 and CAP24.
- Automatic execution authorized: already running under the approved plan; no new job or experiment is authorized.
- Stop condition: CAP20 completion and frozen four-cap aggregation, or a technical integrity failure.

## Latest Research-Action Result

- Action taken: froze cap supervision and submitted all four pipelines.
- Result: 6,801 common-eligible records (5,944 train / 857 validation); jobs
  102858 (CAP18/node07), 102859 (CAP20/node06), 102860 (CAP22/node02), and
  102861 (CAP24/node02) each request exactly one GPU.
- Evidence saved: `outputs/binary_cap_sweep_v1/audits/` and isolated logs under
  `runs/binary_cap_sweep_v1/`.
- Failure or issue: the first GPU submission form expanded a multi-node list
  into three required nodes. Jobs 102853--102856 were cancelled while still
  pending and before outputs; fixed single-node jobs replaced them.
- Lesson learned: in this Slurm configuration, use one explicit `--node` for a
  one-node job rather than a candidate `--nodelist`.
- Next implication: monitor the four immutable pipelines; after all complete,
  aggregate their frozen internal/external results and assign one plan outcome.

### Scheduling amendment (2026-08-20)

- The four still-unmodified scientific jobs inherited the partition default
  time limit of 14 days 12 hours, preventing backfill onto an idle reserved GPU.
- With explicit user approval, jobs 102858--102861 were changed in place to a
  10-hour limit; commands, nodes, GPU count, data, configs, and outputs did not
  change.
- CAP22 job 102860 began running on node02 immediately after the amendment.

### Interim completed-cap result (2026-08-20)

- CAP18, CAP22, and CAP24 each completed 22,307/22,307 external records with
  integrity PASS; CAP20 completed training and is still executing externally.
- Nonoverlapping-suite pooled results (22,307 records): ALL-ON accuracy 75.89%;
  CAP22 75.89% at mean ON 28.00; CAP24 67.10% at mean ON 15.26; CAP18 58.23%
  at mean ON 9.79.
- Confirmed observation: CAP22 is effectively an ALL-ON collapse (99.996%);
  CAP24 and CAP18 save visual-layer execution but incur large net Harm over
  Rescue. This is interim evidence only, not the final plan outcome.
- Evidence: `outputs/binary_cap_sweep_v1/interim_results_20260820.json` and
  `reports/binary_cap{18,22,24}_external_eval.md`.
