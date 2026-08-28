# Four-Action Answer-Alignment Run Status

Last scheduler verification: 2026-08-24 00:18 KST.

## Scientific Objective

This experiment tests whether some Qwen2.5-VL-7B errors are caused by harmful
visual operations at particular transformer layers.

At each of the model's 28 layers, it compares four local actions:

- `FULL` (`M11`): visual READ and WRITE are both enabled.
- `READ_ONLY` (`M10`): text/control rows can read visual K/V, but visual rows
  are not updated.
- `WRITE_ONLY` (`M01`): visual rows are updated, but text/control rows cannot
  directly read visual K/V.
- `IGNORE` (`M00`): neither operation is enabled.

All four actions use the same unified execution machinery, identical pre-layer
state, and unified-FULL suffix. Consequently, the READ, WRITE, and interaction
effects are entirely within-executor causal contrasts. Native FULL and the old
binary single-layer OFF executor are external semantic and numerical
diagnostics only.

## Completed Runs

| Stage | Purpose | Result |
|---|---|---|
| Unit and synthetic checks | Validate mask semantics, branch identity, cache handling, scoring, determinism, and analysis logic | Passed; 67 focused tests |
| Unified preflight | Compare unified FULL with native FULL and unified IGNORE with old binary OFF | Passed; 8 examples |
| Full-layer smoke | Exercise all 28 layers and four actions on real data | Passed; 8 examples |
| Validation/pilot | Broader semantic validation and throughput measurement | Passed; 56 examples |
| Eligibility freeze | Recheck current unified-FULL correctness before freezing cohorts | Passed; 4,890 candidates |
| Primary A+ sweep | Build the complete 28-layer four-action landscape for correctable FULL-wrong errors | Passed; 1,880 samples |
| Control A | Evaluate FULL-wrong samples for which no correction was found under the matched binary-search budget | Passed; 857 analyzable samples |
| Control B | Evaluate FULL-correct, ALL-OFF-wrong vision-required samples | Passed; 2,084 samples |
| Trajectory selection | Select all discrete rescues and strong negative local effects for representation follow-up | Complete; 10,196 cells across 1,579 samples |
| Trajectory rescue | Compare FULL with the relevant single-operation-suppressed trajectory | Passed; 10,196/10,196 cells across all 8 shards |

Authoritative completed stage summaries:

- `primary__unified_v1/stage_summary.json`
- `control_no_correction__unified_v1/stage_summary.json`
- `control_vision_required__unified_v1/stage_summary.json`
- `trajectory_rescue/selection_summary.json`
- `trajectory_rescue/summary.json`

## Primary Cohort

The primary A+ cohort contains samples where:

- FULL is wrong;
- ALL-OFF is also wrong; and
- at least one non-ALL-OFF binary MCTS route is correct.

This isolates vision-correctable errors without including trivial cases where
removing all visual access is itself sufficient. The current-runtime
eligibility pass retained:

- GQA: 1,222 samples
- TextVQA: 658 samples
- Total: 1,880 samples

For every sample, the completed primary sweep stores all four action scores and
answers at every layer, evaluator correctness, factorial effects, the
FULL-model answer-erosion trajectory, and existing binary correcting-route
metadata.

## Control Cohorts

### Control A: No Correction Found

These are FULL-wrong samples for which no correcting route was found under the
matched binary-search budget. This control tests whether harmful local visual
effects are enriched in correctable A+ errors.

Eleven TextVQA candidates had no evaluator-valid correct scoring target and
were explicitly excluded because their correct-versus-FULL-wrong margin is
undefined. This leaves 857 analyzable samples.

The valid interpretation is "no correcting route was found under the matched
search budget," not "no correcting route exists."

### Control B: Vision Required

These samples are FULL correct but ALL-OFF wrong. They test whether the same
apparently harmful visual effects also occur where visual access is required
for the model's correct answer. This sweep is complete for 2,084 samples.

## Why Multiple Slurm Jobs Appear

Most job IDs represent resumable repairs or performance checks, not independent
scientific experiments.

- The original design mixed native-FULL scores with modified-path
  interventions. BF16 drift was large enough to contaminate the small causal
  effects, so all factorial quantities were moved into the unified executor.
- A current-runtime eligibility pass excluded 58 candidates whose FULL
  correctness no longer matched the defining cohort condition.
- A two-model-replicas-per-GPU ramp reached approximately 99% GPU utilization,
  but useful throughput fell to 0.8004 times the one-replica baseline. It was
  rejected, and the append-only sweep resumed with one replica per GPU.
- Eleven evaluator-unscorable TextVQA controls were excluded rather than given
  undefined margins.
- Two BF16 readout-boundary cases and one TextVQA valid-answer phrase-switch
  case required diagnostic/gate repairs. These repairs did not change the
  primary within-unified causal quantities.

Completed shards were retained throughout, and resumed jobs skip already
completed records.

## Completed Trajectory Rescue

GPU job `1572` completed successfully on 2026-08-23 in 18 minutes 18 seconds.
The merged trajectory artifact passed the required integrity gates:

- exactly 10,196 unique selections across 1,579 samples;
- GQA: 5,630 cells; TextVQA: 4,566 cells;
- all eight shards and the eight-worker execution contract present;
- every result gate passed and no disqualifying failures remain;
- 269 fixed-target identity switches (2.64%) are retained as a diagnostic,
  while the evaluator-best endpoint remains preserved in state;
- two legacy boundary rows were recovered under the documented diagnostic
  identity-recheck rule.

## Work Remaining

### Run Final Analysis and Reporting

CPU job `1573` has a satisfied dependency on successful job `1572`, but is
currently pending for CPU resources. It will produce:

- layerwise READ, WRITE, and interaction distributions;
- discrete rescue categories;
- GQA, TextVQA, and combined bootstrap confidence intervals;
- correcting-route OFF-versus-ON and within-sample enrichment analyses;
- Hamming-distance stratification;
- answer-erosion and trajectory-rescue comparisons;
- comparisons with both control cohorts;
- native-versus-unified FULL numerical-drift diagnostics;
- aggregate statistics and plots;
- `numerical_consistency_report.md`;
- `4action_answer_unaligned_report.md`.

## Current Interpretation Boundary

The complete primary causal data, both controls, and the complete trajectory
follow-up exist and have passed their stage-level integrity gates. The final
aggregate analysis has not run because job `1573` is pending for resources.
Therefore, no final scientific conclusion should be stated yet.

For detailed provenance, see `experiment_log.md` and
`implementation_audit.md` in this directory.
