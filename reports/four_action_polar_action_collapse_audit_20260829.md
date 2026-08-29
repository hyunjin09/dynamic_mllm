# Four-Action POLAR External Action-Collapse Audit

## Scope

Deterministic parsing of the completed 14,960-record BCE and NLL merged
external result files. No new model inference or training was run.

## Confirmed action behavior

| Objective | Records | Layer decisions | FULL | Non-FULL | Samples with any non-FULL action | Unique top-1 masks |
|---|---:|---:|---:|---:|---:|---:|
| Duplicated BCE | 14,960 | 418,880 | 418,880 | 0 | 0 | 1 |
| Exact set NLL | 14,960 | 418,880 | 418,880 | 0 | 0 | 1 |

The one unique mask for both objectives is `FULL` at all 28 layers.

## FULL-versus-runner-up logit margin

The margin is the FULL logit minus the largest non-FULL logit at each of the
418,880 sample-layer decisions.

| Objective | Minimum | 1st percentile | Median | Mean | Maximum | Fraction <= 0 |
|---|---:|---:|---:|---:|---:|---:|
| Duplicated BCE | 0.12549 | 0.23779 | 0.59766 | 1.09239 | 5.12500 | 0.00000 |
| Exact set NLL | 0.36719 | 1.45898 | 4.52734 | 4.43887 | 7.42188 | 0.00000 |

Therefore no sample-layer decision was an argmax tie. Exact set NLL was much
more strongly FULL-dominant than duplicated BCE, although both decode to the
same all-FULL mask.

## Behavioral implication

Because the predicted mask equals unified FULL for every record, predicted and
baseline generation are the same route execution. Identical accuracy, zero
wrong-to-correct cases, and zero correct-to-wrong cases follow mechanically.

## Interpretation boundary

- Supported: both deployed top-1 predictors collapsed to the all-FULL route on
  the full external population.
- Supported: the same all-FULL top-1 behavior was already present on all 866
  internal validation records at checkpoint selection.
- Not established: that every learned representation or non-argmax logit is
  input-independent.
- Diagnosis of why training produced FULL dominance: unknown from this audit.
  Candidate causes include supervision frequency/weighting, optimization fit,
  predictor input/capacity, and the categorical factorization, but none is
  distinguished here.

Evidence:

```text
outputs/four_action_polar/node07_20260828/eval_bce/merged/external_results_v1.jsonl
outputs/four_action_polar/node07_20260828/eval_nll/merged/external_results_v1.jsonl
outputs/four_action_polar/node07_20260828/training_bce/best_checkpoint.json
outputs/four_action_polar/node07_20260828/training_nll/best_checkpoint.json
```
