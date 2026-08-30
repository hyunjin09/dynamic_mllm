# Four-Action Generalization Diagnostic Protocol

Frozen before selected-checkpoint diagnostic outcomes were extracted.

## Authority and fixed inputs

- Source plan: `plans/four_action_generalization_diagnostic_plan.md`
- Source-plan SHA-256: `cdc39940a1e19c22f17771bc535d22be792b97cd7f45d7c6128ce816e933e446`
- POLAR checkpoint: epoch 15, `outputs/persistent_corrective_supervision_v1/polar/epoch_15/checkpoint.pt`
- POLAR checkpoint SHA-256: `e22ee666bbc77bfd3c2f2da5b2a789521d9a989e3622637babdc181a8e1da579`
- Online checkpoint: epoch 14, `outputs/persistent_corrective_supervision_v1/online/epoch_14/router_checkpoint.pt`
- Online checkpoint SHA-256: `9ab297a2299cd039d12e80395703a21b94b75782d395bae705457dd6f8bf4a4f`
- Frozen source manifest: `analysis/persistent_corrective_supervision/training_manifest.jsonl`
- Frozen boundary manifest: `analysis/persistent_corrective_supervision/boundary_manifest.jsonl`
- State-manifest SHA-256: `5dfb1fb664b27f7a47b0a6b19e044ad9b5d9c193fddb2b97f9118ddd45b785c6`

No router/base-model parameter, label, data split, or executor contract changes.
No external evaluation is admitted.

## State construction

- Mandatory-deviation positives: all 640 frozen W2C
  mandatory all-FULL-prefix boundaries (512
  train; 128 validation).
- KEEP_FULL negatives: one W2C correcting-trajectory node with unique valid
  next action `FULL`, matched without replacement at the same split, dataset,
  and exact layer, and required to come from a different UID than its positive.
- Total states: 1280 in 640 exact matched pairs.
- Multi-valid non-FULL nodes remain set-valued. Singleton mechanism analyses do
  not assign them an arbitrary class. READ_OFF/WRITE_OFF analyses include a
  state only when every valid action agrees on that bit.
- Depth bins: layers 0--9 early, 10--18 middle, and 19--27 late.

## Selected-router outputs and shuffle

- Action order: `IGNORE`, `READ_ONLY`, `WRITE_ONLY`, `FULL`.
- WHEN score: `1 - P(FULL)`.
- READ_OFF score: `P(WRITE_ONLY) + P(IGNORE)`.
- WRITE_OFF score: `P(READ_ONLY) + P(IGNORE)`.
- Online state shuffle: jointly replace the text query and visual state by the
  frozen partner in the same split/dataset/layer cell; labels and layer identity
  stay fixed. Seed: 20260831.
- Four direct torchrun ranks/GPU devices are required for state extraction.

## Metrics and baselines

- Binary threshold: 0.5; report AUROC, AUPRC, accuracy, balanced accuracy,
  precision, recall, F1, FPR, and FNR.
- Layer-only baseline: train-only empirical class probability by exact layer
  with Jeffreys smoothing alpha=0.5.
- Probe preprocessing uses training-only mean/standard deviation. Linear and
  one-hidden-layer MLP heads use the same native representation, fixed schedule,
  and no validation-based hyperparameter or checkpoint selection.
- Probe seed/epochs/batch/lr/weight decay: 20260832 / 100 /
  64 / 0.001 / 0.0001.
- MLP hidden width: 64; dropout: 0.0.
- kNN uses cosine distance with k=[5, 10, 20]. Candidate pools fall back
  prospectively from exact dataset+layer, to exact layer, to dataset+depth bin,
  then global training states. Preprocessing is training-only.

## Bounded label-incompleteness audit

- Population: validation mandatory-deviation states where a selected router
  predicts non-FULL outside the cached valid-action set.
- Deterministic cap: at most 12 states
  per architecture × predicted action, seed 20260833.
- For each state, execute at most 8 unique
  routes formed by the exact audited prefix, cached-invalid predicted action,
  and a known compatible correcting-route suffix.
- A correct candidate is positive evidence of label incompleteness. Failure of
  every bounded candidate is recorded as `no_bounded_rescue`, not proof that
  the action is globally invalid.

## Stop

Answer Q1--Q9 and identify the dominant supported failure mode. Do not start
the implied next method, a new full router run, or external evaluation.
