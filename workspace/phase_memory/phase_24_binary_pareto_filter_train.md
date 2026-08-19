# Phase 24: Pareto-Filtered Binary Predictor Comparison

## Current Objective

Filter the exact frozen GQA/TextVQA/ChartQA max-50 supervision to its
per-sample score/visual-cost Pareto frontier, then run matched Image+Question
duplicated-BCE and exact-valid-set-NLL ten-epoch train-to-external-evaluation
pipelines.

## Active Constraints

- Preserve all 8,000 records, the 7,000/1,000 image-group split, 6,043/874
  positive train/validation inputs, executor, visual cache, architecture,
  initialization, optimizer, schedule, and threshold decoder.
- The only training variable is duplicated-route BCE versus grouped exact
  valid-set NLL. Both objectives consume one checksum-identical Pareto manifest.
- Evaluate the independently validation-selected Image+Question checkpoints on
  the unchanged 22,307-record no-DocVQA suite with current live ALL-ON.
- Run one complete train-to-eval pipeline per GPU. Never use node03. Node04 is
  allowed again by explicit user amendment on 2026-08-18.
- Preserve all previous MCTS, full10, and external-evaluation artifacts.

## Current State

- Done: deterministic Pareto filter, both matched ten-epoch runs and external
  evaluations, and the all-20-checkpoint train-fitting diagnosis.
- Best train Pareto Hit@1 is 18.27% for BCE and 17.95% for NLL versus the
  frozen train BCE-label oracle of 73.92%.
- Current research action is complete; no new training or follow-on experiment
  is authorized.
- Most recent useful observation: 237,802 selected parent routes become 9,905
  Pareto-efficient routes; the exact weighted BCE label oracle is 73.41% Hit@1
  with mean 9.78 ON layers.

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Every removed route has a retained dominance witness; population/split checks pass | `outputs/binary_pareto_v1/audits/pareto_integrity_audit_v1.json` | Opens training without changing sample membership | confirmed |
| Pareto mean is 1.432 routes/positive and 73.25% are singleton sets | `outputs/binary_pareto_v1/audits/pareto_geometry_v1.json` | Defines remaining multimodality | confirmed |
| Weighted BCE label oracle rises from 5.93% to 73.41% Hit@1 | `outputs/binary_pareto_v1/oracle_analysis/label_oracles_v1.json` | Shows filtering repairs much of the label-level hybrid target | confirmed, diagnostic only |
| BCE/NLL configs match outside objective/protocol and all hashes pass | `outputs/binary_pareto_v1/audits/training_readiness_v1.json` | Isolates the loss/supervision formulation | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Unfiltered max-50 duplicated BCE | Exact label oracle was a cached-valid mask for only 5.93% of positives | supported bitwise hybrid target plus dominated-label pressure | Phase 21 report | Test the approved Pareto-only matched objectives | Do not reuse unfiltered labels in this comparison |
| Unfiltered exact set-NLL full10 | Best checkpoints were near/exact ALL-ON | supported common-route shortcut under old sets; broader cause unresolved | Phase 18/19 reports | Test whether removing dominated FULL routes changes training/execution | Do not interpret Pareto label oracle as learned generalization |
| A4000 NLL checkpoint diagnostic, first parity gate | Epoch-1 decoded validation Hit@1 reproduced exactly, but recomputed NLL differed by 0.000502 on a ~16.96 loss and exceeded the initially fixed `1e-4` absolute tolerance | suspected BF16/reduction-order sensitivity; scientific validation source remains the original A6000 history | `runs/binary_pareto_v1/training_fit_analysis_v1/nll_v3.log` | Use a single scale-aware bounded technical repair; do not replace original logged validation metrics | Do not tune scientific thresholds or checkpoints from the diagnostic |
| A4000 NLL checkpoint diagnostic after scale-aware repair | Epoch-2 Hit@1, original-valid Hit@1, and nearest Hamming again reproduced exactly; continuous NLL differed by 0.001544 and failed the repaired gate | supported that decoded route metrics are stable; exact continuous A6000 loss reproduction on A4000 is unavailable | `runs/binary_pareto_v1/training_fit_analysis_v1/nll_v4.log` | Stop tolerance tuning. Evaluate NLL training checkpoints only and use frozen A6000 histories for validation | Do not repeat A4000 validation-loss parity attempts |
| A4000 BCE validation parity through epoch 4 | Pareto/original-valid Hit@1 reproduced exactly, while four of 874 samples crossed the ALL-OFF threshold at epoch 4 | supported small hardware-sensitive threshold effects; scientific validation trajectory is already frozen | `runs/binary_pareto_v1/training_fit_analysis_v1/bce_v5.log` | Apply the same train-only evaluation rule to both objectives for a matched diagnosis | Do not use A4000-recomputed validation metrics as scientific results |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Pareto duplicated BCE | Most Pareto sets are singleton and its label oracle is much more coherent | Whether dominated labels caused BCE failure | high | queued |
| Pareto exact set-NLL | Complete-mask mass can select among remaining Pareto modes | Whether residual multimodality favors set likelihood | high | queued |

## Next-Step Decision

- Deliberation mode: standard.
- Active objective and bottleneck: determine whether Pareto BCE and exact-set
  NLL fit their 6,043 positive training inputs across all ten epochs; histories
  contain train loss but omit decoded train-set route/probability metrics.
- Confirmed observation: both matched pipelines completed and saved every epoch;
  all validation trajectories and checkpoints are available. Unverified
  interpretation: low held-out route fit may reflect train underfit,
  generalization failure, residual multimodality, or a mixture.
- Diagnosis: unknown pending direct checkpoint-on-training evaluation.
- Viable alternatives considered: infer from train loss alone (insufficient),
  evaluate only best/final checkpoints (misses learning dynamics), or evaluate
  all 20 checkpoints on the frozen train set (chosen).
- Chosen action and strongest objection: add a read-only evaluator and compute
  all requested train/validation trajectories; cost is 20 checkpoint passes,
  but this is the minimum complete discriminator requested by the user.
- How this differs from failed attempts: no training, labels, checkpoint,
  decoder, split, or model state changes; only deterministic evaluation.
- Authorization and stop condition: satisfied. Stop after the training-fit
  diagnosis and do not select a new experiment.

## Latest Research-Action Result

- Action taken: evaluated all ten saved checkpoints for each objective on the
  frozen training set, joined them to the original validation histories, and
  produced trajectory, multiplicity, FULL-status, probability, and collapse
  analyses. No training was run.
- Result: primary training-fit failure for both objectives. BCE best/final
  train Hit@1 is 18.27%/17.69%; NLL is 17.95%/17.86%. Singleton Hit remains
  ~24%, while multi-route Hit is approximately zero. Late train-validation gaps
  are secondary because train fit never becomes strong.
- Evidence saved: `reports/binary_pareto_training_fit_analysis.md` and
  `outputs/binary_pareto_v1/training_fit_analysis_v1/`.
- Failure or issue: exact cross-hardware validation loss/threshold parity was
  unavailable; the analysis therefore uses the frozen original A6000
  validation metrics and does not report reconstructed validation route mass.
- Lesson learned: Pareto filtering removes the ALL-ON shortcut but does not
  make either factorized predictor objective fit coherent complete masks.
- Next implication: more data or another loss-only change is not justified by
  this result; any future authorized action must first isolate the existing
  architecture/optimization/input fitting bottleneck.
