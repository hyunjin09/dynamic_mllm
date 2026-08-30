# Persistent Corrective Supervision Protocol

Frozen before smoke, training, or checkpoint execution on 2026-08-30.

## Authority and scope

- Authorized plan: `plans/four_action_generalization.md`
- Plan SHA-256: `79c159af4aa451cdbb153e95b7145566f77835770c1408765f1fafe1d35837b5`
- Action: one low-budget matched comparison through the architecture decision.
- Excluded: external evaluation, scale-up, lambda tuning, label regeneration,
  DAgger, RL, architecture changes, C2C removal, and follow-up experiments.

## Frozen population

- Exact IDs and cells: `subset_manifest.json`
- Subset manifest SHA-256:
  `92dc068566b799f852c30d0e459aa4f231d6082f2a59c78c64c56c0b9c66936b`
- Training manifest: `training_manifest.jsonl`, SHA-256
  `05716064b11703a339a1bc3f59a54d0e54a50b5490a5748acebbd8f2e91c1583`
- Boundary manifest: `boundary_manifest.jsonl`, SHA-256
  `868bd1981a69e579e5c9cf671d8b36e3f3165ff0066cc1ff609002e851f887fb`
- Train: 512 W2C + 512 C2C.
- Validation: 128 W2C + 128 C2C.
- Selection seed: `20260830`.
- The inherited image-group split is preserved. Selection uses only split,
  dataset, route type, and boundary label metadata; it uses no model outcome.

## Matched intervention

- Both substrates retain their existing architectures and base objectives.
- POLAR base objective: exact-set NLL.
- Online base objective: set-valued next-action loss on one deterministic valid
  teacher route per sample and epoch.
- Every train W2C receives one additional set-valued mandatory-boundary loss in
  every epoch. No C2C receives this term.
- The valid semantic target is the same `valid_nonfull_actions` set from the
  frozen boundary manifest for both substrates.
- `lambda_boundary = 1.0`; it will not be tuned.
- Each epoch visits every selected train sample exactly once in an alternating
  W2C/C2C order. There is no front-loading or replacement sampling.

## Schedule and compute

- Epochs: 20 for both substrates; no early stopping.
- Validation and checkpoint: every epoch.
- Training seed: `20260809` for both substrates.
- Optimizer: AdamW, learning rate `5e-4`, weight decay `0.01`, cosine schedule,
  10 warmup steps, gradient clip norm `1.0`.
- POLAR: physical/effective batch 128, 8 optimizer steps/epoch, 160 total;
  four-GPU `DataParallel` over CUDA devices 0,1,2,3.
- Online: one sample/rank, gradient accumulation 16, effective global batch 64,
  16 optimizer steps/epoch, 320 total; four-GPU DDP with direct `torchrun`.
- The different effective batches retain each architecture's validated runtime
  pattern; epochs, IDs, supervision events, validation, and selection are
  matched.
- This server has no Slurm. GPU execution is direct and must use all four GPUs.

## Prospective checkpoint and architecture rule

For each architecture:

1. Restrict to checkpoints with C2C preservation at least 95%.
2. Among eligible checkpoints, maximize held-out W2C rescue.
3. Tie-break by higher C2C preservation, higher overall routed accuracy, fewer
   C2C regressions, fewer FULL actions, then earlier epoch.
4. If no checkpoint meets 95%, select none and report the complete W2C-rescue
   versus C2C-preservation Pareto frontier.

An architecture is behaviorally viable only if an eligible checkpoint has at
least one held-out W2C rescue. If exactly one is viable, favor it. If both are
viable, compare their selected paired validation outcomes; a winner is claimed
only when the paired bootstrap 95% interval for W2C rescue-rate difference
excludes zero. Otherwise report no supported architecture advantage; when both
behave similarly, POLAR remains operationally simpler. If neither is viable,
select neither. Bootstrap draws: 10,000; seed: `20260830`.

## Metrics and stop rule

- Primary: held-out free-running W2C rescue, C2C preservation, rescues,
  regressions, and net accuracy change.
- Secondary: train and validation boundary Valid-Action@1, validation boundary
  non-FULL recall, first-deviation timing, deployment action distribution, and
  online teacher-forced/free-rollout gap.
- Execute every checkpoint on the same 256 validation records.
- After producing the required comparison and architecture decision, update
  compact research state and stop. Do not run external evaluation or a
  follow-up action.
