# Online Four-Action Router Early-Stop Report

## Decision

Stop the online-router training and do not run external evaluation. The user
explicitly authorized this at 2026-08-29 14:19 KST after nine completed routed
validations showed repeated zero-rescue, effectively all-FULL behavior.

## Validity boundary

- Real eight-H100 semantic smoke 1690 passed every executor, routed-state,
  multi-valid supervision, gradient, frozen-backbone, determinism, loss-
  decrease, and checkpoint-roundtrip gate.
- Training 1691 used exact Git commit `f6a0c42`, the frozen 6,811-record
  GQA/ChartQA/TextVQA manifest, eight-way DDP, and the frozen optimizer/schedule.
- Epochs 1--9 each contain a checksum-valid router checkpoint and exactly 866
  unique validation outputs.
- The preserved history contains epochs 1--9 exactly and has SHA-256
  `a1b9961e5040789174519a9ab9df41eb05baaa43d5f1868cee391caedc6f72c8`.
- No temporary or zero-byte training artifact exists.

The run was therefore valid for interpreting the observed routing behavior.
The cause of that behavior remains unknown.

## Completed epoch results

| Epoch | Step | Train loss | Valid-Action@1 | Routed accuracy | W2C rescue | C2C preserve | FULL | IGNORE | READ_ONLY | WRITE_ONLY |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 1.031217 | 0.690902 | 0.579677 | 0.000000 | 0.984314 | 23,828 | 420 | 0 | 0 |
| 2 | 96 | 0.679224 | 0.712347 | 0.588915 | 0.000000 | 1.000000 | 24,248 | 0 | 0 | 0 |
| 3 | 144 | 0.638322 | 0.715482 | 0.588915 | 0.000000 | 1.000000 | 24,248 | 0 | 0 | 0 |
| 4 | 192 | 0.606559 | 0.707027 | 0.588915 | 0.000000 | 1.000000 | 24,248 | 0 | 0 | 0 |
| 5 | 240 | 0.598854 | 0.714409 | 0.588915 | 0.000000 | 1.000000 | 24,248 | 0 | 0 | 0 |
| 6 | 288 | 0.586317 | 0.719070 | 0.588915 | 0.000000 | 1.000000 | 24,248 | 0 | 0 | 0 |
| 7 | 336 | 0.586094 | 0.710822 | 0.588915 | 0.000000 | 1.000000 | 24,248 | 0 | 0 | 0 |
| 8 | 384 | 0.569411 | 0.716801 | 0.588915 | 0.000000 | 1.000000 | 24,248 | 0 | 0 | 0 |
| 9 | 432 | 0.571311 | 0.724555 | 0.588915 | 0.000000 | 1.000000 | 24,247 | 1 | 0 | 0 |

Training loss and node Valid-Action@1 improved, but actual routed execution did
not. Every completed epoch had zero W2C rescues. Epochs 2--8 were exactly
all-FULL, while epoch 9 differed at only one layer decision.

## Alternatives considered

1. Cancel training and external evaluation immediately.
2. Allow epoch 10 to finish, then cancel external evaluation.
3. Finish training and run the 14,960-record external evaluation.

External evaluation was not decision-relevant after nine valid zero-rescue
internal validations. At the decision boundary, epoch 10 had reached optimizer
step 478/480 but had not created an atomic checkpoint. Its remaining near-zero-
learning-rate tail was unlikely to reverse nine epochs of behavior. Immediate
stop was therefore selected. The strongest objection is loss of the final
planned checkpoint, but that did not justify continued eight-H100 use.

## Terminal execution state

| Job | Purpose | Final state | Runtime |
|---:|---|---|---:|
| 1691 | ten-epoch training | CANCELLED by user/agent | 1:11:23 |
| 1692 | restricted external evaluation | CANCELLED before start | 0:00:00 |

Preserved output boundary:

```text
outputs/four_action_online_router/training_v3/epoch_01 ... epoch_09
outputs/four_action_online_router/training_v3/history.json
outputs/four_action_online_router/training_v3/initialization.json
```

Absent by design:

```text
epoch_10/
best_checkpoint.json
training_summary.json
outputs/four_action_online_router/external_v3/
```

## Interpretation boundary

- Supported: this trained checkpoint family failed to produce useful non-FULL
  internal routing or any W2C rescue despite improved training/node metrics.
- Supported: external evaluation was stopped before outcomes were opened.
- Unknown: why the router converged to FULL.
- Not supported: that every online architecture, supervision objective, or
  four-action router must collapse.
- Next action: none selected. Any architecture, objective, weighting, or label-
  supervision change requires a separately authorized research action.
