# Four-Action POLAR Cross-Server Run Status

## Scope

Read-only status audit performed on 2026-08-28. No training, feature
extraction, evaluation, Slurm submission, cancellation, or checkpoint mutation
was performed.

## Last recorded state on the other server

`workspace/phase_memory/phase_36_four_action_polar_training.md` records Slurm
job `1662` as the end-to-end Image+Question four-action POLAR pipeline:

- fresh projected-visual-feature extraction;
- matched duplicated one-hot BCE and exact valid-set NLL training;
- four H100 GPUs per objective within one eight-GPU allocation;
- ten epochs with per-epoch validation/checkpointing;
- restricted ChartQA, MMMU-Pro Standard/Vision, and POPE evaluation.

The last recorded scheduler state was `PENDING` with reason `AssocGrpGRES`.
There is no recorded transition to running, no recorded completed feature
cache, and no recorded training or evaluation result. The current user reports
that this job never finished. Its exact terminal scheduler state on the other
server cannot be verified from this server.

## Verified state on this server

- `squeue --me` is empty. Job `1662` is not a live job on this server.
- `outputs/four_action_polar/` is absent.
- `runs/four_action_polar/` is absent.
- `checkpoints/four_action_polar/` is absent.
- No job-1662 Slurm log is present.
- No four-action projected-visual-feature cache is present.
- No BCE or NLL epoch checkpoint, training history, validation result, selected
  checkpoint, external-evaluation shard, or merged result is present.

Therefore there is no evidence that either four-action POLAR objective began
training, and there is no checkpoint that can be resumed locally.

## Inputs and implementation that are present

- Four-action labels:
  `datasets/mcts_labels_4action/sequential_branching_v1/full/`
- Local record counts: GQA 3,386; ChartQA 1,785; TextVQA 1,746;
  WeMath2.0 Standard 742; WeMath2.0 Pro 339; total 7,998.
- All 6,917 GQA/ChartQA/TextVQA source records required by the planned VQA
  training population are present.
- Execution contract SHA-256:
  `d8f524b928fb30ea0bb37c6a9389893adb338d4f91992d85255fdfb9bea283cb`.
- Training implementations and matched configs are present:
  `experiments/train_four_action_polar.py`,
  `configs/four_action_polar_image_question_bce_v1.yaml`, and
  `configs/four_action_polar_image_question_nll_v1.yaml`.
- The pinned Qwen2.5-VL model path referenced by the online configuration is
  present locally.

## Required runtime artifacts that are absent locally

- Frozen preparation manifest:
  `outputs/four_action_polar/preparation_v1/manifest_v1.jsonl`
- Frozen manifest audit:
  `outputs/four_action_polar/preparation_v1/manifest_audit_v1.json`
- Fresh four-action visual-feature cache:
  `outputs/four_action_polar/visual_features_v1/`
- Machine-local pipeline wrapper/runbook used on the other server.
- External evaluation bundle and Qwen3 embedding snapshot expected under
  `datasets/eval/`.

The tracked configuration expects 6,811 eligible VQA records, split into 5,945
train and 866 validation records with 248,804 replay-valid routes. It excludes
106 records with no replay-valid four-action route. Those counts are preserved
in the tracked audit evidence, but the exact frozen manifest files themselves
were not transferred to this server.

## Handoff conclusion

The original other-server run should be treated as **submitted but not
completed**, with no reusable training checkpoint found locally. Before a new
launch, either transfer the exact frozen preparation artifacts and runtime
assets from the other server or deterministically reconstruct them and require
the configured checksums to match. A future run would be a fresh execution
from the feature-cache stage, not a checkpoint resume.
