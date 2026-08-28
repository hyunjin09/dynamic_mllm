# Four-Action POLAR Training Preparation Report

Preparation is complete for two Image+Question runs—POLAR-style duplicated BCE
and exact set NLL—but training is not yet authorized or marked ready because
the fresh visual-feature cache has not been extracted.

## Present

- Audited 6,811-record / 248,804-route GQA, ChartQA, TextVQA manifest.
- Complete local Qwen3-Embedding-0.6B snapshot.
- Pinned Qwen2.5-VL-7B-Instruct revision `cc594898...`.
- Exact 14,960-row ChartQA/MMMU-Pro/POPE evaluation population and shared-prefix
  evaluation code.
- Four-action predictor, BCE and NLL objectives, complete-route decoder,
  Image+Question collators, validation metrics, ten-epoch resumable trainer,
  per-epoch checkpoints, best-checkpoint freeze, external evaluator, integrity
  merger, and BCE-vs-NLL comparison.
- Two frozen static-preflight configs and passing static reports.

## Missing by design

- `outputs/four_action_polar/visual_features_v1/cache_audit_v1.json` and its
  6,490 unique-image tensors. The old imported cache is not reused.
- Training checkpoints and external outputs, because the user explicitly said
  not to start training yet.

## Runtime state

The host has eight idle H100 80GB GPUs and no current user jobs. The project
policy still requires GPU work to run through Slurm on this physical host. The
current restricted Codex process has no GPU device nodes because it is not in a
Slurm allocation. No job was queued or submitted.

## Next action

The smallest next action is one authorized eight-GPU Slurm cache-extraction
allocation, followed by CPU cache finalization and both full preflights. This
does not start model training.
