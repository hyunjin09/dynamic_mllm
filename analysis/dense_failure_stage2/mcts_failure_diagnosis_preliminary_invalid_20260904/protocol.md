# Stage-2 MCTS failure diagnosis protocol

- Contract: `de04332defe012cbcb6d442c806d2151b50b49e21b926e37bc628296ea02d99a`
- Parent Stage-2 contract: `b17a81d4749b842fd843a344d3799bffdfc0f41e2c516a835242b656324440f8`
- Checkpoints: frozen Experiment A and B final-update checkpoints; no retraining.
- Population: 2519 replay-valid routes and 34253 oracle entering states.
- Primary rollout diagnostic: P90 across all 725 MCTS routes.
- Exact input rule: reconstruct complete routed token sequences by deterministic replay because Phase-65 saved tensors are pooled summaries only.
- Prefix forcing: C0 free, C1..Ck through each successive non-FULL action, plus full oracle control.
- Ambiguity identity: exact UID + layer + complete entering action prefix.
- Bootstrap: UID-level, 2000 replicates, seed 20260904.
- Stop: diagnosis and one recommendation only; no training, search, threshold change, or held-out test.
