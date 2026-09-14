# Stage-2 observed-valid-set loss result

## Frozen matched experiment

Experiment C used the exact Phase-66 B model, router, seed, optimizer, 3,144-update schedule, sampler, union routes, Stage-1 thresholds, Historical-800 validation set, executor, and LMMS evaluator. The only training change was one-hot CE to exact-prefix observed-valid-set logsumexp loss.

## Answers

1. Exact routed states with 1/2/3/4 observed-valid actions: **19,772 / 785 / 377 / 137**.
2. Overall observed-valid top-1 accuracy B→C: **0.9346 → 0.9374**.
3. Overall observed-valid probability mass B→C: **0.8022 → 0.9056**; MCTS gain +0.0723.
4. MCTS first-intervention nominal / observed-valid recall B→C: **0.0993/0.1352 → 0.1876/0.2593**.
5. MCTS later-intervention nominal / observed-valid recall B→C: **0.0782/0.0900 → 0.1505/0.1925**.
6. Single corrective nominal / observed-valid recall B→C: **0.0136/0.0237 → 0.0136/0.0261**; A nominal reference is 0.1078.
7. Observed-valid top-1 gain is +0.0139 on multi-valid states versus -0.0007 on single-valid states.
8. MCTS FULL-minus-best-non-FULL margin B→C: **0.5698 → 0.6724**.
9. P90 C rollout: W→C/C→W/net = **1/1/+0**, any non-FULL 11.57%, routed accuracy 0.500000.
10. W→C recovery relative to B: **1 versus 0**.
11. C→W under C: **1** versus B's 0.
12. Fixed decision case: **B**. Oracle MCTS observed-valid recall gain is +0.1090; single corrective gain is +0.0024. This determines whether label ambiguity is strengthened as a causal bottleneck or only a descriptive correlate.
13. This experiment cannot establish unseen-source/test generalization, exhaustive action validity, globally optimal MCTS routes, or whether a different architecture/on-policy distribution would work.

## Scope

No test set, new search, Stage-1 change, threshold change, architecture change, layer embedding, Stage-1 latent, class weighting, focal loss, RL, or on-policy training was run.
