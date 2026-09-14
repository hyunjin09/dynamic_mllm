# Next training recommendation

Expanded-corpus health for the planned comparison: **PASS**.

If separately authorized, keep the frozen V1 architecture, optimizer/loss, Stage-1 gate, validation set, and free-rollout evaluator fixed:

1. **Scaled-Single:** Expanded A + Expanded B; exclude Corpus C.
2. **Scaled-Single + MCTS:** the same setup with Expanded C added.

Use a matched optimizer-update budget to the frozen V1-698 run or report the exact difference. This file is a recommendation only; neither training experiment was launched.
