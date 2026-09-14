# Next Stage-2 training recommendation

Threshold-specific corpus health: **PASS**.

If separately authorized, use the same frozen Stage-2 V1 READ/WRITE router architecture, optimizer, sampling logic, loss, and validation evaluator in three independent runs:

1. P98 preservation + single corrective.
2. P95 preservation + single corrective.
3. P90 preservation + single corrective.

Keep MCTS corrective supervision as a matched second ablation after the preservation+single comparison. Select the downstream operating point using validation final accuracy, W-to-C, C-to-W, and C-to-C—not corpus size alone. No training was launched in this phase.
