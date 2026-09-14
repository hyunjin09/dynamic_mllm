# Trigger-conditioned corrective-search pilot protocol

- Contract SHA-256: `6b4eb6de4573a0f06397304721483348523b78a1ce905c5076e49763f15afa2c`
- Source cohort: exactly the Phase-54 train triggered Dense-W manifest (1,881 rows); 120 image-group-unique rows are prospectively frozen for this pilot.
- Diagnostic allocation: 40 per dataset. GQA depth cells L0/L1-8/L9-18/L19-27 = 3/12/12/13; ChartQA and TextVQA = 10/10/10/10 each.
- Primary results include both unweighted pilot estimates and cell-weighted estimates against the full 1,881-row source population.
- Execution begins from the exact native dense FULL state entering the frozen first trigger layer. Layers before the trigger remain FULL.
- Phase A exhausts all three non-FULL single-layer interventions at every layer from trigger through 27. Every correct single route is retained. Single-fixable rows do not enter MCTS.
- Phase B is ordered prefix-tree UCB1 MCTS with actions FULL/READ_ONLY/WRITE_ONLY/IGNORE, binary current LMMS correctness reward, maximum 300 iterations, and checkpoints at 100/200/300.
- An absolute Fixable@300 minus Fixable@200 gain above 0.01 is prospectively defined as material for choosing 300 rather than 200 iterations.
- Rollouts deterministically cycle through 2, 3, and 4 total non-FULL interventions, conditional on the already selected tree prefix. This removes a suffix-length-dependent intervention prior.
- After first success, search continues exactly 25 additional iterations, capped at 300; up to 8 MCTS successes are retained by fewer interventions then stable route key.
- Successful routes are replayed exactly, token/correctness parity is required, and compact pre-layer states entering each chosen suffix action are stored for future Stage-2 training.
- All outputs are bounded-search lower estimates. No Stage-2 training, threshold change, validation/test search, triggered-C search, or external evaluation is authorized.
