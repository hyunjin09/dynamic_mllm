# Phase 52: Stage-1 Gate Winner and Threshold Selection Memory

## Current Objective
Select one frozen Stage-1 gate and one validation operating threshold by treatment-agnostic admission utility at at least 90% failure precision, then confirm all predeclared candidates on the selection-held-out test trajectories.

## Active Constraints
- Reuse only Phase-50 independent and Phase-51 shared validation/test score trajectories; do not retrain or rerun Qwen inference.
- Candidates are independent sequential, shared All-28 sequential, shared Random-4 sequential, and the Phase-51 primary Random-4 score at fixed layer 27.
- Select only on validation by maximum `(wrong triggered - correct triggered) / 800` subject to failure precision at least 0.90, with the plan's ordered tie-breakers.
- The independent candidate retains its established scalar shared-alpha control, which maps to 28 validation-correct higher-quantile thresholds; shared candidates use one raw threshold.
- Freeze the winner before reading Phase-52 test trajectory contents. Existing test trajectories are selection-held-out in this phase but are not historically unopened.
- Stop after test confirmation; no treatment, W-to-C repair, four-action work, MCTS, OOD, or external evaluation.

## Current State
- Done: Completed the frozen validation sweeps, selected one winner before the test command, ran the one planned selection-held-out test comparison, and finalized all requested artifacts.
- In progress: None; stopped at the authorized research-action boundary.
- Blocked: No.
- Most recent useful observation: Fixed L27 won validation by only 3 net utility samples and tied independent sequential on test utility; sequential gating adds no aggregate utility advantage under these controls, while all methods remain highly task-dependent.

## Next-Step Decision
- Deliberation mode: standard.
- Active objective and bottleneck: Choose one treatment-agnostic admission gate without forcing a preselected preservation target.
- Confirmed observation / unverified interpretation: All four frozen candidates have validation/test trajectories; whether fixed L27 or a sequential curve maximizes conservative utility is unverified.
- Diagnosis: unknown; prior calibration mismatch does not identify the winner under the new utility objective.
- Viable alternatives considered: shared-alpha sweep for independent scores; an invalid common raw threshold across independent score spaces; excluding independent from the full curve.
- Chosen action and strongest objection: Sweep every empirical shared-alpha breakpoint for the independent gate and every attainable raw trajectory threshold for shared/fixed gates. Alpha is not a vocabulary-level raw score, but it is the only established coherent scalar control for the 28 independently calibrated spaces.
- How this differs from failed attempts: It compares complete validation Pareto curves and a fixed utility/precision rule instead of choosing from only 99/98/95 points.
- Authorization and stop condition: Explicitly authorized by the user via `plans/stage1_gate_winner_threshold_selection_plan.md`; stop after one validation freeze, selection-held-out test comparison, bootstrap, artifacts, and state update.

## Latest Research-Action Result
- Action taken: Swept every attainable validation control for the four frozen candidates, maximized `(wrong detected - correct false triggers) / 800` subject to aggregate failure precision at least 0.90, froze the winner, and performed one comparison on the existing test trajectories.
- Result: Shared fixed L27 won validation at threshold `0.8497647428417646`: preservation/recall/precision/utility rate `0.9400/0.5400/0.9000/0.2400`. Runner-up shared Random-4 reached `0.9425/0.5300/0.9021/0.23625`, a three-net-sample utility gap. On test, fixed L27 reached `0.9500/0.5275/0.9134/0.23875`; independent sequential tied its utility at `0.23875`, while shared Random-4 and All-28 reached `0.2250` and `0.2175`.
- Evidence saved: `analysis/dense_failure_stage1/gate_winner_selection/` under contract `f065d3728ebce2c95667a566e09ab49d5a4ead901b0334e5e7566df0f2288135`; the 20-file manifest passes.
- Failure or issue: The fixed-L27 minus shared-Random-4 test bootstrap interval spans zero for utility rate `[-0.00375, 0.03125]` and wrong recall `[-0.0125, 0.0475]`. Test performance is heterogeneous: fixed-L27 utility is `0.0775/0.3850/0.4150` on GQA/ChartQA/TextVQA, and GQA precision is `0.7460`. Fixed L27 was already chosen among fixed layers on Phase-51 validation, so this is not an unbiased comparison of every fixed layer. Five leading Phase-51 test rows were accidentally printed before the Phase-52 freeze; no aggregate or selection statistic was computed, and the deviation is documented.
- Lesson learned: Under the frozen aggregate utility, a simple late fixed gate matches or beats the sequential gates; early sequential triggers have no demonstrated utility benefit. The selected operating point is mixture-specific and not a robust per-benchmark admission guarantee.
- Next implication: Carry only shared fixed L27 at threshold `0.8497647428417646` into the next separately authorized four-action treatment-feasibility experiment, retaining the 99/98/95 reference points. Do not claim universal fixed-layer superiority or start treatment automatically.
