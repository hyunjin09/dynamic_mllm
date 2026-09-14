# Stage-2 MCTS failure diagnosis

## Result

- A oracle single non-FULL recall: 0.1078.
- B oracle MCTS non-FULL recall: 0.0845.
- B MCTS first/later intervention recall: 0.0993 / 0.0782.
- B free MCTS-route rescue (C0): 0.0566; best partial-prefix rescue gain: +1.0000.
- B-minus-A single corrective recall: -0.0942, UID-bootstrap 95% CI [-0.2556, -0.1801].
- B multi-valid versus single-valid nominal error gap: +0.2531; valid-set recovery on multi-valid rows: 0.2163.
- Fixed-rule hypothesis support: {"H1_action_learning": true, "H2_exposure": false, "H3_negative_transfer": true, "H4_ambiguity": true, "H5_representation": false}.

## Answers to the plan

1. A's exact oracle single-route action performance is reported above and in `oracle_state/action_metrics_A.csv`.
2. B's exact oracle MCTS performance is reported above and stratified by route complexity/intervention index.
3. First-versus-later failure is quantified in `oracle_state/intervention_index_breakdown.csv` and `first_deviation/summary.csv`.
4. Prefix forcing changes rescue by at most +1.0000 over C0 under the frozen P90 route population.
5. State drift after each first wrong action is quantified for the actual text-query and visual-token inputs in `state_drift/summary.csv`.
6. Paired single-state transfer is quantified with UID bootstrap in `negative_transfer/`.
7. The four A rescues and one A regression are described individually under `case_studies/`.
8. Exact-prefix observed-valid action ambiguity is reported in `ambiguity/`.
9. A/B FULL confidence margins are compared in `confidence/full_margin_by_state_type.csv`.
10. The fixed prospective rules support: H1_action_learning, H3_negative_transfer, H4_ambiguity.
11. This does not establish that MCTS or sequential correction is impossible, that MCTS supervision is inherently harmful, or that single intervention is universally superior.

## Scope

This diagnosis uses frozen training-route states and Historical-validation case studies. It changes no checkpoint, threshold, Stage-1 component, label, evaluator, or test result.
