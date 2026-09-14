# Stage-2 MCTS failure diagnosis

## Result

- A recognizes its single-route corrective actions weakly: state-weighted non-FULL recall is 0.1078.
- B recognizes MCTS corrective actions weakly even on exact oracle states: non-FULL recall is 0.0845; first/later-intervention recall is 0.0993 / 0.0782.
- B's first teacher-forced disagreement is at the first MCTS intervention on 88.83% of routes.
- B free MCTS-route rescue (C0) is 0.0566. Forcing only C1 raises matched UID rescue by +0.0325; C2 and C3 raise it by +0.1814 and +0.4340 among routes that still have a later corrective action.
- Crucially, after release B reproduces only 8.35%, 6.75%, and 8.33% of those remaining C1/C2/C3 oracle non-FULL actions. The accuracy rise is therefore mostly attributable to forced corrections, not recovery of the stored later policy.
- B-minus-A single corrective recall is -0.0942 state-weighted. The UID-weighted delta is -0.2174, 95% CI [-0.2556, -0.1801].
- B multi-valid versus single-valid nominal error gap: +0.2531; valid-set recovery on multi-valid rows: 0.2163.
- Fixed-rule hypothesis support: {"H1_action_learning": true, "H2_exposure": false, "H3_negative_transfer": true, "H4_ambiguity": true, "H5_representation": false}.

## Answers to the plan

1. **A on single routes:** A is better than B on corrective single states, but its own exact oracle recall is still only 10.78%; it does not reliably reproduce its training-route action.
2. **B on MCTS routes:** B reaches 8.45% corrective recall on exact successful MCTS states. That is better than A-on-MCTS (0.66%) but still an action-learning failure in absolute terms.
3. **First versus later:** both are poor (9.93% vs 7.82%), and 88.83% of routes first disagree exactly at the first intervention. There is no large later-only deficit.
4. **Prefix forcing:** C1 gives only +3.25% matched UID rescue. C2/C3 gains are larger (+18.14%/+43.40%), but later corrective-action reproduction remains at most 8.35%. This is not strong evidence that the learned policy recovers after an oracle prefix.
5. **State drift:** at the first post-error layer, text-query relative L2 is 0.103 and visual-sequence relative L2 is 0.233; drift is real, but weak oracle recognition already precedes it.
6. **Negative transfer:** B loses 9.42% state-weighted single corrective recall relative to A, with a strictly negative UID-bootstrap interval. Overall accuracy rises only because B predicts FULL more often.
7. **A's four rescues:** B stays all-FULL on three; on the fourth it intervenes earlier and repeatedly with IGNORE before WRITE_ONLY, but remains wrong. The single A regression is removed by B staying FULL.
8. **Ambiguity:** on B, nominal error is 30.96% for multi-valid states versus 5.65% for single-valid states. Accepting any observed successful action recovers 21.63% of multi-valid rows.
9. **FULL bias:** it is regime-specific, not global. B's mean FULL-minus-best-non-FULL margin increases from 0.322 to 0.569 on single corrective states, but decreases sharply on MCTS first/later states.
10. **Best explanation:** H1 action learning, H3 negative transfer, and H4 ambiguity are jointly supported. H2 exposure is secondary evidence only; H5 is not isolated by the fixed rule. The smallest discriminating next experiment is the fixed observed-valid-set loss described separately.
11. This does not establish that MCTS or sequential correction is impossible, that MCTS supervision is inherently harmful, or that single intervention is universally superior.

## Scope

This diagnosis uses frozen training-route states and Historical-validation case studies. It changes no checkpoint, threshold, Stage-1 component, label, evaluator, or test result.
