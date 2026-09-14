# Phase 68: Stage-2 Observed-Valid-Set Loss Memory

## Current Objective
Run one matched Experiment C that changes only Phase-66 Experiment B's single-label CE objective to exact observed-valid-set loss, then compare oracle-state and P98/P95/P90 free-rollout behavior.

## Active Constraints
- Follow `plans/stage2_observed_valid_set_loss_plan.md` as one bounded research action.
- Freeze Phase-66 B's backbone, router, optimizer, 3,144-update schedule, sampler, union corpora, initialization seed, validation population, Stage-1 thresholds, executor, and LMMS evaluator.
- Merge actions only for exact UID + layer + complete entering action-prefix identity.
- Use `-logsumexp(valid logits) + logsumexp(all logits)` in FP32 and fail on empty/non-finite masks.
- Stop after valid-set audit, smoke, C training, B-vs-C oracle evaluation, P98/P95/P90 Historical-800 rollouts, and one written next decision. Do not open test or launch the recommendation.

## Current State
- Done: Frozen exact-state audit, implementation/unit tests, deliberate smoke, matched 3,144-update C training, paired B-vs-C oracle evaluation, and P98/P95/P90 Historical-800 rollouts.
- Stopped: The plan's one authorized research action is complete. The one next recommendation is written but not executed.
- Blocked: No.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| B exact MCTS non-FULL recall is 0.0845 and single corrective recall is 0.0136 | `analysis/dense_failure_stage2/mcts_failure_diagnosis/` | Defines the primary matched oracle baselines | confirmed |
| Multi-valid B error is 0.3096 versus 0.0565 single-valid; valid-set membership recovers 0.2163 | Phase-67 ambiguity audit | Directly supports testing exact observed-valid supervision | confirmed |
| Phase-66 B completed a fixed 3,144-update schedule and all three Historical-800 rollouts | `analysis/dense_failure_stage2/shared_union_training/` | Provides the frozen matched data/sampler/optimizer/evaluator reference | confirmed |
| C raises MCTS nominal/observed-valid non-FULL recall from 0.0845/0.1036 to 0.1616/0.2126 | `analysis/dense_failure_stage2/observed_valid_set_loss/oracle_eval/overall_metrics.csv` | Confirms that exact accepting-set supervision improves oracle compatibility | confirmed |
| C leaves single corrective nominal recall at 0.0136 and observed-valid recall at 0.0261; A nominal reference is 0.1078 | `analysis/dense_failure_stage2/observed_valid_set_loss/oracle_eval/single_negative_transfer.csv` | The new objective does not repair Phase-66 negative transfer on single-route corrective states | confirmed |
| C P98/P95/P90 net corrections are -1/-1/0; P90 is 1 W→C and 1 C→W | `analysis/dense_failure_stage2/observed_valid_set_loss/free_rollout/threshold_comparison.csv` | Oracle improvement does not translate into material free-rollout benefit | confirmed |

## Next-Step Decision
- Deliberation mode: deep
- Active objective and bottleneck: test whether exact-prefix single-label conflicts materially caused B's weak corrective learning and negative transfer.
- Confirmed observation / unverified interpretation: exact-prefix action ambiguity is directly measured; whether changing only the loss improves learned behavior is unverified.
- Diagnosis: supported for label conflict as a measurable error source; evidence is Phase 67's exact-state ambiguity and B-vs-A transfer results.
- Viable alternatives considered: observed-valid-set loss; A-initialized reweighting; on-policy collection. The latter two change different variables and are outside this plan.
- Chosen action and strongest objection: run matched Experiment C. The strongest objection is that observed sets are bounded-search evidence rather than exhaustive validity, so a negative result cannot rule out unobserved alternatives.
- How this differs from failed attempts: B's routes and sampler remain identical; only one-hot CE is replaced with probability mass over exact-prefix observed-successful actions.
- Authorization and stop condition: explicitly authorized by the user's request to perform the named plan; stop after C's oracle/rollout comparison and one unexecuted recommendation.

## Latest Research-Action Result
- Action taken: trained matched Experiment C with exact observed-valid-set loss and evaluated it against B on identical oracle states and Historical-800 free rollouts.
- Result: Decision Case B. C improves MCTS nominal non-FULL recall from 0.0845 to 0.1616 and observed-valid non-FULL recall from 0.1036 to 0.2126. First/later observed-valid recall improves from 0.1352/0.0900 to 0.2593/0.1925. However, single corrective nominal recall stays 0.0136 (A reference 0.1078), and P90 produces only 1 W→C plus 1 C→W for net zero.
- Evidence saved: authoritative contract `c3e6adfd54a58e5b30eea86757599be49a5dd9879e01a1f7b0c26ad692bd8421`, checkpoint `1f5fd8d776fe63b5b21e8c53b015a3cdcd3acc2853cb0ff5c10729d0b3a21540`, and 86-file artifact audit under `analysis/dense_failure_stage2/observed_valid_set_loss/`.
- Failure or issue: no execution failure. The inherited bootstrap helper's output column is named `mean_delta_B_minus_A`; in this phase its bound inputs are B then C, so the numeric field denotes C-minus-B. The main summaries use the correct B→C semantics.
- Next implication: accepting-set label ambiguity was a causal oracle-learning bottleneck for MCTS states, but not the sole deployment bottleneck. The single-source deficit and oracle-to-rollout gap remain.
- One unexecuted recommendation: a small partial-prefix on-policy collection targeting first-deviation states under fixed C, to isolate exposure shift without combining changes.
