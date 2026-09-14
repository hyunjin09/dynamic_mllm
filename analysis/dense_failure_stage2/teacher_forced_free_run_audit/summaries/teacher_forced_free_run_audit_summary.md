# Teacher-forced versus free-run audit summary

Contract: `3771fd971f07160c95be055f2aff4f4b9016e10c4665811663922e240a69be0f`

## Population and exact expert-state fit

- Audited **569 UIDs**, **4948 trajectories**, and **69178 route-state occurrences** in the seen/full-refit view.
- FULL top-1 accuracy: **0.9947**.
- Non-FULL top-1 accuracy: **0.0183**.
- First-nonFULL top-1 recall: **0.0254** across all successful routes.
- Highest-responsibility-route first-nonFULL recall: **0.1102**.
- First-nonFULL target probability: mean **0.0934**, median **0.0591**.
- First-nonFULL target-vs-FULL margin: mean **-2.4631**, median **-2.6013**.
- Action recall — READ_ONLY **0.0000**, WRITE_ONLY **0.0083**, IGNORE **0.0418**.

## Free rollout and known-route support

- Training Dense-W R0 rescue: **42/463 (0.0907)**.
- Training Dense-C R0 preservation: **106/106 (1.0000)**.
- Left known successful-route support: **429/463 (0.9266)**.
- First off-support delay: median **5.0** layers after trigger.
- Off-support relation to selected first expert intervention: before **4**, at **186**, after **239**, no divergence **34**.
- Supported-action probability mass at divergence: mean **0.15101320252402106**, median **0.10970373451709747**.

## Release experiment

- R0 free from trigger: **42/463 (0.0907)**.
- R1 force only before first non-FULL, release on its exact state: **43/463 (0.0929)**.
- R2 force through the first non-FULL, then release: **264/463 (0.5702)**.
- R1-R0: **+0.0022**; R2-R1: **+0.4773**.
- Removing pre-intervention drift did not materially help. Forcing the first corrective action did; after that intervention the policy preserved a successful outcome for **264/463**, while **199/463** still failed during or after the remaining suffix. Thus later stability is incomplete, but it is downstream of the much earlier first-action failure.

## Held-out generalization and decision

- Frozen internal-dev checkpoint, 115 group-disjoint dev UIDs: first-nonFULL recall **0.0179**, non-FULL recall **0.0088**, W free-run success **0.0860**, off-support rate **0.9247**.
- Seen-to-held-out first-nonFULL recall drop: **+0.0075**.
- Prospectively frozen decision rule: **objective_action_learning**.
- The audit does **not** justify on-policy relabeling as the first response: expert-state first-nonFULL recall is only 2.54%, and exact-state R1 does not improve on R0. Exposure-focused relabeling would become justified only after corrective action selection on exact expert states is made strong while free rollout remains weak.

## Scope limits

Off-support means outside the observed successful route corpus, not provably invalid. The route corpus is not exhaustive, expert actions need not be unique, internal success does not guarantee external success, and this audit does not establish that DAgger or any proposed objective change will work. No external evaluation labels were read or used.
