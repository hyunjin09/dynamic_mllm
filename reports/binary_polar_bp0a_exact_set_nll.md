# Binary POLAR BP-0A Exact Valid-Set NLL Check

Date: 2026-08-09

Decision: `BP0A_PASS`

The direct binary loss computes the probability of each complete valid mask,
adds the normalized route log-weight, and applies a log-sum-exp over complete
masks. It does not average mask bits into a training target.

Checks:

- independent weighted complete-mask formula absolute error: `0.0`;
- padded invalid-route contribution error: `0.0`;
- finite gradients: four of four deterministic runs;
- contradictory-mask coherent top-1 membership: four of four runs;
- loss fell from approximately `2.7725` to `0.7028` in every run;
- label integrity inherited from BP-0: passed.

The contradictory valid set was `{1100, 0011}`. Seeds `7` and `29` selected
`1100`; seeds `13` and `43` selected `0011`. No run selected a marginal hybrid.
This demonstrates objective/implementation consistency only. The logits were
free per example, so the result is not evidence that a shared predictor will
generalize.

Evidence:

- `outputs/binary_polar/preflight/bp0a_exact_set_nll_v1.json`
- `outputs/binary_polar/preflight/bp0a_exact_set_nll_v1.json.sha256`
- `binary_policy/losses.py`
- `binary_policy/objective_audit.py`
- 15 binary-policy contract tests passed.
