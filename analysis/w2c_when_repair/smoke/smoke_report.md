# W2C WHEN Repair Smoke Report

## Decision

**FAIL**. The frozen 12-sample smoke
completed 1,401 route executions, including
312 exact original-route replays, with
0 quarantined samples. The resume replay left all raw
record bytes unchanged.

The failed check is substantive: 37/312 original cached-correct routes replayed
incorrectly under the frozen current runtime, affecting 10/12 samples. The
route identities and order exactly matched the source manifest. Failures span
GQA (26 routes, 3 samples), TextVQA (8 routes, 4 samples), and ChartQA
(3 routes, 3 samples).

## Repair behavior

- Samples with at least one newly verified correct route: 8/12
- Newly verified correct routes: 121
- Samples exercising more than one repair round: 8/12
- Final boundary-shift counts: `{"0": 4, "1": 1, "13": 2, "21": 1, "4": 1, "6": 1, "7": 1, "8": 1}`

## Gate checks

| Check | Result |
|---|---|
| exact old route replay | FAIL |
| full insertion at candidate boundary | PASS |
| compatible suffix enumeration | PASS |
| deduplication | PASS |
| correct cache update | PASS |
| candidate boundary moves after rescue | PASS |
| iterative re evaluation | PASS |
| bounded search after known exhaustion | PASS |
| output determinism | PASS |
| resume restart consistency | PASS |
| coverage and zero quarantine | PASS |

## Decision-changing diagnostic

One failed route per GPU was executed twice under the same current runtime.
All 4/4 pairs reproduced exactly, while 0/4 current generated-token sequences
matched their original cached generated-token sequences. Thus the mismatch is
reproducible current-runtime cache drift, not transient run nondeterminism.
Raw evidence and checksums are under
`/mnt/hyemin/qwen_train_eval/outputs/w2c_when_repair_v1/smoke_replay_diagnostic/`;
the compact result is `replay_failure_diagnostic.json`.

The original label record binds `binary_policy/executor/four_action.py` to
SHA-256 `e8c503618998946b4411fb7beb43c42d1be9f8954527064597b1c34ed2571868`
and `binary_policy/executor/inputs.py` to
`0e22848f56aaaec1c510958eee37e407d0dc51726dc60e5a69a3eea54091d465`.
The current frozen repair config binds those files to
`7f8a076289ef3cc0d09dea09e5c8a5a2606bc758f3ce60eea75928b810759037`
and `b0fa74da3ab134e5e08ae3e36877b64d59d42ff5b50f01cf01914f2daa137716`,
respectively. This establishes code-contract drift but does not by itself prove
which change caused the output differences.

`FULL_UNRESCUED_UNDER_BUDGET` retains bounded one-edit semantics and is not a
claim that FULL is globally invalid. A pass admits only the frozen 640-sample
repair; it does not admit gate/router training or external evaluation.

Because the smoke failed, the 640-sample repair was not started.
