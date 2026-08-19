# P13 Prediction-Level Execution Admission Gate

Frozen before any P13 training outcome was inspected.

The 60-record Qwen execution is admitted only if the selected Image+Question
checkpoint satisfies all of the following on the frozen 150-record validation
set:

1. aligned Image+Question set-NLL is lower than both its question-shuffled and
   image-shuffled conditions;
2. ALL-ON fraction is at most `0.90` (at least 15/150 non-ALL-ON masks);
3. at least 10 unique complete masks are decoded; and
4. relative to the matched P13 Question-only checkpoint, either:
   - cached Hit@1 improves by at least `0.03` absolute, or
   - nearest-valid Hamming improves by at least `0.25` layers;
5. neither cached Hit@1 nor nearest-valid Hamming moves materially in the wrong
   direction: Hit@1 may decrease by at most `0.02`, and nearest Hamming may
   increase by at most `0.25` layers.

These thresholds operationalize the plan's “plausible decoded improvement”
requirement. Lower set-NLL alone cannot admit execution. If the gate fails,
P13 stops without loading Qwen2.5-VL for route execution and assigns Outcome B
or C from the frozen probability/decode evidence.

If admitted, execute both Image-only and Image+Question top-1 masks on the same
frozen 60 records used by P11/P12. Execution outcomes cannot alter checkpoint
selection or the admission thresholds.
