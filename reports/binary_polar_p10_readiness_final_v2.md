# Final P10 Pre-Training Readiness Decision

Status: **READY FOR BOUNDED SMOKE**.

No optimizer step, predictor training, checkpoint fitting, or 7B MLLM evaluation was performed.

## What is verified

- Exact valid-set NLL is the stable weighted probability mass over complete 28-bit masks.
- Duplicated BCE and set-NLL share the same selected masks, equal weights, direct head, data split, optimizer settings, initialization, shuffle generator, and checkpoint-selection rule.
- All 8,000 manifest rows, 6,043 positive train rows, 874 positive validation rows, 1,083 zero-positive rows, masks, weights, caps, and image groups pass the frozen audit.
- The real pinned Qwen3 tokenizer/encoder produced BF16 `[3,11,1024]` features on one GQA, TextVQA, and ChartQA record. Both objectives produced finite losses and finite gradients on all 33 predictor parameter tensors from the same initialization. Encoder gradients remained absent.
- The execution adapter evaluates both selected-set and full raw-cache Hit@1, and actually executes uncached top-1 masks through the repaired binary Qwen executor.

## Controlled boundary

The predictor is question-only because that is the released POLAR architecture. Image conditioning would be an architecture change, not the requested loss-only comparison. The factorized head still does not explicitly model cross-layer dependencies.

## Remaining gate

Only the frozen 300-train/150-validation, two-epoch matched smoke may run next, with 18 actual execution records per objective. Full training remains blocked until that smoke passes and is interpreted.
