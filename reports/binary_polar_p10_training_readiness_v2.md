# P10 Binary Predictor Training-Readiness Audit

Status: **PASS**. No predictor training or MLLM execution was run.

## Confirmed implementation

- Both objectives use the same frozen 8K-derived manifest, 7K/1K image-group split, max-50 route sets, equal within-input weights, direct 28-bit head, optimizer, and inference rule.
- Exact set-NLL computes complete Bernoulli-mask log probabilities with stable `logsigmoid`, masked padding, normalized weights, and `logsumexp`.
- Duplicated BCE encodes each unique question once, evaluates every selected route as an independent predictor/BCE row, and normalizes each input's total route weight to one.
- The runner now binds gates and the predictor manifest to SHA-256, uses a dedicated deterministic DataLoader generator, refuses output overwrite, and applies one common checkpoint-selection rule.
- The execution adapter runs every predicted top-1 mask, including uncached masks, and reports FULL-relative behavior, cached Hit@1, visual-ON layers, and the observed MCTS-oracle gap.

## Important boundary

The predictor is **question-only**, matching released POLAR. Adding image features would be an architecture change and would no longer isolate the supervision loss. Zero-positive samples are excluded from positive training but retained in actual execution evaluation.

## Frozen bounded smoke

The smoke uses 100 positive training and 50 positive validation records per dataset (300/150 total), two epochs, and 18 deterministic validation execution records. The same UIDs, initial seed, order generator, optimizer settings, and checkpoint rule apply to both objectives.

Audit SHA-256: `f4ea0df223d3322cdc1bc2a622a1c73d138412f348b9b47e786056b89864cd22`. Smoke-manifest SHA-256: `ac0d0fa76331511794fe01532e41af97bde4b474f3cff5814ca2a0e7325bafe0`.

## Gate

The implementation is ready for the separately authorized bounded smoke, not full training. Full training remains blocked until the smoke has finite decreasing loss, frozen gradients, working actual mask execution, no leakage, and plausible validation improvement.
