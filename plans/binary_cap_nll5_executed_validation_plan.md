# Binary CAP26/CAP24 Exact-Set-NLL with Executed Validation

## Objective

Compare two otherwise matched Image+Question direct binary predictors trained
with exact valid-set NLL. The only supervision difference is the maximum
VISUAL_ON count retained from the frozen max-50 MCTS routes:

- CAP26: retain valid routes with at most 26 ON layers;
- CAP24: retain valid routes with at most 24 ON layers.

This is one bounded experiment. It does not authorize a new architecture,
objective, dataset, route search, or follow-on experiment.

## Frozen data and model contract

- Source: frozen pre-Pareto GQA/TextVQA/ChartQA max-50 predictor manifest.
- Common population: inputs having at least one CAP24-valid route.
- Frozen counts: 6,007 train and 872 validation records.
- Split remains image-group-disjoint.
- Predictor: existing Image+Question direct factorized 28-bit head.
- Frozen Qwen3 question encoder and frozen Qwen2.5-VL projected visual rows.
- Objective: exact one-of-valid-set NLL with the existing route-weighting
  convention. Since ALL-ON is absent under both caps, the inherited POLAR
  full-route down-weighting is operationally equal weighting.
- Five epochs, effective batch 128, AdamW, cosine schedule, learning rate
  5e-4, warmup 10, seed 20260809, and matched initialization.

## Prospective checkpoint selection

Every epoch checkpoint is preserved. After all five training epochs, execute
the predicted top-1 mask from each checkpoint on every frozen validation
record using deterministic route-conditioned generation and the existing
benchmark-native scorers.

Rank epochs lexicographically by:

1. highest pooled executed validation accuracy;
2. lowest mean VISUAL_ON layers;
3. lowest cached validation exact-set NLL;
4. earliest epoch.

Cached valid-set Hit@1 remains diagnostic only. External/test outcomes are not
available during checkpoint selection.

## External evaluation

Run the selected checkpoint for each cap on the unchanged 22,307-record active
external suite used by the prior full10 and cap experiments. Use current live
ALL-ON as the scientific baseline. Report benchmark accuracy, ratio, Harm,
Rescue, mean ON, ALL-ON/OFF rates, and unique masks. Do not claim latency
improvement without wall-clock sparse-execution measurement.

## Execution

Run CAP26 and CAP24 concurrently, one GPU each on node02. Each pipeline is:

```text
runtime preflight
-> five-epoch exact-set-NLL training
-> five-checkpoint executed validation
-> accuracy-first checkpoint freeze
-> external preflight
-> frozen 22,307-record external evaluation
-> deterministic summary
```

Stop on integrity failure, nonfinite training, failed parity, incomplete
validation execution, or external serialization failure. Do not select another
experiment after completion.
