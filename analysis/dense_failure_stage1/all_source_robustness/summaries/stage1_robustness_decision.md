# Stage-1 Robustness Decision

## B — Source-robust but benchmark-OOD limited

- ALL Historical held-out AUROC: 0.8394
- ALL Canonical OOF AUROC: 0.7689
- Worst-source AUROC: 0.7689
- LODO worst-source AUROCs: ChartQA 0.5691, TextVQA 0.6243, GQA 0.5931

Frozen rule outcome: source robust = `true`; broad OOD transfer = `false`.

A separately authorized next phase may calibrate an in-scope ALL-mixture threshold, while explicitly avoiding a benchmark-universal claim.

This phase stops here. No threshold, trigger map, Stage-2 artifact, corrective search, or deployment evaluation was produced.
