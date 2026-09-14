# Dataset-OOD Summary

| Train datasets | OOD target | Historical AUROC | Canonical AUROC | Average source | Worst source |
|---|---|---:|---:|---:|---:|
| GQA + TextVQA | ChartQA | 0.7016 | 0.5691 | 0.6353 | 0.5691 |
| GQA + ChartQA | TextVQA | 0.6243 | 0.6259 | 0.6251 | 0.6243 |
| ChartQA + TextVQA | GQA | 0.6034 | 0.5931 | 0.5982 | 0.5931 |

Against the historical-only Phase-49 references, Historical-target differences are ChartQA +0.2514, TextVQA -0.0993, and GQA -0.0126. This comparison is qualified because Phase 49 selected single independent-probe layers, whereas this phase evaluates Shared Random-4 maximum trajectories.

The least transferable target is **chartqa** by worst-source AUROC (0.5691). Canonical TextVQA has only 19 wrong examples; its point estimate and bootstrap interval in `lodo/textvqa/metrics.csv` remain high-uncertainty.

Interpretation: Stage-1 is not broadly benchmark-transferable under the frozen rule. Source diversity alone does not justify a universal failure-detector claim when any target/source cell remains weak.
