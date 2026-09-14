# ALL-Source Training Summary

The ALL-source five-fold head ensemble achieved Historical held-out AUROC **0.8394** and Canonical OOF AUROC **0.7689**. Average-source AUROC is **0.8042** and worst-source AUROC is **0.7689**.

| Head | Historical held-out AUROC | Canonical held-out AUROC | Average source | Worst source |
|---|---:|---:|---:|---:|
| Historical specialist | 0.8885 | 0.4056 | 0.6470 | 0.4056 |
| Canonical specialist | 0.5511 | 0.8178 | 0.6845 | 0.5511 |
| ALL-source | 0.8394 | 0.7689 | 0.8042 | 0.7689 |

The weakest dataset×source cell is **canonical chartqa** at AUROC 0.5826 (C=884, W=116). Canonical ChartQA AUROC is 0.5826, so it remains non-inverted.

For canonical Dense-C, the historical-specialist mean/p95 maximum risk was 0.8056/1.0000; under ALL-source training it is 0.6080/0.9447. The five most source-robust layers by worst-source AUROC are L20 (0.796), L19 (0.793), L21 (0.792), L18 (0.790), L22 (0.788).

Conclusion: one shared Stage-1 boundary is viable across both observed source regimes for the aggregate in-scope mixture. No threshold was selected.
