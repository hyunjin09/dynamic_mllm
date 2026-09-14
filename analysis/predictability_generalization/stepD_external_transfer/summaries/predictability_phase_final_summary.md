# Predictability phase A-to-D final summary

- **Measurement (Step A):** controlled local READ/WRITE effects and correctness flips exist under complete four-branch intervention measurement.
- **In-domain learnability (Step B):** eventual Dense failure is learnable (M3 AUROC 0.7869), while immediate READ/WRITE utility is only weakly predictable (rho 0.0416/0.0347).
- **Internal generalization (Step C):** Stage 1 survives semantic distance/clusters but not source/dataset shift; Stage 2 remains weak.
- **External transfer (Step D):** Stage 1 is `D1-C` and Stage 2 is `D2-A` under zero external retuning across ChartQA, TextVQA, MMMU-Pro, and POPE.
- **Deployment:** none of these predictability/measurement results alone establishes positive routed generation performance.
