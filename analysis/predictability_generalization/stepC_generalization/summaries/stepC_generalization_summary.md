# Predictability Step-C generalization summary

Contract: `fafd6442871cdfde3d8cfc19667e2b1b1350b7a436d90000a953e9fddc007ecd`  
Parent Step-B: `792cc760bb41ce8c9b11776f21f43a88e54409ef1543471709000f83ec21bba4`

## Required answers

1. **Encoder.** Frozen `Qwen/Qwen3-Embedding-0.6B` snapshot `c54f2e6e80b2d7b7de06f51cec4959f6b3e03418`, raw Transformers, one shared symmetric instruction, last-token pooling, float32 L2-normalized embeddings.
2. **Nearest similarity.** Mean 0.8830, median 0.8915, P10/P90 0.7494/0.9998 over 10,399 OOF UIDs.
3. **Stage-1 Q1→Q5.** M3 AUROC 0.7828 → 0.8012 (Δ +0.0183; 95% image-group bootstrap CI [-0.0050, +0.0423]).
4. **Question-kNN Stage-1.** Q1 AUROC 0.6008; all quintiles are in `similarity_analysis/stage1_similarity_metrics.csv`.
5. **Least-similar advantage.** M3 Q1 0.7828, kNN 0.6008, nuisance 0.8282; hidden-state advantage is -0.0454 AUROC.
6. **READ similarity.** M3 Q1/Q5 Spearman 0.0144/-0.0126.
7. **WRITE similarity.** M3 Q1/Q5 Spearman 0.0892/0.0373.
8. **High-similarity Stage-2 concentration.** The fixed Q1/Q5 contrasts above provide the direct answer; no post-hoc bin was selected.
9. **K=100 cluster OOD Stage-1.** Concatenated five-fold M3 AUROC 0.7725 (fold mean 0.7735).
10. **K=100 cluster OOD Stage-2.** Concatenated READ/WRITE M3 Spearman 0.0453/0.0164 (fold means 0.0512/0.0300).
11. **Historical→Canonical Stage-1.** M3 AUROC 0.4342.
12. **Canonical→Historical Stage-1.** M3 AUROC 0.5563.
13. **Within-dataset source transfer.** All prospectively supported directions are in `source_transfer/within_dataset_source_transfer.csv`; unsupported cells remain explicit in `splits/support_audit.csv`.
14. **Stage-1 LODO.** chartqa=0.6043, gqa=0.5508, textvqa=0.6006.
15. **READ/WRITE LODO.** Full Spearman/harmful metrics are in the two `dataset_lodo/lodo_stage2_*.csv` tables.
16. **High-C preservation OOD.** Train-calibrated 95/98% operating points are recorded beside every supported Stage-1 regime; calibration drift was not repaired with test labels.
17. **Hidden state versus nuisance.** Per-regime M3−M0 gaps are frozen in `controls/nuisance_generalization.csv`.
18. **Does question-kNN explain transfer?** Exact k=5 label-neighbor controls for every supported regime are in `controls/question_knn_generalization.csv`; M3 is compared without tuning k.
19. **Stage-2 niche.** Largest absolute supported OOD M3 correlation was `dataset_chartqa_to_textvqa` write ρ=0.0968; this is descriptive, not a selected global result.
20. **Stage-1 ladder.** ID 0.7869 → Q1 0.7828 → cluster 0.7735 → source min 0.4342 → LODO min 0.5508.
21. **Stage-2 ladder.** ID READ/WRITE 0.0416/0.0347 → cluster 0.0512/0.0300; source/LODO rows are in the primary tables.
22. **Stage-1 interpretation.** `S1-C source-specific signal`: Q1≈Q5 and cluster OOD≈ID, while bidirectional source transfer and every LODO regime collapse. The prior automatic S1-B label was a reporting-rule defect and is superseded by the plan-defined taxonomy.
23. **Stage-1/Stage-2 asymmetry.** Stage-1 remains materially more learnable than local READ/WRITE utility unless the detailed tables show a narrow Stage-2 exception; the two metric scales are not directly interchangeable.
24. **Not established.** Step C does not establish external transfer, deployment gain, causal mechanism, richer-history Stage-2 feasibility, or on-policy rescue.

## Validity

- Targets, model families, optimization, and three M3 seeds are inherited unchanged from Step B.
- Semantic clustering and role balancing are label-blind and image-group-disjoint.
- OOD preprocessing and calibration use training-side rows only.
- Results are descriptive evidence, not authorization for Step D or a target redesign.

Final audit patch: `89580e7ff35b867f97eefeb7aec2b149299c77e95f95854e33dcdf116d6f5ed9`. No model was refit and no target or metric was changed.
