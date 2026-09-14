# Canonical Stage-1 Refit Diagnostic Protocol

- Frozen contract: `c1a0420a261f96b80a2cbd432d778d7a7caaf942cdf8e752cadf314c1670c826`
- Arm A only: exact historical Shared Random-4 architecture and exact old global normalization.
- Target: frozen current-runtime LMMS dense-wrong label; wrong is positive.
- Five deterministic dataset/label-stratified image-group-disjoint outer folds.
- Each outer-training partition alone supplies its group-disjoint 12.5% internal validation subset.
- Training epochs are C/W balanced by using every minority row and an equal no-replacement majority sample; held-out prevalence remains natural.
- Ten FP32 AdamW epochs, lr 5e-4, weight decay 0.01, epoch-wise cosine schedule, four unique random layers per selected sample, minimum full-28 validation BCE checkpoint.
- Primary canonical evidence is the concatenated 4,000-row OOF max-trajectory score. No threshold is selected.
- The full-canonical model is fit only for historical validation/test cross-evaluation and is not used for canonical claims.
- Arm B, mixed-source training, Stage-2 work, corrective search, and trigger-map regeneration are prohibited.
