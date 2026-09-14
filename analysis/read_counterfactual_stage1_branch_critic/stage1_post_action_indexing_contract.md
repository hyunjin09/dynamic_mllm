# Stage-1 post-action indexing contract

The original DenseFeatureCollector registers forward hooks on decoder layers 0..27. Its entry l is the raw residual OUTPUT of layer l, before final decoder RMSNorm; this is also the input of decoder l+1 when l<27. SharedFailurePredictor uses the matching embedding index l. Thus action l -> post-action state -> Stage-1 index l, NOT l+1. Layer 27 has an explicitly trained output representation and embedding 27: all 1,413 layer-27 states remain eligible; no synthetic index 28.

Feature order is [final user-text token; mean user-text tokens; mean visual tokens], BF16 pooling then concatenation and FP32 normalization with frozen historical mean/std. The final user token is located by the original prompt token-position helper; it is not the final assistant/control prompt token stored in the compact Stage2 cache. Raw states are before decoder final RMSNorm.

Frozen model is the robust ALL-source five-checkpoint probability ensemble. Each head receives the same representation and layer index. Convert each FP32 raw logit to sigmoid in FP64 and average five probabilities, exactly as the P90 trigger pipeline. No additional calibration, retraining or normalization fit. P90 tau=0.9061332901863008 (not the stricter P98 gate default). Store five raw logits, five probabilities, their probability mean and effective ensemble logit. Primary AUROC uses mean probability.

Timing limitation: the frozen P90 trigger l was itself detected from Dense post-layer-l features, while frozen StepA labels intervene at pre-layer l. Preserve the plan's exact UID/action/label identities. This is a retrospective offline rollback comparison; it does not validate a causal forward-only controller. At first-trigger l, ON reproduces the original triggering score, so ON-safe quadrants should be absent there. No label/index shift is allowed to manufacture a next-state-safe transition.

Historical population admission uses strict >tau. This new plan explicitly defines quadrants with >=tau; retain admission and implement its quadrant rule, reporting exact tau ties. All other score ties select ON.
