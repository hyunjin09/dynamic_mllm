# Stage-2 single-label distribution audit

Analysis contract: `32261c07880a0b38a96ce5f25c5af1c5147bb7408f085ea197b2819ad53f7bc5`. Source Phase-56 contract: `6489a0b39af3efb22bcb527d3b43c474dc7bcb01cc453445aab45f420eb9905b`. This audit used only train Corpus A+B and ran no model inference, search, training, validation/test labeling, or external evaluation.

## Answers to the audit questions

1. **Routes per W sample.** The 698 SINGLE_FIXABLE samples have 698 sample records and 7,628 replay-valid routes: mean **10.93**, median **7.0**, IQR **2.0–15.0**, P90/P95 **28.0/35.0**, range **1–60**.
2. **Localization.** Per sample, successful layers have median **4.0** (IQR 2.0–9.0); layer/action pairs have median **7.0**. Correction is not generally a single localized label.
3. **Most common action.** Route weighting favors **IGNORE** (0.416); the primary sample-weighted view also favors **IGNORE** (0.468). Exact alternatives are in the two action tables.
4. **Intervention depth.** Route mass is Early/Middle/Late **0.259/0.381/0.360**. Successful intervention is broad rather than confined to one layer.
5. **Delay.** Trigger-to-intervention delay has mean **12.62**, median **13.0**, and P90 **23.0** layers.
6. **Immediate.** 207/698 = **0.297** have at least one observed rescue at the trigger.
7. **Delayed-only.** 491/698 = **0.703** require a later observed single intervention. FULL must remain a normal post-trigger action.
8. **Same-layer multi-action ambiguity.** 2,685/4,421 = **0.607** successful sample/layer positions have multiple observed non-FULL rescue actions. These are observed sets, not exhaustive validity.
9. **Naive FULL imbalance.** Corpus B is **0.9617 FULL** (25.11:1); A+B is **0.9618 FULL**. Naive all-state CE is structurally collapse-prone.
10. **Route/sample weighting.** Route count spans 1–60; naive expansion therefore gives the most route-rich W sample **60x** the weight of the least route-rich sample before suffix-length effects.
11. **Dataset differences.** GQA/ChartQA/TextVQA breakdowns differ materially in sample count, route multiplicity, action mix, and delay; see `metrics/dataset_breakdown.csv`. This audit does not authorize dataset-specific heads.
12. **Trigger-depth differences.** Delay and immediate-fixable rates vary by L0/L1-8/L9-18/L19-27; see `metrics/trigger_depth_breakdown.csv`.
13. **Simple sampler.** `S3_SAMPLE_BALANCED_S1` gives every W sample equal expected route weight, keeps one mandatory corrective state, and retains up to two FULL timing examples on each side. Its expected FULL fraction is **0.785**, compared with 0.962 naively.
14. **Preservation visibility.** Natural sample frequency exposes C on only **0.053** of sample draws. A 1:2 C:W mix gives C one-third of draws and requires **8.95x** per-C repetition relative to each W sample.
15. **Structural suitability.** **Conditionally yes** for a simple shared V1 only with sample-balanced route sampling and explicit preservation mixing. The corpus is not suitable for naive all-route/all-state CE. Exact routed-state auditing found 90,609 duplicate semantic rows and 4,151 exact entering states with multiple observed route labels; this is valid alternative-route supervision, not a unique deterministic oracle action.

## Interpretation boundary

This establishes corpus structure and a defensible loader contract. It does not establish Stage-2 generalization, C preservation under free rollout, sufficiency of single interventions, or lack of value from Corpus C.
