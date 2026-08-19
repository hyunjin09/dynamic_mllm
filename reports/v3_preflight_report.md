# v3 Held-Out Confirmation Preflight Report

## Decision

The candidate data split and real/null action mechanics are feasible, but the
entry gate is not satisfied. The complete joint READ/WRITE real-donor
caliper/index, null draw count, joint covariance smoke, and exact grounding
control are not yet frozen. Separately, same-image visual WRITE is structurally
query-invariant but not bitwise invariant under unequal stock-BF16 prompt
shapes; equal right-padding repairs the numerical check exactly, but adopting
that Stage C2 rule requires approval.

No new held-out terminal intervention outcome was computed, loaded, aggregated,
or interpreted.

## Frozen/proposed confirmation design

- Primary statistic: per-token accepted-reference
  `max_{l,a!=FULL}[Q_l(a)-Q_l(FULL)]`.
- Layer/action search: layers `[0,4,8,12,16,20,24]`; actions `IGNORE`,
  `READ_ONLY`, `WRITE_ONLY`; 21 cells. Layer 27 is excluded only for structural
  WRITE silence.
- Proposed sample size: 1,600 unique images, balanced 800 GQA/800 TextVQA;
  equal-weight joint primary and dataset-specific secondary results.
- Manifest rule: ascending `SHA256(2026080602:record_id)`, first technically
  valid record per unique image after all inspected image/record/hash
  exclusions. TextVQA uses remaining singleton images.
- Numerical epsilon: `1e-6 nats/token`; practical threshold: `0.05`.
- Null hierarchy: isotropic exact-norm, joint covariance/subspace, and
  same-layer paired real residual; every replicate receives the same 21-cell
  maximum.
- Inference: 10,000-draw stratified image bootstrap. Primary mean CI must be
  above zero; all three paired real-minus-null mean CIs must be above zero;
  robust median, trimmed-mean, prevalence, answer-content, and grounding gates
  also apply.

The final manifest is not frozen because the null entry gate is incomplete.

## Entry criteria

| Criterion | Status | Evidence |
|---|---|---|
| No inspected record/image overlap | PASS | `outputs/v3_preflight/candidate_pool_audit.json` |
| 800 unique images per dataset available | PASS | 10,234 GQA and 2,362 TextVQA images remain |
| TextVQA Stage B ambiguous IDs resolved by canonical hash | PASS | zero Stage A/B canonical-hash overlap; 800 v2 Stage C images excluded |
| Separate Stage C2 reserve | PASS | 800 image groups per dataset, disjoint from proposed Stage C and inspected data |
| Complete real four-action mechanics | PASS | exact FULL parity and finite four-action smoke on GQA/TextVQA at layers 0, 12, 24 |
| Isotropic/covariance/real null insertion and suffix | PARTIAL PASS | all three action analogues finite with norm error below `1e-5`; joint covariance revision not yet smoked |
| Search-budget equality | SPECIFIED | seven layers x three actions for real and every null replicate |
| Exact real-donor caliper/index and coverage | FAIL | joint seven-layer READ/WRITE geometry has not been extracted; arbitrary `2.0` cap rejected |
| Null draw count/seed stability | FAIL | one draw is provisional; full-max seed contribution is unmeasured |
| Exact answer-content/grounding control | FAIL | the candidate GQA answer-margin control is useful but is not a sufficient visual-grounding gate by itself |
| Structural query-invariance premise | PASS | token order, mask, and pre-WRITE K/V graph verified |
| Numerical query invariance under existing unpadded execution | FAIL | unequal lengths diverge from layer 0 and accumulate |
| Fixed-shape diagnostic | PASS, amendment required | right-padding 273 to 281 restores exact equality at all seven layers |
| Accepted-answer scoring path | PASS | finite GQA/TextVQA scoring on inspected calibration smoke |
| Serialization/checksums | PASS | all required JSON/Markdown artifacts written; hashes recorded below |
| Held-out endpoint remained closed | PASS | no held-out terminal Q or null effect was loaded or computed |

## Reviewer reconciliation

The independent review ranked 1,600 images with one null draw above the
800-image/multi-draw alternative only after a calibration seed-stability check.
It rejected the unfitted constant donor caliper and judged a proposed
same-question/different-answer GQA control useful but insufficient as the sole
grounding gate. The final preflight adopts those objections: sample size and
full grid remain proposed, while draw count, caliper/index, joint covariance
smoke, and grounding design stay explicitly unresolved.

## Artifact integrity

- Candidate audit SHA-256:
  `329d02a2aa0938c91818cdd7924fe24efd40532d9dbeae0e6307ccedf090d327`
- Stage C2 reserve audit SHA-256:
  `a6d043da45704de7e4e1140f7831e418b4f296e1c5dd396666c3190f767a68c4`
- Null smoke SHA-256:
  `2a46cf96d2698ad75b1ecb4474af8d482429fc19631627847f7bfb3e1c33fe7c`
- Equal-length diagnostic SHA-256:
  `c83fca1386621fa0e839c859af74494a3086943151abd9154ae2a4063ff60df5`

## Smallest protocol-preserving repair

With explicit approval, run one calibration-only geometry action that extracts
joint READ/WRITE residuals for the 400 inspected Stage B records over the seven
frozen layers, fits and smokes the joint covariance model, measures seed
stability, freezes the minimum supported donor count and exact calibration
calipers, and freezes a sufficient answer-content/visual-grounding control,
then performs outcome-blind donor coverage on the deterministic 1,600-record
manifest. This action must stop before any terminal held-out score. Separately
approve fixed common-length right-padding for future Stage C2 pairs and
preflight its score/parity behavior.

REPAIR_V3_NULL_DESIGN
