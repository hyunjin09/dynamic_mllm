# Phase 60: Stage-1 Historical Population Shortcut Audit Memory

## Current Objective
Reconstruct the historical Stage-1 population and determine, using frozen
artifacts only, whether the old Shared Random-4 head's apparent success depended
on the old selection/source regime or observable nuisance correlations.

## Active Constraints
- Follow `plans/stage1_historical_population_shortcut_audit_plan.md` and stop
  after retrospective diagnosis and a recommendation.
- Use only current LMMS Dense-C/W labels, existing hidden-state features, the
  frozen old scores/normalization, and existing source metadata.
- Do not retrain Stage 1, change its threshold/normalization, train Stage 2,
  rerun corrective search, or perform MLLM inference.
- Treat source/nuisance accessibility and matching as observational evidence,
  not causal intervention.

## Current State
- Done: Reconstructed all 7,999 executable historical identities, their fixed
  old Dense-C/Dense-W quota provenance, Phase-48 split, image SHA groups, source
  split, current label, frozen score trajectory, and observable metadata.
- Done: Compared the historical train/val/test populations with all 4,000 new
  canonical records, including categorical source/image-format distributions.
- Done: Ran group-held-out linear source/nuisance probes, metadata-only
  correctness baselines, exact/coarsened matching, frozen-normalization
  contrasts, and historical C-to-W feature geometry.
- Done/stopped: 30 output artifacts pass SHA-256 verification; no repair
  experiment or model inference ran.
- In progress: none.
- Blocked: none. Natural preselection rates cannot be reconstructed because the
  larger pre-quota candidate population and six raw quota JSONLs are absent;
  this limitation is explicitly recorded rather than inferred.
- Most recent useful observation: canonical correct ChartQA/TextVQA examples
  are highly source-separable and occupy the old failure side of the historical
  feature geometry, while observable nuisance matching reduces but does not
  eliminate old held-out ranking.

## Evidence That Matters
| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| Historical 8K was fixed at GQA 2K C/2K W and ChartQA/TextVQA 1K C/1K W using the previous Qwen dense outcome | `population/historical_construction.md`; `selection_bias_summary.csv` | The 50:50 label prevalence is selected, not natural | confirmed |
| Phase-48 split is 6,399/800/800 with zero UID/SHA-group overlap, but every split shares the same selection mechanism | `population/split_reconstruction.csv` | Old validation/test establish same-regime identity generalization, not source robustness | confirmed |
| Dense-C source probe at L21 is 0.586 GQA, 0.938 ChartQA, 0.937 TextVQA | `metrics/source_probe.csv` | Source is strongly encoded for ChartQA/TextVQA even after conditioning on correctness | confirmed |
| Frozen max-score canonical-minus-old-C mean is +0.089 GQA, +0.689 ChartQA, +0.712 TextVQA, with group-bootstrap intervals excluding zero | `metrics/old_head_score_by_population.csv` | Old head score aligns with source shift within the correct class, especially ChartQA/TextVQA | confirmed association |
| Metadata-only old-test AUROC is 0.673/0.788/0.630 and becomes 0.677/0.381/0.563 on canonical GQA/ChartQA/TextVQA; overall canonical ranking is 0.272 | `metrics/nuisance_only_correctness_probe.csv` | Historical selection created substantial shortcut opportunity and ChartQA/overall inversion resembles the head shift | confirmed |
| Old test max-score AUROC is 0.886 overall, 0.840 after exact token matching, and 0.820 after coarsened token/aspect/question matching | `metrics/matched_head_performance.csv` | Measured nuisances explain part, but not most, of same-regime ranking | confirmed observational |
| L21 new correct projection is 14.70 ChartQA and 4.88 TextVQA versus old C/W references -17.47/+17.47 and -7.49/+7.49 | `metrics/feature_geometry.csv` | New correct states move substantially into the historical wrong region | confirmed descriptive |
| Label/dataset/layer/block-conditioned canonical-vs-old-val normalized mean difference has median 0.063 and maximum 1.323 | `metrics/normalization_shift.csv` | Shift is concentrated rather than a uniform normalization-statistics failure | confirmed descriptive |

## Failed Attempts and Lessons
| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Interpret absolute distance from the pooled old normalization as source shift | Canonical and old validation maxima were both about 3.06 because depth dominates a normalization pooled over all layers | supported metric-design confound | first review pass; corrected `normalization_shift.csv` | Compare source cohorts at the same dataset, correctness, layer, and feature block | Do not use pooled absolute layer distance to claim canonical normalization shift |

## Open Candidates
| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Same head + frozen old normalization + canonical train labels | Directly tests whether the old fitted boundary/selection regime is the problem | Old weights/population versus feature limitation | medium | recommended, not authorized |
| Same head + canonical train-fold normalization | Useful only if the first arm fails | Normalization amplification versus deeper limitation | medium | contingent |
| Source-balanced old+new training | Could improve mixture robustness | Deployment-mixture coverage | medium | defer; confounds the first diagnosis |

## Next-Step Decision
- Deliberation mode: standard.
- Active objective and bottleneck: the retrospective audit is complete; the
  bottleneck is testing whether the unchanged head form can relearn canonical
  failure ranking.
- Relevant memory item used: Phase 49 already rejected a benchmark-general
  interpretation of the strong in-domain signal; Phase 59 exposed severe
  canonical-source score inversion under the frozen head.
- Confirmed observation: source identity and nuisance properties are encoded,
  old scores shift within Dense-C, and matching removes only part of historical
  ranking.
- Unverified interpretation: whether canonical-label fitting with unchanged
  normalization will recover held-out ranking.
- Diagnosis: supported source-sensitive old fitted geometry with historical
  selection shortcut opportunity; causal feature use and the dominant internal
  coordinate remain unresolved.
- Evidence path if diagnosis is not unknown:
  `analysis/dense_failure_stage1/historical_population_shortcut_audit/`.
- Viable alternatives considered: old-normalization canonical fit;
  canonical-normalization fit; source-balanced old+new fit.
- Chosen action: stop. Recommend the old-normalization canonical-label fit as
  the smallest next discriminating experiment, but do not execute it.
- Strongest objection: a successful refit would not by itself prove which
  nuisance/source coordinate the old head used, only that the representation
  remains learnable under the canonical regime.
- How this differs from failed attempts: it would isolate fitted-boundary
  transfer while retaining the architecture, target, and normalization.
- Automatic execution authorized: no.
- Authorization basis: the user authorized only this audit plan.
- Stop condition: reached.

## Latest Research-Action Result
- Action taken: completed the artifact-only population, score, probe, matching,
  normalization, and geometry audit under protocol
  `411a75307ac86efd97a701b3672ffadce693e4e8fa1cc221de622566c63299dd`.
- Result: evidence supports a mixture of genuine same-regime failure signal and
  source/nuisance-sensitive old decision geometry (Case B plus Case-D evidence),
  not visual-token count as a sole explanation.
- Evidence saved:
  `analysis/dense_failure_stage1/historical_population_shortcut_audit/`.
- Failure or issue: natural pre-quota selection rates are unavailable; one
  initially confounded normalization summary was corrected before acceptance.
- Lesson learned: the old 50:50 train/val/test regime made same-regime validation
  optimistic for canonical deployment; source shift is especially strong for
  correct ChartQA/TextVQA, while GQA behaves differently.
- Next implication: wait for explicit authorization before any canonical-label
  Stage-1 refit.
