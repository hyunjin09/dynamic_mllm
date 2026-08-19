# Stage C Outcome-Blind Entry Gate Report

Status: **PASS — Stage C entry gate satisfied on 2026-08-05.** This report
authorizes no experiment by itself. The full Stage C READ/WRITE sweep was not
run, and no held-out primary likelihood effect was computed, aggregated,
inspected, or reported.

## Gate Results

| Entry criterion | Status | Evidence |
|---|---|---|
| Frozen endpoint remains TextVQA, layer 0, per-token `FULL - WRITE_ONLY` | PASS | `workspace/research_plan.md`; `workspace/stage_c_reference_likelihood_proposal.md` |
| Official held-out source and revision pinned | PASS | `configs/stage_c_entry.yaml`; source fingerprint `475bf9de899d571b` |
| Only predeclared technical-invalid rules applied | PASS | 9/5,000 exclusions: 4 empty-after-normalization, 5 over context limit; no new rule |
| Exactly 800 records and 800 unique images frozen | PASS | `outputs/stage_c/manifest/stage_c_manifest_v1.jsonl` |
| No Stage B record/effective-image overlap | PASS | zero record/question/annotation, image-ID, canonical image-hash, image-path, and image-question overlap |
| No duplicate Stage C image | PASS | 800 unique images for 800 rows |
| Manifest and per-record checksums reproduce | PASS | manifest SHA-256 `e3e9e08329fa626bc75706fba6623357f9ca05140bae1f138c98b9cd26e45357`; 800/800 record hashes exact |
| Accepted-answer normalization/weights frozen | PASS | positive duplicate-aggregated annotation-frequency weights sum to one within `1e-9` |
| Prompt, prefix, span, EOS, tokenizer, processor, and template frozen | PASS | `workspace/stage_c_scoring_spec.md` |
| Covariance/subspace null fitted without Stage C outcomes | PASS | 200 Stage B TextVQA records; rank 66; explained variance `0.9017906189` |
| Covariance rank stability | PASS | leave-one-effective-image-out rank range 65–66, median 65 |
| Real-residual donor family frozen and covered | PASS | common max-ratio cap 1.5; 8 donors; Stage B coverage min/median/max 15/186/197; zero failures |
| Structured-null comparison and multiplicity frozen | PASS | paired real-minus-null-mean effects, 10,000 image bootstrap draws, both upper 95% CI endpoints below zero required |
| Unmodified/FULL parity | PASS | maximum prompt-logit difference 0 on both smoke records; tolerance `1e-4` |
| READ reconstruction through suffix | PASS | maximum prompt-logit difference 0 on both smoke records; tolerance `1e-4` |
| WRITE_ONLY execution | PASS | finite unchanged-suffix logits on both smoke records |
| Accepted spans and sequence/image lengths | PASS | processor repeat exact; manifest lengths match; accepted spans nonempty on both smoke records |
| Covariance null shape, native subspace, and norm | PASS | shape `[25,3584]`; native error at most `4.02e-7`; norm error at most `5.05e-7` |
| Real-residual null shape and norm | PASS | shape `[25,3584]`; norm error at most `6.49e-6` |
| Same-sample/image donor exclusion | PASS | zero violations in smoke; enforced in selection code and unit tests |
| Deterministic seeds and serialization | PASS | repeated covariance draws bit-identical; all five frozen artifact checksums verify |
| Held-out primary endpoint remained closed | PASS | both smoke rows and audit record `primary_endpoint_computed: false` |

## First-Failure Record and Bounded Repair

The first smoke attempt is preserved at
`outputs/stage_c/nulls/null_calibration_and_smoke_v1_first_failed.json`. It
passed the causal-path checks but failed two provisional null checks: a
subspace test was performed after lossy row interpolation, and independent
`1.5/1.25/1.25` caps left two calibration targets with fewer than eight
donors. One Stage-B-only geometry diagnostic showed that widening norm alone
through 3.0 could not fix the remaining outlier.

The second attempt changed only the validation location and
outcome-blind matching rule. Native-grid membership is now checked before
interpolation; a composite max-ratio cap is fitted as the worst Stage B
eighth-nearest eligible-donor distance. The frozen fitted value is 1.5. The
second attempt passed; no threshold was relaxed based on a Stage C outcome.

A subsequent code review found that 17 donors without `source_asset_id` had
been conservatively grouped under an empty image ID. The fallback now uses the
already-frozen Stage B `selection_asset_key`; the final audit correctly has 200
unique calibration images. The same bounded smoke was rerun and remained a
pass. The pre-review passing audit is preserved at
`outputs/stage_c/nulls/null_calibration_and_smoke_v1_pre_review_image_id_fix.json`.

## Frozen Artifacts

- Eligibility/overlap audit:
  `outputs/stage_c/manifest/stage_c_eligibility_overlap_audit_v1.json`
- Manifest and checksum:
  `outputs/stage_c/manifest/stage_c_manifest_v1.jsonl`,
  `outputs/stage_c/manifest/stage_c_manifest_v1.sha256`
- Calibration residuals:
  `outputs/stage_c/nulls/stage_b_read_residuals_v1.pt`
- Covariance parameters:
  `outputs/stage_c/nulls/covariance_subspace_parameters_v1.pt`
- Real-residual donor metadata:
  `outputs/stage_c/nulls/real_residual_donor_index_v1.jsonl`
- Per-record deterministic seeds:
  `outputs/stage_c/nulls/deterministic_null_seeds_v1.jsonl`
- Final machine-readable smoke audit:
  `outputs/stage_c/nulls/null_calibration_and_smoke_v1.json`
- Artifact checksums: `outputs/stage_c/nulls/null_artifacts_v1.sha256`

## Unresolved Risks and Fail-Closed Conditions

- Calibration contains 200 Stage B questions from 200 effective images. The
  rank is stable to effective-image deletion, but this remains a finite
  discovery-derived geometry estimate.
- A held-out target outside the frozen 1.5 matching cap has no permitted donor
  fallback and must stop Stage C rather than trigger caliper adaptation.
- Linear row mapping preserves endpoints and exact target norm but not literal
  membership in the native 32-row affine subspace after round-trip mapping;
  that mapping deviation is logged as a descriptor, not misclassified as a
  sampling failure.
- Stage B's heavy-tailed likelihood distribution remains a power and
  interpretation risk; it does not change the frozen n=800 manifest.

None of these risks opens a new endpoint, exclusion rule, or tuning choice.
The next bounded action is the explicitly authorized frozen Stage C execution,
not Stage D and not any new search.
