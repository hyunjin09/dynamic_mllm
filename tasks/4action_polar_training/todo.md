# Four-Action Image+Question POLAR Readiness Tasks

## Task 1: Action/model/loss contract

- [x] RED tests reject malformed actions/routes and require `[B,28,4]` logits.
- [x] RED tests specify duplicated one-hot BCE and exact categorical set NLL.
- [x] Predictor, decoding, and core data contracts pass focused tests.

## Task 2: Predictor manifest

- [x] Audit all 6,917 unique GQA/ChartQA/TextVQA source rows and explicitly
      exclude the 106 rows with zero replay-valid supervision.
- [x] Preserve the resulting 5,945/866 group-disjoint train/validation split.
- [x] Include all 248,804 deduplicated valid routes without a cap.

## Task 3: Image+Question feature contract

- [x] Add a fresh, resumable and sharded feature extractor for the new manifest.
- [ ] Audit projected visual-row dtype, width, checksums, finite values, and
      complete UID/group coverage.
- [x] Make the trainer fail closed when the fresh cache is absent or mismatched.

## Task 4: Training and checkpoint selection

- [x] Match the frozen binary full10 optimizer and schedule settings.
- [x] Save predictor/optimizer/scheduler/config/metrics every epoch.
- [x] Validate all 866 records every epoch and select the best checkpoint using
      the prospective route-quality ordering.

## Task 5: External evaluation

- [x] Load exactly 14,960 rows from ChartQA, MMMU-Pro Standard/Vision, and POPE.
- [x] Predict and execute complete four-action routes through the unified
      executor with a paired live unified-FULL baseline.
- [x] Provide deterministic preflight, resumable shards, integrity merge, and
      per-suite reporting.

## Task 6: Readiness gate

- [x] Focused and existing regression tests pass.
- [x] Both frozen configs pass static/data/model preflight without training.
- [x] Final five-axis code review has no unresolved required code findings.
- [x] No GPU extraction, training, or external generation job has started.

The only intentionally open item is the runtime visual-cache audit. It requires
the separately authorized eight-GPU extraction and is the fail-closed boundary
between preparation and training.
