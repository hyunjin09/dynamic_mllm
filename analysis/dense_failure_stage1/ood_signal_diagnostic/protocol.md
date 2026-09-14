# OOD Dense-Failure Signal Protocol

- Frozen contract: `dfef70e6f802bef02320b034aad995dd6384bf73a71d31e7a747731ce5c64fff`
- Current population: `7,999` rows; labels are current native-dense LMMS correctness only.
- Source train and validation reuse Phase-48 split membership restricted to the two source datasets.
- OOD target is every row of the held-out dataset and is not used for training, checkpointing, threshold calibration, probability calibration, or layer selection.
- Twenty-eight independent linear probes use the exact Phase-48 form and optimization settings.
- Representative layer is the maximum source-validation AUROC with a lower-layer tie break, frozen before OOD scoring.
- 99%, 98%, and 95% preservation thresholds are calibrated on source validation and transferred unchanged.
- The in-domain/OOD comparison uses identical Phase-48 target-test UIDs. Each model uses its own permitted validation-calibrated 99% threshold.
- Fixed interpretation heuristic: meaningful transfer requires source-selected OOD AUROC >= `0.65` on all targets; decoder-added evidence requires at least `2` targets with source-selected hidden-minus-input AUROC >= `0.02` and no target below AUROC `0.55`.

## Native pre-language-decoder control

The control captures the input to language-decoder layer 0 after native token embedding, the learned visual encoder, and multimodal insertion. It pools the same user-final, user-mean, and visual-mean token positions as the hidden-state probe and concatenates them. A private sentinel stops execution inside the layer-0 pre-hook; the smoke requires one capture and zero decoder forward-hook firings.

This is not a raw-pixel/raw-text baseline. It tests information added beyond Qwen's native pre-language-decoder representation.

## Leakage audit

- Cross-dataset image groups: `0`
- Cross-dataset records: `0`
- Passed: `True`

### leave_textvqa_out

- Source train: `4,799`
- Source validation: `600`
- Full OOD target: `2,000`
- Identical-UID in-domain comparison subset: `200`

### leave_chartqa_out

- Source train: `4,800`
- Source validation: `600`
- Full OOD target: `1,999`
- Identical-UID in-domain comparison subset: `200`

### leave_gqa_out

- Source train: `3,199`
- Source validation: `400`
- Full OOD target: `4,000`
- Identical-UID in-domain comparison subset: `400`
