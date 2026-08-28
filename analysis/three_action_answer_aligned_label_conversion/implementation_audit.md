# Three-Action Answer-Aligned Label Conversion Implementation Audit

Date: 2026-08-25 (Asia/Seoul)

Approved specification: `plans/4way_labeling_fix.md`

Git commit at audit: `a3c6a41115490992b4f0cebb40d7e67d857c9286` (dirty research worktree; exact loaded hashes below are authoritative)

## Decision and source boundary

The modified conversion reuses the already-frozen authoritative positive binary-route inventory. It does not run MCTS, add negative routes, use `datasets/mcts_v2`, or overwrite prior labels.

| Dataset | Positive samples | Positive binary routes |
|---|---:|---:|
| GQA | 3,386 | 132,127 |
| TextVQA | 1,746 | 58,789 |
| ChartQA | 1,785 | 46,886 |
| WeMath2.0 Standard | 3,095 | 200,058 |
| WeMath2.0 Pro | 2,266 | 107,671 |
| **Total** | **12,278** | **545,531** |

Source manifest: `datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl`, SHA-256 `a44ca6e8684bc1a559997ce0ea52b2796f3265d19be90e22439c653741f36ed7`.

The known Standard limitation is unchanged: 5,381/5,843 source rows have transferred same-contract terminal records. The absent 462 terminal rows have no existing binary label on this server and are disclosed rather than inferred or regenerated. The 3,095 available positive Standard records and all 200,058 of their declared positive routes are included.

## Action and scoring contract

The implementation uses only the validated unified executor:

| Label action | READ | WRITE | Executor action |
|---|---:|---:|---|
| FULL reference | 1 | 1 | `FULL` |
| READ_OFF | 0 | 1 | `WRITE_ONLY` |
| WRITE_OFF | 1 | 0 | `READ_ONLY` |
| BOTH_OFF | 0 | 0 | `IGNORE` |

FULL is a cached screening/reference state and is never blindly expanded as a fourth decomposition action. For a retained layer, source BOTH_OFF and route-conditioned FULL are already cached; the decomposition normally adds only READ_OFF and WRITE_OFF forwards.

W2C uses the fixed current-FULL-wrong target margin `S(correct)-S(original FULL wrong)`. C2C uses normalized correct-answer support `S_correct` with the evaluator-compatible target policy already validated for GQA, TextVQA, ChartQA, and math. Source route replay and current unified FULL determine W2C/C2C under the current contract.

Numerical epsilon is frozen before conversion as:

`max(1e-6 predeclared mean-score floor, empirical p99 absolute difference from repeated identical unified routes)`.

Native-vs-unified drift is not part of this threshold. Calibration executes identical routes uncached three times and retains every raw signed difference, generation, correctness, and score record.

## Conversion logic

- Screen each binary OFF/BOTH_OFF position in its current correcting-route context.
- W2C retains `HARD_NECESSARY` positions when FULL restoration makes the route wrong and `SOFT_ALIGNMENT_HELPFUL` positions when BOTH_OFF improves margin by more than epsilon; answer-alignment-redundant positions are restored to FULL.
- C2C retains support-gain and context-dependent screening candidates, but admits a positive training route only when it remains evaluator-correct and globally improves `S_correct` over unified FULL by more than epsilon.
- Screening repeats to a deterministic correctness-preserving fixed point after restorations.
- Joint coordinate beam explores only READ_OFF/WRITE_OFF/BOTH_OFF over retained positions. Beam width 8 keeps both a correct low-cost state and a high-margin state, allowing wrong-but-margin-improved W2C states to remain search evidence.
- Wrong partial W2C states are stored separately and never enter positive training supervision.
- Every positive route is executed as a complete trajectory, deduplicated by full 28-layer action sequence, and retains all source-route IDs.
- The independently best locally supported actions are also composed and executed as one complete route. Any local-to-joint failure is retained as a control and never admitted as a positive label.
- The pilot compares beam 8 with beam 16; canonical-route equality and minimum positive-set Jaccard 0.50 are prospective full-launch gates.
- Canonical W2C selection prefers minimum suppression cost within epsilon of the best source-seed margin. If cumulative individually tolerated restorations leave no refined route in that band, the maximum-margin correct refined route is retained and explicitly marked as a fallback.

## Output isolation and compatibility repair

New root: `datasets/mcts_labels_3action/conversion_v1/`. Old four-action outputs remain immutable provenance.

Pending old-semantics job 1605 was canceled at zero elapsed time and zero records. Job 1604's 87 pre-image-fix records remain archived only under the old four-action root.

The server-specific oversized-image repair remains active: `label_regeneration/runtime.py` retries `Image.DecompressionBombError` only after the frozen file matches its content SHA-256, disables the Pillow limit only for that open, and immediately restores it. This preserves exact native geometry for the 16 oversized images affecting 26 positive Standard samples.

## Frozen implementation hashes

| Artifact | SHA-256 |
|---|---|
| `configs/three_action_label_conversion.yaml` | `f586ed249c1ccd4f8641d79726529353f99248bbcf7c4dda98e33e8de6a890c7` |
| `label_regeneration/runtime.py` | `1739b9d0f696ee3da3f601849f54c4d3f2077aa3cb72dea544f11b2ff796f201` |
| `tools/research_analysis/four_action/label_runtime.py` | `785163efe775e7488ffccbe92dee41b7f84b464fb0d61e97f7240859132305b1` |
| `tools/research_analysis/four_action/label_jobs.py` | `a11fddcb62518619f345527b39d7439e77703449eff1744347e1ee11e020e16e` |
| `tools/research_analysis/four_action/three_action_labels.py` | `cc79a27e8cb4b3e39d5aa396334e1eac6c1ac7f2ce141e2b0802bd3dc3867e97` |
| `tools/research_analysis/four_action/three_action_jobs.py` | `398ecce48a40883cc9aab0bd157e1d40b65262d25f2a47c591dfe0b1965205e7` |
| `experiments/run_three_action_label_conversion.py` | `8eff94a73aa387a01454df08edab8ccb6037eca2a9198822362d61b10df38e7a` |
| `experiments/finalize_three_action_noise_calibration.py` | `37cc9216bcfd5fba66a60b1bb2fe1766043875deec098e419d72080332725fb4` |
| `experiments/audit_three_action_label_pilot.py` | `6aa7d1c49ff28ea2649863591871818eea5aae6b0d8e3651d13cca3d8dba8b78` |
| `experiments/estimate_three_action_label_conversion.py` | `f15b5723134c83d1971c2facd7ece1ef6702460ab26f88ecb226319b0b60320c` |
| `experiments/run_three_action_calibration_pilot.sh` | `3dd215e35ba259d45be9991224e164c3f1094b7a32843062364af6f6310833e7` |
| `experiments/run_three_action_full.sh` | `47bc2ddd1cf92b16af92edff69d4a38b6500e7911c13ebaf865e27dd6c7c215e` |

Calibration execution-contract SHA-256: `2cbee2bab1da18ee3be8eb46cbd99f61253e063d298b466c4d04acc829c386f1`.

## Validation and pilot freeze

- New focused tests: 24/24 pass.
- Complete active project suite: 404/404 passes.
- Pilot manifest: 56 samples, 4,026 source routes, all five datasets, SHA-256 `890ddbf933396267251d5023e04828b91f80ecc36c6fb80147478970aeb6dfc9`.
- Pilot proxies: 26 source-W2C, 30 source-C2C, 17 ALL-OFF W2C, and 49 multi-route samples. Current hard/soft/C2C alignment paths remain live pilot gates rather than inferred from historical labels.

## Environment and live server

- Project-local uv environment: Python 3.12.7, PyTorch `2.6.0+cu124`, Transformers `5.3.0`.
- Model: Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16 SDPA.
- Server: one Slurm-visible host with 192 CPUs, about 1.78 TiB RAM, and eight H100 80GB GPUs.
- Required datasets, images, model snapshot, and external symlinks are the same verified assets as the frozen source inventory; no download is required.
- At the prelaunch audit, another user's exclusive eight-GPU job 1600 occupied the server. The calibration/pilot allocation may remain pending and must not interfere with it.
