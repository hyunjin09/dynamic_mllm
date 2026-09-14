# READ versus WRITE: completed bounded diagnostics

Both interventions can rescue and damage answers. Neither tested local feature family nor the tested short-horizon predictors provides a useful harmful-action selector. WRITE has modest adjacent structure and a verified delayed effect on text, so the causal pathways differ even though both learnability gates fail.

| Question | READ | WRITE |
|---|---|---|
| Controlled intervention | FULL versus WRITE_ONLY; WRITE held ON | FULL versus READ_ONLY; READ held ON |
| Frozen continuous target | HR=q_WRITE_ONLY−q_FULL | HW=q_READ_ONLY−q_FULL |
| Complete dense population | 15,185 states / 1,413 UIDs / 1,385 image groups | Same exact population |
| Harmful / beneficial / zero signs | 7,285 / 7,900 / 0 | 6,501 / 7,271 / 1,413 |
| Harmful / beneficial correctness flips | 625 / 41 | 448 / 40 |
| Harmful adjacency | 0.5174, below shuffled 97.5th percentile 0.5185 | 0.4975 versus shuffled mean 0.4561, 95% interval [0.4473,0.4648] |
| Harmful spans | Mean 1.896, maximum 13; R-STRUCT-B under the READ compound gate | Mean 1.990, median 1, maximum 13; modest within-UID organization |
| Matched local mechanism | Only 1/34 feature intervals excludes zero; matched probe AUROC 0.4186 | 0/85 intervals excludes zero; logistic/MLP AUROC 0.4215/0.4132, only 22 matched pairs |
| Full local F_ALL/MLP Spearman / harmful AUROC | 0.0697 / 0.5338 | 0.0475 / 0.4819 |
| Generic prestate baseline | 0.0416 / 0.5193 | 0.0359 / 0.4984 after exact sign and denominator alignment |
| One-step delta baseline | 0.0773 / 0.5321 | 0.0380 / 0.4897 after alignment |
| Local harmful-flip AUROC | 0.3484 | 0.3911 |
| Propagation common support | 6,916 states / 872 UIDs | 6,044 states / 772 UIDs |
| Immediate → longest common-support delta Spearman | 0.0756 → 0.1097 | 0.0412 → 0.0973 |
| Immediate → longest harmful AUROC | 0.5316 → 0.5483 | 0.5162 → 0.5387 |
| Longest Dense-W delta Spearman / AUROC | 0.1142 / 0.5504 | 0.1038 / 0.5443 |
| Useful high-precision subset | No | No |
| Propagation classification | H-READ-D | Qualified W-PROP-C / WRITE-S4 |
| Source/task transfer | Measured and weak; routed results secondary | Not run because the positive dense gate failed; transfer remains unknown |

The table compares completed studies descriptively; it is not a paired significance test of READ versus WRITE. The READ compound structural gate was not retrospectively imposed on WRITE. Different marginal signs and the terminal zero states also make raw adjacency and AUROC insufficient for an operation ranking. Every WRITE zero occurs at layer 27, where no later layer can read the changed visual state. Filtering the existing WRITE OOF predictions to the 13,772 nonzero states gives Spearman 0.0425 and AUROC 0.5219 without refitting; local usefulness still fails.

Horizon labels differ. READ H1 is immediately after intervention and READ H8 includes seven subsequent FULL layers (eligible l≤20). WRITE H0 is immediate and WRITE H8 includes eight subsequent FULL layers (l≤19). Each curve uses its own exact longest-horizon support. No extrapolation from these subsets to all dense states, or direct H8 superiority claim, is warranted. The comparison figure plots the actual number of subsequent layers to expose this difference.

WRITE text equality holds at H0; every one of its 6,044 common-support states has nonzero control-text divergence at H1. Absolute median visual difference grows from 612.48 to 1,272.82 by H8, while mean relative difference changes from 0.2642 to 0.2456. This supports delayed transmission of the intervention through subsequent computation, but does not identify a specific mediator or explain which updates harm answers. Absolute growth is not relative amplification. READ's reported 2.268× growth is a pooled norm ratio, so it is not directly comparable to WRITE's all-visual norm.

WRITE's H8−H0 Spearman gain is +0.0561, with image-group bootstrap 95% CI [0.0267,0.0852]. The AUROC gain is +0.0226 [0.0051,0.0396]. These gains survive common support and Dense-W refits, but fall below the prospective +0.10 Spearman / +0.08 AUROC **improvement** thresholds. READ's smaller gains have intervals crossing zero. Neither result supports a useful critic. READ additionally tested a token comparator, random pairs, and permuted targets; WRITE's propagation conclusion covers the fixed pooled input/predictor family, not identical architectural coverage.

The continuous utility signs are not correctness labels: 84 of WRITE's 448 harmful correctness flips have negative HW. Ranking the continuous target cannot be assumed to rank correctness rescue. This observation does not establish why the predictors are weak.

The justified common conclusion is to close the tested local/short-horizon routing hypothesis. Intrinsic nonlocality, longer-horizon usefulness, richer representations, and joint trajectory-level action selection remain unknown. The stopped Phase85 READ search remains PRELIMINARY and does not establish saturation. No WRITE search or joint policy was tested.

Evidence: [WRITE structure](write_structure_summary.md), [WRITE mechanism](write_mechanism_summary.md), [WRITE learnability](write_learnability_summary.md), [WRITE propagation](write_propagation_summary.md), [READ structure](../../read_harm_structure_learnability/summaries/read_harm_structure_summary.md), [READ learnability](../../read_harm_structure_learnability/summaries/read_harm_learnability_summary.md), and [READ propagation](../../read_harm_short_horizon_propagation/summaries/read_short_horizon_propagation_summary.md).
