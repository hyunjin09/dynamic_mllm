# READ-harm structure and learnability protocol

- Contract: `eaaef86021f06f032a4e809b3db468fb4ea281f5aa0ce921ee15f8e0e256e4d6`
- Target: `H_R = q_WRITE_ONLY - q_FULL`; positive means READ is harmful.
- Primary: all 15,185 dense states / 1,413 UIDs.
- Secondary: all 35,565 exact routed states / 569 UIDs.
- READ pair only: FULL versus WRITE_ONLY. No WRITE study or deployment router.
- F4-F7 must pass operation-level SDPA reconstruction validation before extraction.
- The routed feature/model is selected by frozen dense OOF Spearman, harmful-AUROC, then fixed group/model tie-breaks.
