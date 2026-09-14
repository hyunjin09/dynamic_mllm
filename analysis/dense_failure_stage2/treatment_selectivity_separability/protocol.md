# Stage-2 treatment-selectivity separability protocol

- Contract: `082c9f459da798c419b50b637c0afb1c1b388b403ea2b51c73092b1df3e44edd`
- Frozen checkpoint: Experiment A final update 3,144, `48536bfbf6ebf62071898aea91952b6fe6a3f13b5723b09f6d814eb0ad9bbc7f`.
- Population: 21,071 unique exact prefix-states from 34,253 successful route-state occurrences and 569 UIDs.
- Labels: 18,438 KEEP_REQUIRED, 1,657 INTERVENE_REQUIRED, 976 MIXED; no search miss is labeled KEEP.
- Evaluation: five UID/image-group-disjoint folds; UID- and class-balanced training, unbalanced held-out metrics, fold-local normalization.
- Inputs: scalar margin, four logits, z_R, z_W, and [z_R;z_W]. No layer/dataset/source/Stage-1 inputs.
- Matched sensitivity: reweight held-out KEEP/INTERVENE within supported dataset/source/layer-bin/route-source-signature cells.
- Scope: diagnostic probes only. No deployment head, router retraining, search, Stage-1 change, or external evaluation.
