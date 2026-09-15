# Current-Server 3B Route Corpus Audit — V2, in progress

Source inventory and completed V1 census were verified against unchanged files.
Unique samples: 10,000. Unique image groups: 9,273. Unique routes: 3,907,717. Duplicate routes: 0.

Routes/sample: mean 390.7717, median 400.0, p90 407.0, max 409.

OLD routes: Correct 1,880,345, Wrong 2,027,372. Exactly one immutable dense route per sample. Dataset-specific source counts are in `census/dataset_breakdown.csv`. These are historical labels.

CURRENT dense transitions, routed transitions, Dense-C/W counts, route categories and current geometry eligibility: NOT AVAILABLE; Gate A has not passed. No zero counts or old labels are substituted for missing current results.

Focused one-GPU diagnostic submitted as Slurm job 2941; no full replay submitted.
