# Proposed 6400/800/800 Split Feasibility Audit

No proposed split manifest was created.

The physical population has 7,477 exact image-content groups and therefore
appears large enough to support an approximately 6,400/800/800 group-disjoint
partition. That observation is insufficient to freeze a scientific split:

- 1,083 authoritative source rows and their UID/question/answer/image-group
  metadata are missing;
- the complete current-dense labels are reconstructed rather than physically
  re-audited;
- exact duplicate question/source-example relationships cannot be checked for
  the missing rows;
- the preserved historical P7 split is 7,000/1,000 and has no untouched test
  partition, so it cannot silently substitute for this plan's target.

`proposed_split_manifest.jsonl` is intentionally absent. After the canonical
8K source is restored, use the union of source image-group IDs and image-content
hash equivalence classes, then solve for the nearest group-valid
6,400/800/800 allocation while preserving dataset and correctness proportions.

