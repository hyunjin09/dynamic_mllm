# Frozen split audit

PASS: no UID, image-byte, RGB-pixel, native-image, or connected content-group overlap between CAL and TEST, including MMMU Standard/Vision native question counterparts and all POPE variants. All external image byte hashes match the frozen source manifests. No prior intervention outcomes or correctness strata were used to select the pool. ChartQA/TextVQA retain their full existing evaluation sets. Their train pool is limited to locally available, previously prospectively acquired images/shards; representativeness of this frame is a limitation.

MMMU-Pro and POPE have no matching separate train/dev split in the pinned evaluation contract, so their test results will exclude complete selected calibration groups. Current official dataset cards corroborate the test-only/task-variant configurations: https://huggingface.co/datasets/MMMU/MMMU_Pro/blob/main/README.md and https://huggingface.co/datasets/lmms-lab/POPE/blob/main/README.md . No dataset revision was changed.

Nested pools use whole-group prefixes up to N=32/64/128/256, so actual sizes may fall below the target. Split-half assignment alternates frozen groups and is independent of labels.
