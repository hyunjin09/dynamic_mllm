# Population Construction Summary

1. The raw authority is the frozen 10K previous-Qwen all-on manifest; the Stage-1 subset removes 2K DocVQA and keeps fixed GQA 4K, ChartQA 2K, TextVQA 2K quotas.
2. Correct/wrong membership was defined by the previous dense all-on task score, then exact per-dataset C/W quotas were consumed. It was not a natural prevalence sample.
3. One missing ChartQA-C image produces the executable 7,999 = 3,999 C + 4,000 W population.
4. Phase-48 train/val/test are 6,399/800/800 group-disjoint identities from the same quota-selected source mechanism. UID and SHA-256 image-content-group overlap are zero.
5. The canonical 4K was selected outcome-blind from pinned canonical training sources, disjoint from the legacy content population, then retained its observed current outcome skew (3,129 C / 871 W).
6. Observable shifts are fully tabulated in `metadata_distribution.csv`, including categorical image format/source strata. The largest absolute numeric standardized mean differences per dataset are {'gqa': ('W', 'image_size_bytes', 0.6286356951637775), 'chartqa': ('C', 'image_width', 0.8415013991867336), 'textvqa': ('C', 'image_height', 0.33532477885950845)}; none is assumed causal.
7. The original larger pre-quota harvesting pool and six raw quota JSONLs are not present, so natural prevalence and selection rates cannot be reconstructed.
8. Old validation/test success is evidence for same-selection-regime identity generalization; it is not source-shift robustness. Phase-49 and this audit directly show the distinction.
