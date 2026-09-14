# Program predictor beam-oracle audit summary

## Result

- Main bottleneck: **generation/representation**.
- Triggered population: **901** = 496 Dense-W + 405 Dense-C.
- Frozen unique candidates executed: **6,944 / 6,944**.
- Mean unique programs per triggered beam: **7.7070**; duplicate-entry rate: **0.0000%**.
- Dense-W top-1 all-FULL / any-non-FULL: **398 / 98** (80.24% / 19.76%).
- Dense-W mean unique programs / mean pairwise Hamming distance: **7.798387096774194 / 1.6630904377880185**.
- W-to-C@1/@2/@4/@8: **3 / 10 / 24 / 33**.
- Additional rescues at ranks 2-8: **30**.
- W top-1 success / ranking failure / generation failure: **3 / 30 / 463**.

## Required answers

1. Stage-1-triggered samples audited: **901**.
2. Triggered Dense-W / Dense-C: **496 / 405**.
3. P(all-FULL | triggered W): **0.802419**.
4. P(any non-FULL | triggered W): **0.197581**.
5. W-to-C@1/@2/@4/@8: **3/10/24/33**.
6. Additional rank-2-8 rescues: **30**.
7. Ranking failures among top-1 W failures: **30/493 (6.09%)**.
8. Generation failures among top-1 W failures: **463/493 (93.91%)**.
9. First-correct-rank counts are in `metrics/first_correct_rank.csv`.
10. Ranking-failure top1-minus-correct score gap median/IQR: **1.886015884578228 / [0.8444309942424297, 2.55528013035655]**.
11. Generation-failure W mean beam Hamming diversity: **1.6618327676643012**.
12. Top-1-wrong versus best-correct median non-FULL actions: **0.0 versus 1.0**.
13. All-FULL is present in **396/405** triggered-C beams.
14. The 8 top-1 C-to-W regressions contain **8 ranking failures** and **0 generation failures**; all-FULL C1/C2 is **6/2**.
15. TextVQA W-to-C@1/@8: **0/19**.
16. MMMU-Pro W-to-C@1/@8: **2/12**.
17. The quantitatively dominant W failure mode is **generation/representation** (30 ranking versus 463 generation failures).
18. Exactly one next experiment is recommended: **one minimal trigger-state representation enrichment experiment**.
19. This oracle audit does not establish a deployable reranker, an inference-time oracle, or prospective benchmark improvement.

Layer-27 samples retain their denominator with exhaustive four-program support; no duplicate ranks were fabricated. External labels were used only after the complete candidate set was frozen.
